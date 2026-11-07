"""Milestone 4 - explanation layer for the real-data temporal graph model (notebook 05).

For one alert (a vessel-week with a high investigation priority) it answers:

  * WHICH INPUTS drove the score?    Integrated Gradients on the node features of the
                                     alert's 2-hop computation subgraph, grouped into
                                     human-readable feature groups and three roles:
                                     the vessel this week / its own past weeks / other vessels.
  * DID ITS HISTORY MATTER?          temporal masking: remove each past week (and all of
                                     them) from the graph and re-score.
  * DID OTHER VESSELS MATTER?        exact edge occlusion of encounter and co-location
                                     edges (leave-one-out and all at once).
  * WHAT WOULD CHANGE THE SCORE?     counterfactuals: "had it not done X this week"
                                     (that feature group set to zero) -> new priority.
  * CAN THE EXPLANATION BE TRUSTED?  completeness (IG sums to the score change), deletion
                                     fidelity (removing the top-attributed inputs must drop
                                     the score more than removing random ones), sparsity,
                                     and stability across independently trained seeds.

Exact occlusion is used for edges instead of a learned GNNExplainer mask: the computation
subgraph of one vessel-week is small, so every edge can be removed and re-scored exactly.
Attention weights are never used as explanations.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd
import torch

from .real_model import COUNT_FEATURES, EDGE_COLS, REAL_MODEL_CAVEAT, TASKS

FEATURE_GROUPS = {
    "fishing outside its flag state's EEZ": ["fishing_outside_flag_h"],
    "fishing inside a protected area": ["fishing_mpa_h", "n_in_mpa"],
    "fishing in a non-Indian EEZ": ["fishing_foreign_h"],
    "fishing effort": ["n_fishing", "fishing_h"],
    "AIS-off gaps": ["n_gaps", "gap_h", "n_long_gaps", "n_extreme_gaps", "max_gap_km", "max_implied_speed_kn",
                     "min_sat_ppd"],
    "encounters": ["n_enc", "n_carrier_enc", "n_fishing_enc", "enc_min_dist_km"],
    "loitering": ["n_loiter", "loiter_h"],
    "port visits": ["n_port", "port_h"],
    "distance from shore / port": ["dist_shore_km", "dist_port_km"],
    "location": ["lat", "lon"],
    "season": ["season_sin", "season_cos"],
    "vessel type": [],            # class_* columns
    "overall activity": ["n_events"],
    "weeks since last active": ["weeks_since_prev"],
}
# groups a vessel can "not do" in a counterfactual (behaviours, not identity/place/time)
ACTIONABLE = ["fishing outside its flag state's EEZ", "fishing inside a protected area", "fishing in a non-Indian EEZ",
              "fishing effort", "AIS-off gaps", "encounters", "loitering", "port visits"]
ROLES = ["this vessel, this week", "this vessel, past weeks", "other vessels"]


def feature_group(col: str) -> str:
    base = col[:-len("_missing")] if col.endswith("_missing") else col
    if base.startswith("class_"):
        return "vessel type"
    for g, cols in FEATURE_GROUPS.items():
        if base in cols:
            return g
    return "other"


def fit_platt(P: np.ndarray, G: Dict, split: np.ndarray, tasks: Sequence[str]):
    """Per-task Platt parameters (a, b) on validation logits, so priority is differentiable."""
    from sklearn.linear_model import LogisticRegression
    a, b = np.ones(len(tasks)), np.zeros(len(tasks))
    va = split == "val"
    for j, t in enumerate(tasks):
        k = list(TASKS).index(t)
        r = va & G["M"][:, k]
        y = G["Y"][r, k]
        if 0 < y.sum() < len(y):
            z = np.log(np.clip(P[r, j], 1e-6, 1)) - np.log(np.clip(1 - P[r, j], 1e-6, 1))
            lr = LogisticRegression(C=1.0).fit(z[:, None], y)
            a[j], b[j] = lr.coef_[0, 0], lr.intercept_[0]
    return a, b


@dataclass
class Subgraph:
    node: int                 # global id of the alert node
    subset: np.ndarray        # global ids of subgraph nodes
    center: int               # local id of the alert node
    ei: torch.Tensor          # local edge index
    ea: torch.Tensor
    lag: torch.Tensor
    etype: np.ndarray
    x: torch.Tensor           # standardised features of subgraph nodes
    x0: torch.Tensor          # baseline (training mean = 0 after standardisation)
    zero_raw: torch.Tensor    # standardised value of a raw 0 for each feature (for counterfactuals)
    roles: np.ndarray         # role index per subgraph node


class AlertExplainer:
    def __init__(self, G: Dict, model, Z: np.ndarray, stats: Dict, tasks: Sequence[str], platt, weights: Dict,
                 device, layers: int = 2):
        self.G, self.model, self.tasks = G, model.eval(), list(tasks)
        self.device = device
        self.Z = torch.as_tensor(Z, device=device)
        self.cols = list(G["X_raw"].columns)
        self.groups = np.array([feature_group(c) for c in self.cols])
        self.a = torch.as_tensor(platt[0], dtype=torch.float32, device=device)
        self.b = torch.as_tensor(platt[1], dtype=torch.float32, device=device)
        self.w = torch.as_tensor([weights.get(t, 0.0) for t in self.tasks], dtype=torch.float32, device=device)
        self.ei = torch.as_tensor(G["edge_index"], device=device)
        self.ea = torch.as_tensor(G["edge_attr"], device=device)
        self.lag = torch.as_tensor(G["edge_lag"], device=device)
        self.layers = layers
        mu, sd = np.asarray(stats["mean"]), np.asarray(stats["sd"])
        self.zero_raw = torch.as_tensor(np.clip((0 - mu) / sd, -10, 10), dtype=torch.float32, device=device)
        nd = G["nodes"]
        self.vessel, self.week = nd.vessel_id.values, nd.week.values

    # ---------------------------------------------------------------- core
    def subgraph(self, node: int) -> Subgraph:
        from torch_geometric.utils import k_hop_subgraph
        subset, ei, mapping, emask = k_hop_subgraph(int(node), self.layers, self.ei, relabel_nodes=True,
                                                    num_nodes=self.G["n"], flow="source_to_target")
        sub = subset.cpu().numpy()
        roles = np.where(self.vessel[sub] == self.vessel[node],
                         np.where(sub == node, 0, 1), 2)
        x = self.Z[subset]
        em = emask.cpu().numpy()
        return Subgraph(int(node), sub, int(mapping[0]), ei, self.ea[emask], self.lag[emask],
                        self.G["edge_type"][em], x, torch.zeros_like(x), self.zero_raw, roles)

    def priority(self, s: Subgraph, x=None, keep: Optional[np.ndarray] = None):
        x = s.x if x is None else x
        ei, ea, lag = s.ei, s.ea, s.lag
        if keep is not None:
            k = torch.as_tensor(keep, device=self.device)
            ei, ea, lag = ei[:, k], ea[k], lag[k]
        logit = self.model(x, ei, ea, lag)[s.center]
        p = torch.sigmoid(self.a * logit + self.b)
        return 1 - torch.exp((self.w * torch.log(torch.clamp(1 - p, min=1e-9))).sum(-1))

    def per_task(self, s: Subgraph) -> Dict[str, float]:
        with torch.no_grad():
            logit = self.model(s.x, s.ei, s.ea, s.lag)[s.center]
            p = torch.sigmoid(self.a * logit + self.b).cpu().numpy()
        return {t: round(float(v), 4) for t, v in zip(self.tasks, p)}

    # ---------------------------------------------------------------- attributions
    def integrated_gradients(self, s: Subgraph, steps: int = 48) -> np.ndarray:
        diff = s.x - s.x0
        total = torch.zeros_like(s.x)
        for alpha in (torch.arange(steps, device=self.device, dtype=torch.float32) + 0.5) / steps:
            xi = (s.x0 + alpha * diff).detach().requires_grad_(True)
            pr = self.priority(s, xi)
            (g,) = torch.autograd.grad(pr, xi)
            total += g
        return (diff * total / steps).detach().cpu().numpy()

    def attribution_table(self, s: Subgraph, ig: np.ndarray) -> pd.DataFrame:
        rows = []
        for r, role in enumerate(ROLES):
            m = s.roles == r
            if not m.any():
                continue
            per_col = ig[m].sum(0)
            for g in pd.unique(self.groups):
                rows.append({"role": role, "group": g, "contribution": float(per_col[self.groups == g].sum())})
        t = pd.DataFrame(rows)
        return t.reindex(t.contribution.abs().sort_values(ascending=False).index).reset_index(drop=True)

    # ---------------------------------------------------------------- masking / occlusion
    def temporal_masking(self, s: Subgraph, base: float) -> pd.DataFrame:
        rows = []
        src = s.ei[0].cpu().numpy()
        dst = s.ei[1].cpu().numpy()
        hist = np.flatnonzero(s.roles == 1)
        with torch.no_grad():
            for h in hist:
                keep = (src != h) & (dst != h)
                rows.append({"removed": f"past week {pd.Timestamp(self.week[s.subset[h]]).date()}",
                             "priority": float(self.priority(s, keep=keep)), "delta": 0.0})
            if len(hist):
                keep = ~(np.isin(src, hist) | np.isin(dst, hist))
                rows.append({"removed": "all past weeks", "priority": float(self.priority(s, keep=keep)), "delta": 0.0})
        t = pd.DataFrame(rows, columns=["removed", "priority", "delta"])
        t["delta"] = t.priority - base
        return t

    def edge_occlusion(self, s: Subgraph, base: float, max_edges: int = 400) -> pd.DataFrame:
        names = {0: "temporal", 1: "encounter", 2: "co-location"}
        rows = []
        n = s.ei.shape[1]
        with torch.no_grad():
            for t, name in names.items():
                m = s.etype == t
                if m.any():
                    rows.append({"removed": f"all {name} edges ({int(m.sum())})",
                                 "priority": float(self.priority(s, keep=~m))})
            social = np.flatnonzero(s.etype > 0)
            if 0 < len(social) <= max_edges:
                best = None
                for e in social:
                    keep = np.ones(n, bool)
                    keep[e] = False
                    p = float(self.priority(s, keep=keep))
                    if best is None or abs(p - base) > abs(best[1] - base):
                        best = (e, p)
                if best is not None:
                    e = best[0]
                    other = self.vessel[s.subset[int(s.ei[0, e])]]
                    rows.append({"removed": f"single most influential {names[int(s.etype[e])]} edge (from {other})",
                                 "priority": best[1]})
            rows.append({"removed": "every edge (vessel scored alone)",
                         "priority": float(self.priority(s, keep=np.zeros(n, bool)))})
        t = pd.DataFrame(rows)
        t["delta"] = t.priority - base
        return t

    def counterfactuals(self, s: Subgraph, base: float) -> pd.DataFrame:
        rows = []
        with torch.no_grad():
            for g in ACTIONABLE:
                cols = np.flatnonzero(self.groups == g)
                cols = [c for c in cols if not self.cols[c].endswith("_missing")]
                if not cols:
                    continue
                raw_vals = self.G["X_raw"].iloc[s.node, cols].values
                if np.all(raw_vals == 0):
                    continue      # the vessel did not do this - nothing to take away
                x = s.x.clone()
                x[s.center, cols] = s.zero_raw[cols]
                rows.append({"had it not (this week)": g, "priority": float(self.priority(s, x))})
        t = pd.DataFrame(rows, columns=["had it not (this week)", "priority"])
        t["delta"] = t.priority - base
        return t.sort_values("delta").reset_index(drop=True)

    # ---------------------------------------------------------------- quality checks
    def deletion_fidelity(self, s: Subgraph, ig: np.ndarray, k: int = 5, n_random: int = 20, seed: int = 0):
        """Replace the k most positively attributed inputs with the baseline; compare with k random inputs."""
        rng = np.random.default_rng(seed)
        flat = ig.ravel()
        with torch.no_grad():
            base = float(self.priority(s))
            top = np.argsort(-flat)[:k]
            x = s.x.clone().view(-1)
            x[top] = s.x0.view(-1)[top]
            drop_top = base - float(self.priority(s, x.view_as(s.x)))
            nz = np.flatnonzero(np.abs(s.x.cpu().numpy().ravel() - s.x0.cpu().numpy().ravel()) > 1e-6)
            rnd = []
            for _ in range(n_random):
                pick = rng.choice(nz, size=min(k, len(nz)), replace=False)
                x = s.x.clone().view(-1)
                x[pick] = s.x0.view(-1)[pick]
                rnd.append(base - float(self.priority(s, x.view_as(s.x))))
        return {"drop_top_k": drop_top, "drop_random_k": float(np.mean(rnd)), "k": k}

    @staticmethod
    def sparsity(ig: np.ndarray, share: float = 0.8) -> float:
        """Fraction of inputs needed to cover `share` of the total absolute attribution (lower = sparser)."""
        a = np.sort(np.abs(ig.ravel()))[::-1]
        if a.sum() == 0:
            return 0.0
        return float((np.searchsorted(np.cumsum(a) / a.sum(), share) + 1) / len(a))

    # ---------------------------------------------------------------- full card
    def explain(self, node: int, steps: int = 48) -> Dict:
        s = self.subgraph(node)
        with torch.no_grad():
            base = float(self.priority(s))
            ref = float(self.priority(s, s.x0))
        ig = self.integrated_gradients(s, steps)
        att = self.attribution_table(s, ig)
        card = {
            "node": int(node), "priority": base, "baseline_priority": ref,
            "ig_sum": float(ig.sum()), "completeness_error": float(abs(ig.sum() - (base - ref))),
            "per_task": self.per_task(s), "attribution": att,
            "temporal": self.temporal_masking(s, base), "edges": self.edge_occlusion(s, base),
            "counterfactual": self.counterfactuals(s, base),
            "fidelity": self.deletion_fidelity(s, ig), "sparsity_80pct": self.sparsity(ig),
            "subgraph_nodes": len(s.subset), "subgraph_edges": int(s.ei.shape[1]),
            "n_past_weeks": int((s.roles == 1).sum()), "n_other_vessels": int(len(set(self.vessel[s.subset[s.roles == 2]]))),
        }
        card["reason"] = reason_text(self.G["nodes"], s, card)
        return card


# ==========================================================================
# Plain-language reasons
# ==========================================================================
_DESCRIBE = {
    "fishing outside its flag state's EEZ": ("fishing_outside_flag_h", "fished {v:.0f} h in an EEZ that is not its flag state's"),
    "fishing inside a protected area": ("fishing_mpa_h", "fished {v:.0f} h inside a marine protected area"),
    "fishing in a non-Indian EEZ": ("fishing_foreign_h", "fished {v:.0f} h in a non-Indian EEZ"),
    "encounters": ("n_enc", "had {v:.0f} encounter(s) with other vessels"),
    "AIS-off gaps": ("n_gaps", "had {v:.0f} AIS-off gap(s)"),
    "loitering": ("loiter_h", "loitered for {v:.0f} h"),
    "fishing effort": ("fishing_h", "fished {v:.0f} h in total"),
    "port visits": ("n_port", "made {v:.0f} port visit(s)"),
}


def reason_text(nodes: pd.DataFrame, s: Subgraph, card: Dict) -> str:
    att = card["attribution"]
    pos = att[att.contribution > 0]
    parts = []
    for r in pos.head(3).itertuples():
        if r.group in _DESCRIBE:
            col, tmpl = _DESCRIBE[r.group]
            if r.role == ROLES[0]:
                v = nodes[col].values[s.node]
                if v > 0:
                    parts.append("this week it " + tmpl.format(v=v))
            elif r.role == ROLES[1]:
                hist = s.subset[s.roles == 1]
                k = int((nodes[col].values[hist] > 0).sum())
                if k:
                    parts.append(f"in {k} of its {len(hist)} previous active weeks it also showed {r.group}")
            else:
                parts.append(f"vessels it was linked to showed {r.group}")
        elif r.role == ROLES[2]:
            parts.append(f"the {r.group} of vessels it was linked to")
        else:
            ctx = {"season": "the time of year", "location": "where it was operating",
                   "vessel type": "its vessel type", "distance from shore / port": "how far from shore it was",
                   "overall activity": "its overall activity level", "weeks since last active": "how recently it was active"}
            when = "this week" if r.role == ROLES[0] else "in past weeks"
            parts.append(f"{ctx.get(r.group, r.group)} ({when})")
    cf = card["counterfactual"]
    tail = ""
    if len(cf) and cf.delta.iloc[0] < -0.02:
        tail = (f" Without the {cf['had it not (this week)'].iloc[0]} this week, the priority would fall from "
                f"{card['priority']:.2f} to {cf.priority.iloc[0]:.2f}.")
    head = "Flagged mainly because " + ("; ".join(dict.fromkeys(parts)) if parts else "of its overall activity pattern")
    return head + "." + tail


def card_to_json(card: Dict, nodes: pd.DataFrame, alert_row: Optional[Dict] = None) -> Dict:
    i = card["node"]
    att = card["attribution"]
    out = {
        "vessel_id": nodes.vessel_id.values[i], "vessel_class": nodes.vessel_class.values[i],
        "week": str(pd.Timestamp(nodes.week.values[i]).date()),
        "approx_position": [round(float(nodes.lat.values[i]), 1), round(float(nodes.lon.values[i]), 1)],
        "investigation_priority": round(card["priority"], 3),
        "forecast_next_week": card["per_task"],
        "reason": card["reason"],
        "top_contributions": att.head(6).round(4).to_dict("records"),
        "history_effect": card["temporal"].round(4).to_dict("records"),
        "other_vessels_effect": card["edges"].round(4).to_dict("records"),
        "counterfactuals": card["counterfactual"].round(4).to_dict("records"),
        "explanation_quality": {"completeness_error": round(card["completeness_error"], 4),
                                "deletion_drop_top5": round(card["fidelity"]["drop_top_k"], 4),
                                "deletion_drop_random5": round(card["fidelity"]["drop_random_k"], 4),
                                "inputs_for_80pct_of_attribution": round(card["sparsity_80pct"], 4)},
        "caveat": "Investigation priority only - not evidence of illegality. Requires human verification. "
                  "Explanation describes the model, not the vessel's intent.",
    }
    if alert_row is not None:
        out["observed_next_week (evaluation only)"] = alert_row.get("next_week_observed")
    return out


def save_cards(cards: List[Dict], path: Path, meta: Dict):
    Path(path).write_text(json.dumps({"caveat": REAL_MODEL_CAVEAT, "meta": meta, "alerts": cards}, indent=2,
                                     default=str))


# ==========================================================================
# Global importance and stability
# ==========================================================================
def group_permutation_importance(predict, G: Dict, rows: np.ndarray, tasks: Sequence[str], X: np.ndarray,
                                 repeats: int = 3, seed: int = 0) -> pd.DataFrame:
    """Drop in macro PR-AUC on `rows` when one feature group is shuffled across rows (all nodes)."""
    from sklearn.metrics import average_precision_score
    rng = np.random.default_rng(seed)
    cols = [list(TASKS).index(t) for t in tasks]
    groups = np.array([feature_group(c) for c in G["X_raw"].columns])

    def score(P):
        aps = []
        for j, k in enumerate(cols):
            r = rows & G["M"][:, k]
            y = G["Y"][r, k]
            if 0 < y.sum() < len(y):
                aps.append(average_precision_score(y, P[r, j]))
        return float(np.mean(aps))

    base = score(predict(X))
    out = []
    for g in pd.unique(groups):
        idx = np.flatnonzero(groups == g)
        drops = []
        for _ in range(repeats):
            Xp = X.copy()
            perm = rng.permutation(len(X))
            Xp[:, idx] = X[perm][:, idx]
            drops.append(base - score(predict(Xp)))
        out.append({"group": g, "importance": float(np.mean(drops)), "sd": float(np.std(drops))})
    return pd.DataFrame(out).sort_values("importance", ascending=False).reset_index(drop=True)


def seed_stability(tables: List[pd.DataFrame]) -> float:
    """Mean pairwise Spearman correlation of (role, group) attributions across seeds."""
    from scipy.stats import spearmanr
    keyed = [t.set_index(["role", "group"]).contribution for t in tables]
    idx = keyed[0].index
    for k in keyed[1:]:
        idx = idx.union(k.index)
    vals = [k.reindex(idx).fillna(0).values for k in keyed]
    rs = [spearmanr(vals[i], vals[j]).correlation for i in range(len(vals)) for j in range(i + 1, len(vals))]
    rs = [r for r in rs if np.isfinite(r)]
    return float(np.mean(rs)) if rs else float("nan")
