"""Milestone 3 - explainable temporal graph-attention network (practical TGAT-style).

Graph
  * node  = one vessel in one 6-h window (the same samples as Milestone 2);
  * edges = (a) proximity interactions inside the window (>= 20 min within 2 km),
            (b) *possible dark encounters* (space-time prism during an AIS gap),
            (c) temporal edges from the same vessel's previous windows (t-6 h, t-12 h).
    Interaction edges carry features (minimum separation, time close, low-speed
    time close, relative speed, parallel motion, port distance); every edge
    carries a learned TGAT-style time encoding of its time lag.
  * Temporal edges only point forward in time, so a node never sees the future.

Model (trained end-to-end with gradient descent)
  node encoder : dilated TCN over the 36-step sequence  +  MLP over window
                 aggregates  +  static vessel attributes
  message pass : 2 x GATv2 layers with edge features and time encoding
                 (residual + LayerNorm), PyTorch Geometric
  heads        : fishing activity (node), suspicious interaction (edge),
                 AIS-gap anomaly (node), neural gap-trajectory decoder, and the
                 final vessel investigation score, which is learned from the node
                 embedding PLUS the other heads' outputs (fishing, gap anomaly,
                 strongest incident interaction, reappearance error) - mirroring
                 the flowchart's "base investigation risk".
Loss = BCE(risk) + BCE(fishing) + BCE(gap, gap windows only) + BCE(edge)
       + 0.3 * Gaussian-NLL(reappearance position); each term is reported.
Training is inductive: only edges between training nodes are used when
fitting; validation/test nodes are scored on the full graph without labels.
Attention weights are NOT treated as explanations (Milestone 4 adds proper XAI).
"""
from __future__ import annotations

import copy
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as Fn
from sklearn.metrics import average_precision_score

from .baselines import WindowTensors, _CausalBlock, set_seed
from .graph_builder import EDGE_FEATURES, window_interaction_edges
from .trajectory import GAP_FEATURES, K_POINTS, build_gap_dataset

try:
    from torch_geometric.nn import GATv2Conv
except ImportError:  # pragma: no cover
    GATv2Conv = None


@dataclass
class GNNConfig:
    hidden: int = 64
    heads: int = 4
    layers: int = 2
    dropout: float = 0.2
    lr: float = 2e-3
    weight_decay: float = 1e-4
    epochs: int = 80
    patience: int = 12
    seeds: tuple = (0, 1, 2)
    temporal_lags: tuple = (1, 2)        # connect to the vessel's windows 6 h and 12 h earlier
    w_fishing: float = 1.0
    w_gap: float = 1.0
    w_edge: float = 1.0
    w_traj: float = 0.3


# ==========================================================================
# Graph construction
# ==========================================================================
def build_graph(d: Dict, wcfg, lags=(1, 2)) -> Dict:
    lab = d["labels"]
    T0 = d["T0"]
    node_of = {k: i for i, k in enumerate(zip(lab.vessel_id, lab.win))}
    src, dst, feat, etype, lag_h = [], [], [], [], []

    def add(a, b, f, t, h):
        src.extend([a, b]); dst.extend([b, a]); feat.extend([f, f]); etype.extend([t, t]); lag_h.extend([h, h])

    # (a) proximity interactions
    inter = window_interaction_edges(d["ais"], d["layers"], T0, wcfg)
    T = wcfg.n_steps
    for r in inter.itertuples(index=False):
        a, b = node_of.get((r.vessel_a, r.win)), node_of.get((r.vessel_b, r.win))
        if a is None or b is None:
            continue
        f = [np.log1p(r.min_dist_m / 1000), r.n_2km / T, r.n_close / T, r.slow_close / T,
             min(r.rel_speed / 10, 2), r.parallel, np.log1p(r.port_min_m / 1000) / 4, 0, 0, 0]
        add(a, b, f, 0, 0.0)
    # (b) possible dark encounters, per overlapped window
    wh = pd.Timedelta(hours=wcfg.window_h)
    for r in d["prism"].itertuples(index=False):
        for w in range((r.first_bin - T0) // wh, (r.last_bin - T0) // wh + 1):
            a, b = node_of.get((r.vessel_a, w)), node_of.get((r.vessel_b, w))
            if a is not None and b is not None:
                add(a, b, [0, 0, 0, 0, 0, 0, 0, 1, 0, min(r.n_bins / T, 1)], 1, 0.0)
    n_inter = len(src)
    # (c) temporal edges (directed, past -> present only)
    for (v, w), i in node_of.items():
        for k in lags:
            j = node_of.get((v, w - k))
            if j is not None:
                src.append(j); dst.append(i); etype.append(2); lag_h.append(k * wcfg.window_h)
                feat.append([0, 0, 0, 0, 0, 0, 0, 0, 1, 0])
    edge_index = np.array([src, dst], dtype=np.int64)
    etype = np.array(etype)
    # suspicious-interaction label: both nodes belong to the same planted rendezvous
    ev_type = dict(zip(d["events"].event_id, d["events"].event_type))
    evs = [set(e for e in s.split(",") if e and ev_type.get(e) == "low_speed_rendezvous") for s in lab.event_ids]
    edge_y = np.array([bool(evs[a] & evs[b]) if t < 2 else False
                       for a, b, t in zip(edge_index[0], edge_index[1], etype)], dtype=np.float32)
    gap_ds = build_gap_dataset(d, node_of, T0, wcfg.window_h)
    agg = d["agg"]
    gap_mask = ((agg.frac_in_gap > 0) | (agg.max_gap_min >= 60)).values
    return {
        "node_of": node_of, "edge_index": edge_index, "edge_attr": np.array(feat, np.float32),
        "edge_type": etype, "edge_lag_h": np.array(lag_h, np.float32), "edge_y": edge_y,
        "n_interaction_edges": n_inter, "y": lab.y.values.astype(np.float32),
        "y_fishing": (lab.fishing_frac.values >= 0.25).astype(np.float32),
        "y_gap": lab.dark_in_window.values.astype(np.float32), "gap_mask": gap_mask, "gap_ds": gap_ds,
    }


def graph_summary(G: Dict) -> pd.Series:
    et = G["edge_type"]
    return pd.Series({
        "nodes (vessel-windows)": len(G["y"]),
        "proximity edges (directed)": int((et == 0).sum()),
        "possible-dark-encounter edges (directed)": int((et == 1).sum()),
        "temporal edges": int((et == 2).sum()),
        "rendezvous-labelled interaction edges": int(G["edge_y"].sum()),
        "at-sea gaps for trajectory head": len(G["gap_ds"]["gaps"]),
    })


# ==========================================================================
# Model
# ==========================================================================
class TimeEncoding(nn.Module):
    """TGAT-style learnable harmonic encoding of a time lag (hours)."""

    def __init__(self, dim: int = 8):
        super().__init__()
        self.w = nn.Parameter(torch.logspace(0, -2, dim))
        self.b = nn.Parameter(torch.zeros(dim))

    def forward(self, dt_h):
        return torch.cos(dt_h[:, None] * self.w[None, :] + self.b[None, :])


class GapDecoder(nn.Module):
    """Neural gap-trajectory decoder: K cumulative position offsets + endpoint uncertainty.

    Per-step displacements are bounded by a 15-knot speed limit, so every
    predicted path is physically reachable."""

    def __init__(self, n_in: int, h_node: int, hidden: int = 64, k: int = K_POINTS):
        super().__init__()
        self.k = k
        self.net = nn.Sequential(nn.Linear(n_in + h_node + 1, hidden), nn.GELU(),
                                 nn.Linear(hidden, hidden), nn.GELU())
        self.steps = nn.Linear(hidden, 2 * k)
        self.logsig = nn.Linear(hidden, 2)
        nn.init.constant_(self.logsig.bias, float(np.log(10.0)))   # start with a ~10 km uncertainty

    def forward(self, f, h_prev, has_prev, dur_h):
        z = self.net(torch.cat([f, h_prev * has_prev[:, None], has_prev[:, None]], -1))
        vmax = 15 * 1.852 * dur_h / self.k                          # km per step at 15 kn
        steps = torch.tanh(self.steps(z)).view(-1, self.k, 2) * vmax[:, None, None]
        path = steps.cumsum(1)
        logsig = self.logsig(z).clamp(np.log(0.3), np.log(150.0))
        return path, logsig


class TemporalGAT(nn.Module):
    def __init__(self, n_seq: int, n_agg: int, n_edge: int, n_gapf: int, cfg: GNNConfig, use_graph: bool = True):
        super().__init__()
        if GATv2Conv is None:
            raise ImportError("torch_geometric is required: pip install torch_geometric")
        H = cfg.hidden
        self.use_graph = use_graph
        self.tcn = nn.Sequential(_CausalBlock(n_seq, H // 2, 3, 1, cfg.dropout), _CausalBlock(H // 2, H // 2, 3, 2, cfg.dropout),
                                 _CausalBlock(H // 2, H // 2, 3, 4, cfg.dropout), _CausalBlock(H // 2, H // 2, 3, 8, cfg.dropout))
        self.seq_proj = nn.Linear(H, H)
        self.agg_enc = nn.Sequential(nn.Linear(n_agg, H), nn.LayerNorm(H), nn.GELU(), nn.Dropout(cfg.dropout), nn.Linear(H, H))
        self.norm0 = nn.LayerNorm(H)
        self.tenc = TimeEncoding(8)
        self.convs = nn.ModuleList([GATv2Conv(H, H, heads=cfg.heads, concat=False, edge_dim=n_edge + 8,
                                              dropout=cfg.dropout, add_self_loops=True, fill_value="mean")
                                    for _ in range(cfg.layers)])
        self.norms = nn.ModuleList([nn.LayerNorm(H) for _ in range(cfg.layers)])
        self.drop = nn.Dropout(cfg.dropout)
        self.head_fish = nn.Linear(H, 1)
        self.head_gap = nn.Linear(H, 1)
        self.head_edge = nn.Sequential(nn.Linear(2 * H + n_edge, H), nn.GELU(), nn.Linear(H, 1))
        self.decoder = GapDecoder(n_gapf, H)
        self.head_risk = nn.Sequential(nn.Linear(H + 6, H), nn.GELU(), nn.Dropout(cfg.dropout), nn.Linear(H, 1))

    def encode(self, seq, agg):
        z = self.tcn(seq.transpose(1, 2))
        hs = self.seq_proj(torch.cat([z.mean(-1), z.amax(-1)], -1))
        return self.norm0(hs + self.agg_enc(agg))

    def forward(self, g: Dict, edge_keep: Optional[torch.Tensor] = None, return_attention: bool = False):
        h = self.encode(g["seq"], g["agg"])
        ei, ea, lag = g["edge_index"], g["edge_attr"], g["edge_lag_h"]
        if edge_keep is not None:
            ei, ea, lag = ei[:, edge_keep], ea[edge_keep], lag[edge_keep]
        if not self.use_graph:
            ei, ea, lag = ei[:, :0], ea[:0], lag[:0]
        eattr = torch.cat([ea, self.tenc(lag)], -1)
        att = []
        for conv, norm in zip(self.convs, self.norms):
            if return_attention:
                m, (aei, aw) = conv(h, ei, eattr, return_attention_weights=True)
                att.append((aei, aw))
            else:
                m = conv(h, ei, eattr)
            h = norm(h + self.drop(Fn.elu(m)))
        out = {"h": h, "fish": self.head_fish(h).squeeze(-1), "gap": self.head_gap(h).squeeze(-1)}
        # suspicious-interaction head on interaction edges (types 0/1)
        et = g["edge_type"] if edge_keep is None else g["edge_type"][edge_keep]
        inter = et < 2 if self.use_graph else torch.zeros_like(et, dtype=torch.bool)
        ii = ei[:, inter] if self.use_graph else ei
        e_logit = self.head_edge(torch.cat([h[ii[0]], h[ii[1]], ea[inter] if self.use_graph else ea], -1)).squeeze(-1)
        out["edge_logit"], out["edge_index_inter"] = e_logit, ii
        e_node = torch.zeros(h.shape[0], device=h.device)
        if e_logit.numel():
            e_node = e_node.scatter_reduce(0, ii[1], torch.sigmoid(e_logit), reduce="amax", include_self=True)
        # gap-trajectory decoder (causal: previous-window embedding only)
        gd = g["gap"]
        prev = gd["prev_node"]
        has_prev = (prev >= 0).float()
        h_prev = h[prev.clamp(min=0)]
        path, logsig = self.decoder(gd["F"], h_prev, has_prev, gd["dur_h"])
        out["gap_path"], out["gap_logsig"] = path, logsig
        z = (((gd["end_km"] - path[:, -1]) / logsig.exp()) ** 2).sum(-1).sqrt().detach()
        err = (gd["end_km"] - path[:, -1]).norm(dim=-1).detach()
        z_node = torch.zeros(h.shape[0], device=h.device).scatter_reduce(
            0, gd["span_node"], z[gd["span_gap"]], reduce="amax", include_self=True)
        err_node = torch.zeros(h.shape[0], device=h.device).scatter_reduce(
            0, gd["span_node"], err[gd["span_gap"]], reduce="amax", include_self=True)
        has_gap = torch.zeros(h.shape[0], device=h.device).index_fill(0, gd["span_node"], 1.0)
        side = torch.stack([torch.sigmoid(out["fish"]).detach(), torch.sigmoid(out["gap"]).detach(), e_node.detach(),
                            torch.log1p(z_node), torch.log1p(err_node) / 3, has_gap], -1)
        out["risk"] = self.head_risk(torch.cat([h, side], -1)).squeeze(-1)
        out["side"] = side
        if return_attention:
            out["attention"] = att
        return out


# ==========================================================================
# Training
# ==========================================================================
def to_torch_graph(G: Dict, T: WindowTensors, device) -> Dict:
    gd = G["gap_ds"]
    tt = lambda a, dt=torch.float32: torch.as_tensor(a, dtype=dt, device=device)  # noqa: E731
    return {
        "seq": T.seq.to(device), "agg": T.agg.to(device),
        "edge_index": tt(G["edge_index"], torch.long), "edge_attr": tt(G["edge_attr"]),
        "edge_lag_h": tt(G["edge_lag_h"]), "edge_type": tt(G["edge_type"], torch.long),
        "gap": {"F": tt(gd["F"]), "end_km": tt(gd["end_km"]), "dur_h": tt(gd["dur_h"]),
                "prev_node": tt(gd["prev_node"], torch.long), "span_gap": tt(gd["span_gap"], torch.long),
                "span_node": tt(gd["span_node"], torch.long)},
    }


def _pw(y):
    p = float(y.sum())
    return torch.tensor((len(y) - p) / max(p, 1.0))


def train_tgat(G: Dict, T: WindowTensors, split: np.ndarray, cfg: GNNConfig, seed: int, device,
               use_graph: bool = True, ckpt: Optional[Path] = None, verbose: bool = False) -> Dict:
    set_seed(seed)
    g = to_torch_graph(G, T, device)
    n_edge, n_gapf = G["edge_attr"].shape[1], G["gap_ds"]["F"].shape[1]
    model = TemporalGAT(T.n_seq, T.n_agg, n_edge, n_gapf, cfg, use_graph).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="max", factor=0.5, patience=5)
    is_tr = torch.as_tensor(split == "train", device=device)
    va_idx = np.flatnonzero(split == "val")
    ei = g["edge_index"]
    keep_train = is_tr[ei[0]] & is_tr[ei[1]]                       # inductive: train-only subgraph
    y = torch.as_tensor(G["y"], device=device)
    yf = torch.as_tensor(G["y_fishing"], device=device)
    yg = torch.as_tensor(G["y_gap"], device=device)
    gmask = torch.as_tensor(np.array(G["gap_mask"], dtype=bool), device=device) & is_tr
    ey_all = torch.as_tensor(G["edge_y"], device=device)
    gap_end = torch.as_tensor(G["gap_ds"]["end_node"], device=device)
    gap_tr = (gap_end >= 0) & is_tr[gap_end.clamp(min=0)]
    bce = lambda logit, target, pw: Fn.binary_cross_entropy_with_logits(logit, target, pos_weight=pw.to(device))  # noqa
    pw_r, pw_f = _pw(G["y"][split == "train"]), _pw(G["y_fishing"][split == "train"])
    pw_g = _pw(G["y_gap"][(split == "train") & G["gap_mask"]])
    et_np = G["edge_type"]
    tr_np = split == "train"
    ktr = tr_np[G["edge_index"][0]] & tr_np[G["edge_index"][1]] & (et_np < 2)
    pw_e = _pw(G["edge_y"][ktr])
    hist, best, best_state, bad = [], -1.0, None, 0
    t0 = time.time()
    for ep in range(1, cfg.epochs + 1):
        model.train()
        opt.zero_grad()
        out = model(g, keep_train)
        l_risk = bce(out["risk"][is_tr], y[is_tr], pw_r)
        l_fish = bce(out["fish"][is_tr], yf[is_tr], pw_f)
        l_gap = bce(out["gap"][gmask], yg[gmask], pw_g) if gmask.any() else torch.zeros((), device=device)
        et_kept = g["edge_type"][keep_train]
        ey = ey_all[keep_train][et_kept < 2]
        l_edge = bce(out["edge_logit"], ey, pw_e) if (use_graph and out["edge_logit"].numel()) else torch.zeros((), device=device)
        gd = g["gap"]
        mu, ls = out["gap_path"][gap_tr, -1], out["gap_logsig"][gap_tr]
        l_traj = (0.5 * ((gd["end_km"][gap_tr] - mu) / ls.exp()) ** 2 + ls).sum(-1).mean() if gap_tr.any() \
            else torch.zeros((), device=device)
        loss = l_risk + cfg.w_fishing * l_fish + cfg.w_gap * l_gap + cfg.w_edge * l_edge + cfg.w_traj * l_traj
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        opt.step()
        model.eval()
        with torch.no_grad():
            o = model(g)
            pv = torch.sigmoid(o["risk"]).cpu().numpy()[va_idx]
        vap = average_precision_score(G["y"][va_idx], pv)
        sched.step(vap)
        hist.append({"epoch": ep, "loss_total": loss.item(), "loss_risk": l_risk.item(), "loss_fishing": l_fish.item(),
                     "loss_gap": l_gap.item(), "loss_edge": l_edge.item(), "loss_traj": l_traj.item(),
                     "val_pr_auc": vap})
        if verbose:
            print(f"ep{ep:02d} total {loss.item():.3f} risk {l_risk.item():.3f} val PR-AUC {vap:.3f}")
        if vap > best + 1e-4:
            best, best_state, bad = vap, copy.deepcopy(model.state_dict()), 0
        else:
            bad += 1
            if bad >= cfg.patience:
                break
    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        o = model(g)
    if ckpt is not None:
        Path(ckpt).parent.mkdir(parents=True, exist_ok=True)
        torch.save({"state_dict": best_state, "config": asdict(cfg), "seed": seed, "use_graph": use_graph,
                    "best_val_pr_auc": best}, ckpt)
    return {
        "model": model, "history": pd.DataFrame(hist), "best_val_pr_auc": best, "seconds": time.time() - t0,
        "n_params": sum(p.numel() for p in model.parameters()),
        "risk": torch.sigmoid(o["risk"]).cpu().numpy(), "fish": torch.sigmoid(o["fish"]).cpu().numpy(),
        "gap": torch.sigmoid(o["gap"]).cpu().numpy(), "edge": torch.sigmoid(o["edge_logit"]).cpu().numpy(),
        "edge_index_inter": o["edge_index_inter"].cpu().numpy(),
        "gap_path": o["gap_path"].cpu().numpy(), "gap_logsig": o["gap_logsig"].cpu().numpy(),
        "side": o["side"].cpu().numpy(),
    }
