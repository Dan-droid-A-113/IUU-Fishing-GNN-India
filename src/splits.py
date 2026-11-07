"""Leakage-safe train / validation / test splitting (brief section 8.10 and 13).

* vessel-disjoint: whole vessels go to one split; vessels linked by an
  encounter (edge) are grouped so an interaction never straddles splits.
* time-disjoint: contiguous time blocks with an embargo buffer; segments
  (voyages) that overlap a boundary or buffer are dropped, never cut.
* region-disjoint: whole region_ids held out.
No voyage/segment ever appears in two splits.
"""
from __future__ import annotations

from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np
import pandas as pd


def connected_groups(vessels: Sequence[str], pairs: Iterable[Tuple[str, str]]) -> Dict[str, int]:
    """Union-find: vessel -> group id; vessels sharing an encounter share a group."""
    parent = {v: v for v in vessels}

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for a, b in pairs:
        if a in parent and b in parent:
            ra, rb = find(a), find(b)
            if ra != rb:
                parent[rb] = ra
    roots = {v: find(v) for v in vessels}
    ids = {r: i for i, r in enumerate(sorted(set(roots.values())))}
    return {v: ids[r] for v, r in roots.items()}


def vessel_disjoint_split(vessels: Sequence[str], pairs: Iterable[Tuple[str, str]] = (),
                          strata: Dict[str, int] = None, fractions=(0.6, 0.2, 0.2),
                          seed: int = 42) -> Dict[str, str]:
    """Assign encounter-connected vessel groups to train/val/test.

    `strata` (optional) marks vessels of interest (e.g. 1 = has any positive
    label); groups are sorted so each split receives a share of them.
    """
    rng = np.random.default_rng(seed)
    grp = connected_groups(list(vessels), pairs)
    gdf = pd.DataFrame({"vessel": list(grp), "group": list(grp.values())})
    gdf["stratum"] = gdf.vessel.map(strata or {}).fillna(0).astype(int)
    groups = gdf.groupby("group").agg(size=("vessel", "size"), stratum=("stratum", "max")).reset_index()
    names = ["train", "val", "test"]
    assign = {}
    for s, sub in groups.groupby("stratum"):
        sub = sub.sample(frac=1.0, random_state=int(rng.integers(1e9)))
        total = sub["size"].sum()
        filled = {n: 0 for n in names}
        for g, size in zip(sub.group, sub["size"]):
            # greedy: put group where the split is furthest below its target share
            deficit = {n: fractions[i] * total - filled[n] for i, n in enumerate(names)}
            n = max(deficit, key=deficit.get)
            assign[g] = n
            filled[n] += size
    return {v: assign[g] for v, g in grp.items()}


def time_disjoint_split(segments: pd.DataFrame, boundaries: Sequence[pd.Timestamp],
                        embargo: pd.Timedelta = pd.Timedelta("6h")) -> pd.Series:
    """segments: one row per segment with columns segment_id, start, end.

    boundaries: [t_train_end, t_val_end]. Returns split per segment_id
    ('train'/'val'/'test'/'dropped'); segments touching a boundary +/- embargo are dropped.
    """
    b1, b2 = boundaries
    out = {}
    for r in segments.itertuples(index=False):
        if r.end < b1 - embargo:
            out[r.segment_id] = "train"
        elif r.start > b1 + embargo and r.end < b2 - embargo:
            out[r.segment_id] = "val"
        elif r.start > b2 + embargo:
            out[r.segment_id] = "test"
        else:
            out[r.segment_id] = "dropped"
    return pd.Series(out, name="split")


def region_disjoint_split(region_ids: Sequence[str], test_regions: Sequence[str],
                          val_regions: Sequence[str] = ()) -> Dict[str, str]:
    out = {}
    for r in set(region_ids):
        out[r] = "test" if r in test_regions else "val" if r in val_regions else "train"
    return out


def segment_table(ais: pd.DataFrame) -> pd.DataFrame:
    return (ais.groupby("segment_id")
            .agg(vessel_id=("vessel_id", "first"), start=("timestamp", "min"), end=("timestamp", "max"),
                 region_id=("region_id", "first"), n_msgs=("timestamp", "size"))
            .reset_index())


def check_no_leakage(ais: pd.DataFrame, split_col: str = "split") -> Dict[str, bool]:
    """Sanity checks: each vessel and each segment belongs to exactly one split."""
    d = ais[ais[split_col] != "dropped"]
    return {
        "segments_unique": bool((d.groupby("segment_id")[split_col].nunique() <= 1).all()),
        "vessels_unique": bool((d.groupby("vessel_id")[split_col].nunique() <= 1).all()),
    }


# --------------------------------------------------------------------------
# Milestone 2 helpers (window-level splits)
# --------------------------------------------------------------------------
def labelled_interaction_pairs(events: pd.DataFrame, id_map: pd.DataFrame, vessel_type: Dict[str, str],
                               exclude_types=("tug",)) -> List[Tuple[str, str]]:
    """Vessel pairs from labelled interaction events (e.g. rendezvous, fleet pairs,
    tug assists). Service vessels (tugs) are excluded because they meet nearly every
    arriving ship and would chain unrelated vessels into one group."""
    to_v = dict(zip(id_map.raw_id, id_map.vessel_id))
    pairs = []
    for vs in events.vessels:
        ids = [to_v[x] for x in vs.split(",") if x in to_v]
        ids = [v for v in ids if vessel_type.get(v) not in exclude_types]
        pairs += [(ids[0], b) for b in ids[1:]]
    return pairs


def window_vessel_split(meta: pd.DataFrame, vsplit: Dict[str, str]) -> np.ndarray:
    return meta.vessel_id.map(vsplit).values


def window_time_split(meta: pd.DataFrame, ais: pd.DataFrame, boundaries, embargo=pd.Timedelta("6h")) -> np.ndarray:
    """Split windows by time; a window is dropped if any voyage segment it touches
    straddles a boundary (+/- embargo), so no voyage appears in two splits."""
    seg_split = time_disjoint_split(segment_table(ais), boundaries, embargo)
    out = []
    for s in meta.segments:
        labs = {seg_split.get(x, "dropped") for x in s.split(",") if x}
        out.append(labs.pop() if len(labs) == 1 else "dropped")
    return np.array(out)


def cross_split_contacts(meta: pd.DataFrame, split: np.ndarray, pairs_df: pd.DataFrame,
                         target: str = "test") -> int:
    """Number of sustained-proximity vessel pairs linking the target split to another split."""
    vs = dict(zip(meta.vessel_id, split))
    return int(sum((vs.get(a) == target) != (vs.get(b) == target)
                   for a, b in zip(pairs_df.vessel_a, pairs_df.vessel_b)))
