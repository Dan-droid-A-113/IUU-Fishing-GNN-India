"""Milestone 2 - baselines before the graph model.

* `rule_score`      - transparent rule-based REFERENCE (not a learning method,
                      not the contribution).
* `MLPClassifier`   - feed-forward network on window aggregates.
* `GRUClassifier`   - recurrent sequence model over the 36-step window.
* `TCNClassifier`   - temporal convolutional network (dilated causal convs).

All neural models are trained end-to-end with gradient descent (AdamW,
BCE-with-logits with positive-class weighting, ReduceLROnPlateau, early
stopping on validation PR-AUC, checkpointing, fixed seeds, CPU/GPU selection).
None of them sees other vessels' features directly - only scalar neighbour
summaries - which is the gap the Milestone-3 graph model is meant to close.
"""
from __future__ import annotations

import copy
import time
from dataclasses import asdict
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import average_precision_score

from .config import TrainConfig


def get_device(prefer_gpu: bool = True) -> torch.device:
    return torch.device("cuda" if prefer_gpu and torch.cuda.is_available() else "cpu")


def set_seed(seed: int):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# --------------------------------------------------------------------------
# Rule-based reference
# --------------------------------------------------------------------------
RULES = {
    "dark_at_sea": "AIS gap >= 3 h at sea, not port-to-port, few neighbours also silent",
    "slow_encounter": ">= 60 min within 500 m of another vessel while slow (< 2 kn)",
    "carrier_encounter": ">= 30 min within 500 m of a reefer/fish carrier",
    "fishing_in_mpa": "fishing-like motion inside the (approx.) MPA",
    "fishing_across_boundary": "fishing-like motion on the far side of the (approx.) boundary",
    "dark_link": "possible dark encounter (space-time prism) during the window",
}


def rule_components(agg: pd.DataFrame) -> pd.DataFrame:
    gap_ok = (agg.max_gap_min >= 180) & (agg.gap_start_port < 1) & (agg.gap_end_port < 1) & \
             ((agg.gap_silence < 0.3) | (agg.gap_silence_known < 1))
    return pd.DataFrame({
        "dark_at_sea": 0.35 * gap_ok * np.clip(agg.max_gap_min / 360, 0, 1),
        "slow_encounter": 0.30 * (agg.close_slow_min >= 60),
        "carrier_encounter": 0.15 * (agg.close_carrier_min >= 30),
        "fishing_in_mpa": 0.35 * np.clip(agg.frac_fish_in_mpa / 0.2, 0, 1),
        "fishing_across_boundary": 0.35 * np.clip(agg.frac_fish_other_side / 0.2, 0, 1),
        "dark_link": 0.20 * (agg.dark_link_frac > 0),
    })


def rule_score(agg: pd.DataFrame) -> np.ndarray:
    return np.clip(rule_components(agg).sum(1).values, 0, 1)


# --------------------------------------------------------------------------
# Neural models
# --------------------------------------------------------------------------
class MLPClassifier(nn.Module):
    def __init__(self, n_in: int, hidden: int = 64, dropout: float = 0.2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_in, hidden), nn.LayerNorm(hidden), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(hidden, hidden // 2), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(hidden // 2, 1))

    def forward(self, agg, seq=None):
        return self.net(agg).squeeze(-1)


class GRUClassifier(nn.Module):
    """Bidirectional GRU; the masked mean and the final state feed a linear head."""

    def __init__(self, n_in: int, hidden: int = 32, dropout: float = 0.2):
        super().__init__()
        self.gru = nn.GRU(n_in, hidden, batch_first=True, bidirectional=True)
        self.head = nn.Sequential(nn.Dropout(dropout), nn.Linear(4 * hidden, hidden), nn.ReLU(),
                                  nn.Linear(hidden, 1))

    def forward(self, agg, seq):
        out, h = self.gru(seq)                                    # out: B,T,2H
        last = torch.cat([h[0], h[1]], dim=-1)
        pooled = out.mean(1)
        return self.head(torch.cat([last, pooled], -1)).squeeze(-1)


class _CausalBlock(nn.Module):
    def __init__(self, c_in, c_out, k, d, dropout):
        super().__init__()
        self.pad = (k - 1) * d
        self.c1 = nn.Conv1d(c_in, c_out, k, dilation=d)
        self.c2 = nn.Conv1d(c_out, c_out, k, dilation=d)
        self.drop = nn.Dropout(dropout)
        self.res = nn.Conv1d(c_in, c_out, 1) if c_in != c_out else nn.Identity()

    def forward(self, x):
        y = torch.relu(self.c1(nn.functional.pad(x, (self.pad, 0))))
        y = self.drop(torch.relu(self.c2(nn.functional.pad(y, (self.pad, 0)))))
        return torch.relu(y + self.res(x))


class TCNClassifier(nn.Module):
    """Dilated causal TCN (dilations 1,2,4,8 -> receptive field 31 steps ~ 5 h)."""

    def __init__(self, n_in: int, hidden: int = 32, dropout: float = 0.2, k: int = 3,
                 dilations=(1, 2, 4, 8)):
        super().__init__()
        layers, c = [], n_in
        for d in dilations:
            layers.append(_CausalBlock(c, hidden, k, d, dropout))
            c = hidden
        self.tcn = nn.Sequential(*layers)
        self.head = nn.Sequential(nn.Linear(2 * hidden, hidden), nn.ReLU(), nn.Linear(hidden, 1))

    def forward(self, agg, seq):
        z = self.tcn(seq.transpose(1, 2))                         # B,H,T
        return self.head(torch.cat([z.mean(-1), z.amax(-1)], -1)).squeeze(-1)


MODEL_CLASSES = {"MLP": MLPClassifier, "GRU": GRUClassifier, "TCN": TCNClassifier}


# --------------------------------------------------------------------------
# Data preparation
# --------------------------------------------------------------------------
class WindowTensors:
    """Standardised tensors for one dataset; statistics are fitted on TRAIN only."""

    def __init__(self, X: np.ndarray, S: np.ndarray, agg: pd.DataFrame, y: np.ndarray, train_idx: np.ndarray):
        A = agg.values.astype(np.float32)
        A = np.sign(A) * np.log1p(np.abs(A))          # tame heavy tails (e.g. 60-h gaps)
        mu, sd = A[train_idx].mean(0), A[train_idx].std(0)
        sd = np.where(sd < 1e-3, 1.0, sd)             # constant-in-train columns stay unscaled
        self.agg = torch.tensor(np.clip((A - mu) / sd, -10, 10), dtype=torch.float32)
        seq = np.concatenate([X, np.repeat(S[:, None, :], X.shape[1], axis=1)], axis=-1)
        self.seq = torch.tensor(seq, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)
        self.agg_names = list(agg.columns)
        self.n_agg, self.n_seq = self.agg.shape[1], self.seq.shape[2]


def build_model(name: str, data: WindowTensors, cfg: TrainConfig) -> nn.Module:
    if name == "MLP":
        return MLPClassifier(data.n_agg, hidden=2 * cfg.hidden, dropout=cfg.dropout)
    return MODEL_CLASSES[name](data.n_seq, hidden=cfg.hidden, dropout=cfg.dropout)


# --------------------------------------------------------------------------
# Training
# --------------------------------------------------------------------------
def _batches(idx, bs, shuffle, gen):
    idx = idx[torch.randperm(len(idx), generator=gen)] if shuffle else idx
    for i in range(0, len(idx), bs):
        yield idx[i:i + bs]


@torch.no_grad()
def predict(model: nn.Module, data: WindowTensors, idx: np.ndarray, device, bs: int = 1024) -> np.ndarray:
    model.eval()
    out = []
    idx = torch.as_tensor(idx)
    for b in _batches(idx, bs, False, None):
        out.append(torch.sigmoid(model(data.agg[b].to(device), data.seq[b].to(device))).cpu())
    return torch.cat(out).numpy() if out else np.array([])


def train_model(name: str, data: WindowTensors, train_idx, val_idx, cfg: TrainConfig, seed: int,
                device=None, ckpt: Optional[Path] = None, verbose: bool = False) -> Dict:
    """Train one model; returns best model (by validation PR-AUC) and its history."""
    device = device or get_device()
    set_seed(seed)
    model = build_model(name, data, cfg).to(device)
    ytr = data.y[train_idx]
    pos_weight = ((ytr.numel() - ytr.sum()) / ytr.sum().clamp(min=1)).to(device)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="max", factor=0.5, patience=4)
    gen = torch.Generator().manual_seed(seed)
    tr = torch.as_tensor(train_idx)
    va = torch.as_tensor(val_idx)
    hist, best, best_state, bad = [], -1.0, None, 0
    t_start = time.time()
    for ep in range(1, cfg.epochs + 1):
        model.train()
        tot, n = 0.0, 0
        for b in _batches(tr, cfg.batch_size, True, gen):
            opt.zero_grad()
            logit = model(data.agg[b].to(device), data.seq[b].to(device))
            loss = loss_fn(logit, data.y[b].to(device))
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            tot += loss.item() * len(b)
            n += len(b)
        model.eval()
        with torch.no_grad():
            vlogit = torch.cat([model(data.agg[b].to(device), data.seq[b].to(device)).cpu()
                                for b in _batches(va, 1024, False, None)])
            vloss = loss_fn(vlogit.to(device), data.y[va].to(device)).item()
        vap = average_precision_score(data.y[va].numpy(), torch.sigmoid(vlogit).numpy())
        sched.step(vap)
        hist.append({"epoch": ep, "train_loss": tot / n, "val_loss": vloss, "val_pr_auc": vap,
                     "lr": opt.param_groups[0]["lr"]})
        if verbose:
            print(f"{name} seed{seed} ep{ep:02d} train {tot / n:.4f} val {vloss:.4f} PR-AUC {vap:.3f}")
        if vap > best + 1e-4:
            best, best_state, bad = vap, copy.deepcopy(model.state_dict()), 0
        else:
            bad += 1
            if bad >= cfg.patience:
                break
    model.load_state_dict(best_state)
    if ckpt is not None:
        Path(ckpt).parent.mkdir(parents=True, exist_ok=True)
        torch.save({"model": name, "state_dict": best_state, "config": asdict(cfg), "seed": seed,
                    "best_val_pr_auc": best, "n_agg": data.n_agg, "n_seq": data.n_seq}, ckpt)
    return {"model": model, "history": pd.DataFrame(hist), "best_val_pr_auc": best,
            "seconds": time.time() - t_start, "n_params": sum(p.numel() for p in model.parameters())}
