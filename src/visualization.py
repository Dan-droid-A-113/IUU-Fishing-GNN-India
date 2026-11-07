"""Maps and diagnostic figures (Matplotlib static + Folium interactive)."""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import matplotlib

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

from .india_layers import IndiaLayers

# Validated categorical order (blue, orange, aqua, yellow, magenta, ...)
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"]
INK, INK2, MUTED = "#0b0b0b", "#52514e", "#8a8984"
LAND, SEA = "#e6dfcf", "#fcfcfb"
MPA_C, IMBL_C, GROUND_C = "#008300", "#4a3aa7", "#8a8984"

TRAFFIC_GROUPS = {"fishing": "Fishing vessels", "reefer_carrier": "Reefer / fish carriers"}
GROUP_COLOR = {"Fishing vessels": SERIES[0], "Reefer / fish carriers": SERIES[1], "Other traffic": SERIES[2]}


def _style(ax):
    ax.set_facecolor(SEA)
    for s in ax.spines.values():
        s.set_color(MUTED); s.set_linewidth(0.6)
    ax.tick_params(colors=INK2, labelsize=8)


def draw_base_map(ax, layers: IndiaLayers, show_grounds=True, label_ports=True):
    lm = layers.land
    lat0, lat1, lon0, lon1 = layers.bbox
    LO, LA = np.meshgrid(lm.lons, lm.lats)
    _style(ax)
    ax.contourf(LO, LA, lm.grid.astype(float), levels=[0.5, 1.5], colors=[LAND], zorder=1)
    ax.contour(LO, LA, lm.grid.astype(float), levels=[0.5], colors=[MUTED], linewidths=0.5, zorder=2)
    for g in layers.mpa.values():
        x, y = g.exterior.xy
        ax.fill(x, y, color=MPA_C, alpha=0.13, zorder=2)
        ax.plot(x, y, color=MPA_C, lw=1.0, zorder=3)
    x, y = layers.imbl.xy
    ax.plot(x, y, color=IMBL_C, lw=1.4, ls=(0, (6, 3)), zorder=3)
    if show_grounds:
        for g in layers.fishing_grounds.values():
            x, y = g.exterior.xy
            ax.plot(x, y, color=GROUND_C, lw=0.8, ls=":", zorder=3)
    ax.scatter(layers.ports.lon, layers.ports.lat, s=22, marker="s", color=INK, zorder=6)
    if label_ports:
        for r in layers.ports.itertuples():
            ax.annotate(r.name, (r.lon, r.lat), xytext=(4, 3), textcoords="offset points",
                        fontsize=7, color=INK2, zorder=7)
    ax.set_xlim(lon0, lon1); ax.set_ylim(lat0, lat1)
    ax.set_aspect(1 / np.cos(np.deg2rad((lat0 + lat1) / 2)))
    ax.set_xlabel("Longitude (°E)", fontsize=8, color=INK2)
    ax.set_ylabel("Latitude (°N)", fontsize=8, color=INK2)
    return ax


def draw_india_inset(main_ax, layers: IndiaLayers, rect=(0.01, 0.66, 0.30, 0.33)):
    """Small locator map of peninsular India with the pilot box, placed over the
    Tamil Nadu land area (upper-left) so it never hides tracks."""
    from global_land_mask import globe
    ax = main_ax.inset_axes(rect)
    lats = np.arange(5.5, 24.5, 0.1); lons = np.arange(67.5, 90.5, 0.1)
    LO, LA = np.meshgrid(lons, lats)
    ax.set_facecolor(SEA)
    ax.contourf(LO, LA, globe.is_land(LA, LO).astype(float), levels=[0.5, 1.5], colors=[LAND])
    lat0, lat1, lon0, lon1 = layers.bbox
    ax.add_patch(plt.Rectangle((lon0, lat0), lon1 - lon0, lat1 - lat0, fill=False, ec=SERIES[7], lw=1.4))
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_aspect(1 / np.cos(np.deg2rad(15)))
    for s in ax.spines.values():
        s.set_color(MUTED)
    ax.set_title("India - pilot box", fontsize=7, color=INK2, pad=2)
    return ax


def plot_tracks_map(ais: pd.DataFrame, layers: IndiaLayers, gaps: Optional[pd.DataFrame] = None,
                    title: str = "", path: Optional[Path] = None, inset=True, min_gap_min=60):
    """Tracks by traffic group; AIS gaps drawn as thin dashed straight connectors
    (the straight line is NOT an observed path)."""
    fig, ax = plt.subplots(figsize=(10, 9.4))
    draw_base_map(ax, layers)
    grp = ais.vessel_type.map(TRAFFIC_GROUPS).fillna("Other traffic")
    piece = ais.track_piece if "track_piece" in ais else 0
    for (g_name, _, _), d in ais.assign(_g=grp, _p=piece).groupby(["_g", "vessel_id", "_p"], sort=False):
        ax.plot(d.longitude, d.latitude, color=GROUP_COLOR[g_name], lw=0.6, alpha=0.75, zorder=4)
    handles = [Line2D([], [], color=c, lw=2, label=n) for n, c in GROUP_COLOR.items()]
    if gaps is not None and len(gaps):
        gg = gaps[(gaps.gap_duration_min >= min_gap_min)]
        if "gap_context" in gg:
            gg = gg[gg.gap_context == "at_sea"]
        for r in gg.itertuples():
            ax.plot([r.start_lon, r.end_lon], [r.start_lat, r.end_lat], color=INK, lw=0.9,
                    ls=(0, (2, 2)), zorder=5)
        ax.scatter(gg.end_lon, gg.end_lat, s=14, facecolor="white", edgecolor=INK, lw=0.8, zorder=6)
        handles.append(Line2D([], [], color=INK, lw=0.9, ls=(0, (2, 2)), marker="o", mfc="white",
                              ms=4, label=f"AIS gap ≥{min_gap_min} min at sea (connector, not a path)"))
    handles += [Patch(fc=MPA_C, alpha=0.25, ec=MPA_C, label="Gulf of Mannar MNP (approx.)"),
                Line2D([], [], color=IMBL_C, lw=1.4, ls=(0, (6, 3)), label="India-Sri Lanka boundary (approx.)"),
                Line2D([], [], color=GROUND_C, lw=0.8, ls=":", label="Illustrative fishing grounds"),
                Line2D([], [], color=INK, marker="s", lw=0, ms=5, label="Ports / harbours")]
    ax.legend(handles=handles, loc="lower right", fontsize=7.5, frameon=True, facecolor="white",
              edgecolor=MUTED, framealpha=0.95)
    ax.set_title(title, fontsize=11, color=INK, loc="left")
    if inset:
        draw_india_inset(ax, layers)
    fig.text(0.01, 0.005, "SYNTHETIC DEMO DATA - simulated vessels, not real observations. "
             "Boundaries approximate.", fontsize=7, color=INK2)
    if path:
        fig.savefig(path, dpi=150, bbox_inches="tight")
    return fig


def plot_gap_diagnostics(gaps_truth: pd.DataFrame, threshold_min: float, path: Optional[Path] = None):
    """Left: gap duration by simulated cause. Right: duration vs concurrent silence."""
    from .evaluation import CAUSE_ORDER
    labels = {"dark_intentional": "Planted blackout (suspicious)", "equipment_glitch": "Transmitter glitch",
              "receiver_outage": "Receiver outage", "coverage": "Coverage / reception",
              "between_voyages": "Port stay / region exit"}
    colors = dict(zip(CAUSE_ORDER, SERIES[:5]))
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6))
    ax = axes[0]
    bins = np.logspace(np.log10(threshold_min), np.log10(max(gaps_truth.gap_duration_min.max(), threshold_min * 2)), 18)
    data = [gaps_truth.loc[gaps_truth.truth_cause == c, "gap_duration_min"] for c in CAUSE_ORDER]
    ax.hist(data, bins=bins, stacked=True, color=[colors[c] for c in CAUSE_ORDER],
            label=[labels[c] for c in CAUSE_ORDER], edgecolor="white", linewidth=1.0)
    ax.set_xscale("log"); ax.set_xlabel("Gap duration (min, log scale)", fontsize=9, color=INK2)
    ax.set_ylabel("Detected gaps", fontsize=9, color=INK2)
    ax.set_title("Detected AIS gaps by simulated cause", fontsize=10, color=INK, loc="left")
    ax.legend(fontsize=7.5, frameon=False)
    ax2 = axes[1]
    within = gaps_truth[gaps_truth.truth_cause != "between_voyages"]
    for c, m in zip(CAUSE_ORDER[:-1], ["o", "s", "^", "D"]):
        d = within[within.truth_cause == c]
        ax2.scatter(d.gap_duration_min, d.concurrent_silence_frac.fillna(-0.08), s=36, marker=m,
                    color=colors[c], edgecolor="white", linewidth=0.8, label=labels[c], zorder=3)
    ax2.axhline(-0.08, color=MUTED, lw=0.5, ls=":")
    ax2.text(ax2.get_xlim()[0] if False else within.gap_duration_min.min() if len(within) else 60, -0.13,
             "no neighbours nearby (unknown)", fontsize=7, color=INK2)
    ax2.set_xscale("log")
    ax2.set_xlabel("Gap duration (min, log scale)", fontsize=9, color=INK2)
    ax2.set_ylabel("Share of nearby vessels also silent", fontsize=9, color=INK2)
    ax2.set_title("Within-voyage gaps: reception context separates causes", fontsize=10, color=INK, loc="left")
    ax2.legend(fontsize=7.5, frameon=False, loc="center right")
    for a in axes:
        _style(a); a.set_facecolor("white"); a.grid(color="#e9e8e4", lw=0.5, zorder=0)
    fig.text(0.01, -0.02, "Synthetic demo: causes come from hidden simulation truth; used for pipeline "
             "validation only.", fontsize=7, color=INK2)
    fig.tight_layout()
    if path:
        fig.savefig(path, dpi=150, bbox_inches="tight")
    return fig


def plot_behaviour_features(ais_truth: pd.DataFrame, path: Optional[Path] = None):
    """Speed and fishing-motion score by simulated behaviour (sanity check of features)."""
    order = ["transit", "ferry", "steaming", "fishing", "loitering", "rendezvous", "tug_assist", "anchored"]
    order = [b for b in order if b in set(ais_truth.behavior)]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2), sharey=True)
    for ax, col, lab in [(axes[0], "sog", "Speed over ground (kn)"),
                         (axes[1], "fishing_motion_score", "Fishing-motion score (heuristic, 0-1)")]:
        data = [ais_truth.loc[ais_truth.behavior == b, col].dropna().values for b in order]
        bp = ax.boxplot(data, vert=False, tick_labels=order, showfliers=False, patch_artist=True,
                        medianprops=dict(color=INK, lw=1.2), whiskerprops=dict(color=MUTED),
                        capprops=dict(color=MUTED))
        for p in bp["boxes"]:
            p.set_facecolor(SERIES[0]); p.set_alpha(0.35); p.set_edgecolor(SERIES[0])
        ax.set_xlabel(lab, fontsize=9, color=INK2)
        _style(ax); ax.set_facecolor("white"); ax.grid(axis="x", color="#e9e8e4", lw=0.5)
    axes[0].set_title("Observed kinematics by simulated behaviour", fontsize=10, color=INK, loc="left")
    fig.tight_layout()
    if path:
        fig.savefig(path, dpi=150, bbox_inches="tight")
    return fig


def folium_map(ais: pd.DataFrame, layers: IndiaLayers, gaps: Optional[pd.DataFrame] = None,
               path: Optional[Path] = None, max_points_per_track: int = 400):
    """Interactive Folium map (Esri basemaps; needs internet when viewed)."""
    import folium
    lat0, lat1, lon0, lon1 = layers.bbox
    # Esri basemaps: OSM's own tile servers block notebook hosts such as Colab ("403 Access
    # blocked") and CARTO now needs an API key. Esri Ocean needs neither.
    m = folium.Map(location=[(lat0 + lat1) / 2, (lon0 + lon1) / 2], zoom_start=8, tiles=None)
    esri = "https://server.arcgisonline.com/ArcGIS/rest/services/{}/MapServer/tile/{{z}}/{{y}}/{{x}}"
    folium.TileLayer(esri.format("Ocean/World_Ocean_Base"), name="Esri Ocean",
                     attr="Esri, GEBCO, NOAA, National Geographic, DeLorme, HERE, Geonames.org").add_to(m)
    folium.TileLayer(esri.format("World_Imagery"), name="Esri Imagery",
                     attr="Esri, Maxar, Earthstar Geographics").add_to(m)
    folium.GeoJson(layers.geojson(), name="Context layers (approx.)",
                   style_function=lambda f: {
                       "mpa": {"color": MPA_C, "fillOpacity": 0.15, "weight": 1},
                       "imbl": {"color": IMBL_C, "weight": 2, "dashArray": "6 4"},
                       "ground": {"color": GROUND_C, "weight": 1, "dashArray": "2 3", "fillOpacity": 0},
                   }[f["properties"]["layer"]],
                   tooltip=folium.GeoJsonTooltip(fields=["name"])).add_to(m)
    grp = ais.vessel_type.map(TRAFFIC_GROUPS).fillna("Other traffic")
    fgs = {g: folium.FeatureGroup(name=g) for g in GROUP_COLOR}
    for (g, vid, piece), d in ais.assign(_g=grp).groupby(["_g", "vessel_id", "track_piece"]):
        step = max(1, len(d) // max_points_per_track)
        pts = d.iloc[::step][["latitude", "longitude"]].values.tolist()
        if len(pts) > 1:
            folium.PolyLine(pts, color=GROUP_COLOR[g], weight=1.5, opacity=0.8,
                            tooltip=f"{vid} ({d.vessel_type.iloc[0]})").add_to(fgs[g])
    for fg in fgs.values():
        fg.add_to(m)
    if gaps is not None and len(gaps):
        fg = folium.FeatureGroup(name="AIS gaps at sea")
        for r in gaps[gaps.get("gap_context", "at_sea") == "at_sea"].itertuples():
            folium.PolyLine([[r.start_lat, r.start_lon], [r.end_lat, r.end_lon]], color=INK, weight=1.5,
                            dash_array="4 4",
                            tooltip=f"{r.vessel_id}: silent {r.gap_duration_min:.0f} min").add_to(fg)
        fg.add_to(m)
    for r in layers.ports.itertuples():
        folium.CircleMarker([r.lat, r.lon], radius=4, color=INK, fill=True, tooltip=r.name).add_to(m)
    folium.LayerControl().add_to(m)
    if path:
        m.save(str(path))
    return m


def plot_gap_example(ais: pd.DataFrame, gap, layers: IndiaLayers, true_path: Optional[pd.DataFrame] = None,
                     hours: float = 4.0, path: Optional[Path] = None, title: str = ""):
    """Zoom on one AIS gap: observed fixes before/after, straight connector, and
    (synthetic only) the hidden true path that the pipeline never sees."""
    v = ais[(ais.vessel_id == gap.vessel_id) &
            (ais.timestamp >= gap.gap_start - pd.Timedelta(hours=hours)) &
            (ais.timestamp <= gap.gap_end + pd.Timedelta(hours=hours))]
    pts = [v[["latitude", "longitude"]].values]
    if true_path is not None and len(true_path):
        pts.append(true_path[["lat_true", "lon_true"]].values)
    allp = np.vstack(pts)
    pad = 0.06
    fig, ax = plt.subplots(figsize=(8.5, 7.5))
    draw_base_map(ax, layers, label_ports=True)
    before, after = v[v.timestamp <= gap.gap_start], v[v.timestamp >= gap.gap_end]
    ax.plot(before.longitude, before.latitude, color=SERIES[0], lw=1.4, marker="o", ms=2.2, zorder=5,
            label="Observed AIS before gap")
    ax.plot(after.longitude, after.latitude, color=SERIES[2], lw=1.4, marker="o", ms=2.2, zorder=5,
            label="Observed AIS after reappearance")
    ax.plot([gap.start_lon, gap.end_lon], [gap.start_lat, gap.end_lat], color=INK, lw=1.0,
            ls=(0, (2, 2)), zorder=6, label=f"AIS gap: {gap.gap_duration_min:.0f} min silent")
    if true_path is not None and len(true_path):
        ax.plot(true_path.lon_true, true_path.lat_true, color=SERIES[1], lw=1.3, ls=(0, (1, 1.5)), zorder=4,
                label="Hidden simulation truth (never shown to models)")
    ax.scatter([gap.start_lon], [gap.start_lat], s=60, marker="v", color=INK, zorder=7)
    ax.scatter([gap.end_lon], [gap.end_lat], s=60, marker="^", color=INK, zorder=7)
    ax.set_xlim(allp[:, 1].min() - pad, allp[:, 1].max() + pad)
    ax.set_ylim(allp[:, 0].min() - pad, allp[:, 0].max() + pad)
    ax.legend(fontsize=7.5, loc="best", facecolor="white", edgecolor=MUTED)
    ax.set_title(title, fontsize=10, color=INK, loc="left")
    if path:
        fig.savefig(path, dpi=150, bbox_inches="tight")
    return fig


# ==========================================================================
# Milestone 2 figures
# ==========================================================================
MODEL_COLORS = {"Rule (reference)": MUTED, "MLP": SERIES[0], "GRU": SERIES[1], "TCN": SERIES[2]}


def _best_preds(res, split, model):
    from .experiments import best_run
    return best_run(res, split, model)


def plot_pr_curves(res, labels, path=None):
    from sklearn.metrics import precision_recall_curve, average_precision_score
    splits = list(res["runs"].split.unique())
    fig, axes = plt.subplots(1, len(splits), figsize=(6 * len(splits), 4.8), squeeze=False)
    y = labels.y.values
    for ax, sp in zip(axes[0], splits):
        for m in res["runs"][res["runs"].split == sp].model.unique():
            p = _best_preds(res, sp, m)
            yt = y[p["test_idx"]]
            pr, rc, _ = precision_recall_curve(yt, p["test"])
            ls = "--" if m.startswith("Rule") else "-"
            ax.plot(rc, pr, color=MODEL_COLORS.get(m, INK), lw=2, ls=ls,
                    label=f"{m} (AP {average_precision_score(yt, p['test']):.2f})")
        ax.axhline(yt.mean(), color=MUTED, lw=0.8, ls=":")
        ax.text(0.99, yt.mean() + 0.015, f"prevalence {yt.mean():.2f}", fontsize=7, color=INK2, ha="right")
        ax.set(xlim=(0, 1), ylim=(0, 1.02), xlabel="Recall", ylabel="Precision")
        ax.set_title(f"Test precision-recall - {sp.replace('_', '-')} split (best seed)", fontsize=10, loc="left")
        ax.legend(fontsize=8, frameon=True, facecolor="white", edgecolor=MUTED, loc="center left")
        _style(ax); ax.set_facecolor("white"); ax.grid(color="#e9e8e4", lw=0.5)
    fig.tight_layout()
    if path:
        fig.savefig(path, dpi=150, bbox_inches="tight")
    return fig


def plot_confusion_matrices(res, labels, split, path=None):
    from sklearn.metrics import confusion_matrix
    models = list(res["runs"][res["runs"].split == split].model.unique())
    fig, axes = plt.subplots(1, len(models), figsize=(3.3 * len(models), 3.3))
    y = labels.y.values
    for ax, m in zip(np.atleast_1d(axes), models):
        p = _best_preds(res, split, m)
        cm = confusion_matrix(y[p["test_idx"]], (p["test"] >= p["thr"]).astype(int), labels=[0, 1])
        ax.imshow(cm / cm.sum(1, keepdims=True), cmap="Blues", vmin=0, vmax=1)
        for i in range(2):
            for j in range(2):
                frac = cm[i, j] / max(cm[i].sum(), 1)
                ax.text(j, i, f"{cm[i, j]}\n({frac:.0%})", ha="center", va="center", fontsize=9,
                        color="white" if frac > 0.6 else INK)
        ax.set_xticks([0, 1], ["normal", "suspicious"], fontsize=8)
        ax.set_yticks([0, 1], ["normal", "suspicious"], fontsize=8)
        ax.set_xlabel("Predicted", fontsize=8); ax.set_ylabel("Synthetic label", fontsize=8)
        ax.set_title(m, fontsize=9, loc="left")
    fig.suptitle(f"Test confusion matrices - {split.replace('_', '-')} split (threshold from validation)",
                 fontsize=10, x=0.01, ha="left")
    fig.tight_layout()
    if path:
        fig.savefig(path, dpi=150, bbox_inches="tight")
    return fig


def plot_learning_curves(res, split, path=None):
    hist = {k: v for k, v in res["histories"].items() if k[0] == split}
    models = sorted({k[1] for k in hist}, key=["MLP", "GRU", "TCN"].index)
    fig, axes = plt.subplots(2, len(models), figsize=(4.2 * len(models), 6), sharex="col", squeeze=False)
    for j, m in enumerate(models):
        for (sp, mm, seed), h in hist.items():
            if mm != m:
                continue
            a = 1.0 if seed == 0 else 0.35
            axes[0, j].plot(h.epoch, h.train_loss, color=SERIES[0], lw=1.6, alpha=a,
                            label="train loss" if seed == 0 else None)
            axes[0, j].plot(h.epoch, h.val_loss, color=SERIES[1], lw=1.6, alpha=a,
                            label="validation loss" if seed == 0 else None)
            axes[1, j].plot(h.epoch, h.val_pr_auc, color=SERIES[2], lw=1.6, alpha=a,
                            label="validation PR-AUC" if seed == 0 else None)
        axes[0, j].set_title(f"{m}", fontsize=10, loc="left")
        axes[1, j].set_xlabel("Epoch", fontsize=8)
        for i in range(2):
            _style(axes[i, j]); axes[i, j].set_facecolor("white"); axes[i, j].grid(color="#e9e8e4", lw=0.5)
            axes[i, j].legend(fontsize=7, frameon=False)
    axes[0, 0].set_ylabel("Weighted BCE loss", fontsize=8)
    axes[1, 0].set_ylabel("PR-AUC", fontsize=8)
    fig.suptitle(f"Learning curves - {split.replace('_', '-')} split (faint lines = other seeds; "
                 "early stopping on validation PR-AUC)", fontsize=10, x=0.01, ha="left")
    fig.tight_layout()
    if path:
        fig.savefig(path, dpi=150, bbox_inches="tight")
    return fig


def fp_by_context(res, labels, split) -> pd.DataFrame:
    rows = []
    for m in res["runs"][res["runs"].split == split].model.unique():
        p = _best_preds(res, split, m)
        L = labels.iloc[p["test_idx"]].assign(alert=p["test"] >= p["thr"])
        neg = L[L.y == 0]
        for c, g in neg.groupby("context"):
            rows.append({"model": m, "context": c, "negatives": len(g), "false_alerts": int(g.alert.sum()),
                         "fp_rate": g.alert.mean()})
    return pd.DataFrame(rows)


def plot_fp_by_context(tab: pd.DataFrame, split, path=None):
    """Horizontal grouped bars: false-alert rate per benign context and model."""
    ctx = (tab.groupby("context").negatives.first().sort_values()).index.tolist()
    models = list(tab.model.unique())
    fig, ax = plt.subplots(figsize=(10, 0.62 * len(ctx) + 1.3))
    h = 0.8 / len(models)
    for i, m in enumerate(models):
        t = tab[tab.model == m].set_index("context").reindex(ctx)
        ax.barh(np.arange(len(ctx)) + (len(models) - 1 - i) * h, t.fp_rate.fillna(0) * 100, height=h * 0.9,
                color=MODEL_COLORS.get(m, INK), label=m, edgecolor="white", linewidth=0.6)
    n = tab.groupby("context").negatives.first().reindex(ctx)
    ax.set_yticks(np.arange(len(ctx)) + h * (len(models) - 1) / 2,
                  [f"{c}  (n={int(k)})" for c, k in zip(ctx, n)], fontsize=8)
    ax.set_xlabel("False-alert rate among normal test windows (%)", fontsize=8.5)
    ax.set_title(f"Where false alerts come from - {split.replace('_', '-')} split", fontsize=10, loc="left")
    ax.legend(fontsize=8, frameon=False, loc="lower right")
    _style(ax); ax.set_facecolor("white"); ax.grid(axis="x", color="#e9e8e4", lw=0.5)
    fig.tight_layout()
    if path:
        fig.savefig(path, dpi=150, bbox_inches="tight")
    return fig


def plot_fp_gallery(d, pred, split, model_name, path=None, n=6):
    """Top-scoring false alerts, one per benign context where possible, drawn on the map."""
    labels, ais, layers = d["labels"], d["ais"], d["layers"]
    L = labels.iloc[pred["test_idx"]].assign(score=pred["test"])
    fp = L[(L.y == 0) & (L.score >= pred["thr"])].sort_values("score", ascending=False)
    pick = fp.groupby("context").head(1).head(n)
    if len(pick) < n:
        pick = pd.concat([pick, fp.drop(pick.index).head(n - len(pick))])
    if pick.empty:
        return None
    cols = 3
    rows = int(np.ceil(len(pick) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(5.2 * cols, 4.8 * rows), squeeze=False)
    for ax, r in zip(axes.ravel(), pick.itertuples()):
        v = ais[(ais.vessel_id == r.vessel_id) & (ais.timestamp >= r.t0 - pd.Timedelta("2h")) &
                (ais.timestamp < r.t1 + pd.Timedelta("2h"))]
        inw = v[(v.timestamp >= r.t0) & (v.timestamp < r.t1)]
        others = ais[(ais.timestamp >= r.t0) & (ais.timestamp < r.t1) & (ais.vessel_id != r.vessel_id)]
        draw_base_map(ax, layers, show_grounds=False, label_ports=True)
        gw = d["gaps"][(d["gaps"].vessel_id == r.vessel_id) & (d["gaps"].gap_start < r.t1) &
                       (d["gaps"].gap_end > r.t0)]
        for gr in gw.itertuples():
            ax.plot([gr.start_lon, gr.end_lon], [gr.start_lat, gr.end_lat], color=INK, lw=1.0, ls=(0, (2, 2)))
        ext = inw if len(inw) else v
        if len(ext) or len(gw):
            la = np.r_[ext.latitude.values, gw.start_lat.values, gw.end_lat.values]
            lo = np.r_[ext.longitude.values, gw.start_lon.values, gw.end_lon.values]
            lat0, lat1, lon0, lon1 = la.min(), la.max(), lo.min(), lo.max()
            near = others[others.latitude.between(lat0 - 0.05, lat1 + 0.05) & others.longitude.between(lon0 - 0.05, lon1 + 0.05)]
            for _, g in near.groupby("vessel_id"):
                ax.plot(g.longitude, g.latitude, color=MUTED, lw=0.8, alpha=0.8)
            ax.plot(v.longitude, v.latitude, color=SERIES[0], lw=0.8, alpha=0.4)
            ax.plot(inw.longitude, inw.latitude, color=SERIES[0], lw=1.8, marker="o", ms=2)
            pad = max(0.04, 0.25 * max(lat1 - lat0, lon1 - lon0))
            ax.set_xlim(lon0 - pad, lon1 + pad); ax.set_ylim(lat0 - pad, lat1 + pad)
        silent = " | no fixes in window (dark)" if not len(inw) else ""
        ax.set_title(f"{r.context}\n{r.vessel_type} | score {r.score:.2f} | {r.t0:%d %b %H:%M} UTC{silent}",
                     fontsize=8.5, loc="left")
        ax.tick_params(labelsize=6)
        ax.set_xlabel(""); ax.set_ylabel("")
    for ax in axes.ravel()[len(pick):]:
        ax.axis("off")
    fig.suptitle(f"False-alert gallery - {model_name}, {split.replace('_', '-')} split. Blue = flagged vessel in the 6-h "
                 "window, dashed = AIS gap connector, grey = nearby vessels. SYNTHETIC DATA.", fontsize=10, x=0.01, ha="left")
    fig.tight_layout()
    if path:
        fig.savefig(path, dpi=130, bbox_inches="tight")
    return fig


# ==========================================================================
# Milestone 3 figures
# ==========================================================================
MODEL_COLORS.update({"TGAT (temporal graph)": SERIES[6], "TGAT without graph (ablation)": SERIES[3]})


def plot_task_losses(histories, split, model="TGAT (temporal graph)", path=None):
    """Each multi-task loss term and validation PR-AUC (small multiples; faint = other seeds)."""
    terms = [("loss_risk", "Investigation score (BCE)"), ("loss_fishing", "Fishing activity (BCE)"),
             ("loss_gap", "AIS-gap anomaly (BCE)"), ("loss_edge", "Suspicious interaction, edges (BCE)"),
             ("loss_traj", "Gap trajectory (Gaussian NLL)"), ("val_pr_auc", "Validation PR-AUC (risk)")]
    hs = {k[2]: v for k, v in histories.items() if k[0] == split and k[1] == model}
    fig, axes = plt.subplots(2, 3, figsize=(13, 6.5))
    for ax, (col, title) in zip(axes.ravel(), terms):
        for seed, h in hs.items():
            ax.plot(h.epoch, h[col], color=SERIES[2] if col == "val_pr_auc" else SERIES[6],
                    lw=1.6, alpha=1.0 if seed == min(hs) else 0.35)
        ax.set_title(title, fontsize=9, loc="left")
        ax.set_xlabel("Epoch", fontsize=8)
        _style(ax); ax.set_facecolor("white"); ax.grid(color="#e9e8e4", lw=0.5)
    fig.suptitle(f"Multi-task training of the temporal GAT - {split.replace('_', '-')} split "
                 "(training losses; faint = other seeds)", fontsize=10, x=0.01, ha="left")
    fig.tight_layout()
    if path:
        fig.savefig(path, dpi=150, bbox_inches="tight")
    return fig


def plot_model_comparison(runs: pd.DataFrame, metric="pr_auc", path=None, label="Test PR-AUC"):
    order = ["Rule (reference)", "MLP", "GRU", "TCN", "TGAT without graph (ablation)", "TGAT (temporal graph)"]
    splits = list(runs.split.unique())
    fig, axes = plt.subplots(1, len(splits), figsize=(6.2 * len(splits), 3.6), squeeze=False)
    for ax, sp in zip(axes[0], splits):
        r = runs[runs.split == sp]
        ms = [m for m in order if m in set(r.model)]
        g = r.groupby("model")[metric]
        mean, sd = g.mean().reindex(ms), g.std().reindex(ms).fillna(0)
        ypos = np.arange(len(ms))[::-1]
        ax.barh(ypos, mean, xerr=sd, color=[MODEL_COLORS.get(m, INK) for m in ms], height=0.62,
                error_kw=dict(ecolor=INK2, lw=1, capsize=3))
        for yv, m, v in zip(ypos, ms, mean):
            ax.text(min(v, 1) - 0.01, yv, f"{v:.3f}", va="center", ha="right", fontsize=8, color="white")
        ax.set_yticks(ypos, ms, fontsize=8)
        lo = max(0.0, float(mean.min()) - 0.15)
        ax.set_xlim(lo, 1.0)
        ax.set_xlabel(label + " (mean ± sd over seeds)", fontsize=8)
        ax.set_title(f"{sp.replace('_', '-')} split", fontsize=10, loc="left")
        _style(ax); ax.set_facecolor("white"); ax.grid(axis="x", color="#e9e8e4", lw=0.5)
    fig.tight_layout()
    if path:
        fig.savefig(path, dpi=150, bbox_inches="tight")
    return fig


def plot_gap_reconstruction(d, G, raw, k, path=None, title=""):
    """One AIS gap: observed track, hidden true path, dead reckoning, neural path + 90% endpoint ellipse."""
    from matplotlib.patches import Ellipse
    gd = G["gap_ds"]
    g = gd["gaps"].iloc[k]
    lat0, lon0 = g.start_lat, g.start_lon
    kx = 111.32 * np.cos(np.deg2rad(lat0))

    def to_ll(P):
        return lat0 + P[..., 1] / 111.32, lon0 + P[..., 0] / kx

    ais = d["ais"]
    v = ais[(ais.vessel_id == g.vessel_id) & (ais.timestamp >= g.gap_start - pd.Timedelta("3h")) &
            (ais.timestamp <= g.gap_end + pd.Timedelta("3h"))]
    fig, ax = plt.subplots(figsize=(8.5, 7.5))
    draw_base_map(ax, d["layers"], show_grounds=False)
    b, a = v[v.timestamp <= g.gap_start], v[v.timestamp >= g.gap_end]
    ax.plot(b.longitude, b.latitude, color=SERIES[0], lw=1.6, marker="o", ms=2, label="Observed before gap", zorder=6)
    ax.plot(a.longitude, a.latitude, color=SERIES[2], lw=1.6, marker="o", ms=2, label="Observed after reappearance", zorder=6)
    pts = [np.c_[v.latitude, v.longitude]]
    zero = np.zeros((1, 2))
    for P, c, ls, lab in [(gd["truth_path"][k], SERIES[1], (0, (1, 1.5)), "Hidden true path (evaluation only)"),
                          (gd["dead_reckoning"][k], MUTED, "--", "Dead reckoning (reference)"),
                          (raw["gap_path"][k], SERIES[6], "-", "Neural decoder (predicted)")]:
        if np.isnan(P).any():
            continue
        la, lo = to_ll(np.vstack([zero, P]))
        ax.plot(lo, la, color=c, ls=ls, lw=2, label=lab, zorder=5)
        pts.append(np.c_[la, lo])
    mu = raw["gap_path"][k][-1]
    sig = np.exp(raw["gap_logsig"][k]) * np.sqrt(4.605)
    la, lo = to_ll(mu)
    ax.add_patch(Ellipse((lo, la), 2 * sig[0] / kx, 2 * sig[1] / 111.32, fc=SERIES[6], alpha=0.12, ec=SERIES[6],
                         lw=1, label="90% reappearance region", zorder=4))
    P = np.vstack(pts)
    pad = 0.05
    ax.set_xlim(P[:, 1].min() - pad, P[:, 1].max() + pad); ax.set_ylim(P[:, 0].min() - pad, P[:, 0].max() + pad)
    ax.legend(fontsize=7.5, loc="best", facecolor="white", edgecolor=MUTED)
    ax.set_title(title or f"{g.vessel_id} silent {g.gap_duration_min:.0f} min", fontsize=10, loc="left")
    if path:
        fig.savefig(path, dpi=150, bbox_inches="tight")
    return fig


# ==========================================================================
# Real-data (Global Fishing Watch) figures
# ==========================================================================
def plot_real_events_map(ev: pd.DataFrame, sar: Optional[pd.DataFrame], layers: IndiaLayers, title: str = "",
                         path=None, max_gap_lines: int = 400):
    """Real GFW events on the India-focused pilot map."""
    fig, ax = plt.subplots(figsize=(10, 9.4))
    draw_base_map(ax, layers, show_grounds=False)
    handles = []
    if sar is not None and len(sar):
        ax.scatter(sar.lon, sar.lat, s=np.clip(sar.detections, 1, 20) * 3, marker="x", color=MUTED, lw=0.7,
                   alpha=0.6, zorder=3)
        handles.append(Line2D([], [], marker="x", color=MUTED, lw=0, label="SAR detections without AIS match"))
    style = {"fishing": (SERIES[0], "o", 6, "Apparent fishing events"),
             "loitering": (SERIES[2], "s", 12, "Loitering events"),
             "encounter": (SERIES[1], "D", 26, "Encounters (vessel meetings)"),
             "port_visit": (INK, "^", 10, "Port visits")}
    for k, (c, m, sz, lab) in style.items():
        x = ev[ev.event_type == k]
        if len(x):
            ax.scatter(x.lon, x.lat, s=sz, marker=m, color=c, alpha=0.65 if k == "fishing" else 0.9,
                       edgecolor="white" if k == "encounter" else "none", lw=0.6, zorder=5)
            handles.append(Line2D([], [], marker=m, color=c, lw=0, ms=6, label=f"{lab} (n={len(x):,})"))
    g = ev[ev.event_type == "gap"].dropna(subset=["off_lat", "on_lat"])
    if len(g):
        gi = g.sort_values("gap_duration_h", ascending=False).head(max_gap_lines)
        for r in gi.itertuples():
            ax.plot([r.off_lon, r.on_lon], [r.off_lat, r.on_lat], color=SERIES[7], lw=0.8, ls=(0, (2, 2)), zorder=4)
        ax.scatter(gi.off_lon, gi.off_lat, s=14, color=SERIES[7], marker="v", zorder=6)
        handles.append(Line2D([], [], color=SERIES[7], ls=(0, (2, 2)), marker="v", ms=5,
                              label=f"AIS-off gaps: off → on (n={len(g):,})"))
    handles += [Patch(fc=MPA_C, alpha=0.25, ec=MPA_C, label="Gulf of Mannar MNP (approx.)"),
                Line2D([], [], color=IMBL_C, lw=1.4, ls=(0, (6, 3)), label="India-Sri Lanka boundary (approx.)")]
    ax.legend(handles=handles, loc="lower right", fontsize=7.5, facecolor="white", edgecolor=MUTED, framealpha=0.95)
    ax.set_title(title, fontsize=11, loc="left")
    fig.text(0.01, 0.005, "REAL DATA: Global Fishing Watch (AIS-derived events, Sentinel-1 SAR). Activity patterns only; "
             "not evidence of illegality.", fontsize=7, color=INK2)
    if path:
        fig.savefig(path, dpi=150, bbox_inches="tight")
    return fig


def folium_real_map(ev: pd.DataFrame, sar: Optional[pd.DataFrame], layers: IndiaLayers, path=None, max_points=3000):
    import folium
    lat0, lat1, lon0, lon1 = layers.bbox
    m = folium.Map(location=[(lat0 + lat1) / 2, (lon0 + lon1) / 2], zoom_start=8, tiles=None)
    esri = "https://server.arcgisonline.com/ArcGIS/rest/services/{}/MapServer/tile/{{z}}/{{y}}/{{x}}"
    folium.TileLayer(esri.format("Ocean/World_Ocean_Base"), name="Esri Ocean", attr="Esri, GEBCO, NOAA").add_to(m)
    folium.TileLayer(esri.format("World_Imagery"), name="Esri Imagery", attr="Esri, Maxar").add_to(m)
    colors = {"fishing": SERIES[0], "loitering": SERIES[2], "encounter": SERIES[1], "port_visit": INK, "gap": SERIES[7]}
    for k, c in colors.items():
        x = ev[ev.event_type == k].head(max_points)
        if not len(x):
            continue
        fg = folium.FeatureGroup(name=f"{k} ({len(x)})")
        for r in x.itertuples():
            if k == "gap" and pd.notna(getattr(r, "on_lat", np.nan)):
                folium.PolyLine([[r.off_lat, r.off_lon], [r.on_lat, r.on_lon]], color=c, weight=1.5, dash_array="4 4",
                                tooltip=f"{r.vessel_id} AIS off {r.gap_duration_h:.0f} h").add_to(fg)
            else:
                folium.CircleMarker([r.lat, r.lon], radius=3, color=c, fill=True, weight=0,
                                    tooltip=f"{k} | {r.vessel_id} | {r.vessel_type} | {r.start:%Y-%m-%d}").add_to(fg)
        fg.add_to(m)
    if sar is not None and len(sar):
        fg = folium.FeatureGroup(name="SAR detections without AIS match")
        for r in sar.head(max_points).itertuples():
            folium.CircleMarker([r.lat, r.lon], radius=2, color=MUTED, fill=True, weight=0,
                                tooltip=f"SAR {r.date:%Y-%m-%d}: {r.detections} detection(s)").add_to(fg)
        fg.add_to(m)
    folium.GeoJson(layers.geojson(), name="Context layers (approx.)").add_to(m)
    folium.LayerControl().add_to(m)
    if path:
        m.save(str(path))
    return m


# --------------------------------------------------------------------------
# Real-data graph model (notebook 04b)
# --------------------------------------------------------------------------
REAL_MODEL_COLORS = {"Temporal GAT (full graph)": SERIES[0], "Temporal GAT (temporal edges only)": SERIES[6],
                     "Same model, no graph (MLP)": SERIES[2], "Gradient boosting (no graph)": SERIES[3],
                     "Persistence (this week = next week)": MUTED, "Activity only (n events this week)": "#c9c8c3"}
FISHING_BANS = (("East coast ban (15 Apr - 14 Jun)", "04-15", "06-14", SERIES[3]),
                ("West coast ban (1 Jun - 31 Jul)", "06-01", "07-31", SERIES[4]))


def plot_weekly_activity(nodes: pd.DataFrame, path=None, split_lines=None):
    """Active fishing vessels and AIS-off gaps per week, with India's seasonal fishing bans shaded."""
    f = nodes[nodes.vessel_class == "fishing"]
    wk = f.groupby("week").agg(vessels=("vessel_id", "nunique"), gaps=("n_gaps", "sum"))
    fig, ax = plt.subplots(figsize=(11, 3.8))
    _style(ax)
    years = sorted(set(wk.index.year))
    for i, (lab, a, b, c) in enumerate(FISHING_BANS):
        for y in years:
            ax.axvspan(pd.Timestamp(f"{y}-{a}"), pd.Timestamp(f"{y}-{b}"), color=c, alpha=0.12, lw=0,
                       label=lab if y == years[0] else None)
    ax.plot(wk.index, wk.vessels, color=SERIES[0], lw=1.6, label="active fishing vessels (with >= 1 event)")
    ax2 = ax.twinx()
    ax2.plot(wk.index, wk.gaps, color=SERIES[7], lw=1.0, alpha=0.8, label="AIS-off gaps by fishing vessels (1 h - 30 d)")
    ax2.tick_params(colors=INK2, labelsize=8)
    ax.set_ylim(0, wk.vessels.max() * 1.1)
    ax2.set_ylim(0, max(wk.gaps.max(), 1) * 1.1)
    for name, t in (split_lines or {}).items():
        ax.axvline(pd.Timestamp(t), color=INK, lw=0.8, ls="--")
        ax.annotate(name, (pd.Timestamp(t), 0), xytext=(3, 4), textcoords="offset points",
                    fontsize=7.5, color=INK2)
    ax.set_ylabel("fishing vessels per week", fontsize=8, color=INK2)
    ax2.set_ylabel("gaps per week", fontsize=8, color=INK2)
    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, fontsize=7.5, loc="upper center", bbox_to_anchor=(0.5, -0.1), frameon=False, ncol=4)
    ax.set_title("Real weekly fishing activity (GFW), with India's seasonal fishing bans shaded (approx. dates)",
                 loc="left", fontsize=10)
    fig.tight_layout()
    if path:
        fig.savefig(path, dpi=150, bbox_inches="tight")
    return fig


def _model_bars(tab, cat_col, val_col, models, ax, ref_col=None, ref_label=None, fmt="{:.2f}"):
    cats = list(dict.fromkeys(tab[cat_col]))
    h = 0.8 / len(models)
    for i, m in enumerate(models):
        s = tab[tab.model == m].set_index(cat_col)[val_col].reindex(cats)
        y = np.arange(len(cats)) + (i - (len(models) - 1) / 2) * h
        ax.barh(y, s.values, height=h * 0.9, color=REAL_MODEL_COLORS.get(m, MUTED), label=m)
        for yy, v in zip(y, s.values):
            if np.isfinite(v):
                ax.annotate(fmt.format(v), (v, yy), xytext=(2, 0), textcoords="offset points", va="center",
                            fontsize=6.5, color=INK2)
    if ref_col is not None:
        r = tab.drop_duplicates(cat_col).set_index(cat_col)[ref_col].reindex(cats)
        for k, v in enumerate(r.values):
            ax.plot([v, v], [k - 0.42, k + 0.42], color=INK, lw=1.2, label=ref_label if k == 0 else None)
    ax.set_yticks(np.arange(len(cats)))
    ax.set_yticklabels(cats, fontsize=8)
    ax.invert_yaxis()
    ax.legend(fontsize=7, frameon=False, loc="lower right")


def plot_real_task_scores(tsum: pd.DataFrame, split: str, path=None):
    from .real_model import TASKS
    t = tsum[tsum.split == split].copy()
    t["task_label"] = t.task.map(lambda k: TASKS.get(k, k))
    models = [m for m in REAL_MODEL_COLORS if m in set(t.model)]
    fig, ax = plt.subplots(figsize=(10, 1.0 + 0.75 * t.task.nunique()))
    _style(ax)
    _model_bars(t, "task_label", "pr_auc", models, ax, ref_col="base_rate", ref_label="base rate (random ranking)")
    ax.set_xlabel("Test PR-AUC for next-week forecast (higher is better)", fontsize=8, color=INK2)
    ax.set_title(f"Forecasting next-week behaviour on real data ({split} split)", loc="left", fontsize=10)
    fig.tight_layout()
    if path:
        fig.savefig(path, dpi=150, bbox_inches="tight")
    return fig


def plot_evidence_lift(esum: pd.DataFrame, split: str, path=None, metric="lift_top5pct"):
    from .real_model import EVIDENCE
    e = esum[esum.split == split].copy()
    e["ev_label"] = e.evidence.map(lambda k: EVIDENCE.get(k, k))
    models = [m for m in REAL_MODEL_COLORS if m in set(e.model)]
    fig, ax = plt.subplots(figsize=(10, 1.2 + 0.9 * e.evidence.nunique()))
    _style(ax)
    _model_bars(e, "ev_label", metric, models, ax, fmt="{:.1f}x")
    ax.axvline(1, color=INK, lw=1, ls="--")
    ax.set_xlabel("Lift in the top 5% of priority scores (1 = no better than random)", fontsize=8, color=INK2)
    ax.set_title(f"Held-out evidence the model never saw ({split} split, test weeks)", loc="left", fontsize=10)
    fig.tight_layout()
    if path:
        fig.savefig(path, dpi=150, bbox_inches="tight")
    return fig


def plot_real_alerts_map(nodes: pd.DataFrame, alerts: pd.DataFrame, gaps: pd.DataFrame, bbox, layers=None,
                         title: str = "", path=None):
    """Top-priority vessel-weeks and fishing-vessel AIS-off positions on the southern Indian EEZ.

    Gaps are drawn as the position where AIS went off only; straight off->on lines are
    not drawn because they are not the route the vessel took."""
    from global_land_mask import globe
    lat0, lat1, lon0, lon1 = bbox
    lats = np.arange(lat0, lat1 + 0.02, 0.03)
    lons = np.arange(lon0, lon1 + 0.02, 0.03)
    LO, LA = np.meshgrid(lons, lats)
    fig, ax = plt.subplots(figsize=(10, 9.4))
    _style(ax)
    land = globe.is_land(LA, LO).astype(float)
    ax.contourf(LO, LA, land, levels=[0.5, 1.5], colors=[LAND], zorder=1)
    ax.contour(LO, LA, land, levels=[0.5], colors=[MUTED], linewidths=0.5, zorder=2)
    if layers is not None:
        for g in layers.mpa.values():
            x, y = g.exterior.xy
            ax.fill(x, y, color=MPA_C, alpha=0.2, zorder=2)
        x, y = layers.imbl.xy
        ax.plot(x, y, color=IMBL_C, lw=1.2, ls=(0, (6, 3)), zorder=3)
    handles = []
    f = nodes[nodes.vessel_class == "fishing"]
    ax.hexbin(f.lon, f.lat, gridsize=70, extent=(lon0, lon1, lat0, lat1), mincnt=1, bins="log",
              cmap="Blues", alpha=0.55, zorder=2, linewidths=0)
    handles.append(Patch(fc="#9ecae1", label="fishing-vessel activity (vessel-weeks, log scale)"))
    if len(gaps):
        long_ = gaps.gap_h > 24
        for m, c, lab in ((~long_, SERIES[3], "AIS off 1-24 h"), (long_, SERIES[7], "AIS off 1-30 days")):
            g = gaps[m]
            ax.scatter(g.off_lon, g.off_lat, s=12, marker="v", color=c, alpha=0.8, lw=0, zorder=5)
            handles.append(Line2D([], [], marker="v", color=c, lw=0, ms=5,
                                  label=f"{lab}: where a fishing vessel's AIS went off (n={len(g):,})"))
    if len(alerts):
        s = 30 + 250 * (alerts.priority - alerts.priority.min()) / max(np.ptp(alerts.priority), 1e-9)
        ax.scatter(alerts.lon, alerts.lat, s=s, marker="*", color=SERIES[1], edgecolor=INK, lw=0.5, zorder=7)
        for r in alerts.head(10).itertuples():
            ax.annotate(str(r.rank), (r.lon, r.lat), xytext=(5, 4), textcoords="offset points", fontsize=7,
                        color=INK, zorder=8)
        handles.append(Line2D([], [], marker="*", color=SERIES[1], mec=INK, lw=0, ms=11,
                              label=f"top {len(alerts)} test vessel-weeks by investigation priority"))
    ax.set_xlim(lon0, lon1); ax.set_ylim(lat0, lat1)
    ax.set_aspect(1 / np.cos(np.deg2rad((lat0 + lat1) / 2)))
    ax.set_xlabel("Longitude (°E)", fontsize=8, color=INK2)
    ax.set_ylabel("Latitude (°N)", fontsize=8, color=INK2)
    ax.legend(handles=handles, loc="lower left", fontsize=7.5, facecolor="white", edgecolor=MUTED, framealpha=0.95)
    ax.set_title(title, fontsize=11, loc="left")
    fig.text(0.01, 0.005, "REAL DATA: Global Fishing Watch. Positions are weekly means (0.1°). Priority is for "
             "investigation only, not evidence of illegality. Vessel IDs anonymised.", fontsize=7, color=INK2)
    if path:
        fig.savefig(path, dpi=150, bbox_inches="tight")
    return fig


def plot_real_training(histories: dict, split: str, path=None):
    fig, ax = plt.subplots(1, 2, figsize=(11, 3.4))
    for a in ax:
        _style(a)
    for (sp, model, seed), h in histories.items():
        if sp != split:
            continue
        c = REAL_MODEL_COLORS.get(model, MUTED)
        ax[0].plot(h.epoch, h.loss, color=c, lw=1, alpha=0.8, label=model if seed == min(
            s for (p, m, s) in histories if p == sp and m == model) else None)
        ax[1].plot(h.epoch, h.val_macro_pr_auc, color=c, lw=1, alpha=0.8)
    ax[0].set_title("training loss (mean over tasks)", loc="left", fontsize=9)
    ax[1].set_title("validation macro PR-AUC", loc="left", fontsize=9)
    ax[0].legend(fontsize=7, frameon=False)
    for a in ax:
        a.set_xlabel("epoch", fontsize=8, color=INK2)
    fig.tight_layout()
    if path:
        fig.savefig(path, dpi=150, bbox_inches="tight")
    return fig


# --------------------------------------------------------------------------
# Milestone 4: explanations (notebook 05)
# --------------------------------------------------------------------------
ROLE_COLORS = {"this vessel, this week": SERIES[0], "this vessel, past weeks": SERIES[6], "other vessels": SERIES[3]}


def plot_alert_explanation(card: dict, title: str = "", path=None, top: int = 8):
    """Left: what drove the score (Integrated Gradients by role x feature group).
    Right: what-if tests (remove history / other vessels / one behaviour) and the new priority."""
    import textwrap
    att = card["attribution"].head(top).iloc[::-1]
    fig, ax = plt.subplots(1, 2, figsize=(12.5, 4.4), gridspec_kw={"width_ratios": [1.15, 1]})
    for a in ax:
        _style(a)
    ax[0].barh([f"{g}\n({r})" for r, g in zip(att.role, att.group)], att.contribution,
               color=[ROLE_COLORS.get(r, MUTED) for r in att.role])
    ax[0].axvline(0, color=INK, lw=0.8)
    ax[0].tick_params(axis="y", labelsize=7)
    ax[0].set_xlabel("contribution to the priority score (Integrated Gradients)", fontsize=8, color=INK2)
    ax[0].set_title("What drove the score", loc="left", fontsize=9.5)
    rows = [("as scored", card["priority"])]
    t = card["temporal"]
    if len(t) and (t.removed == "all past weeks").any():
        rows.append(("without its past weeks", float(t[t.removed == "all past weeks"].priority.iloc[0])))
    e = card["edges"]
    for lab, key in (("without encounter links", "all encounter"), ("without co-location links", "all co-location"),
                     ("scored alone (no graph)", "every edge")):
        m = e.removed.str.startswith(key)
        if m.any():
            rows.append((lab, float(e[m].priority.iloc[0])))
    for r in card["counterfactual"].head(3).itertuples():
        rows.append((f"had it not: {r[1]}", r.priority))
    labs, vals = zip(*rows)
    y = np.arange(len(rows))[::-1]
    cols = [INK] + [SERIES[1] if v < card["priority"] - 0.01 else MUTED for v in vals[1:]]
    ax[1].barh(y, vals, color=cols)
    ax[1].set_yticks(y)
    ax[1].set_yticklabels(["\n".join(textwrap.wrap(l, 34)) for l in labs], fontsize=7)
    for yy, v in zip(y, vals):
        ax[1].annotate(f"{v:.2f}", (v, yy), xytext=(3, 0), textcoords="offset points", va="center", fontsize=7)
    ax[1].set_xlim(0, 1.05)
    ax[1].set_xlabel("investigation priority", fontsize=8, color=INK2)
    ax[1].set_title("What-if tests", loc="left", fontsize=9.5)
    fig.suptitle(title, x=0.01, ha="left", fontsize=10.5)
    fig.text(0.01, -0.06, "\n".join(textwrap.wrap(card["reason"], 170)), fontsize=8, color=INK)
    fig.text(0.01, -0.12, "Explains the model's score, not the vessel's intent. Not evidence of illegality.",
             fontsize=7, color=INK2)
    fig.tight_layout()
    if path:
        fig.savefig(path, dpi=150, bbox_inches="tight")
    return fig


def plot_global_importance(tables: dict, path=None, top: int = 10):
    fig, ax = plt.subplots(figsize=(10, 0.9 + 0.42 * top))
    _style(ax)
    groups = list(dict.fromkeys(pd.concat(tables.values()).sort_values("importance", ascending=False).group))[:top]
    h = 0.8 / len(tables)
    colors = [SERIES[0], SERIES[3], SERIES[2], SERIES[6]]
    for i, (name, t) in enumerate(tables.items()):
        s = t.set_index("group").reindex(groups)
        yy = np.arange(len(groups)) + (i - (len(tables) - 1) / 2) * h
        ax.barh(yy, s.importance, xerr=s.sd, height=h * 0.9, color=colors[i % 4], label=name,
                error_kw={"elinewidth": 0.6, "ecolor": INK2})
    ax.set_yticks(np.arange(len(groups)))
    ax.set_yticklabels(groups, fontsize=8)
    ax.invert_yaxis()
    ax.axvline(0, color=INK, lw=0.8)
    ax.set_xlabel("drop in test macro PR-AUC when the feature group is shuffled", fontsize=8, color=INK2)
    ax.set_title("Which information the models rely on (global, test weeks)", loc="left", fontsize=10)
    ax.legend(fontsize=7.5, frameon=False, loc="lower right")
    fig.tight_layout()
    if path:
        fig.savefig(path, dpi=150, bbox_inches="tight")
    return fig


def plot_explanation_quality(q: pd.DataFrame, path=None):
    fig, ax = plt.subplots(1, 2, figsize=(11, 3.4))
    for a in ax:
        _style(a)
    x = np.arange(len(q))
    ax[0].bar(x - 0.2, q.drop_top_k, 0.4, color=SERIES[0], label="remove top-5 attributed inputs")
    ax[0].bar(x + 0.2, q.drop_random_k, 0.4, color=MUTED, label="remove 5 random inputs")
    ax[0].set_xticks(x)
    ax[0].set_xticklabels(q["rank"].astype(str), fontsize=7)
    ax[0].set_xlabel("alert rank", fontsize=8, color=INK2)
    ax[0].set_ylabel("priority drop", fontsize=8, color=INK2)
    ax[0].set_title("Deletion fidelity (higher blue = faithful)", loc="left", fontsize=9.5)
    ax[0].legend(fontsize=7, frameon=False)
    ax[1].bar(x, q.stability, color=SERIES[2])
    ax[1].set_ylim(-1, 1)
    ax[1].axhline(0, color=INK, lw=0.6)
    ax[1].set_xticks(x)
    ax[1].set_xticklabels(q["rank"].astype(str), fontsize=7)
    ax[1].set_xlabel("alert rank", fontsize=8, color=INK2)
    ax[1].set_title("Stability across 3 trained seeds (Spearman)", loc="left", fontsize=9.5)
    fig.tight_layout()
    if path:
        fig.savefig(path, dpi=150, bbox_inches="tight")
    return fig
