# Data

This folder starts empty. The notebooks fill it when they run. No raw data are committed:
- the synthetic data are regenerated from a seed;
- the real data belong to Global Fishing Watch and must be downloaded with your own free token.

| Sub-folder | Filled by | Contents |
|---|---|---|
| `synthetic/` | notebook 01 | Raw synthetic AIS (7 days, 50 vessels) and `hidden_truth/`, which is for evaluation only |
| `synthetic_m2/` | notebooks 03, 04 | 60-day synthetic AIS (62 vessels) used for the baselines and the synthetic graph model |
| `processed/` | notebooks 01, 02 | Cleaned AIS with features, gap tables, and `real_gfw/` (cleaned real events, SAR tables, `provenance.json`) |
| `processed/_private/` | notebook 02 | Mapping from anonymised IDs to real IDs. **Git-ignored. Never share it.** |
| `raw/` | notebook 02 (local runs) | Month-by-month GFW download cache |

In Colab, the real-data files live on Google Drive instead, under `MyDrive/india-fishing-gnn-data/`:
- `real_gfw/` (cleaned data);
- `gfw_raw_cache/` (download cache);
- `outputs_04b/` and `outputs_05/` (results).

## 1. Synthetic AIS (generated)

- **Area:** Gulf of Mannar – Palk Bay, 8.0–10.2°N, 77.9–80.2°E. Distances are computed in UTM 44N (EPSG:32644).
- **Fleets:**
  - 7-day run: 50 vessels (32 fishing, 8 of them scripted "actors"; 4 carriers; 10 cargo/tanker; 2 tugs; 2 ferries).
  - 60-day run: 62 vessels (40 fishing, 12 actors; 6 carriers, 4 suspicious; 12 cargo/tanker; 2 tugs; 2 ferries).
- **Simulation:** each vessel is moved every minute. A coverage model decides which AIS messages are received: 7 coastal receivers, satellite passes, Class B range, faults, and a recurring receiver outage. Realistic errors are planted: duplicates, jumps, impossible coordinates, and the 102.3 kn and 511° "not available" codes.
- **Hidden truth:** true positions, the cause of every lost message, and every planted event. It is used only for evaluation.
- **Size:**
  - 7-day run: 38,281 raw rows, 37,792 after cleaning.
  - 60-day run: 9,618 six-hour windows, 7.4% with planted suspicious behaviour.
- **Labels** are prefixed `synthetic_demo_` and are never presented as real.

## 2. Global Fishing Watch (real, 2021–2025)

- **Source:** GFW API v3 through `gfw-api-python-client`.
- **Datasets:**
  - fishing, encounter, loitering, port-visit and AIS-gap events;
  - `public-global-sar-presence` (Sentinel-1);
  - `public-global-fishing-effort`.
- **Area:** 5–16°N, 70–82°E. This covers Kerala, the Lakshadweep Sea, Tamil Nadu and the Gulf of Mannar, plus parts of the Sri Lankan and Maldivian EEZs.
- **Period:** 1 Jan 2021 – 31 Dec 2025.
- **5-year volume:**
  - 1,878,474 events;
  - 136,950 Sentinel-1 cell-days with detections that match no AIS vessel;
  - 181,002 cell-days with any detection.
- **Cleaning** (notebook 04b):
  - drop events that started before the study period;
  - keep gaps of 1 h – 30 days (913 of 1,004);
  - cap single-event durations at 7 days;
  - mirror each encounter to both vessels;
  - keep fishing-related vessels and their encounter partners (14,198 vessels, 995,433 events).
- **Anonymisation:** vessel IDs become `vessel_xxxxxxxx`, a keyed HMAC-SHA256 hash. Set a private `AIS_ID_SALT`.
- **Licence:** GFW non-commercial terms; attribution to Global Fishing Watch is required.

See [`../DATA_SOURCE.md`](../DATA_SOURCE.md) for columns, map layers and known limitations.
