# Multi-station machine-learning system for short-term eruption forecasting at Ubinas volcano (Peru)

This repository contains the code, catalogs, and figure scripts accompanying
the article:

> Centeno, R., Ibáñez, J. M., et al. (2026). *A transferable multi-station machine-learning
> system for short-term eruption forecasting: application to Ubinas volcano (Peru).*
> Frontiers in Earth Science. [DOI to be added upon publication]

It reproduces the classification of volcanic states and the early-warning
results reported in the paper, training exclusively on the 2023 eruptive crisis
and validating, without retraining, on the independent 2019 crisis.

---

## 1. Repository structure

```
.
├── README.md
├── LICENSE.txt                 (MIT, for the code)
├── requirements.txt            (Python dependencies)
├── pipeline/                   (the processing pipeline, stages 0–7)
│   ├── config.py               Stage 0 — single source of truth (paths, catalogs, parameters)
│   ├── load.py                 Stage 1 — loading and normalization to a 1-min grid
│   ├── label.py                Stage 2 — labeling of the four volcanic states (2019 and 2023)
│   ├── features.py             Stage 3 — causal z-score normalization + 85 features
│   ├── train.py                Stage 4 — Random Forest training + LOSO validation
│   ├── validate.py             Stage 5 — blind validation on 2019 (26-column output)
│   ├── postprocess.py          Stage 6 — causal HMM, thresholds, unrest and event anticipation
│   ├── event_type.py           Stage 7 — prospective explosion/emission classifier
│   ├── sustained.py            Shared alert-density definition (used across stages)
│   ├── run_all.py              Runs the full pipeline (stages 0–7)
│   ├── run_stages_0_1_2.py     Runs only loading + labeling (quick check)
│   └── diagnostics/            Optional diagnostic and threshold-sweep scripts
├── figures/                    Scripts to reproduce the paper figures
│   ├── plot_fig2_fig3_split.py     Figures 2–3 (parameters and normalization)
│   ├── plot_fig4.py                Figure 5 (model performance)
│   └── plot_fig5_fig6_split_v2.py  Figures 6–7 (unrest and eruptive alert)
├── catalogs/
│   ├── ubinas_catalog_2019.csv     9 episodes (3 explosions, 6 emissions)
│   └── ubinas_catalog_2023.csv     76 episodes (17 explosions, 59 emissions)
├── supplementary/
│   └── Figure_S3_threshold_sweep.png
└── data/                       (INPUT DATA — see Section 3)
```

---

## 2. Installation

```bash
python -m venv venv && source venv/bin/activate   # (or conda)
pip install -r requirements.txt
```
Tested with Python 3.9–3.11.

---

## 3. Input data

The pipeline operates on **per-minute seismic parameter series** (Shannon
entropy, kurtosis, frequency index, and SSAM) for stations UBI1, UBI2, and
UBI4, one CSV per station per period:

```
data/Parametros_7params_UB1_BHZ_20230101_to_20231231_wd.csv
data/Parametros_7params_UB2_BHZ_20230101_to_20231231_wd.csv
data/Parametros_7params_UB4_BHZ_20230101_to_20231231_wd.csv
data/Parametros_7params_UB4_BHZ_20180814_to_20191231_wd.csv
```

Each file has 1-min resolution with columns:
`DateTime_UTC, SH_Shannon, Curtosis, FreqIndex, SSAM_Mean, SH_Shannon_Smooth,
FreqIndex_Smooth, Station, Has_Gaps`.

These derived-parameter files are sufficient to reproduce **all** the figures
and results in the paper.

> **Raw seismic waveforms** are the property of the Instituto Geofísico del
> Perú (IGP) and are available from the IGP upon reasonable request. The
> derived parameters provided here allow full reproduction of the analysis
> without the raw traces.

Before running, edit `pipeline/config.py` → `BASE_DATA` to point to your `data/`
folder.

---

## 4. How to reproduce the results

```bash
cd pipeline
python run_all.py
```

This runs the seven stages in order and writes all intermediate outputs
(Parquet files, trained models, and the 2019 validation CSV) to
`pipeline/intermediate/`. The console reports the key results of the paper:

- LOSO AUC per station and mean (Stage 4)
- Blind 2019 operational AUC (Stage 5)
- Operational threshold, unrest declaration, and per-event anticipation (Stage 6)
- Explosion/emission cross-validated accuracy (Stage 7)

To regenerate the figures (after `run_all.py`):

```bash
cd ../figures
python plot_fig4.py                 # Figure 5
python plot_fig5_fig6_split_v2.py   # Figures 6–7
python plot_fig2_fig3_split.py      # Figures 2–3
```
(Adjust the data paths at the top of each figure script to your environment.)

---

## 5. Key parameters (in config.py)

| Parameter | Value | Meaning |
|---|---|---|
| W_ROLLING_DAYS | 30 | Causal z-score normalization window |
| N_TREES | 400 | Random Forest trees |
| MIN_LEAF | 20 | Minimum leaf size |
| N_FEATURES_FINAL | 45 | Features retained (from 85) |
| PERSIST_HOURS | 6 | Sustained-alert window |
| DENSITY_FRAC | 0.85 | Alert-density criterion |
| ASSOC_WINDOW_H | 24 | Event–alert association window |

The operational threshold (th_op = 0.62) is calibrated as the lowest value with
no sustained false declarations during the 2019 quiescence (see Figure S3).

---

## 6. Reproducibility notes

- All computations are strictly causal (only present and past information).
- The Random Forest uses `random_state=0` for reproducibility.
- Stage 6 uses a causal Viterbi decoding for operational states; a retrospective
  decoding is computed only for descriptive comparison.

---

## 7. License and citation

Code is released under the MIT License (see LICENSE.txt). If you use this code
or the catalogs, please cite the article above and this repository
(Zenodo DOI: 10.5281/zenodo.23051412).
