# SCI — Structural Coherence Index

**Author:** Parker J. Lee  
**Patent:** US Provisional 63/904,444 (filed Oct 23, 2025)  
**Version history:** v1 (bearing paper, Zenodo May 2026) → v2 (seed=42, Sleep-EDF paper) → v3 (full diagnostic output, this repo)

---

## What is SCI?

SCI measures excess Hilbert-envelope autocorrelation relative to a phase-randomized surrogate baseline that preserves the power spectrum exactly.

**Core formula:**

```
c_obs  = mean ACF of Hilbert envelope across lags 1–12
gap    = c_obs − c_surr_mean        ← THE scientific object
z      = gap / c_surr_std           (clipped to ±6)
SCI    = 1 / (1 + exp(−0.9 × z))   (logistic, range [0, 1])
```

**What gap measures:** nonlinear amplitude organization beyond what the power spectrum predicts.  
A positive gap means the signal's amplitude envelope has temporal structure that phase-randomized surrogates — which match the real signal's power spectrum exactly — do not produce.

**Classification thresholds:**

| Bucket     | SCI range |
|------------|-----------|
| CORE       | ≥ 0.75    |
| CORE_MID   | ≥ 0.65    |
| TACTICAL   | ≥ 0.55    |
| INELIGIBLE | < 0.55    |

---

## Locked parameters (DO NOT CHANGE for pre-registered work)

| Parameter | Value | Meaning                        |
|-----------|-------|--------------------------------|
| W         | 12    | Envelope smoothing window      |
| L         | 12    | ACF lag count (lags 1–L)       |
| S         | 40    | Phase-randomized surrogates    |
| k         | 0.9   | Logistic slope                 |
| seed      | 42    | RNG seed (v2+)                 |
| z_clip    | 6.0   | z-score clip before logistic   |

See `params_locked.py` for the authoritative constants.

Domain-specific adaptations (W=6/L=6 for cells, W=7/L=10 for daily finance) are documented in `params_locked.py` — the core math is identical across all domains.

---

## Installation

```bash
pip install numpy scipy pandas matplotlib seaborn mne
```

---

## Repository structure

```
SCI_Project/
├── sci_score_v3.py          # Canonical SCI function (import this)
├── params_locked.py         # Frozen parameter constants
├── scripts/
│   ├── sci_bearing_full_rerun_v2.py    # Bearings (CWRU/MFPT/IMS/FEMTO)
│   ├── sci_cells_full_rerun_v2.py      # Cell death (Huh7 + HeLa/Huh7 images)
│   ├── sci_eeg_full_rerun_v2.py        # CHB-MIT seizure EEG
│   ├── sci_eeg_full_analysis.py        # Sleep-EDF (26 subjects, AUC=0.959)
│   ├── sci_killbox.py                  # Pre-registered kill box (May 2, 2026)
│   ├── sci_garch_gap_mapper.py         # GARCH(1,1) parameter sweep
│   └── finance/
│       ├── financial_garch_sci_mapper.py
│       ├── garch_sci_postprocess.py
│       ├── forward_return_by_amplitude_bucket.py
│       ├── amplitude_bucket_portfolio_backtest.py
│       ├── amplitude_transaction_cost_analysis.py
│       ├── amplitude_risk_quality_analysis.py
│       ├── factor_control_analysis.py
│       ├── add_spy_benchmark_analysis.py
│       ├── make_report_grade_portfolio_outputs.py
│       └── live_sci_watchlist_generator.py
└── results/                 # Output CSVs and figures (gitignored if large)
```

---

## Quick start — import SCI v3

```python
from sci_score_v3 import sci_score_v3, sci_score_v3_scalar

result = sci_score_v3(signal)
# Returns dict: {c_obs, c_surr_mean, c_surr_std, gap, z, SCI, bucket}

sci = sci_score_v3_scalar(signal)  # backwards-compatible float
```

Run the built-in self-test:

```bash
python sci_score_v3.py
# Expected:
#   White noise:   gap≈0   SCI<0.6
#   GARCH(1,1):    gap>0.01  SCI>0.75  bucket=CORE
#   Self-test passed.
```

---

## How to run each domain script

### Bearings (CWRU / MFPT / IMS / FEMTO)

**Data:** Download from the respective public repositories and set paths in the script.  
**Parameters:** W=12, L=12, S=40, seed=42 on 500 Hz bandpass-filtered vibration segments (~6000 samples).

```bash
python scripts/sci_bearing_full_rerun_v2.py
# Outputs: results/bearing_sci_results.csv (segment-level diagnostic stack)
```

**Key result:** Healthy bearings → gap ≈ 0; race-fault bearings → gap clearly positive (CORE bucket).

---

### Cell death (Huh7 text + HeLa/Huh7 image sequences)

**Data:** Huh7 text time series + fluorescence image sequences.  
**Parameters:** W=6, L=6 (appropriate for short fluorescence series, ~20–200 points). Core math identical.

```bash
python scripts/sci_cells_full_rerun_v2.py
# Outputs: results/cells_sci_results.csv
```

**Key result:** NP25/NP100 treatment vs. control separation confirmed.

---

### EEG — CHB-MIT seizure (boundary/refinement result)

**Data:** CHB-MIT Scalp EEG Database (physionet.org). Set `DATA_DIR` in script.  
**Parameters:** W and L derived from sample rate (smooth_sec × fs, lag_sec × fs). Core math identical.

```bash
python scripts/sci_eeg_full_rerun_v2.py
# Outputs: results/eeg_seizure_sci_results.csv
```

**Key result:** Gap *collapses* during seizure — hypersynchrony is spectrum-dominated, so surrogates replicate it. This is a pre-registered boundary/refinement finding.

---

### EEG — Sleep-EDF consciousness (main result)

**Data:** PhysioNet Sleep-EDF Database (26 subjects). Default path: `~/Desktop/SCI_Consciousness/mne_data/physionet-sleep-data/`

```bash
python scripts/sci_eeg_full_analysis.py
# Outputs: results/sleep_edf_sci_results.csv + mixed-effects model summary
```

**Key result:** Wake SCI=0.970, Stage 4 SCI=0.707, Cohen d=3.991, AUC=0.959 (LOSO-CV, 26 subjects).

---

### Kill box (pre-registered, run once)

**Pre-registration date:** May 2, 2026.  
**Parameters:** W=12, L=12, S=40, seed=42, N=10,000 synthetic samples at 1000 Hz.

```bash
python scripts/sci_killbox.py
# Outputs: results/killbox_results/ (one-time run, do not re-run with modified params)
```

**Key results:**
- GARCH(1,1): SCI=0.996, gap=0.163 ✓ (CORE, as predicted)
- AR(1): boundary case — gap real but too small ✓ (as predicted)
- AM signal, Chirp: failed (as predicted)

---

### GARCH parameter sweep

```bash
python scripts/sci_garch_gap_mapper.py
# Outputs: results/garch_heatmaps/ (gap vs alpha/beta grid)
```

**Key result:** Gap rises monotonically with GARCH persistence (alpha+beta). Provides theoretical grounding.

---

### Finance (daily returns backtest)

**Data:** Daily OHLCV prices in `cache_prices/*.csv`, ticker list in `tickers.txt`.  
**Parameters:** W=7 (smooth), L=10 (lag), window=500 days — scaled for daily returns. Core math identical.

Run in order:

```bash
# 1. Theory validation: does vol clustering predict SCI gap?
python scripts/finance/financial_garch_sci_mapper.py

# 2. Add percentile buckets + amplitude scores
python scripts/finance/garch_sci_postprocess.py

# 3. Main backtest: rolling SCI gaps → forward returns
python scripts/finance/forward_return_by_amplitude_bucket.py

# 4. Equal-weight portfolio equity curves
python scripts/finance/amplitude_bucket_portfolio_backtest.py

# 5. Transaction cost sensitivity (0–50 bps)
python scripts/finance/amplitude_transaction_cost_analysis.py

# 6. Quality metrics: Sharpe, win rate, dispersion per bucket
python scripts/finance/amplitude_risk_quality_analysis.py

# 7. OLS controlling for momentum/vol/drawdown
python scripts/finance/factor_control_analysis.py

# 8. CORE_TOP10 vs SPY benchmark
python scripts/finance/add_spy_benchmark_analysis.py

# 9. Report-grade charts + summary
python scripts/finance/make_report_grade_portfolio_outputs.py

# 10. Live forward-test snapshot (today's CORE_TOP10)
python scripts/finance/live_sci_watchlist_generator.py
```

**Key result:** Sharpe ~0.50 on pre-model screening (before any factor control).  
Output dirs: `results_forward_amplitude/`, `results_garch_finance/`, `live_watchlists/`

---

## Reproducing results end-to-end

1. Install dependencies: `pip install numpy scipy pandas matplotlib seaborn mne`
2. Acquire data (links in each script's header or comments)
3. Run `python sci_score_v3.py` — confirm self-test passes
4. Run domain scripts in any order; outputs go to `results/`

All scripts use the locked parameters from `params_locked.py`. Do not modify locked scripts — create a new version label for any parameter change.

---

## Citation / prior art

```
Lee, P. J. (2026). Structural Coherence Index: bearing fault detection via 
Hilbert-envelope surrogate gap. Zenodo. (May 2, 2026)

US Provisional Patent 63/904,444 (filed October 23, 2025)
```
