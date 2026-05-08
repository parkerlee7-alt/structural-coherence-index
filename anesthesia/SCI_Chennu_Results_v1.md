# Structural Coherence Index During Propofol Sedation
## Pre-registered analysis of the Chennu et al. (2014) Cambridge dataset

**Author:** Parker J. Lee  
**Date:** 2026-05-07  
**Patent:** US Provisional 63/904,444  
**Pre-registration:** `SCI_Anesthesia_PreRegistration_v1.md` (written 2026-05-05, before data download)  
**Script:** `scripts/sci_chennu_v1.py`  
**Data DOI:** https://doi.org/10.17863/CAM.68959

---

## Abstract

We applied the Structural Coherence Index (SCI) to resting-state EEG recorded during graded propofol sedation in 20 healthy volunteers (Chennu et al., 2014). SCI measures excess Hilbert-envelope autocorrelation relative to a phase-randomized surrogate baseline that preserves the power spectrum exactly. The pre-registered primary prediction — that SCI would decrease monotonically with sedation depth, mirroring the Wake > Stage 4 ordering from the Sleep-EDF result — was not confirmed. Instead, SCI increased significantly from baseline (0.751 ± 0.099) to moderate sedation (0.838 ± 0.084; paired t-test: t(19)=−3.11, p=0.006, Cohen d=−0.695; Wilcoxon p=0.006). The Friedman test across all four states was highly significant (χ²=20.58, p=0.0001), and the mild-to-moderate transition was the largest single contrast (d=−1.295, p<0.0001). This outcome — pre-registered as a "surprising positive finding" — indicates that propofol-induced sedation imposes amplitude envelope organization beyond what the power spectrum predicts, producing a distinct SCI signature from both natural sleep and seizure.

---

## 1. Introduction

The Structural Coherence Index (SCI) was developed to quantify the degree to which a signal's amplitude envelope is organized beyond its own power spectral structure. The core measure is a *gap* between observed Hilbert-envelope autocorrelation and the mean autocorrelation of an ensemble of phase-randomized surrogates that preserve the power spectrum exactly:

```
c_obs  = mean ACF of Hilbert envelope across lags 1–L
gap    = c_obs − c_surr_mean
z      = gap / c_surr_std           (clipped to ±6)
SCI    = 1 / (1 + exp(−0.9 × z))   (logistic, range [0, 1])
```

A positive gap means the signal contains amplitude modulation structure that its spectrum does not predict. Prior validated results across domains are:

| Domain | Key result |
|--------|-----------|
| Sleep-EDF (26 subjects) | Wake SCI=0.970, Stage 4 SCI=0.707, Cohen d=3.991, AUC=0.959 |
| CHB-MIT seizure EEG | Gap *collapses* during seizure — hypersynchrony is spectrum-dominated |
| Bearing fault detection | Healthy → gap≈0; race fault → gap strongly positive |
| GARCH(1,1) kill box | SCI=0.996, gap=0.163 (pre-registered May 2, 2026) |

The sleep result established that SCI tracks conscious state across natural sleep stages, declining monotonically from wakefulness through Stage 4. Propofol anesthesia provides a pharmacologically controlled model with graded depth and a known return of consciousness at recovery, making it a logical next test of the SCI-consciousness hypothesis.

---

## 2. Pre-registered Predictions

The pre-registration document (`SCI_Anesthesia_PreRegistration_v1.md`) was written on **2026-05-05** before any data was downloaded or analyzed. The primary prediction was:

**P1 — Monotonic decrease:**
```
SCI(Baseline) > SCI(Mild) > SCI(Moderate)
gap(Baseline) > gap(Mild) > gap(Moderate)
```

Secondary predictions:

**P2 — Recovery approximates Baseline:**
```
|SCI(Recovery) − SCI(Baseline)| < |SCI(Moderate) − SCI(Baseline)|
```

**P3 — Moderate SCI in Stage 3–4 range:**
```
0.65 ≤ SCI(Moderate) ≤ 0.85
```

**P4 — Statistical significance:**
```
Paired t-test (Baseline vs. Moderate): p < 0.05
```

The pre-registration also explicitly acknowledged an alternative: *"If gap is positive in Moderate sedation and exceeds Baseline, that would be a surprising positive finding — propofol imposing amplitude coherence beyond spectrum prediction. Not predicted but would be reported."*

---

## 3. Methods

### 3.1 Dataset

The Chennu et al. (2014) Cambridge propofol sedation dataset was used (DOI: 10.17863/CAM.68959). Twenty healthy adult volunteers underwent graded propofol sedation. EEG was recorded at 250 Hz using a 91-channel BioSemi ActiveTwo system and preprocessed into 10-second epochs. Four sedation conditions were collected per subject in chronological order: **baseline** (eyes-closed wakefulness), **mild sedation**, **moderate sedation**, and **recovery** (return to full responsiveness). A median of 39 epochs were available per condition per subject (range: 25–43).

### 3.2 Signal Processing

For each epoch, the Fz electrode (channel index 8) was selected — a frontal midline channel consistent with the Sleep-EDF analysis. Epochs were bandpass-filtered (1–40 Hz, 4th-order Butterworth, zero-phase via `scipy.signal.filtfilt`).

### 3.3 SCI Computation

SCI v3 was computed on each filtered epoch using locked parameters scaled to 250 Hz:

| Parameter | Value | Derivation |
|-----------|-------|-----------|
| W (smoothing) | 25 samples | round(0.10 s × 250 Hz) |
| L (ACF lags) | 25 samples | round(0.10 s × 250 Hz) |
| S (surrogates) | 40 | locked |
| k (logistic slope) | 0.9 | locked |
| seed | 42 | locked |
| z_clip | 6.0 | locked |

Note: The core parameters W=12/L=12 in `params_locked.py` are specified in samples at the original domain's native sample rate. For EEG at 250 Hz, the time-equivalent scaling (smooth_sec=0.10 s) yields W=L=25, exactly as applied in `sci_eeg_full_analysis.py` for the Sleep-EDF dataset (256 Hz → W=L=26).

The full diagnostic stack (c_obs, c_surr_mean, c_surr_std, gap, z, SCI, bucket) was computed per epoch. Subject-level means were computed by averaging across all epochs within each condition, yielding one SCI value per subject per state (N=20 observations per state) as the unit of analysis.

### 3.4 Statistical Tests

Per the pre-registration analysis plan:

- **Primary:** Paired t-test, Baseline vs. Moderate (two-tailed)
- **Omnibus:** Friedman test across all four states
- **Pairwise:** Wilcoxon signed-rank tests for all six pairs with Bonferroni correction (α=0.05/6=0.0083)
- **Effect sizes:** Cohen's d for all comparisons

Data files read from `.set` (MATLAB v7.3 HDF5 format) and paired `.fdt` (float32 binary, shape [channels × samples × trials] in Fortran/column-major order) using h5py, as MNE-Python's EEGLAB reader does not support MATLAB v7.3 HDF5.

---

## 4. Results

### 4.1 SCI and Gap by State

**Table 1.** Subject-level means (N=20) by sedation state.

| State | SCI (mean ± SD) | gap (mean) | z (mean) |
|-------|-----------------|-----------|---------|
| Baseline | 0.7513 ± 0.0990 | 0.0715 | 1.575 |
| Mild | 0.7128 ± 0.1041 | 0.0584 | 1.268 |
| **Moderate** | **0.8379 ± 0.0840** | **0.1020** | **2.344** |
| Recovery | 0.7151 ± 0.1103 | 0.0601 | 1.298 |

SCI was highest at moderate sedation, not lowest as predicted. The pattern across states was: Moderate > Baseline > Recovery > Mild.

### 4.2 Statistical Tests

**Omnibus:** Friedman χ²(3)=20.58, **p=0.0001** — the four states differ significantly.

**Table 2.** Pairwise comparisons (Bonferroni-corrected threshold α=0.0083).

| Comparison | t | p (t-test) | p (Wilcoxon) | Cohen d |
|-----------|---|-----------|-------------|---------|
| Baseline vs. Mild | 1.750 | 0.096 | 0.105 | 0.391 |
| **Baseline vs. Moderate** ★ | **−3.107** | **0.006** | **0.006** | **−0.695** |
| Baseline vs. Recovery | 2.052 | 0.054 | 0.105 | 0.459 |
| Mild vs. Moderate | −5.793 | <0.0001 | <0.0001 | −1.295 |
| Mild vs. Recovery | −0.130 | 0.898 | 0.870 | −0.029 |
| Moderate vs. Recovery | 4.929 | 0.0001 | 0.0001 | 1.102 |

★ Primary pre-registered test. Significant after Bonferroni correction.

### 4.3 Pre-registration Scorecard

| Prediction | Outcome |
|-----------|---------|
| P1 — Monotonic decrease (Baseline > Mild > Moderate) | **Not confirmed** — Moderate was highest |
| P2 — Recovery closer to Baseline than Moderate | **Confirmed** (Δ=0.063 vs. Δ=0.117) |
| P3 — 0.65 ≤ SCI(Moderate) ≤ 0.85 | **Confirmed** (0.838) |
| P4 — Paired t-test p<0.05 | **Confirmed** (p=0.006) — but opposite direction |

---

## 5. Discussion

### 5.1 SCI Increases with Moderate Propofol Sedation

The primary pre-registered prediction (monotonic decrease) was not confirmed. Instead, SCI increased significantly at moderate sedation relative to wakefulness. This is the "surprising positive finding" acknowledged in the pre-registration: propofol-induced sedation imposes amplitude envelope organization that phase-randomized surrogates cannot replicate, driving gap higher.

This result is physiologically interpretable. Moderate propofol sedation is characterized by structured slow oscillations (~0.5–1 Hz) and frontal alpha spindles (~10 Hz) — neural signatures that are qualitatively different from waking EEG. These oscillations carry temporally organized amplitude modulation: during a slow oscillation, local neural populations transition in a coordinated fashion between up and down states, producing envelope patterns with more autocorrelation than the spectrum predicts. The surrogates, which randomize phase while preserving power, cannot reproduce this structure — hence the gap rises.

### 5.2 Contrast with the Sleep Result

The Sleep-EDF result showed Wake SCI=0.970 and Stage 4 SCI=0.707 — SCI decreasing with depth. The current result shows the opposite. These are not contradictory: they reflect different oscillatory regimes.

| State | Dominant oscillation | SCI direction vs. waking |
|-------|---------------------|------------------------|
| NREM Stage 4 (sleep) | High-amplitude slow waves, K-complexes | Decreases |
| Propofol moderate sedation | Structured alpha + slow oscillations | Increases |
| Seizure (ictal) | Hypersynchronous rhythmic discharge | Collapses |

The unifying principle is that SCI measures *nonlinear amplitude organization beyond the power spectrum* — it does not map linearly onto any single clinical dimension. Deep NREM sleep produces slow waves whose envelope is largely captured by the power spectrum, so gap falls. Propofol sedation at moderate doses produces oscillations with additional amplitude structure not fully captured by the spectrum, so gap rises. Seizure produces hypersynchrony that is almost entirely spectrum-dominated, so gap collapses to near zero or below.

### 5.3 Mild Sedation as a Transition Point

The largest single effect was the mild-to-moderate transition (d=−1.295, p<0.0001 after Bonferroni correction). Mild sedation (SCI=0.713) was statistically indistinguishable from both baseline and recovery. Moderate sedation (SCI=0.838) was clearly elevated above all other states. This suggests SCI may be sensitive to the threshold crossing into moderate sedation, which is the clinically relevant transition for anesthesia monitoring.

### 5.4 Spatial Distribution of the SCI Effect

A follow-up analysis (`scripts/sci_chennu_spatial_v1.py`) ran SCI on all 91 channels for all 20 subjects using the same locked parameters. The moderate − baseline SCI difference was computed per channel and visualized as a topographic map.

The largest increases at moderate sedation were concentrated over **frontal and fronto-central regions**: channels near FCz (E6, Δ=+0.099), bilateral frontal (E5/E12 ≈ F3/F4 region, Δ=+0.094–0.096), and the canonical Fz (Δ=+0.089). This spatial pattern is consistent with the known frontally maximal distribution of propofol-induced alpha oscillations (~10 Hz), which are the dominant oscillatory signature at moderate sedation doses. The fronto-central topography also parallels the spatial distribution of propofol-induced slow oscillations (<1 Hz), which travel anteriorly in the cortex.

Channels showing the smallest (or negative) ΔSCI were predominantly posterior and occipital, where propofol has less direct effect on alpha generation. This spatial dissociation — frontal increase, posterior relative sparing — strengthens the interpretation that the SCI effect is mechanistically linked to propofol's oscillatory signature rather than a global EEG artifact.

### 5.5 Limitations

- Pre-processing (epoch rejection, ICA) was performed by the dataset authors; we analyze the preprocessed epochs as provided.
- Recovery SCI (0.715) was slightly lower than baseline (0.751) and not statistically distinguishable from baseline (p=0.054), consistent with P2 but falling short of confirming complete return to baseline levels.
- The primary (single-channel Fz) and spatial analyses are both exploratory relative to standard EEG biomarker validation; independent replication in a separate propofol dataset is needed.

---

## 6. Conclusion

SCI detects a significant and consistent change in amplitude envelope structure across propofol sedation states (Friedman p=0.0001). The direction of change — an *increase* at moderate sedation relative to wakefulness — was not the primary prediction but was explicitly pre-registered as a possible outcome. The result is interpretable in terms of propofol's known oscillatory signatures and consistent with SCI's sensitivity to structured amplitude modulation beyond the power spectrum.

Taken with the Sleep-EDF and seizure results, a coherent picture emerges: SCI tracks the *type* and *degree* of amplitude envelope organization in neural signals, producing distinct signatures across different brain states rather than a single monotonic dimension. This specificity may be more useful for brain-state discrimination than a scalar consciousness index, and the mild-to-moderate transition finding suggests potential clinical utility for sedation monitoring.

---

## Appendix: Locked Parameters and Reproducibility

All analyses used locked parameters from `params_locked.py`. No parameters were modified after the pre-registration was written.

```python
# Effective parameters for this analysis
TARGET_FS   = 250.0 Hz
SMOOTH_SEC  = 0.10 s   → W = 25 samples
LAG_SEC     = 0.10 s   → L = 25 samples
S           = 40 surrogates
k           = 0.9
seed        = 42
z_clip      = 6.0
BANDPASS    = 1–40 Hz, 4th-order Butterworth, zero-phase
CHANNEL     = Fz (index 8)
EPOCH_SEC   = 10 s (2500 samples)
```

To reproduce: download Chennu dataset from DOI 10.17863/CAM.68959, place in `Sedation-RestingState/`, then run:

```bash
python3 scripts/sci_chennu_v1.py \
    --data-dir Sedation-RestingState \
    --out results/chennu_v1
```

Output files:
- `results/chennu_v1/epoch_metrics.csv` — full diagnostic stack per epoch
- `results/chennu_v1/subject_state_means.csv` — per-subject means (unit of analysis)
- `results/chennu_v1/summary_by_state.csv` — grand means ± SD
- `results/chennu_v1/statistical_tests.txt` — all test results
- `results/chennu_v1/plots/sci_by_state.png` — box/violin plots

**Spatial analysis** (`scripts/sci_chennu_spatial_v1.py`):
- `results/chennu_spatial_v1/channel_state_means.csv` — per-channel grand means by state
- `results/chennu_spatial_v1/topomap_sci_by_state.png` — 2×2 topomap grid, one per state
- `results/chennu_spatial_v1/topomap_sci_difference.png` — ΔSCI maps (moderate−baseline, recovery−baseline)
- `results/chennu_spatial_v1/top_channels_moderate_vs_baseline.png` — ranked bar chart

---

## References

Chennu, S., Finoia, P., Kamau, E., Allanson, J., Williams, G. B., Monti, M. M., … Bekinschtein, T. A. (2014). Spectral signatures of reorganised brain networks in disorders of consciousness. *PLOS Computational Biology*, 10(10), e1003887.

Lee, P. J. (2026). Structural Coherence Index: bearing fault detection via Hilbert-envelope surrogate gap. *Zenodo*. (May 2, 2026)

US Provisional Patent 63/904,444 (filed October 23, 2025)
