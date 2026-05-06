# SCI Anesthesia Pre-Registration
## Structural Coherence Index Applied to Propofol Sedation EEG

**Author:** Parker J. Lee  
**Date written:** 2026-05-05  
**Status:** PRE-REGISTRATION — written before any data analysis  
**Patent:** US Provisional 63/904,444

---

## 1. Background and Motivation

The Structural Coherence Index (SCI) measures excess Hilbert-envelope autocorrelation relative to a phase-randomized surrogate baseline:

```
gap    = c_obs − c_surr_mean
z      = gap / c_surr_std
SCI    = 1 / (1 + exp(−0.9 × z))
```

Prior validated results:
- **Sleep-EDF (26 subjects):** Wake SCI=0.970, Stage 4 SCI=0.707, Cohen d=3.991, AUC=0.959
- **CHB-MIT seizure EEG:** gap *collapses* during seizure — hypersynchrony is spectrum-dominated
- **Bearings:** Fault states → positive gap; healthy → near-zero gap
- **GARCH(1,1):** SCI=0.996, gap=0.163 (kill box confirmed)

The sleep result shows SCI tracks depth of unconsciousness across natural sleep stages. Propofol anesthesia offers a clean dose-controlled model of consciousness loss that tests whether SCI tracks pharmacologically induced unconsciousness.

---

## 2. Dataset

**Chennu et al. (2014/2016) Cambridge propofol sedation dataset**  
- Repository: University of Cambridge Data Repository  
- DOI: https://doi.org/10.17863/CAM.68959  
- 20 healthy volunteers  
- 4 states per subject: **Baseline** (awake), **Mild sedation**, **Moderate sedation**, **Recovery**  
- 91-channel EEG, 250 Hz, preprocessed 10-second epochs  
- Format: EEGLAB .set files  
- License: CC BY 2.0 UK (open access, no DUA required)

---

## 3. Pre-registered Predictions

### Primary prediction (must hold for result to be positive)

**P1 — Monotonic decrease in mean SCI/gap with sedation depth:**

```
SCI(Baseline) > SCI(Mild) > SCI(Moderate)
gap(Baseline) > gap(Mild) > gap(Moderate)
```

Rationale: Propofol induces increasingly regular slow oscillations as dose increases. Regular oscillations are more spectrum-predictable, so phase-randomized surrogates replicate them better — gap decreases. This mirrors the sleep result (Wake > Stage 1 > Stage 2 > Stage 3 > Stage 4).

### Secondary predictions

**P2 — Recovery approximates Baseline:**
```
|SCI(Recovery) − SCI(Baseline)| < |SCI(Moderate) − SCI(Baseline)|
```

Rationale: SCI should recover when consciousness returns, as it did going from Stage 4 → REM → Wake in the sleep result.

**P3 — Moderate sedation SCI falls in the Stage 3–4 range:**
```
0.65 ≤ SCI(Moderate) ≤ 0.85
```

Rationale: Moderate propofol sedation produces slow oscillations similar to Stage 3-4 NREM sleep (delta power dominant). SCI for Stage 3 was 0.835, Stage 4 was 0.707 in the Sleep-EDF result.

**P4 — Statistical significance at the subject level:**
```
Paired t-test (Baseline vs Moderate): p < 0.05
```

This is the correct unit of analysis (one mean SCI per subject per state), matching the sleep-EDF approach.

### Pre-registered null / boundary conditions

- If gap is *positive* in Moderate sedation and exceeds Baseline, that would be a **surprising positive finding** — propofol imposing amplitude coherence beyond spectrum prediction. Not predicted but would be reported.
- If gap collapses to near-zero or negative (like seizure), that would indicate propofol's slow oscillations are fully spectrum-predictable — also reportable.
- If there is no monotonic ordering, that is a **null result** and will be reported as such.

---

## 4. Analysis Plan

### SCI parameters (locked — DO NOT CHANGE)

| Parameter | Value | Source |
|-----------|-------|--------|
| W (smooth) | 12 | params_locked.py |
| L (ACF lags) | 12 | params_locked.py |
| S (surrogates) | 40 | params_locked.py |
| k (logistic) | 0.9 | params_locked.py |
| seed | 42 | params_locked.py |
| z_clip | 6.0 | params_locked.py |

Note: The EEG script will scale W and L by sample rate if needed (smooth_sec × fs, lag_sec × fs), exactly as in sci_eeg_full_analysis.py. For 250 Hz data with smooth_sec=0.048s → W=12; lag_sec=0.048s → L=12. No parameter change — the locked values apply directly in samples.

### Channel selection

Use a single representative frontal channel (e.g., Fz or equivalent) per epoch, consistent with the sleep-EDF approach. If the dataset provides epochs already, use the epoch mean across frontal channels.

### Per-subject computation

For each subject and each state:
1. Load all 10-second epochs for that state
2. For each epoch: compute SCI v3 full diagnostic (c_obs, c_surr_mean, c_surr_std, gap, z, SCI)
3. Aggregate: mean SCI, mean gap, mean z per subject per state
4. Final analysis on the subject-level means (N=20)

### Statistical tests

- Paired t-test (Baseline vs Moderate) — primary test for P1 and P4
- Friedman test across all 4 states — non-parametric omnibus
- Post-hoc Wilcoxon signed-rank with Bonferroni correction for pairwise state comparisons
- Report effect sizes (Cohen's d) for all comparisons

### Output files

```
results/anesthesia_full_v1/
  epoch_metrics.csv          — full diagnostic stack per epoch per subject per state
  subject_state_means.csv    — per-subject means (unit of analysis)
  summary_by_state.csv       — grand means ± SD per state
  statistical_tests.txt      — all test results
  plots/sci_by_state.png     — box/violin plot of subject means
```

---

## 5. What Would Falsify SCI Here

- SCI does not differ significantly between Baseline and Moderate sedation (p > 0.05)
- SCI is *higher* in Moderate sedation than Baseline (gap increases with anesthesia)
- There is no monotonic ordering across states

Any of these outcomes will be reported honestly. The pre-registration is the record that predictions were written before analysis.

---

## 6. Relationship to Prior Work

This experiment extends the sleep-EDF result into a pharmacologically controlled setting. If P1 holds, it strengthens the interpretation that SCI tracks conscious state rather than just sleep-stage-specific physiology. If it fails, it constrains the theory.

This is explicitly not a pre-registered kill box (that was done May 2, 2026 for GARCH). This is an exploratory-confirmatory extension — the prediction is directional (monotonic decrease) but the dataset was not used in any prior SCI analysis.

---

*This document was written on 2026-05-05 before any data was downloaded or analyzed.*  
*Script: `scripts/sci_anesthesia_v1.py` (to be written after this document)*  
*Data: Chennu et al. Cambridge repository, DOI 10.17863/CAM.68959*
