"""
SCI Locked Parameters
=====================
These values are frozen for all pre-registered work.
DO NOT MODIFY. New experiments must use a new version label.

Version history:
  v1 — original, no seed (bearing paper, Zenodo May 2026)
  v2 — seed=42 added (Sleep-EDF paper, 26 subjects)
  v3 — full diagnostic output, same numerical params as v2

Patent: US Provisional 63/904,444 (filed Oct 23, 2025)
"""

# SCI operator parameters (numerical — identical across v1/v2/v3)
W = 12        # envelope smoothing window (samples)
L = 12        # ACF lag count (lags 1..L)
S = 40        # number of phase-randomized surrogates
K = 0.9       # logistic slope
SEED = 42     # RNG seed (v2+ only)
Z_CLIP = 6.0  # z-score clip before logistic

# Classification thresholds
THRESH_CORE     = 0.75   # SCI >= 0.75 → CORE
THRESH_CORE_MID = 0.65   # SCI >= 0.65 → CORE_MID
THRESH_TACTICAL = 0.55   # SCI >= 0.55 → TACTICAL
                          # SCI <  0.55 → INELIGIBLE

# Domain-specific notes (parameters stay the same; context changes)
#
# Bearings (sci_bearing_full_rerun_v2.py):
#   W=12, L=12, S=40 on bandpass-filtered vibration segments (~6000 samples)
#
# EEG (sci_eeg_full_rerun_v2.py, sci_eeg_full_analysis.py):
#   W and L derived from seconds × fs (smooth_sec=0.10, lag_sec=0.10 × 256 Hz)
#   → smooth=26 samples, max_lag=26 samples for 256 Hz data
#   SCI core math is identical; only the sample-rate scaling changes.
#
# Cells (sci_cells_full_rerun_v2.py):
#   Uses smooth=6, acf_lag=6 — appropriate for short fluorescence time series
#   (~20-200 points). Core math identical.
#
# Finance (sci_garch_gap_mapper.py, forward_return_by_amplitude_bucket.py):
#   Uses window=500 days, smooth=7, lag=10 — appropriate for daily return series.
#   Core math identical.
#
# Kill box (sci_killbox.py):
#   W=12, L=12, S=40, seed=42, N=10000 synthetic samples at FS=1000 Hz.
