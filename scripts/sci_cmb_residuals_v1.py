#!/usr/bin/env python3
"""
sci_cmb_residuals_v1.py
SCI on Planck 2018 CMB Temperature Power Spectrum Residuals

The test:
    After subtracting the best-fit ΛCDM model from the Planck 2018 TT
    power spectrum, the remaining residuals δD_l = D_l^obs − D_l^ΛCDM
    should be consistent with noise (cosmic variance + instrument noise)
    if standard cosmology is complete.

    If SCI on δD_l is positive and significant beyond phase-randomized
    surrogates, the residuals contain organized amplitude-envelope
    coherence — a candidate signal inconsistent with pure noise.

Data (Planck 2018, public release R3):
    Observed:  COM_PowerSpect_CMB-TT-full_R3.01.txt  (l=2..2508, unbinned)
    Theory:    COM_PowerSpect_CMB-base-plikHM-TTTEEE-
               lowl-lowE-lensing-minimum-theory_R3.01.txt
    Binned:    COM_PowerSpect_CMB-TT-binned_R3.01.txt  (84 bins, with BestFit col)
Source: IRSA / NASA LAMBDA, Planck Collaboration 2020

Two series tested:
    (A) Unbinned standardized residuals:  δD_l / σ_l,  l=2..2508, N=2507
    (B) Binned residuals: (D_l^obs − D_l^BestFit) / σ_l, 83 bins

Multiple SCI window sizes to assess sensitivity.

Locked params: S=200, k=0.9, seed=42  (W, L varied per series length)
"""

import sys, os
sys.path.insert(0, "/Users/parkerlee/Desktop/If Im Right/SCI_Project")

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats, signal

from sci_score_v3 import sci_score_v3

# ── Paths ─────────────────────────────────────────────────────────────────────
DATA  = "/Users/parkerlee/Desktop/If Im Right/SCI_Project/results/cmb_residuals_v1/data"
OUT   = "/Users/parkerlee/Desktop/If Im Right/SCI_Project/results/cmb_residuals_v1"
os.makedirs(f"{OUT}/plots", exist_ok=True)

S, K, SEED = 200, 0.9, 42

# ── Load data ─────────────────────────────────────────────────────────────────
print("Loading Planck 2018 TT power spectrum…")
full = np.loadtxt(f"{DATA}/planck_TT_full.txt", comments="#")
# cols: l, Dl_obs, -dDl, +dDl
l_full   = full[:, 0].astype(int)
Dl_obs   = full[:, 1]
dDl_lo   = full[:, 2]
dDl_hi   = full[:, 3]
sigma    = 0.5 * (dDl_lo + dDl_hi)  # symmetric error estimate

theory = np.loadtxt(f"{DATA}/planck_theory.txt", comments="#")
# cols: L, TT, TE, EE, BB, PP
l_th     = theory[:, 0].astype(int)
Dl_th    = theory[:, 1]

# Align on common l range (both l=2..2508)
assert np.all(l_full == l_th), "l arrays must match"
l = l_full

# Raw and standardized residuals
resid_raw  = Dl_obs - Dl_th           # μK²
resid_std  = resid_raw / (sigma + 1e-6)  # signal-to-noise weighted

print(f"  Unbinned series: {len(l)} multipoles (l={l[0]}..{l[-1]})")
print(f"  Residual range: {resid_raw.min():.1f} to {resid_raw.max():.1f} μK²")
print(f"  Standardized residuals: μ={resid_std.mean():.3f}, σ={resid_std.std():.3f}")

# Binned data
binned   = np.loadtxt(f"{DATA}/planck_TT_binned.txt", comments="#")
# cols: l, Dl_obs, -dDl, +dDl, BestFit
l_bin    = binned[:, 0]
Dl_b_obs = binned[:, 1]
sig_b    = 0.5 * (binned[:, 2] + binned[:, 3])
Dl_b_th  = binned[:, 4]
resid_b  = Dl_b_obs - Dl_b_th
resid_b_std = resid_b / (sig_b + 1e-6)
print(f"  Binned series: {len(l_bin)} bins (l≈{l_bin[0]:.0f}..{l_bin[-1]:.0f})")

# ── SCI analysis function ─────────────────────────────────────────────────────
def run_sci(series, label, w_list, L_list):
    results = []
    for w, Lv in zip(w_list, L_list):
        r = sci_score_v3(series, w=w, L=Lv, S=S, k=K, seed=SEED)
        results.append({
            "series": label,
            "N": len(series),
            "W": w, "L": Lv,
            "c_obs":       round(r["c_obs"], 5),
            "c_surr_mean": round(r["c_surr_mean"], 5),
            "c_surr_std":  round(r["c_surr_std"], 6),
            "gap":         round(r["gap"], 5),
            "z":           round(r["z"], 4),
            "SCI":         round(r["SCI"], 5),
            "bucket":      r["bucket"],
        })
        print(f"  [{label}] W={w:4d} L={Lv:4d} → "
              f"c_obs={r['c_obs']:.4f}  gap={r['gap']:.4f}  "
              f"z={r['z']:.3f}  SCI={r['SCI']:.4f}  [{r['bucket']}]")
    return results

# ── Window parameter sets ─────────────────────────────────────────────────────
# Unbinned (N=2507): test 3 scales
# W = smoothing window for Hilbert envelope, L = coherence lag
# Scale 1: "fine" — detects modulation over ~50-multipole stretches
# Scale 2: "acoustic" — matches spacing of acoustic peaks (~300 multipoles)
# Scale 3: "broad" — detects slow envelope drift across full spectrum
W_unb  = [20,  75, 200]
L_unb  = [40, 150, 400]

# Binned (N=83): 2 scales
W_bin  = [5,  15]
L_bin  = [10, 25]

print("\n── Unbinned standardized residuals (δD_l/σ_l, l=2..2508) ──────────────")
res_unb_std = run_sci(resid_std, "unbinned_std", W_unb, L_unb)

print("\n── Unbinned raw residuals (δD_l μK², l=2..2508) ────────────────────────")
res_unb_raw = run_sci(resid_raw, "unbinned_raw", W_unb, L_unb)

print("\n── Binned standardized residuals (84 bins) ──────────────────────────────")
res_bin_std = run_sci(resid_b_std, "binned_std", W_bin, L_bin)

print("\n── Binned raw residuals (84 bins) ───────────────────────────────────────")
res_bin_raw = run_sci(resid_b, "binned_raw", W_bin, L_bin)

# ── Also test the RAW (non-residual) spectrum for baseline comparison ─────────
print("\n── Raw observed D_l (no model subtraction) — baseline ───────────────────")
# Normalize by mean to remove the large-scale trend
Dl_norm = Dl_obs / Dl_obs.mean()
res_raw_baseline = run_sci(Dl_norm, "raw_Dl_normalized", W_unb, L_unb)

# ── Save results ──────────────────────────────────────────────────────────────
all_results = res_unb_std + res_unb_raw + res_bin_std + res_bin_raw + res_raw_baseline
df = pd.DataFrame(all_results)
df.to_csv(f"{OUT}/sci_results.csv", index=False)

# ── Plots ─────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(4, 1, figsize=(14, 16),
                         gridspec_kw={"height_ratios": [2, 2, 1.5, 1.5]})
plt.subplots_adjust(hspace=0.4)

# Panel 1: Observed vs ΛCDM
ax = axes[0]
ax.semilogy(l, Dl_obs, color="#2c3e50", lw=0.6, alpha=0.7, label="Planck 2018 observed")
ax.semilogy(l, Dl_th, color="#e74c3c", lw=1.5, alpha=0.9, label="ΛCDM best-fit (CAMB)")
ax.set_ylabel("$D_\\ell$ (μK²)", fontsize=11)
ax.set_title("Planck 2018 TT Power Spectrum vs. ΛCDM Best-Fit", fontsize=12, fontweight="bold")
ax.legend(fontsize=9)
ax.set_xlim(2, 2508)
ax.set_xlabel("Multipole $\\ell$", fontsize=10)

# Panel 2: Residuals with ±1σ band
ax2 = axes[1]
ax2.plot(l, resid_raw, color="#2980b9", lw=0.5, alpha=0.7, label="$\\delta D_\\ell = D_\\ell^{obs} - D_\\ell^{\\Lambda CDM}$")
ax2.fill_between(l, -sigma, sigma, alpha=0.2, color="#95a5a6", label="$\\pm 1\\sigma$ error")
ax2.axhline(0, color="#c0392b", lw=1, ls="--", alpha=0.8)
ax2.set_ylabel("$\\delta D_\\ell$ (μK²)", fontsize=11)
ax2.set_title("CMB TT Residuals After ΛCDM Subtraction", fontsize=12, fontweight="bold")
ax2.legend(fontsize=9)
ax2.set_xlim(2, 2508)
ax2.set_xlabel("Multipole $\\ell$", fontsize=10)

# Panel 3: Standardized residuals
ax3 = axes[2]
ax3.plot(l, resid_std, color="#8e44ad", lw=0.5, alpha=0.8,
         label="$\\delta D_\\ell / \\sigma_\\ell$ (standardized)")
ax3.axhline(0,  color="#c0392b", lw=1, ls="--", alpha=0.8)
ax3.axhline( 2, color="#e67e22", lw=0.8, ls=":", alpha=0.6)
ax3.axhline(-2, color="#e67e22", lw=0.8, ls=":", alpha=0.6)
ax3.set_ylabel("Std. residual (σ)", fontsize=11)
ax3.set_title("Standardized Residuals", fontsize=11)
ax3.legend(fontsize=9)
ax3.set_xlim(2, 2508)
ax3.set_ylim(-6, 6)
ax3.set_xlabel("Multipole $\\ell$", fontsize=10)

# Panel 4: SCI results bar chart
ax4 = axes[3]
key_runs = df[df["series"].isin(["unbinned_std", "binned_std"])].copy()
labels  = [f"{row['series']}\nW={row['W']} L={row['L']}" for _, row in key_runs.iterrows()]
sci_vals = key_runs["SCI"].values
colors  = ["#27ae60" if s > 0.5 else "#e74c3c" if s < 0.3 else "#e67e22" for s in sci_vals]
bars = ax4.barh(range(len(sci_vals)), sci_vals, color=colors, alpha=0.75)
ax4.set_yticks(range(len(sci_vals)))
ax4.set_yticklabels(labels, fontsize=8)
ax4.axvline(0.5, color="gray", ls="--", lw=1, label="Neutral")
ax4.axvline(0.75, color="#c0392b", ls="--", lw=1, label="CORE threshold")
ax4.set_xlabel("SCI", fontsize=10)
ax4.set_title("SCI Scores — CMB Residuals (multiple window sizes)", fontsize=11)
ax4.set_xlim(0, 1)
ax4.legend(fontsize=8)
for i, (sci, row) in enumerate(zip(sci_vals, key_runs.itertuples())):
    ax4.text(sci + 0.01, i, f"z={row.z:.2f}", va="center", fontsize=7.5)

fig.savefig(f"{OUT}/plots/cmb_residuals_sci.png", dpi=150, bbox_inches="tight")
plt.close(fig)

# ── Hilbert envelope visualization ────────────────────────────────────────────
from scipy.signal import hilbert

fig2, axes2 = plt.subplots(2, 1, figsize=(14, 8), sharex=True)
plt.subplots_adjust(hspace=0.3)

# Smooth with W=75 window for visualization
from numpy.lib.stride_tricks import sliding_window_view
W_vis = 75
analytic  = hilbert(resid_std)
envelope  = np.abs(analytic)
env_smooth = np.convolve(envelope, np.ones(W_vis)/W_vis, mode="same")

axes2[0].plot(l, resid_std, color="#8e44ad", lw=0.5, alpha=0.6, label="Std. residuals")
axes2[0].plot(l, env_smooth, color="#e74c3c", lw=1.8, label=f"Hilbert envelope (smooth W={W_vis})")
axes2[0].axhline(0, color="gray", lw=0.8, ls="--")
axes2[0].set_ylabel("$\\delta D_\\ell / \\sigma_\\ell$", fontsize=11)
axes2[0].set_title("Hilbert Envelope of Standardized CMB TT Residuals", fontsize=12, fontweight="bold")
axes2[0].legend(fontsize=9)
axes2[0].set_xlim(2, 2508)

# Surrogate envelope (one phase-randomized surrogate)
rng = np.random.default_rng(42)
ph = np.fft.rfft(resid_std)
rand_phase = np.exp(2j * np.pi * rng.uniform(0, 1, len(ph)))
rand_phase[0] = 1.0
if len(resid_std) % 2 == 0:
    rand_phase[-1] = 1.0
surr = np.fft.irfft(ph * rand_phase, n=len(resid_std))
surr_env = np.abs(hilbert(surr))
surr_env_smooth = np.convolve(surr_env, np.ones(W_vis)/W_vis, mode="same")

axes2[1].plot(l, env_smooth, color="#e74c3c", lw=1.5, label="Real envelope", alpha=0.9)
axes2[1].plot(l, surr_env_smooth, color="#7f8c8d", lw=1.2, ls="--",
              label="Surrogate envelope (phase-randomized)", alpha=0.8)
axes2[1].set_ylabel("Hilbert envelope", fontsize=11)
axes2[1].set_xlabel("Multipole $\\ell$", fontsize=11)
axes2[1].set_title("Real vs. Surrogate Hilbert Envelope", fontsize=12, fontweight="bold")
axes2[1].legend(fontsize=9)
axes2[1].set_xlim(2, 2508)

fig2.savefig(f"{OUT}/plots/hilbert_envelope.png", dpi=150, bbox_inches="tight")
plt.close(fig2)

# ── Written summary ───────────────────────────────────────────────────────────
best_run = df[df["series"] == "unbinned_std"].sort_values("z", ascending=False).iloc[0]

summary = f"""SCI on Planck 2018 CMB TT Power Spectrum Residuals
====================================================
Script: sci_cmb_residuals_v1.py
Date:   2026-05-14

Data:
  Source:   IRSA / NASA LAMBDA, Planck Collaboration 2020 (public R3)
  Observed: COM_PowerSpect_CMB-TT-full_R3.01.txt (l=2..2508, N=2507)
  Theory:   COM_PowerSpect_CMB-base-plikHM-TTTEEE-lowl-lowE-lensing-minimum-theory_R3.01.txt
  Binned:   COM_PowerSpect_CMB-TT-binned_R3.01.txt (N=83 bins, incl. BestFit)

Residuals:  δD_l = D_l^obs − D_l^ΛCDM  (μK²)
Standardized: δD_l / σ_l  (signal-to-noise weighted)

SCI Parameters: S={S}, k={K}, seed={SEED}  [W, L varied per series]

Results Summary
───────────────────────────────────────────────────────────────────
"""
for _, row in df.iterrows():
    summary += (f"  {row['series']:25s} W={row['W']:4d} L={row['L']:4d} | "
                f"c_obs={row['c_obs']:.4f}  gap={row['gap']:+.4f}  "
                f"z={row['z']:+.3f}  SCI={row['SCI']:.4f}  [{row['bucket']}]\n")

summary += f"""
Best unbinned_std run: W={best_run['W']}, L={best_run['L']} → z={best_run['z']:.4f}, SCI={best_run['SCI']:.4f}, [{best_run['bucket']}]

Interpretation
──────────────
Null hypothesis: δD_l is white (Gaussian) noise → SCI ≈ 0.5 (INELIGIBLE).
Alternative: δD_l has organized amplitude-envelope coherence → SCI > 0.5.

If SCI is INELIGIBLE across all window sizes: residuals are consistent with pure noise.
This is evidence AGAINST the Oscillatory Arrow claim at CMB scales.

If SCI is positive (CORE/EMERGING): residuals contain organized amplitude modulation.
This is a candidate anomaly requiring further scrutiny (systematic effects,
non-ΛCDM physics, or noise correlation artifacts from Planck data processing).
"""

print(summary)
with open(f"{OUT}/summary.txt", "w") as f:
    f.write(summary)

print(f"\n✓ Outputs saved to {OUT}/")
print(f"  sci_results.csv, summary.txt, plots/cmb_residuals_sci.png, plots/hilbert_envelope.png")
