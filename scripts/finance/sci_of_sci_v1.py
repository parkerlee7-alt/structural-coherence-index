#!/usr/bin/env python3
"""
sci_of_sci_v1.py
SCI of SCI — Does the Market's Coherence Level Oscillate Coherently?

Input signal: monthly universe-wide CORE fraction (and PC1 from Triadic PCA)
              across 56 monthly rebalance dates (Sep 2021 – Apr 2026).

Hypothesis: coherence_level[t] — the fraction of the 1,339-instrument universe
            classified CORE at each month — is itself coherent over time.
            SCI applied to this series returns CORE (>0.75).

Locked parameters (per experiment spec):
    W=3, L=5, S=500, k=0.9, seed=42

Controls:
    - Random walk of same length (null, should be INELIGIBLE)
    - INELIGIBLE fraction series (opposite of CORE, should mirror)
    - PC2 from Triadic PCA (distributional shape mode, not level)
    - Median SCI (q50) per month
    - Shuffled CORE fraction (destroys temporal order, should be INELIGIBLE)
"""

import sys, os
sys.path.insert(0, "/Users/parkerlee/Desktop/If Im Right/SCI_Project")

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from scipy import stats

from sci_score_v3 import sci_score_v3

# ── Paths ─────────────────────────────────────────────────────────────────────
RESULTS = "/Users/parkerlee/Desktop/If Im Right/SCI_Project/results"
OUT     = f"{RESULTS}/sci_of_sci_v1"
os.makedirs(f"{OUT}/plots", exist_ok=True)

# ── Locked parameters ─────────────────────────────────────────────────────────
W, L, S, K, SEED = 3, 5, 500, 0.9, 42

# ── Load data ─────────────────────────────────────────────────────────────────
print("Loading data…")
bf  = pd.read_csv(f"{RESULTS}/triadic_law_v1/bucket_fractions.csv", parse_dates=["date"])
hm  = pd.read_csv(f"{RESULTS}/triadic_law_dist_v1/hist_matrix.csv")
qm  = pd.read_csv(f"{RESULTS}/triadic_law_dist_v1/quantile_matrix.csv")

dates            = pd.to_datetime(hm["Unnamed: 0"].values)
H                = hm.iloc[:, 1:].values.astype(float)  # (56, 10)
core_frac        = bf["core"].values
inelig_frac      = bf["ineligible"].values
q50              = qm["q50"].values

print(f"  {len(dates)} monthly dates: {dates[0].date()} → {dates[-1].date()}")
print(f"  CORE fraction: min={core_frac.min():.3f}, max={core_frac.max():.3f}, "
      f"mean={core_frac.mean():.3f}")

# ── Recompute PC1 / PC2 from histogram matrix ─────────────────────────────────
print("\nRecomputing PCA on histogram matrix (56×10)…")
scaler   = StandardScaler()
H_std    = scaler.fit_transform(H)
pca      = PCA(n_components=5)
scores   = pca.fit_transform(H_std)   # (56, 5)
pc1      = scores[:, 0]
pc2      = scores[:, 1]
var_exp  = pca.explained_variance_ratio_

print(f"  PC1 variance explained: {var_exp[0]*100:.2f}%")
print(f"  PC2 variance explained: {var_exp[1]*100:.2f}%")
print(f"  PC1-3 cumulative:       {var_exp[:3].sum()*100:.2f}%")

# Verify: PC1 should correlate strongly with INELIGIBLE fraction
r1_core,   _ = stats.pearsonr(pc1, core_frac)
r1_inelig, _ = stats.pearsonr(pc1, inelig_frac)
print(f"  PC1 ↔ CORE fraction:       r = {r1_core:.3f}")
print(f"  PC1 ↔ INELIGIBLE fraction: r = {r1_inelig:.3f}")

# Sign-align PC1 so that it increases with INELIGIBLE (noise regime)
if r1_inelig < 0:
    pc1 = -pc1; r1_core = -r1_core; r1_inelig = -r1_inelig
    print("  (PC1 sign-flipped to align with INELIGIBLE fraction)")

# ── SCI runner ────────────────────────────────────────────────────────────────
def run_one(series, label):
    r = sci_score_v3(series, w=W, L=L, S=S, k=K, seed=SEED)
    print(f"  [{label:35s}]  c_obs={r['c_obs']:.4f}  c_surr={r['c_surr_mean']:.4f}±{r['c_surr_std']:.4f}"
          f"  gap={r['gap']:+.4f}  z={r['z']:+.4f}  SCI={r['SCI']:.4f}  [{r['bucket']}]")
    return {
        "series":      label,
        "N":           len(series),
        "c_obs":       round(r["c_obs"],      5),
        "c_surr_mean": round(r["c_surr_mean"],5),
        "c_surr_std":  round(r["c_surr_std"], 6),
        "gap":         round(r["gap"],         5),
        "z":           round(r["z"],           4),
        "SCI":         round(r["SCI"],         5),
        "bucket":      r["bucket"],
    }

# ── Null-control surrogate distributions (for plotting) ────────────────────────
def get_surrogates(series, n_surr=500, seed=42):
    """Return array of c_surr values for histogram."""
    rng = np.random.default_rng(seed)
    surrs = []
    for _ in range(n_surr):
        ph   = np.fft.rfft(series)
        rand = np.exp(2j * np.pi * rng.uniform(0, 1, len(ph)))
        rand[0] = 1.0
        if len(series) % 2 == 0: rand[-1] = 1.0
        s = np.fft.irfft(ph * rand, n=len(series))
        from scipy.signal import hilbert
        env  = np.abs(hilbert(s))
        kern = np.ones(W) / W
        env  = np.convolve(env, kern, mode="same")
        surrs.append(float(np.corrcoef(env[:-L], env[L:])[0, 1]))
    return np.array(surrs)

# ── Run all series ────────────────────────────────────────────────────────────
print("\n── Main signal ─────────────────────────────────────────────────────────")
res_core   = run_one(core_frac,   "CORE fraction (main signal)")
res_pc1    = run_one(pc1,         "PC1 (Triadic PCA, ~60% var)")

print("\n── Controls ────────────────────────────────────────────────────────────")
res_inelig = run_one(inelig_frac, "INELIGIBLE fraction")
res_pc2    = run_one(pc2,         "PC2 (distributional shape)")
res_q50    = run_one(q50,         "q50 median SCI per month")

# Shuffled control (destroys temporal order)
rng_shuf   = np.random.default_rng(99)
core_shuf  = core_frac.copy(); rng_shuf.shuffle(core_shuf)
res_shuf   = run_one(core_shuf,  "CORE fraction (shuffled — null)")

# Random walk null
rng_rw     = np.random.default_rng(SEED)
rw         = np.cumsum(rng_rw.standard_normal(len(core_frac)))
rw         = (rw - rw.min()) / (rw.max() - rw.min())   # scale to [0,1]
res_rw     = run_one(rw,          "Random walk [0,1] — null control")

# White noise null
wn         = rng_rw.uniform(0, 1, len(core_frac))
res_wn     = run_one(wn,          "White noise [0,1] — null control")

# ── Collect results ───────────────────────────────────────────────────────────
all_results = [res_core, res_pc1, res_inelig, res_pc2, res_q50,
               res_shuf, res_rw, res_wn]
df = pd.DataFrame(all_results)
df.to_csv(f"{OUT}/sci_of_sci_results.csv", index=False)

# ── Get surrogate distribution for CORE fraction (main signal) ─────────────────
print("\nBuilding surrogate distribution for CORE fraction…")
surr_dist = get_surrogates(core_frac, n_surr=500, seed=SEED)
print(f"  Surrogate c: mean={surr_dist.mean():.4f}, std={surr_dist.std():.4f}")
print(f"  c_obs={res_core['c_obs']:.4f}, percentile={(surr_dist < res_core['c_obs']).mean()*100:.1f}th")

# ── Plot 1: Main figure (3-panel) ─────────────────────────────────────────────
fig = plt.figure(figsize=(14, 12))
gs  = gridspec.GridSpec(3, 2, figure=fig, hspace=0.42, wspace=0.35)

# Panel A: CORE fraction time series
ax_a = fig.add_subplot(gs[0, :])
ax_a.plot(dates, core_frac,   color="#1a6e3c", lw=2.0, marker="o", ms=4, label="CORE fraction")
ax_a.plot(dates, inelig_frac, color="#b83232", lw=1.5, ls="--", alpha=0.7, label="INELIGIBLE fraction")
ax_a.fill_between(dates, core_frac, alpha=0.12, color="#1a6e3c")
ax_a.axhline(core_frac.mean(), color="#1a6e3c", ls=":", lw=1, alpha=0.6)
ax_a.set_ylabel("Fraction of universe", fontsize=11)
ax_a.set_title(
    f"Universe-Wide Coherence Level — 56 Monthly Dates (Sep 2021 – Apr 2026)\n"
    f"SCI of CORE fraction: gap={res_core['gap']:+.4f}, z={res_core['z']:+.3f}, "
    f"SCI={res_core['SCI']:.4f}  [{res_core['bucket']}]",
    fontsize=11, fontweight="bold")
ax_a.legend(fontsize=9)
ax_a.set_xlim(dates[0], dates[-1])
ax_a.set_ylim(0, 0.55)
ax_a.tick_params(axis="x", labelrotation=30, labelsize=8)
# annotate SCI badge
ax_a.text(0.98, 0.93, f"SCI={res_core['SCI']:.4f}", transform=ax_a.transAxes,
          ha="right", va="top", fontsize=13, fontweight="bold",
          color="#1a6e3c" if res_core['SCI'] > 0.75 else "#c0392b",
          bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor="#aaa"))

# Panel B: Surrogate distribution
ax_b = fig.add_subplot(gs[1, 0])
ax_b.hist(surr_dist, bins=30, color="#7f8c8d", alpha=0.65, edgecolor="white",
          label=f"Surrogates (N=500)\nμ={surr_dist.mean():.4f}, σ={surr_dist.std():.4f}")
ax_b.axvline(res_core["c_obs"], color="#1a6e3c", lw=2.5,
             label=f"c_obs={res_core['c_obs']:.4f}\n(z={res_core['z']:+.3f})")
ax_b.set_xlabel("Envelope autocorrelation (lag L=5)", fontsize=10)
ax_b.set_ylabel("Count", fontsize=10)
ax_b.set_title("Surrogate Distribution — CORE Fraction", fontsize=10, fontweight="bold")
ax_b.legend(fontsize=8)

# Panel C: PC1 time series
ax_c = fig.add_subplot(gs[1, 1])
ax_c2 = ax_c.twinx()
ax_c.plot(dates, -pc1, color="#8e44ad", lw=2, marker="s", ms=3.5, label="−PC1 (≈ CORE level)")
ax_c2.plot(dates, core_frac, color="#1a6e3c", lw=1.2, ls="--", alpha=0.7, label="CORE frac")
ax_c.set_ylabel("−PC1 score", fontsize=9, color="#8e44ad")
ax_c2.set_ylabel("CORE fraction", fontsize=9, color="#1a6e3c")
ax_c.set_title(
    f"PC1 Triadic PCA ({var_exp[0]*100:.1f}% var)\n"
    f"SCI={res_pc1['SCI']:.4f}, z={res_pc1['z']:+.3f}  [{res_pc1['bucket']}]",
    fontsize=9, fontweight="bold")
ax_c.tick_params(axis="x", labelrotation=30, labelsize=7)
lines1, labs1 = ax_c.get_legend_handles_labels()
lines2, labs2 = ax_c2.get_legend_handles_labels()
ax_c.legend(lines1+lines2, labs1+labs2, fontsize=7)

# Panel D: All-series SCI bar chart
ax_d = fig.add_subplot(gs[2, :])
series_labels = [r["series"] for r in all_results]
sci_vals      = [r["SCI"]    for r in all_results]
z_vals        = [r["z"]      for r in all_results]
bucket_colors = {
    "CORE":        "#1a6e3c",
    "CORE_MID":    "#27ae60",
    "TACTICAL":    "#e67e22",
    "INELIGIBLE":  "#c0392b",
}
bar_colors = [bucket_colors.get(r["bucket"].split("_")[0] + ("_MID" if "MID" in r["bucket"] else ""),
              "#7f8c8d") for r in all_results]
bar_colors = [bucket_colors.get(r["bucket"], "#7f8c8d") for r in all_results]

bars = ax_d.barh(range(len(sci_vals)), sci_vals, color=bar_colors, alpha=0.7, height=0.65)
ax_d.set_yticks(range(len(series_labels)))
ax_d.set_yticklabels(series_labels, fontsize=8.5)
ax_d.axvline(0.75, color="#c0392b", ls="--", lw=1.2, label="CORE threshold (0.75)")
ax_d.axvline(0.50, color="#7f8c8d", ls=":",  lw=1.0, label="Neutral (0.50)")
ax_d.set_xlabel("SCI", fontsize=10)
ax_d.set_title("SCI of SCI — All Series", fontsize=11, fontweight="bold")
ax_d.set_xlim(0, 1.05)
ax_d.legend(fontsize=9, loc="lower right")
for i, (sci, z) in enumerate(zip(sci_vals, z_vals)):
    ax_d.text(sci + 0.01, i, f"z={z:+.2f}", va="center", fontsize=8, color="#333")

fig.savefig(f"{OUT}/plots/sci_of_sci_main.png", dpi=150, bbox_inches="tight")
plt.close(fig)

# ── Plot 2: CORE fraction with Hilbert envelope ────────────────────────────────
from scipy.signal import hilbert

fig2, axes2 = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
plt.subplots_adjust(hspace=0.3)

analytic = hilbert(core_frac - core_frac.mean())
envelope = np.abs(analytic)
kern     = np.ones(W) / W
env_sm   = np.convolve(envelope, kern, mode="same")

# Surrogate envelope
rng_s = np.random.default_rng(SEED)
ph = np.fft.rfft(core_frac)
rph = np.exp(2j * np.pi * rng_s.uniform(0, 1, len(ph)))
rph[0] = 1.0
if len(core_frac) % 2 == 0: rph[-1] = 1.0
surr_sig = np.fft.irfft(ph * rph, n=len(core_frac))
surr_env = np.abs(hilbert(surr_sig))
surr_env_sm = np.convolve(surr_env, kern, mode="same")

axes2[0].plot(dates, core_frac,   color="#1a6e3c", lw=1.5, label="CORE fraction")
axes2[0].plot(dates, core_frac.mean() + env_sm,  color="#e74c3c", lw=2,
              label=f"Hilbert envelope (W={W})")
axes2[0].axhline(core_frac.mean(), color="gray", lw=0.8, ls="--")
axes2[0].set_ylabel("CORE fraction", fontsize=10)
axes2[0].set_title("CORE Fraction + Hilbert Envelope", fontsize=11, fontweight="bold")
axes2[0].legend(fontsize=9)

axes2[1].plot(dates, env_sm,      color="#e74c3c", lw=1.8, label="Real envelope")
axes2[1].plot(dates, surr_env_sm, color="#7f8c8d", lw=1.4, ls="--",
              label="Surrogate envelope (1 phase-randomization)")
axes2[1].set_ylabel("Envelope amplitude", fontsize=10)
axes2[1].set_xlabel("Date", fontsize=10)
axes2[1].set_title("Real vs. Surrogate Envelope", fontsize=11, fontweight="bold")
axes2[1].legend(fontsize=9)
axes2[1].tick_params(axis="x", labelrotation=30, labelsize=8)

fig2.savefig(f"{OUT}/plots/hilbert_envelope.png", dpi=150, bbox_inches="tight")
plt.close(fig2)

# ── Written summary ───────────────────────────────────────────────────────────
percentile = (surr_dist < res_core["c_obs"]).mean() * 100

summary = f"""SCI of SCI — Universe-Wide Coherence Level Self-Coherence Test
================================================================
Script: sci_of_sci_v1.py
Date:   2026-05-14

Input: 56 monthly CORE/INELIGIBLE/TACTICAL fractions across 1,339 instruments
       Dates: {dates[0].date()} to {dates[-1].date()}
       Source: results/triadic_law_v1/bucket_fractions.csv
               results/triadic_law_dist_v1/hist_matrix.csv

Locked SCI parameters: W={W}, L={L}, S={S}, k={K}, seed={SEED}

Triadic PCA check (on 56×10 histogram matrix):
  PC1 variance explained: {var_exp[0]*100:.2f}%
  PC2 variance explained: {var_exp[1]*100:.2f}%
  PC1-3 cumulative:       {var_exp[:3].sum()*100:.2f}%
  PC1 ↔ INELIGIBLE:       r={r1_inelig:.3f}
  PC1 ↔ CORE:             r={r1_core:.3f}

Main Result
───────────
CORE fraction (main signal):
  c_obs      = {res_core['c_obs']:.5f}
  c_surr     = {res_core['c_surr_mean']:.5f} ± {res_core['c_surr_std']:.6f}
  gap        = {res_core['gap']:+.5f}
  z          = {res_core['z']:+.4f}
  SCI        = {res_core['SCI']:.5f}
  bucket     = {res_core['bucket']}
  percentile = {percentile:.1f}th (vs 500 surrogates)

PC1 (Triadic PCA):
  z={res_pc1['z']:+.4f}, SCI={res_pc1['SCI']:.5f}  [{res_pc1['bucket']}]

Controls
────────
"""
for r in all_results[2:]:
    summary += (f"  {r['series']:40s}  z={r['z']:+.4f}  SCI={r['SCI']:.5f}  [{r['bucket']}]\n")

summary += f"""
Interpretation
──────────────
Prediction: CORE fraction SCI > 0.75 (CORE bucket).
Result:     SCI = {res_core['SCI']:.4f}  →  {res_core['bucket']}
"""
if res_core["SCI"] > 0.75:
    summary += (
        "CONFIRMED. The market's coherence level is itself coherent over time.\n"
        "High-coherence months cluster with high-coherence months.\n"
        "The oscillatory arrow of time is detectable in the financial SCI data.\n"
    )
elif res_core["SCI"] > 0.5:
    summary += (
        "PARTIAL. Coherence level shows above-null envelope autocorrelation\n"
        "but below the CORE threshold. Weak clustering present.\n"
    )
else:
    summary += (
        "NOT CONFIRMED. Coherence level fluctuates without structured oscillation.\n"
        "The oscillatory arrow is not detectable at monthly resolution.\n"
    )

print("\n" + summary)
with open(f"{OUT}/summary.txt", "w") as f:
    f.write(summary)

np.save(f"{OUT}/surr_dist_core_frac.npy", surr_dist)
print(f"\n✓ Outputs saved to {OUT}/")
