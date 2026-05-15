"""
Triadic Law — Distributional PCA v1

Instead of tracking each instrument's SCI individually (56×1339),
track the SHAPE of the cross-sectional SCI distribution at each date.

Feature vector at each date (10 features):
  - 10 histogram bins [0.0-0.1, 0.1-0.2, ..., 0.9-1.0]
    (fraction of universe in each bin)

This gives a 56×10 matrix. PCA on this tests whether the distribution
evolves in a low-rank way — the actual Triadic Law claim.

Surrogate control: phase-randomize each column of the 56×10 matrix
independently (preserves each feature's own temporal autocorrelation
but destroys cross-feature phase structure).

Loads existing sci_matrix.csv — no SCI recomputation needed.

Usage:
    python3 scripts/finance/triadic_law_dist_v1.py \
        --matrix results/triadic_law_v1/sci_matrix.csv \
        --out    results/triadic_law_dist_v1 \
        --n-surr 500
"""

import sys, argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from scipy import stats as scipy_stats
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.gridspec as gridspec

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

SEED = 42
BINS = np.linspace(0, 1, 11)          # 10 bins: [0-0.1, 0.1-0.2, ..., 0.9-1.0]
BIN_LABELS = [f"{BINS[i]:.1f}–{BINS[i+1]:.1f}" for i in range(10)]

# bucket boundaries for annotation
CORE_THRESH      = 0.75
TACTICAL_THRESH  = 0.55
K                = 0.9


# ── helpers ──────────────────────────────────────────────────────────────────

def build_hist_matrix(sci_matrix):
    """
    sci_matrix: np.ndarray (n_dates, n_instruments) — may contain NaN
    Returns: H of shape (n_dates, 10) — fraction of universe in each SCI bin
    """
    n_dates = sci_matrix.shape[0]
    H = np.zeros((n_dates, 10))
    for i in range(n_dates):
        row = sci_matrix[i]
        row = row[np.isfinite(row)]
        if len(row) == 0:
            H[i] = np.nan
        else:
            counts, _ = np.histogram(row, bins=BINS)
            H[i] = counts / len(row)
    return H


def build_quantile_matrix(sci_matrix, quantiles=None):
    """
    Returns Q of shape (n_dates, len(quantiles)).
    Useful as a cross-check alongside the histogram.
    """
    if quantiles is None:
        quantiles = [0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95]
    n_dates = sci_matrix.shape[0]
    Q = np.zeros((n_dates, len(quantiles)))
    for i in range(n_dates):
        row = sci_matrix[i]
        row = row[np.isfinite(row)]
        if len(row) == 0:
            Q[i] = np.nan
        else:
            Q[i] = np.nanquantile(row, quantiles)
    return Q


def phase_randomize(x, rng):
    n  = len(x)
    ft = np.fft.rfft(x)
    ph = rng.uniform(0, 2 * np.pi, len(ft))
    ph[0] = 0.0
    if n % 2 == 0:
        ph[-1] = 0.0
    mag = np.abs(ft)
    return np.fft.irfft(mag * np.exp(1j * ph), n=n)


def run_pca(X, n_components=10):
    """PCA with standardization. Returns (pca, scores, cumvar_pct)."""
    # drop any rows with NaN
    valid = ~np.any(np.isnan(X), axis=1)
    Xv   = X[valid]
    # standardize each feature (column)
    mu  = Xv.mean(axis=0)
    sig = Xv.std(axis=0) + 1e-10
    Xs  = (Xv - mu) / sig
    n_comp = min(n_components, min(Xs.shape))
    pca    = PCA(n_components=n_comp)
    scores_v = pca.fit_transform(Xs)
    # expand back to full date range
    scores = np.full((X.shape[0], n_comp), np.nan)
    scores[valid] = scores_v
    cumvar = np.cumsum(pca.explained_variance_ratio_) * 100
    return pca, scores, cumvar, valid


def surrogate_pca(H, n_surr, seed, n_components=10):
    rng  = np.random.default_rng(seed)
    results = []
    for _ in range(n_surr):
        Hs = np.zeros_like(H)
        for j in range(H.shape[1]):
            col = H[:, j]
            valid = np.isfinite(col)
            surr  = col.copy()
            if valid.sum() > 4:
                surr[valid] = phase_randomize(col[valid], rng)
            Hs[:, j] = surr
        _, _, cumvar, _ = run_pca(Hs, n_components)
        results.append(cumvar.tolist())
    return results


# ── plotting ──────────────────────────────────────────────────────────────────

def plot_heatmap(H, dates, out_dir):
    """Heatmap of SCI distribution over time."""
    fig, ax = plt.subplots(figsize=(13, 5))
    im = ax.imshow(H.T, aspect="auto", origin="lower",
                   extent=[0, len(dates), 0, 10],
                   cmap="RdYlGn", vmin=0, vmax=0.25)
    ax.set_yticks(np.arange(10) + 0.5)
    ax.set_yticklabels(BIN_LABELS, fontsize=8)
    # x-axis: show year labels
    year_ticks = [i for i, d in enumerate(dates) if d.month == 1]
    ax.set_xticks([t + 0.5 for t in year_ticks])
    ax.set_xticklabels([dates[t].year for t in year_ticks])
    plt.colorbar(im, ax=ax, label="Fraction of universe")
    # mark bucket thresholds
    ax.axhline(TACTICAL_THRESH * 10, color="orange", linewidth=1.5,
               linestyle="--", label="TACTICAL threshold (0.55)")
    ax.axhline(CORE_THRESH * 10,     color="green",  linewidth=1.5,
               linestyle="--", label="CORE threshold (0.75)")
    ax.legend(fontsize=8, loc="upper right")
    ax.set_xlabel("Date")
    ax.set_ylabel("SCI bin")
    ax.set_title("Universe-wide SCI Distribution Over Time\n(fraction of instruments in each bin)")
    plt.tight_layout()
    fig.savefig(out_dir / "distribution_heatmap.png", dpi=150)
    plt.close(fig)
    print("  Saved distribution_heatmap.png")


def plot_scree_and_cumvar(pca, surr_cumvars, out_dir):
    surr = np.array(surr_cumvars)
    surr_mean = surr.mean(axis=0)
    surr_p5   = np.percentile(surr, 5,  axis=0)
    surr_p95  = np.percentile(surr, 95, axis=0)
    real_cumvar = np.cumsum(pca.explained_variance_ratio_) * 100
    n = len(real_cumvar)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    # scree
    ax = axes[0]
    ax.bar(range(1, n+1), pca.explained_variance_ratio_*100,
           color="steelblue", alpha=0.8, label="Real")
    ax.set_xlabel("Principal Component")
    ax.set_ylabel("Variance Explained (%)")
    ax.set_title("Scree Plot — Distributional PCA (56×10)")
    ax.set_xticks(range(1, n+1))

    # cumulative vs surrogate
    ax = axes[1]
    ax.plot(range(1, n+1), real_cumvar, "o-", color="tomato",
            linewidth=2.5, label="Real distribution", zorder=5)
    ax.plot(range(1, len(surr_mean)+1), surr_mean[:n], "s--",
            color="steelblue", linewidth=1.5, label="Surrogate mean", zorder=4)
    ax.fill_between(range(1, len(surr_mean)+1),
                    surr_p5[:n], surr_p95[:n],
                    color="steelblue", alpha=0.2, label="Surrogate 5–95%")
    ax.axhline(80, color="grey", linestyle="--", linewidth=1, label="80%")
    ax.axhline(90, color="grey", linestyle=":",  linewidth=1, label="90%")
    # annotate triadic
    if len(real_cumvar) >= 3:
        ax.annotate(f"PC3 = {real_cumvar[2]:.1f}%",
                    xy=(3, real_cumvar[2]),
                    xytext=(4.5, real_cumvar[2]-6),
                    arrowprops=dict(arrowstyle="->", color="tomato"),
                    color="tomato", fontsize=10)
    ax.set_xlabel("Number of Components")
    ax.set_ylabel("Cumulative Variance (%)")
    ax.set_title("Triadic Law Test — Real vs Surrogate\n(distributional PCA on 56×10 matrix)")
    ax.legend(fontsize=9)
    ax.set_ylim(0, 105)
    ax.set_xlim(0.5, n + 0.5)
    plt.tight_layout()
    fig.savefig(out_dir / "triadic_law_dist_scree.png", dpi=150)
    plt.close(fig)
    print("  Saved triadic_law_dist_scree.png")


def plot_pc_timeseries(scores, dates, pca, H, out_dir):
    valid_mask = ~np.isnan(scores[:, 0])
    valid_dates = [d for d, v in zip(dates, valid_mask) if v]

    fig = plt.figure(figsize=(14, 12))
    gs  = gridspec.GridSpec(4, 1, hspace=0.45)

    colors = ["#c0392b", "#2980b9", "#27ae60"]
    for k in range(3):
        ax = fig.add_subplot(gs[k])
        s  = scores[valid_mask, k]
        ax.plot(valid_dates, s, color=colors[k], linewidth=1.5)
        ax.axhline(0, color="black", linewidth=0.5)
        ax.fill_between(valid_dates, 0, s,
                        where=s > 0, color=colors[k], alpha=0.20)
        ax.fill_between(valid_dates, 0, s,
                        where=s < 0, color=colors[k], alpha=0.10)
        var_pct = pca.explained_variance_ratio_[k] * 100
        ax.set_ylabel(f"PC{k+1}")
        ax.set_title(f"PC{k+1} — {var_pct:.1f}% of distributional variance")
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
        ax.xaxis.set_major_locator(mdates.MonthLocator(bymonth=[1, 7]))
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=30, ha="right")

    # stacked area of CORE/TACTICAL/INELIGIBLE fractions
    ax = fig.add_subplot(gs[3])
    Hv  = H[valid_mask]
    # CORE = bins 7+8+9+10 (0.7-1.0), TACTICAL = bins 5+6+7 (0.5-0.7), rest = INELIGIBLE
    core     = Hv[:, 7:].sum(axis=1)   # 0.7–1.0
    tactical = Hv[:, 5:7].sum(axis=1)  # 0.5–0.7
    inelig   = Hv[:, :5].sum(axis=1)   # 0.0–0.5

    ax.stackplot(valid_dates, core, tactical, inelig,
                 labels=["CORE (≥0.70)", "TACTICAL (0.50–0.70)", "INELIGIBLE (<0.50)"],
                 colors=["#2ecc71", "#f39c12", "#e74c3c"], alpha=0.75)
    ax.set_ylabel("Fraction")
    ax.set_title("Universe SCI bucket fractions")
    ax.legend(loc="upper left", fontsize=8)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    ax.xaxis.set_major_locator(mdates.MonthLocator(bymonth=[1, 7]))
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=30, ha="right")

    plt.tight_layout()
    fig.savefig(out_dir / "triadic_law_dist_pc_timeseries.png", dpi=150)
    plt.close(fig)
    print("  Saved triadic_law_dist_pc_timeseries.png")


def plot_loadings(pca, out_dir):
    """Bar chart of PC loadings on each histogram bin."""
    fig, axes = plt.subplots(1, 3, figsize=(14, 4), sharey=False)
    colors = ["#c0392b", "#2980b9", "#27ae60"]
    for k, ax in enumerate(axes):
        loadings = pca.components_[k]
        bar_colors = ["tomato" if l > 0 else "steelblue" for l in loadings]
        ax.bar(range(10), loadings, color=bar_colors, alpha=0.8)
        ax.set_xticks(range(10))
        ax.set_xticklabels(BIN_LABELS, rotation=45, ha="right", fontsize=8)
        ax.axhline(0, color="black", linewidth=0.5)
        ax.set_xlabel("SCI bin")
        ax.set_ylabel("Loading")
        ax.set_title(f"PC{k+1} Loadings\n"
                     f"({pca.explained_variance_ratio_[k]*100:.1f}% variance)")
        # mark thresholds
        ax.axvline(5.5, color="orange", linestyle="--", linewidth=1,
                   alpha=0.6, label="TACTICAL (0.55)")
        ax.axvline(7.5, color="green",  linestyle="--", linewidth=1,
                   alpha=0.6, label="CORE (0.75)")
        if k == 0:
            ax.legend(fontsize=7)
    plt.suptitle("PC Loadings on SCI Histogram Bins\n"
                 "(positive = high PC score when this bin is elevated)",
                 fontsize=11)
    plt.tight_layout()
    fig.savefig(out_dir / "triadic_law_dist_loadings.png", dpi=150)
    plt.close(fig)
    print("  Saved triadic_law_dist_loadings.png")


# ── main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--matrix", default="results/triadic_law_v1/sci_matrix.csv")
    parser.add_argument("--out",    default="results/triadic_law_dist_v1")
    parser.add_argument("--n-surr", type=int, default=500)
    args = parser.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── load SCI matrix ───────────────────────────────────────────────────────
    df = pd.read_csv(args.matrix, index_col=0, parse_dates=True)
    dates      = [pd.Timestamp(d) for d in df.index]
    sci_matrix = df.values.astype(float)
    print(f"Loaded SCI matrix: {sci_matrix.shape}  "
          f"({dates[0].date()} → {dates[-1].date()})")

    # ── build histogram matrix (56×10) ───────────────────────────────────────
    H = build_hist_matrix(sci_matrix)
    print(f"Histogram matrix shape: {H.shape}")
    pd.DataFrame(H, index=[d.date() for d in dates],
                 columns=BIN_LABELS).to_csv(out_dir / "hist_matrix.csv")

    # also build quantile matrix for cross-check
    Q_quantiles = [0.10, 0.25, 0.50, 0.75, 0.90]
    Q = build_quantile_matrix(sci_matrix, Q_quantiles)
    pd.DataFrame(Q, index=[d.date() for d in dates],
                 columns=[f"q{int(q*100)}" for q in Q_quantiles]
                 ).to_csv(out_dir / "quantile_matrix.csv")

    # ── heatmap ───────────────────────────────────────────────────────────────
    plot_heatmap(H, dates, out_dir)

    # ── PCA on histogram matrix ───────────────────────────────────────────────
    print("\nRunning PCA on histogram matrix (56×10)...")
    pca, scores, cumvar, valid = run_pca(H, n_components=10)

    var_explained = pca.explained_variance_ratio_ * 100
    print("\nReal PCA results (distributional):")
    for k in range(len(var_explained)):
        print(f"  PC{k+1}: {var_explained[k]:.2f}%  (cumulative: {cumvar[k]:.2f}%)")

    pc3_cumvar = cumvar[2] if len(cumvar) >= 3 else np.nan
    print(f"\n  PC1-3 cumulative variance: {pc3_cumvar:.2f}%")
    print(f"  Triadic Law prediction (>80%): "
          f"{'CONFIRMED ✓' if pc3_cumvar > 80 else 'NOT CONFIRMED'}")

    # ── PC-bucket correlations ────────────────────────────────────────────────
    print("\nPC correlations with bucket fractions:")
    core_frac  = H[valid, 7:].sum(axis=1)
    inelig_frac = H[valid, :5].sum(axis=1)
    sc_valid   = scores[valid]
    for k in range(min(3, sc_valid.shape[1])):
        rc, pc = scipy_stats.pearsonr(sc_valid[:, k], core_frac)
        ri, pi = scipy_stats.pearsonr(sc_valid[:, k], inelig_frac)
        print(f"  PC{k+1} vs CORE:       r={rc:+.3f}  p={pc:.4f}")
        print(f"  PC{k+1} vs INELIGIBLE: r={ri:+.3f}  p={pi:.4f}")

    # ── quantile PCA cross-check ──────────────────────────────────────────────
    print("\nCross-check: PCA on quantile matrix (56×5)...")
    pca_q, _, cumvar_q, _ = run_pca(Q, n_components=5)
    for k in range(len(pca_q.explained_variance_ratio_)):
        print(f"  PC{k+1}: {pca_q.explained_variance_ratio_[k]*100:.2f}%  "
              f"cumulative: {cumvar_q[k]:.2f}%")

    # ── surrogate control ─────────────────────────────────────────────────────
    print(f"\nRunning {args.n_surr} surrogate PCA runs on histogram matrix...")
    surr_cumvars = surrogate_pca(H, args.n_surr, SEED, n_components=10)
    surr_arr     = np.array(surr_cumvars)
    surr_mean    = surr_arr.mean(axis=0)
    surr_std     = surr_arr.std(axis=0)

    surr_pc3       = surr_arr[:, 2] if surr_arr.shape[1] >= 3 else np.array([np.nan])
    surr_pc3_mean  = surr_pc3.mean()
    surr_pc3_std   = surr_pc3.std()
    z_score        = (pc3_cumvar - surr_pc3_mean) / (surr_pc3_std + 1e-8)

    print(f"\n  Surrogate PC1-3 cumulative: {surr_pc3_mean:.2f}% ± {surr_pc3_std:.2f}%")
    print(f"  z-score (real vs surrogate): {z_score:.2f}")

    # ── plots ─────────────────────────────────────────────────────────────────
    plot_scree_and_cumvar(pca, surr_cumvars, out_dir)
    plot_pc_timeseries(scores, dates, pca, H, out_dir)
    plot_loadings(pca, out_dir)

    # ── save summary ──────────────────────────────────────────────────────────
    lines = [
        "TRIADIC LAW — DISTRIBUTIONAL PCA RESULTS",
        "=" * 55,
        f"Matrix: histogram (56 dates × 10 SCI bins)",
        f"Surrogates: {args.n_surr}",
        "",
        "REAL PCA (histogram matrix):",
    ]
    for k in range(len(var_explained)):
        lines.append(f"  PC{k+1}: {var_explained[k]:.2f}%  "
                     f"cumulative: {cumvar[k]:.2f}%")
    lines += [
        "",
        f"PC1-3 cumulative variance (real):      {pc3_cumvar:.2f}%",
        f"PC1-3 cumulative variance (surrogate): {surr_pc3_mean:.2f}% ± {surr_pc3_std:.2f}%",
        f"z-score:                               {z_score:.2f}",
        f"Triadic Law >80% prediction:           {'CONFIRMED' if pc3_cumvar > 80 else 'NOT CONFIRMED'}",
        "",
        "CROSS-CHECK — quantile matrix (56×5):",
    ]
    for k in range(len(pca_q.explained_variance_ratio_)):
        lines.append(f"  PC{k+1}: {pca_q.explained_variance_ratio_[k]*100:.2f}%  "
                     f"cumulative: {cumvar_q[k]:.2f}%")

    summary = "\n".join(lines)
    print("\n" + summary)
    with open(out_dir / "triadic_law_dist_summary.txt", "w") as f:
        f.write(summary)
    print(f"\nAll outputs in {out_dir}/")


if __name__ == "__main__":
    main()
