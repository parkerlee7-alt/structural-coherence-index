"""
Triadic Law Test v1 — Do universe-wide SCI distributions have a low-rank triadic structure?

Hypothesis (Triadic Law): Market coherence self-organizes into three recurrent modes —
broad CORE epoch, transitional epoch, broad INELIGIBLE epoch — and these three modes
account for >80% of variance in the time-evolving SCI distribution.

Experiment:
  1. Compute rolling 90-day SCI for all instruments at each monthly rebalance date
  2. At each date: record the full cross-sectional SCI distribution (vector of ~1,300 scores)
  3. PCA on the (n_dates × n_instruments) SCI matrix
  4. Test: do PC1-3 explain >80% variance?
  5. Test: are the PC time series interpretable as coherence epochs?
  6. Control: phase-randomize each instrument's SCI time series → redo PCA
     Real triadic structure should NOT appear in the surrogate.

This is the SCI-within-SCI test: using the surrogate operator on SCI's own output.

Usage:
    python3 scripts/finance/triadic_law_v1.py \\
        --prices pulse_garch_finance/cache_prices \\
        --out    results/triadic_law_v1 \\
        --window 90 \\
        --workers 8
"""

import sys, argparse, warnings
from pathlib import Path
from collections import defaultdict
import multiprocessing as mp

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from scipy import stats as scipy_stats
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.gridspec as gridspec

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))
from sci_score_v3 import sci_score_v3

# ── locked parameters ─────────────────────────────────────────────────────────
W_SMOOTH = 7
L_LAGS   = 10
S        = 40
K        = 0.9
SEED     = 42
MIN_OBS  = 90       # minimum days required for a valid SCI computation


# ── helpers ──────────────────────────────────────────────────────────────────

def compute_sci_series(args):
    """
    Worker: compute SCI at each monthly date for one instrument.
    Returns (ticker, {date: SCI}) dict.
    """
    ticker, csv_path, monthly_dates, window = args
    try:
        df = pd.read_csv(csv_path, index_col=0, parse_dates=True)
        col = "Close" if "Close" in df.columns else df.columns[0]
        prices = df[col].dropna().sort_index()
        returns = prices.pct_change(fill_method=None).dropna()
    except Exception:
        return ticker, {}

    results = {}
    for date in monthly_dates:
        # take `window` trading days ending on or before `date`
        mask  = returns.index <= date
        seg   = returns[mask].iloc[-window:]
        if len(seg) < MIN_OBS:
            continue
        vals = seg.values.astype(float)
        if not np.all(np.isfinite(vals)):
            vals = vals[np.isfinite(vals)]
        if len(vals) < MIN_OBS:
            continue
        try:
            r = sci_score_v3(vals, w=W_SMOOTH, L=L_LAGS, S=S, k=K, seed=SEED)
            results[date] = r["SCI"]
        except Exception:
            pass
    return ticker, results


def phase_randomize_col(x, rng):
    """Phase-randomize a 1-D array preserving power spectrum."""
    n  = len(x)
    ft = np.fft.rfft(x)
    ph = rng.uniform(0, 2 * np.pi, len(ft))
    ph[0] = 0.0
    if n % 2 == 0:
        ph[-1] = 0.0
    return np.fft.irfft(ft * np.exp(1j * ph) / (np.abs(ft) + 1e-30) * np.abs(ft), n=n)


def run_pca(matrix, n_components=10):
    """Run PCA on a (dates × instruments) matrix with median imputation for NaN."""
    # impute column-wise NaN with column median
    col_medians = np.nanmedian(matrix, axis=0)
    nan_mask    = np.isnan(matrix)
    imputed     = matrix.copy()
    for j in range(matrix.shape[1]):
        imputed[nan_mask[:, j], j] = col_medians[j]

    # row-wise z-score (remove date-level mean shift, test distributional shape)
    row_mean = imputed.mean(axis=1, keepdims=True)
    row_std  = imputed.std(axis=1, keepdims=True) + 1e-8
    X = (imputed - row_mean) / row_std

    pca = PCA(n_components=min(n_components, min(X.shape)))
    scores = pca.fit_transform(X)
    return pca, scores, X


def bucket_fractions(sci_vec):
    """Return fraction of scores in CORE (≥0.75), TACTICAL (0.55-0.75), INELIGIBLE (<0.55)."""
    v = sci_vec[np.isfinite(sci_vec)]
    if len(v) == 0:
        return np.nan, np.nan, np.nan
    return (v >= 0.75).mean(), ((v >= 0.55) & (v < 0.75)).mean(), (v < 0.55).mean()


# ── plotting ──────────────────────────────────────────────────────────────────

def plot_variance(pca, out_dir, label=""):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    # scree plot
    n = min(15, len(pca.explained_variance_ratio_))
    ax = axes[0]
    ax.bar(range(1, n+1), pca.explained_variance_ratio_[:n] * 100,
           color="steelblue", alpha=0.8)
    ax.set_xlabel("Principal Component")
    ax.set_ylabel("Variance Explained (%)")
    ax.set_title(f"Scree Plot — {label}")
    ax.set_xticks(range(1, n+1))

    # cumulative
    ax = axes[1]
    cumvar = np.cumsum(pca.explained_variance_ratio_) * 100
    ax.plot(range(1, len(cumvar)+1), cumvar[:15], "o-", color="tomato")
    ax.axhline(80, color="grey", linestyle="--", linewidth=0.8, label="80%")
    ax.axhline(90, color="grey", linestyle=":",  linewidth=0.8, label="90%")
    ax.set_xlabel("Number of Components")
    ax.set_ylabel("Cumulative Variance (%)")
    ax.set_title(f"Cumulative Variance — {label}")
    ax.legend()
    ax.set_ylim(0, 105)

    plt.tight_layout()
    fname = f"variance_{label.lower().replace(' ','_')}.png"
    fig.savefig(out_dir / fname, dpi=150)
    plt.close(fig)
    print(f"  Saved {fname}")
    return cumvar


def plot_pc_timeseries(scores, dates, pca, bucket_df, out_dir, label=""):
    """Plot PC1-3 over time alongside CORE/TACTICAL/INELIGIBLE fractions."""
    fig = plt.figure(figsize=(15, 10))
    gs  = gridspec.GridSpec(4, 1, hspace=0.4)

    colors = ["#1f77b4", "#ff7f0e", "#2ca02c"]
    for k in range(3):
        ax = fig.add_subplot(gs[k])
        ax.plot(dates, scores[:, k], color=colors[k], linewidth=1.2)
        ax.axhline(0, color="black", linewidth=0.5)
        var_pct = pca.explained_variance_ratio_[k] * 100
        ax.set_ylabel(f"PC{k+1}")
        ax.set_title(f"PC{k+1} — {var_pct:.1f}% variance  [{label}]")
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
        ax.xaxis.set_major_locator(mdates.YearLocator())

    # bucket fractions
    ax = fig.add_subplot(gs[3])
    ax.stackplot(
        dates,
        bucket_df["core"],
        bucket_df["tactical"],
        bucket_df["ineligible"],
        labels=["CORE (≥0.75)", "TACTICAL (0.55–0.75)", "INELIGIBLE (<0.55)"],
        colors=["#2ecc71", "#f39c12", "#e74c3c"],
        alpha=0.75,
    )
    ax.set_ylabel("Fraction of universe")
    ax.set_title("Universe-wide SCI bucket fractions over time")
    ax.legend(loc="upper left", fontsize=8)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax.xaxis.set_major_locator(mdates.YearLocator())

    fname = f"pc_timeseries_{label.lower().replace(' ','_')}.png"
    fig.savefig(out_dir / fname, dpi=150)
    plt.close(fig)
    print(f"  Saved {fname}")


def plot_comparison(real_var, surr_var_list, out_dir):
    """Compare cumulative variance of real PCA vs surrogate PCA."""
    fig, ax = plt.subplots(figsize=(10, 5))

    surr_arr = np.array(surr_var_list)
    surr_mean = surr_arr.mean(axis=0)
    surr_p5   = np.percentile(surr_arr, 5, axis=0)
    surr_p95  = np.percentile(surr_arr, 95, axis=0)
    n = len(real_var)

    ax.plot(range(1, n+1), real_var[:n], "o-", color="tomato",
            linewidth=2, label="Real SCI distributions", zorder=5)
    ax.plot(range(1, len(surr_mean)+1), surr_mean, "s--", color="steelblue",
            linewidth=1.5, label="Surrogate mean", zorder=4)
    ax.fill_between(range(1, len(surr_mean)+1), surr_p5, surr_p95,
                    color="steelblue", alpha=0.2, label="Surrogate 5–95%")
    ax.axhline(80, color="grey", linestyle="--", linewidth=0.8, label="80% threshold")

    # annotate PC3 variance for real data
    if len(real_var) >= 3:
        ax.annotate(f"Real PC3 = {real_var[2]:.1f}%",
                    xy=(3, real_var[2]), xytext=(5, real_var[2] - 8),
                    arrowprops=dict(arrowstyle="->", color="tomato"), color="tomato")

    ax.set_xlabel("Number of Principal Components")
    ax.set_ylabel("Cumulative Variance Explained (%)")
    ax.set_title("Real vs Surrogate PCA — Triadic Law Test\n"
                 "Real curve above surrogate band = low-rank structure is not noise")
    ax.legend(fontsize=9)
    ax.set_ylim(0, 105)
    ax.set_xlim(0.5, min(n, 15) + 0.5)

    fig.savefig(out_dir / "triadic_law_real_vs_surrogate.png", dpi=150)
    plt.close(fig)
    print("  Saved triadic_law_real_vs_surrogate.png")


# ── main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Triadic Law PCA test on universe SCI")
    parser.add_argument("--prices",   default="pulse_garch_finance/cache_prices")
    parser.add_argument("--out",      default="results/triadic_law_v1")
    parser.add_argument("--window",   type=int, default=90)
    parser.add_argument("--workers",  type=int, default=min(8, mp.cpu_count()))
    parser.add_argument("--n-surr",   type=int, default=20,
                        help="Number of surrogate PCA runs for the control")
    args = parser.parse_args()

    prices_dir = Path(args.prices)
    out_dir    = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── find all price files ──────────────────────────────────────────────────
    csv_files = sorted(prices_dir.glob("*.csv"))
    print(f"Found {len(csv_files)} price files.")

    # ── determine monthly rebalance dates from the data ───────────────────────
    # load one large file to get date range
    sample_df   = pd.read_csv(csv_files[0], index_col=0, parse_dates=True)
    all_dates   = sample_df.index.sort_values()
    date_min    = all_dates[args.window]           # first valid date
    date_max    = all_dates[-1]

    # monthly end-of-month dates
    monthly = pd.date_range(start=date_min, end=date_max, freq="ME")
    # snap to actual trading days (take last trading day of each calendar month)
    trading_days = pd.DatetimeIndex(all_dates)
    monthly_dates = []
    for m in monthly:
        before = trading_days[trading_days <= m]
        if len(before):
            monthly_dates.append(before[-1])
    monthly_dates = sorted(set(monthly_dates))
    print(f"Monthly rebalance dates: {monthly_dates[0].date()} → "
          f"{monthly_dates[-1].date()}  ({len(monthly_dates)} dates)")

    # ── compute rolling SCI for every instrument ──────────────────────────────
    work = [(f.stem, f, monthly_dates, args.window) for f in csv_files]
    print(f"\nComputing rolling {args.window}-day SCI for {len(work)} instruments "
          f"using {args.workers} workers...")

    with mp.Pool(processes=args.workers) as pool:
        results = pool.map(compute_sci_series, work)

    # ── assemble SCI matrix (dates × instruments) ─────────────────────────────
    tickers   = [r[0] for r in results]
    sci_dict  = {r[0]: r[1] for r in results}

    # filter: keep instruments with data on ≥50% of dates
    min_dates = len(monthly_dates) // 2
    valid_tickers = [t for t in tickers
                     if sum(1 for d in monthly_dates if d in sci_dict[t]) >= min_dates]
    print(f"Instruments with ≥50% date coverage: {len(valid_tickers)}")

    # build matrix
    sci_matrix = np.full((len(monthly_dates), len(valid_tickers)), np.nan)
    for j, ticker in enumerate(valid_tickers):
        for i, date in enumerate(monthly_dates):
            if date in sci_dict[ticker]:
                sci_matrix[i, j] = sci_dict[ticker][date]

    # save raw matrix
    df_matrix = pd.DataFrame(
        sci_matrix,
        index=[d.date() for d in monthly_dates],
        columns=valid_tickers
    )
    df_matrix.to_csv(out_dir / "sci_matrix.csv")
    print(f"SCI matrix shape: {sci_matrix.shape}  → saved sci_matrix.csv")

    # ── bucket fractions over time ────────────────────────────────────────────
    bucket_records = []
    for i in range(len(monthly_dates)):
        row = sci_matrix[i]
        c, t, inelig = bucket_fractions(row)
        bucket_records.append({"date": monthly_dates[i], "core": c,
                                "tactical": t, "ineligible": inelig})
    bucket_df = pd.DataFrame(bucket_records).set_index("date")
    bucket_df.to_csv(out_dir / "bucket_fractions.csv")

    # ── PCA on real data ──────────────────────────────────────────────────────
    print("\nRunning PCA on real SCI matrix...")
    pca_real, scores_real, X_real = run_pca(sci_matrix, n_components=15)

    var_explained = pca_real.explained_variance_ratio_ * 100
    cum_var       = np.cumsum(var_explained)
    pc3_cumvar    = cum_var[2] if len(cum_var) >= 3 else np.nan

    print(f"\nReal PCA results:")
    for k in range(min(10, len(var_explained))):
        print(f"  PC{k+1}: {var_explained[k]:.2f}%  (cumulative: {cum_var[k]:.2f}%)")
    print(f"\n  PC1-3 cumulative variance: {pc3_cumvar:.2f}%")
    print(f"  Triadic Law prediction (>80%): {'CONFIRMED' if pc3_cumvar > 80 else 'NOT CONFIRMED'}")

    # bucket fractions aligned to monthly dates
    bucket_aligned = bucket_df.reindex(monthly_dates).fillna(method="ffill")

    # ── PCA plots ─────────────────────────────────────────────────────────────
    real_cumvar = plot_variance(pca_real, out_dir, label="Real")
    plot_pc_timeseries(scores_real, monthly_dates, pca_real, bucket_aligned,
                       out_dir, label="Real")

    # ── PC-bucket correlations ────────────────────────────────────────────────
    print("\nCorrelations: PC scores vs bucket fractions")
    for k in range(3):
        for col in ["core", "tactical", "ineligible"]:
        	vals = bucket_aligned[col].values
        	mask = np.isfinite(vals) & np.isfinite(scores_real[:, k])
        	if mask.sum() < 5:
        	    continue
        	r, p = scipy_stats.pearsonr(scores_real[mask, k], vals[mask])
        	print(f"  PC{k+1} vs {col:10s}: r={r:+.3f}  p={p:.4f}")

    # ── Surrogate control ─────────────────────────────────────────────────────
    print(f"\nRunning {args.n_surr} surrogate PCA runs...")
    rng = np.random.default_rng(SEED)
    surr_cumvars = []

    # impute once for surrogates
    col_med = np.nanmedian(sci_matrix, axis=0)
    nan_mask = np.isnan(sci_matrix)
    X_imp = sci_matrix.copy()
    for j in range(X_imp.shape[1]):
        X_imp[nan_mask[:, j], j] = col_med[j]

    for s in range(args.n_surr):
        surr = np.zeros_like(X_imp)
        for j in range(X_imp.shape[1]):
            surr[:, j] = phase_randomize_col(X_imp[:, j], rng)

        pca_s, _, _ = run_pca(surr, n_components=15)
        surr_cumvars.append(np.cumsum(pca_s.explained_variance_ratio_ * 100).tolist())
        if (s + 1) % 5 == 0:
            print(f"  Surrogate {s+1}/{args.n_surr}  "
                  f"PC3 cumvar={np.cumsum(pca_s.explained_variance_ratio_*100)[2]:.2f}%")

    # ── comparison plot ───────────────────────────────────────────────────────
    plot_comparison(real_cumvar[:15].tolist(), surr_cumvars, out_dir)

    # ── summary stats ─────────────────────────────────────────────────────────
    surr_pc3 = [cv[2] for cv in surr_cumvars if len(cv) >= 3]
    surr_pc3_mean = np.mean(surr_pc3)
    surr_pc3_std  = np.std(surr_pc3)
    z_score = (pc3_cumvar - surr_pc3_mean) / (surr_pc3_std + 1e-8)

    lines = [
        "TRIADIC LAW TEST — RESULTS",
        "=" * 55,
        f"Instruments: {len(valid_tickers)}",
        f"Monthly dates: {len(monthly_dates)}  "
        f"({monthly_dates[0].date()} → {monthly_dates[-1].date()})",
        f"Rolling window: {args.window} days",
        f"Surrogates: {args.n_surr}",
        "",
        "REAL PCA:",
    ]
    for k in range(min(10, len(var_explained))):
        lines.append(f"  PC{k+1}: {var_explained[k]:.2f}%  cumulative: {cum_var[k]:.2f}%")
    lines += [
        "",
        f"PC1-3 cumulative variance (real):      {pc3_cumvar:.2f}%",
        f"PC1-3 cumulative variance (surrogate): {surr_pc3_mean:.2f}% ± {surr_pc3_std:.2f}%",
        f"z-score (real vs surrogate):           {z_score:.2f}",
        f"Triadic Law prediction (>80%):         {'CONFIRMED' if pc3_cumvar > 80 else 'NOT CONFIRMED'}",
        "",
        "BUCKET FRACTION SUMMARY (grand means):",
        f"  CORE      (≥0.75): {bucket_df['core'].mean()*100:.1f}%",
        f"  TACTICAL  (0.55–0.75): {bucket_df['tactical'].mean()*100:.1f}%",
        f"  INELIGIBLE (<0.55): {bucket_df['ineligible'].mean()*100:.1f}%",
    ]

    print("\n" + "\n".join(lines))
    with open(out_dir / "triadic_law_summary.txt", "w") as f:
        f.write("\n".join(lines))
    print(f"\nAll outputs in {out_dir}/")


if __name__ == "__main__":
    main()
