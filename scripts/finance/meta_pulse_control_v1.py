"""
Meta-Pulse Control Analysis v1

The decisive test: do CORE stocks have higher cross-envelope coherence than
equally-correlated non-CORE stocks?

Design:
  1. Sample many random groups of N stocks from the full universe
  2. For each group compute:
       r_returns  = mean pairwise Pearson correlation of raw returns
                    (measures ordinary market co-movement)
       r_envelope = mean pairwise Pearson correlation of Hilbert envelopes
                    (= Meta-Pulse c_obs)
       gap        = r_envelope - surrogate_mean
  3. Fit OLS: r_envelope ~ r_returns (across all random groups)
  4. The regression captures: "given ordinary correlation, how much envelope
     correlation is expected?"
  5. For the CORE group, compute residual = actual r_envelope - predicted
  6. If CORE residual > 0 and above the bulk of random groups → Meta-Pulse is real

Key scatter: r_returns (x) vs r_envelope (y), colored by CORE status.
CORE group above the regression line = envelope coherence beyond return correlation.

Usage:
    python3 scripts/finance/meta_pulse_control_v1.py \\
        --prices  pulse_garch_finance/cache_prices \\
        --results pulse_garch_finance/results_garch_finance/financial_garch_sci_results_v2_ranked.csv \\
        --out     results/meta_pulse_control_v1 \\
        --n-stocks 20 \\
        --n-samples 300 \\
        --surrogates 100
"""

import sys, argparse
from pathlib import Path
from itertools import combinations

import numpy as np
import pandas as pd
import scipy.signal as sg
from scipy.stats import pearsonr
from scipy import stats as scipy_stats
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

# ── locked parameters ─────────────────────────────────────────────────────────
W_SMOOTH = 7
K        = 0.9
SEED     = 42


# ── helpers ──────────────────────────────────────────────────────────────────

def phase_randomize(x, rng):
    n  = len(x)
    ft = np.fft.rfft(x)
    ph = rng.uniform(0, 2 * np.pi, len(ft))
    ph[0] = 0.0
    if n % 2 == 0:
        ph[-1] = 0.0
    return np.fft.irfft(ft * np.exp(1j * ph) / (np.abs(ft) + 1e-30) * np.abs(ft), n=n)


def hilbert_envelope(x, w=W_SMOOTH):
    env = np.abs(sg.hilbert(x))
    if w > 1:
        env = np.convolve(env, np.ones(w) / w, mode="same")
    return env


def mean_pairwise_corr(matrix):
    """Mean Pearson r across all unique column pairs. matrix: (T, N)."""
    N = matrix.shape[1]
    corrs = []
    for i, j in combinations(range(N), 2):
        c, _ = pearsonr(matrix[:, i], matrix[:, j])
        if np.isfinite(c):
            corrs.append(c)
    return float(np.mean(corrs)) if corrs else np.nan


def meta_pulse_gap(returns_matrix, S=100, seed=SEED, w=W_SMOOTH):
    """Return (r_returns, r_envelope, gap, surrogate_std)."""
    T, N = returns_matrix.shape
    rng  = np.random.default_rng(seed)

    r_returns  = mean_pairwise_corr(returns_matrix)

    envelopes  = np.stack([hilbert_envelope(returns_matrix[:, i], w) for i in range(N)], axis=1)
    r_envelope = mean_pairwise_corr(envelopes)

    surr_corrs = np.zeros(S)
    for s in range(S):
        surr_env = np.stack(
            [hilbert_envelope(phase_randomize(returns_matrix[:, i], rng), w)
             for i in range(N)], axis=1
        )
        surr_corrs[s] = mean_pairwise_corr(surr_env)

    gap = r_envelope - surr_corrs.mean()
    z   = np.clip(gap / (surr_corrs.std() + 1e-12), -6, 6)
    mp  = 1.0 / (1.0 + np.exp(-K * z))

    return {
        "r_returns":  r_returns,
        "r_envelope": r_envelope,
        "c_surr_mean":surr_corrs.mean(),
        "c_surr_std": surr_corrs.std(),
        "gap":        gap,
        "z":          z,
        "meta_pulse": mp,
    }


def load_prices(prices_dir, tickers):
    frames = {}
    for t in tickers:
        p = Path(prices_dir) / f"{t}.csv"
        if not p.exists():
            continue
        df = pd.read_csv(p, index_col=0, parse_dates=True)
        col = "Close" if "Close" in df.columns else df.columns[0]
        frames[t] = df[col].rename(t)
    if not frames:
        return pd.DataFrame()
    return pd.DataFrame(frames).sort_index().dropna(how="all")


# ── main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--prices",     default="pulse_garch_finance/cache_prices")
    parser.add_argument("--results",    default="pulse_garch_finance/results_garch_finance/financial_garch_sci_results_v2_ranked.csv")
    parser.add_argument("--out",        default="results/meta_pulse_control_v1")
    parser.add_argument("--n-stocks",   type=int, default=20)
    parser.add_argument("--n-samples",  type=int, default=300,
                        help="Random group samples to build the baseline distribution")
    parser.add_argument("--surrogates", type=int, default=100)
    args = parser.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    N = args.n_stocks
    rng = np.random.default_rng(SEED)

    results_df = pd.read_csv(args.results)
    print(f"Loaded {len(results_df)} tickers.")

    # ── identify CORE and non-CORE ────────────────────────────────────────────
    core_mask     = results_df["amplitude_bucket_v2"].str.contains("CORE", na=False)
    core_tickers  = results_df[core_mask]["ticker"].tolist()
    noncore_tickers = results_df[~core_mask & (results_df["amplitude_bucket_v2"] != "NO_DATA")]["ticker"].tolist()

    print(f"CORE: {len(core_tickers)}  |  Non-CORE: {len(noncore_tickers)}")

    # ── load all prices ───────────────────────────────────────────────────────
    all_tickers = core_tickers + noncore_tickers
    print(f"Loading prices for {len(all_tickers)} tickers...")
    prices = load_prices(args.prices, all_tickers)
    returns_all = prices.pct_change(fill_method=None).dropna(how="all")

    # find common date range where we have enough stocks
    valid_cols = returns_all.columns[returns_all.notna().mean() > 0.8]
    returns_all = returns_all[valid_cols].dropna()
    print(f"Aligned date range: {returns_all.index[0].date()} → "
          f"{returns_all.index[-1].date()}  ({len(returns_all)} days, "
          f"{len(valid_cols)} stocks)")

    available_core    = [t for t in core_tickers    if t in returns_all.columns]
    available_noncore = [t for t in noncore_tickers if t in returns_all.columns]
    print(f"Available — CORE: {len(available_core)}  Non-CORE: {len(available_noncore)}")

    # ── CORE group: top-N by w500_gap ─────────────────────────────────────────
    gap_map   = results_df.set_index("ticker")["w500_gap"].to_dict()
    top_core  = sorted(available_core,
                       key=lambda t: gap_map.get(t, -999), reverse=True)[:N]
    print(f"\nTop-{N} CORE stocks: {top_core}")

    core_returns = returns_all[top_core].values.astype(float)
    print(f"Computing Meta-Pulse for CORE group (S={args.surrogates})...")
    core_result = meta_pulse_gap(core_returns, S=args.surrogates, seed=SEED)
    print(f"  r_returns  = {core_result['r_returns']:.4f}")
    print(f"  r_envelope = {core_result['r_envelope']:.4f}")
    print(f"  gap        = {core_result['gap']:.4f}")
    print(f"  z          = {core_result['z']:.4f}")
    print(f"  MP         = {core_result['meta_pulse']:.4f}")

    # ── Random samples: CORE vs non-CORE groups ───────────────────────────────
    print(f"\nSampling {args.n_samples} random groups each from CORE and non-CORE...")

    def sample_groups(ticker_pool, n_stocks, n_samples, S, label):
        records = []
        pool = [t for t in ticker_pool if t in returns_all.columns]
        if len(pool) < n_stocks:
            print(f"  WARNING: {label} pool too small ({len(pool)} < {n_stocks})")
            return records
        for i in range(n_samples):
            chosen = rng.choice(pool, size=n_stocks, replace=False)
            mat    = returns_all[chosen].values.astype(float)
            r      = meta_pulse_gap(mat, S=S, seed=int(rng.integers(1e6)))
            r["group"]   = label
            r["tickers"] = ", ".join(chosen[:5])
            records.append(r)
            if (i + 1) % 50 == 0:
                print(f"  {label}: {i+1}/{n_samples}")
        return records

    core_samples    = sample_groups(available_core,    N, args.n_samples,
                                    args.surrogates // 2, "CORE")
    noncore_samples = sample_groups(available_noncore, N, args.n_samples,
                                    args.surrogates // 2, "NON_CORE")

    df = pd.DataFrame(core_samples + noncore_samples)
    df.to_csv(out_dir / "control_samples.csv", index=False)
    print(f"\nSaved {len(df)} sample records → control_samples.csv")

    # ── OLS regression: r_envelope ~ r_returns ───────────────────────────────
    slope, intercept, r_val, p_val, se = scipy_stats.linregress(
        df["r_returns"], df["r_envelope"]
    )
    print(f"\nOLS across all {len(df)} samples:")
    print(f"  r_envelope = {intercept:.4f} + {slope:.4f} × r_returns")
    print(f"  R²={r_val**2:.4f}  p={p_val:.4e}")

    # CORE group residual
    core_predicted  = intercept + slope * core_result["r_returns"]
    core_residual   = core_result["r_envelope"] - core_predicted
    core_pct_rank   = (df["r_envelope"] - (intercept + slope * df["r_returns"])
                       ).lt(core_residual).mean()

    print(f"\nTop-{N} CORE group:")
    print(f"  r_returns        = {core_result['r_returns']:.4f}")
    print(f"  r_envelope obs   = {core_result['r_envelope']:.4f}")
    print(f"  r_envelope pred  = {core_predicted:.4f}")
    print(f"  residual         = {core_residual:+.4f}")
    print(f"  percentile rank  = {core_pct_rank:.1%}  "
          f"(beats {core_pct_rank*100:.0f}% of random {N}-stock groups)")

    # ── CORE vs non-CORE distribution comparison ──────────────────────────────
    core_df    = df[df["group"] == "CORE"]
    noncore_df = df[df["group"] == "NON_CORE"]

    t_stat, t_p = scipy_stats.ttest_ind(
        core_df["gap"], noncore_df["gap"], alternative="greater"
    )
    t_env, p_env = scipy_stats.ttest_ind(
        core_df["r_envelope"], noncore_df["r_envelope"], alternative="greater"
    )

    print(f"\nCORE vs non-CORE (random samples, N={N} each group, "
          f"{args.n_samples} draws):")
    print(f"  CORE    gap: {core_df['gap'].mean():.4f} ± {core_df['gap'].std():.4f}")
    print(f"  NonCORE gap: {noncore_df['gap'].mean():.4f} ± {noncore_df['gap'].std():.4f}")
    print(f"  t-test (gap, one-sided CORE>NonCORE): t={t_stat:.3f} p={t_p:.4f}")
    print()
    print(f"  CORE    r_envelope: {core_df['r_envelope'].mean():.4f} ± {core_df['r_envelope'].std():.4f}")
    print(f"  NonCORE r_envelope: {noncore_df['r_envelope'].mean():.4f} ± {noncore_df['r_envelope'].std():.4f}")
    print(f"  t-test (r_envelope, one-sided CORE>NonCORE): t={t_env:.3f} p={p_env:.4f}")

    # ── save summary ──────────────────────────────────────────────────────────
    summary_lines = [
        "META-PULSE CONTROL ANALYSIS",
        "=" * 50,
        f"N stocks per group: {N}",
        f"Random samples per group: {args.n_samples}",
        f"Surrogates per computation: {args.surrogates // 2}",
        "",
        f"Top-{N} CORE group (fixed):",
        f"  Tickers:       {', '.join(top_core)}",
        f"  r_returns    = {core_result['r_returns']:.4f}",
        f"  r_envelope   = {core_result['r_envelope']:.4f}",
        f"  gap          = {core_result['gap']:.4f}",
        f"  z            = {core_result['z']:.4f}",
        f"  Meta-Pulse   = {core_result['meta_pulse']:.4f}",
        f"  OLS residual = {core_residual:+.4f}",
        f"  Percentile   = {core_pct_rank:.1%} above random groups",
        "",
        f"OLS (all {len(df)} samples): r_envelope = {intercept:.4f} + {slope:.4f}×r_returns  "
        f"R²={r_val**2:.4f}  p={p_val:.2e}",
        "",
        "CORE vs non-CORE random samples:",
        f"  CORE    gap: {core_df['gap'].mean():.4f} ± {core_df['gap'].std():.4f}",
        f"  NonCORE gap: {noncore_df['gap'].mean():.4f} ± {noncore_df['gap'].std():.4f}",
        f"  t-test gap (CORE>NonCORE): t={t_stat:.3f}  p={t_p:.4f}",
        f"  CORE    r_env: {core_df['r_envelope'].mean():.4f} ± {core_df['r_envelope'].std():.4f}",
        f"  NonCORE r_env: {noncore_df['r_envelope'].mean():.4f} ± {noncore_df['r_envelope'].std():.4f}",
        f"  t-test r_env (CORE>NonCORE): t={t_env:.3f}  p={p_env:.4f}",
    ]
    with open(out_dir / "control_summary.txt", "w") as f:
        f.write("\n".join(summary_lines))
    print("\nSaved control_summary.txt")

    # ── plots ─────────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    # 1. Scatter: r_returns vs r_envelope, colored by group
    ax = axes[0]
    ax.scatter(noncore_df["r_returns"], noncore_df["r_envelope"],
               alpha=0.3, s=15, color="steelblue", label="Non-CORE samples", zorder=2)
    ax.scatter(core_df["r_returns"], core_df["r_envelope"],
               alpha=0.4, s=15, color="tomato", label="CORE samples", zorder=3)

    # regression line
    x_line = np.linspace(df["r_returns"].min(), df["r_returns"].max(), 100)
    ax.plot(x_line, intercept + slope * x_line,
            color="black", linewidth=1.5, linestyle="--",
            label=f"OLS (R²={r_val**2:.3f})", zorder=4)

    # top-N CORE group star
    ax.scatter([core_result["r_returns"]], [core_result["r_envelope"]],
               color="red", s=200, marker="*", zorder=5,
               label=f"Top-{N} CORE\n(residual={core_residual:+.3f})")

    ax.set_xlabel("Mean pairwise return correlation (ordinary market co-movement)")
    ax.set_ylabel("Mean pairwise envelope correlation (Meta-Pulse c_obs)")
    ax.set_title("Return Correlation vs. Envelope Correlation\nby Group")
    ax.legend(fontsize=8)

    # 2. Gap distribution: CORE vs non-CORE
    ax = axes[1]
    ax.hist(noncore_df["gap"], bins=30, alpha=0.6, color="steelblue",
            label=f"Non-CORE (μ={noncore_df['gap'].mean():.3f})", density=True)
    ax.hist(core_df["gap"], bins=30, alpha=0.6, color="tomato",
            label=f"CORE (μ={core_df['gap'].mean():.3f})", density=True)
    ax.axvline(core_result["gap"], color="red", linewidth=2, linestyle="-",
               label=f"Top-{N} CORE gap={core_result['gap']:.3f}")
    ax.set_xlabel("Meta-Pulse gap (r_envelope − c_surr_mean)")
    ax.set_ylabel("Density")
    ax.set_title(f"Meta-Pulse Gap Distribution\nCORE vs Non-CORE (N={N} per group)\n"
                 f"t={t_stat:.2f}  p={t_p:.4f}")
    ax.legend(fontsize=8)

    # 3. OLS residuals distribution: does CORE systematically sit above the line?
    ax = axes[2]
    all_residuals = df["r_envelope"] - (intercept + slope * df["r_returns"])
    core_res    = all_residuals[df["group"] == "CORE"]
    noncore_res = all_residuals[df["group"] == "NON_CORE"]

    ax.hist(noncore_res, bins=30, alpha=0.6, color="steelblue",
            label=f"Non-CORE residual (μ={noncore_res.mean():.3f})", density=True)
    ax.hist(core_res, bins=30, alpha=0.6, color="tomato",
            label=f"CORE residual (μ={core_res.mean():.3f})", density=True)
    ax.axvline(core_residual, color="red", linewidth=2,
               label=f"Top-{N} CORE residual={core_residual:+.3f}\n"
                     f"({core_pct_rank:.0%} percentile)")
    ax.axvline(0, color="black", linewidth=0.8, linestyle="--")
    ax.set_xlabel("OLS residual (actual − predicted envelope corr)")
    ax.set_ylabel("Density")
    ax.set_title("Envelope Coherence Residual\n(controlling for ordinary return correlation)")
    ax.legend(fontsize=8)

    plt.tight_layout()
    fig.savefig(out_dir / "meta_pulse_control.png", dpi=150)
    plt.close(fig)
    print("Saved → meta_pulse_control.png")

    print(f"\nDone. All outputs in {out_dir}/")


if __name__ == "__main__":
    main()
