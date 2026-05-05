#!/usr/bin/env python3
"""
Amplitude Risk/Quality Analysis

Reads:
  results_forward_amplitude/bucket_forward_stats.csv
  results_forward_amplitude/forward_amplitude_returns.csv

Creates:
  results_forward_amplitude/risk_quality_summary.csv
  results_forward_amplitude/core_vs_low_spreads.csv
  results_forward_amplitude/core_vs_spectral_spreads.csv
  results_forward_amplitude/summary_risk_quality.txt

Purpose:
  Show whether AMPLITUDE_CORE_TOP10 improves return quality:
    - mean return
    - median return
    - volatility / dispersion
    - sharpe-like ratio
    - win rate
    - CORE vs LOW spread
    - CORE vs SPECTRAL spread
"""

import os
import numpy as np
import pandas as pd

OUTDIR = "results_forward_amplitude"
STATS_FILE = os.path.join(OUTDIR, "bucket_forward_stats.csv")
RETURNS_FILE = os.path.join(OUTDIR, "forward_amplitude_returns.csv")

CORE = "AMPLITUDE_CORE_TOP10"
LOW = "AMPLITUDE_LOW_BOTTOM40"
SPECTRAL = "SPECTRAL_DOMINANT"
MID = "AMPLITUDE_MID_70_90"
TACTICAL = "AMPLITUDE_TACTICAL_40_70"

def pct(x):
    if pd.isna(x):
        return ""
    return f"{100*x:.2f}%"

def num(x):
    if pd.isna(x):
        return ""
    return f"{x:.4f}"

def load():
    if not os.path.exists(STATS_FILE):
        raise FileNotFoundError(f"Missing {STATS_FILE}")
    if not os.path.exists(RETURNS_FILE):
        raise FileNotFoundError(f"Missing {RETURNS_FILE}")

    stats = pd.read_csv(STATS_FILE)
    returns = pd.read_csv(RETURNS_FILE, parse_dates=["date"])
    return stats, returns

def make_spread_table(stats, compare_bucket):
    rows = []

    for h in sorted(stats["horizon"].unique()):
        core = stats[(stats["horizon"] == h) & (stats["bucket"] == CORE)]
        other = stats[(stats["horizon"] == h) & (stats["bucket"] == compare_bucket)]

        if core.empty or other.empty:
            continue

        c = core.iloc[0]
        o = other.iloc[0]

        rows.append({
            "horizon": h,
            "compare": f"{CORE} vs {compare_bucket}",

            "core_mean": c["mean_return"],
            "other_mean": o["mean_return"],
            "mean_spread_core_minus_other": c["mean_return"] - o["mean_return"],

            "core_median": c["median_return"],
            "other_median": o["median_return"],
            "median_spread_core_minus_other": c["median_return"] - o["median_return"],

            "core_std": c["std_return"],
            "other_std": o["std_return"],
            "std_reduction_core_vs_other": o["std_return"] - c["std_return"],
            "std_reduction_pct": (o["std_return"] - c["std_return"]) / o["std_return"] if o["std_return"] else np.nan,

            "core_sharpe_like": c["sharpe_like"],
            "other_sharpe_like": o["sharpe_like"],
            "sharpe_like_spread": c["sharpe_like"] - o["sharpe_like"],
            "sharpe_like_ratio": c["sharpe_like"] / o["sharpe_like"] if o["sharpe_like"] else np.nan,

            "core_win_rate": c["win_rate"],
            "other_win_rate": o["win_rate"],
            "win_rate_spread": c["win_rate"] - o["win_rate"],

            "core_avg_gap": c["avg_gap"],
            "other_avg_gap": o["avg_gap"],
        })

    return pd.DataFrame(rows)

def make_ranked_quality_table(stats):
    df = stats.copy()

    # Higher is better for mean, median, sharpe, win rate.
    # Lower is better for std.
    out = []

    for h in sorted(df["horizon"].unique()):
        sub = df[df["horizon"] == h].copy()

        sub["mean_rank"] = sub["mean_return"].rank(ascending=False)
        sub["median_rank"] = sub["median_return"].rank(ascending=False)
        sub["std_rank"] = sub["std_return"].rank(ascending=True)
        sub["sharpe_rank"] = sub["sharpe_like"].rank(ascending=False)
        sub["win_rate_rank"] = sub["win_rate"].rank(ascending=False)

        sub["quality_rank_average"] = sub[
            ["mean_rank", "median_rank", "std_rank", "sharpe_rank", "win_rate_rank"]
        ].mean(axis=1)

        sub["quality_score"] = 1 / sub["quality_rank_average"]

        out.append(sub)

    ranked = pd.concat(out, ignore_index=True)
    ranked = ranked.sort_values(["horizon", "quality_rank_average"])
    return ranked

def make_date_level_spreads(returns):
    """
    Date-level portfolio spreads:
    equal-weight bucket return each rebalance date, then compare CORE vs others.
    """
    rows = []

    for h in [21, 63, 126, 252]:
        col = f"fwd_{h}"
        if col not in returns.columns:
            continue

        sub = returns[["date", "amplitude_bucket_v2", col]].dropna().copy()

        piv = sub.pivot_table(
            index="date",
            columns="amplitude_bucket_v2",
            values=col,
            aggfunc="mean"
        )

        if CORE not in piv.columns:
            continue

        for other in [LOW, MID, TACTICAL, SPECTRAL]:
            if other not in piv.columns:
                continue

            spread = piv[CORE] - piv[other]
            spread = spread.dropna()

            if len(spread) == 0:
                continue

            rows.append({
                "horizon": h,
                "spread": f"{CORE} minus {other}",
                "n_rebalance_dates": len(spread),
                "mean_spread": spread.mean(),
                "median_spread": spread.median(),
                "std_spread": spread.std(ddof=1),
                "spread_sharpe_like": spread.mean() / (spread.std(ddof=1) + 1e-12),
                "spread_win_rate": (spread > 0).mean(),
                "worst_spread": spread.min(),
                "best_spread": spread.max(),
            })

    return pd.DataFrame(rows)

def write_report(stats, ranked, spread_low, spread_spectral, date_spreads):
    lines = []
    lines.append("=" * 80)
    lines.append("AMPLITUDE RISK / QUALITY ANALYSIS")
    lines.append("=" * 80)
    lines.append("")
    lines.append("Main question:")
    lines.append("Does AMPLITUDE_CORE_TOP10 improve return quality, not just raw return?")
    lines.append("")

    lines.append("Bucket quality ranking by horizon:")
    show = ranked[
        [
            "horizon", "bucket", "n", "mean_return", "median_return",
            "std_return", "sharpe_like", "win_rate", "quality_rank_average"
        ]
    ].copy()
    lines.append(show.to_string(index=False))
    lines.append("")

    lines.append("=" * 80)
    lines.append("CORE vs LOW_BOTTOM40")
    lines.append("=" * 80)
    lines.append(spread_low.to_string(index=False))
    lines.append("")

    lines.append("=" * 80)
    lines.append("CORE vs SPECTRAL_DOMINANT")
    lines.append("=" * 80)
    lines.append(spread_spectral.to_string(index=False))
    lines.append("")

    lines.append("=" * 80)
    lines.append("Date-level equal-weight spread tests")
    lines.append("=" * 80)
    lines.append(date_spreads.to_string(index=False))
    lines.append("")

    lines.append("Plain-English read:")
    lines.append("- If CORE has lower std than LOW with similar mean, it is a cleaner return bucket.")
    lines.append("- If CORE has higher sharpe-like and higher win rate than LOW, amplitude gap is useful as a quality filter.")
    lines.append("- If CORE beats SPECTRAL_DOMINANT, negative SCI gap may be an avoid/noise bucket.")
    lines.append("- Date-level spreads are more important than pooled rows because they treat each rebalance date like a portfolio snapshot.")

    with open(os.path.join(OUTDIR, "summary_risk_quality.txt"), "w") as f:
        f.write("\n".join(lines))

def main():
    stats, returns = load()

    ranked = make_ranked_quality_table(stats)
    ranked.to_csv(os.path.join(OUTDIR, "risk_quality_summary.csv"), index=False)

    spread_low = make_spread_table(stats, LOW)
    spread_low.to_csv(os.path.join(OUTDIR, "core_vs_low_spreads.csv"), index=False)

    spread_spectral = make_spread_table(stats, SPECTRAL)
    spread_spectral.to_csv(os.path.join(OUTDIR, "core_vs_spectral_spreads.csv"), index=False)

    date_spreads = make_date_level_spreads(returns)
    date_spreads.to_csv(os.path.join(OUTDIR, "date_level_bucket_spreads.csv"), index=False)

    write_report(stats, ranked, spread_low, spread_spectral, date_spreads)

    print("DONE")
    print(f"Open: {os.path.join(OUTDIR, 'summary_risk_quality.txt')}")
    print(f"Open: {os.path.join(OUTDIR, 'risk_quality_summary.csv')}")
    print(f"Open: {os.path.join(OUTDIR, 'date_level_bucket_spreads.csv')}")

if __name__ == "__main__":
    main()
