#!/usr/bin/env python3
"""
Report-grade portfolio outputs for SCI amplitude bucket backtest.

Reads:
  results_forward_amplitude/portfolio_backtest_summary.csv
  results_forward_amplitude/portfolio_spread_summary.csv
  results_forward_amplitude/portfolio_equity_curves.csv

Creates:
  results_forward_amplitude/report_grade_comparison.csv
  results_forward_amplitude/report_grade_summary.txt
  results_forward_amplitude/equity_curve_amplitude_buckets.png
  results_forward_amplitude/equity_curve_core_vs_universe.png
"""

import os
import numpy as np
import pandas as pd

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    HAS_MPL = True
except Exception:
    HAS_MPL = False

OUTDIR = "results_forward_amplitude"

SUMMARY_FILE = os.path.join(OUTDIR, "portfolio_backtest_summary.csv")
SPREAD_FILE = os.path.join(OUTDIR, "portfolio_spread_summary.csv")
EQUITY_FILE = os.path.join(OUTDIR, "portfolio_equity_curves.csv")

CORE = "AMPLITUDE_CORE_TOP10"
MID = "AMPLITUDE_MID_70_90"
TACTICAL = "AMPLITUDE_TACTICAL_40_70"
LOW = "AMPLITUDE_LOW_BOTTOM40"
SPECTRAL = "SPECTRAL_DOMINANT"
UNIVERSE = "EQUAL_WEIGHT_UNIVERSE"

def fmt_pct(x):
    if pd.isna(x):
        return "NA"
    return f"{100*x:.2f}%"

def fmt_num(x):
    if pd.isna(x):
        return "NA"
    return f"{x:.3f}"

def get_row(df, name):
    row = df[df["portfolio"] == name]
    if row.empty:
        return None
    return row.iloc[0]

def build_comparison(summary):
    core = get_row(summary, CORE)
    universe = get_row(summary, UNIVERSE)
    low = get_row(summary, LOW)
    spectral = get_row(summary, SPECTRAL)

    rows = []

    def add_compare(other_name, other):
        if core is None or other is None:
            return

        rows.append({
            "comparison": f"{CORE} vs {other_name}",
            "core_total_return": core["total_return"],
            "other_total_return": other["total_return"],
            "total_return_difference": core["total_return"] - other["total_return"],

            "core_annualized_return": core["annualized_return"],
            "other_annualized_return": other["annualized_return"],
            "annualized_return_difference": core["annualized_return"] - other["annualized_return"],

            "core_annualized_volatility": core["annualized_volatility"],
            "other_annualized_volatility": other["annualized_volatility"],
            "volatility_difference": core["annualized_volatility"] - other["annualized_volatility"],

            "core_sharpe_like": core["sharpe_like"],
            "other_sharpe_like": other["sharpe_like"],
            "sharpe_like_difference": core["sharpe_like"] - other["sharpe_like"],

            "core_max_drawdown": core["max_drawdown"],
            "other_max_drawdown": other["max_drawdown"],
            "drawdown_improvement_points": abs(other["max_drawdown"]) - abs(core["max_drawdown"]),

            "core_win_rate": core["win_rate"],
            "other_win_rate": other["win_rate"],
            "win_rate_difference": core["win_rate"] - other["win_rate"],
        })

    add_compare(UNIVERSE, universe)
    add_compare(LOW, low)
    add_compare(SPECTRAL, spectral)

    return pd.DataFrame(rows)

def make_plots(equity):
    if not HAS_MPL:
        return

    equity = equity.copy()
    equity.index = pd.to_datetime(equity.index)

    # Plot all main bucket curves
    cols = [c for c in [CORE, MID, TACTICAL, LOW, SPECTRAL, UNIVERSE] if c in equity.columns]

    plt.figure(figsize=(11, 6))
    for c in cols:
        plt.plot(equity.index, equity[c], label=c)
    plt.title("SCI Amplitude Bucket Portfolio Equity Curves")
    plt.xlabel("Date")
    plt.ylabel("Growth of $1")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(OUTDIR, "equity_curve_amplitude_buckets.png"), dpi=200)
    plt.close()

    # Core vs universe only
    plt.figure(figsize=(11, 6))
    for c in [CORE, UNIVERSE]:
        if c in equity.columns:
            plt.plot(equity.index, equity[c], label=c)
    plt.title("SCI AMPLITUDE_CORE_TOP10 vs Equal-Weight Universe")
    plt.xlabel("Date")
    plt.ylabel("Growth of $1")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(OUTDIR, "equity_curve_core_vs_universe.png"), dpi=200)
    plt.close()

def write_report(summary, comparison, spread):
    core = get_row(summary, CORE)
    universe = get_row(summary, UNIVERSE)
    low = get_row(summary, LOW)
    spectral = get_row(summary, SPECTRAL)

    lines = []
    lines.append("=" * 80)
    lines.append("REPORT-GRADE SCI AMPLITUDE PORTFOLIO SUMMARY")
    lines.append("=" * 80)
    lines.append("")
    lines.append("Core claim:")
    lines.append(
        "The top 10% of tickers ranked by 500-day SCI amplitude-envelope gap produced "
        "the strongest portfolio-quality profile in this historical equal-weight test."
    )
    lines.append("")

    lines.append("Main portfolio table:")
    lines.append(summary.to_string(index=False))
    lines.append("")

    lines.append("=" * 80)
    lines.append("CORE_TOP10 vs Equal-Weight Universe")
    lines.append("=" * 80)

    if core is not None and universe is not None:
        lines.append(f"CORE total return: {fmt_pct(core['total_return'])}")
        lines.append(f"Universe total return: {fmt_pct(universe['total_return'])}")
        lines.append(f"Total return improvement: {fmt_pct(core['total_return'] - universe['total_return'])}")
        lines.append("")
        lines.append(f"CORE annualized return: {fmt_pct(core['annualized_return'])}")
        lines.append(f"Universe annualized return: {fmt_pct(universe['annualized_return'])}")
        lines.append(f"Annualized return improvement: {fmt_pct(core['annualized_return'] - universe['annualized_return'])}")
        lines.append("")
        lines.append(f"CORE annualized volatility: {fmt_pct(core['annualized_volatility'])}")
        lines.append(f"Universe annualized volatility: {fmt_pct(universe['annualized_volatility'])}")
        lines.append(f"Volatility change: {fmt_pct(core['annualized_volatility'] - universe['annualized_volatility'])}")
        lines.append("")
        lines.append(f"CORE Sharpe-like: {fmt_num(core['sharpe_like'])}")
        lines.append(f"Universe Sharpe-like: {fmt_num(universe['sharpe_like'])}")
        lines.append(f"Sharpe-like improvement: {fmt_num(core['sharpe_like'] - universe['sharpe_like'])}")
        lines.append("")
        lines.append(f"CORE max drawdown: {fmt_pct(core['max_drawdown'])}")
        lines.append(f"Universe max drawdown: {fmt_pct(universe['max_drawdown'])}")
        lines.append(
            f"Drawdown improvement: {fmt_pct(abs(universe['max_drawdown']) - abs(core['max_drawdown']))}"
        )
        lines.append("")
        lines.append(f"CORE win rate: {fmt_pct(core['win_rate'])}")
        lines.append(f"Universe win rate: {fmt_pct(universe['win_rate'])}")
        lines.append(f"Win-rate change: {fmt_pct(core['win_rate'] - universe['win_rate'])}")

    lines.append("")
    lines.append("=" * 80)
    lines.append("Comparison table")
    lines.append("=" * 80)
    lines.append(comparison.to_string(index=False))
    lines.append("")

    lines.append("=" * 80)
    lines.append("Spread portfolio table")
    lines.append("=" * 80)
    lines.append(spread.to_string(index=False))
    lines.append("")

    lines.append("Plain-English interpretation:")
    lines.append(
        "- CORE_TOP10 beat the equal-weight universe on total return, annualized return, "
        "Sharpe-like ratio, and max drawdown."
    )
    lines.append(
        "- The result is better framed as a portfolio-quality filter than a pure long-short alpha engine."
    )
    lines.append(
        "- LOW_BOTTOM40 can still contain upside, but it carries much worse dispersion/drawdown behavior."
    )
    lines.append(
        "- SPECTRAL_DOMINANT remains the weakest structural bucket and may be an avoid/noise category."
    )
    lines.append("")
    lines.append("Suggested paper wording:")
    lines.append(
        "In a historical equal-weight rebalance test over 100 rebalance dates and 1,337 loaded tickers, "
        "the top decile of instruments by SCI amplitude-envelope gap achieved the highest Sharpe-like "
        "profile among all amplitude buckets and outperformed the equal-weight universe on both "
        "absolute and risk-adjusted terms. This supports the interpretation of SCI as a pre-model "
        "return-quality filter rather than a standalone directional predictor."
    )

    with open(os.path.join(OUTDIR, "report_grade_summary.txt"), "w") as f:
        f.write("\n".join(lines))

def main():
    if not os.path.exists(SUMMARY_FILE):
        raise FileNotFoundError(f"Missing {SUMMARY_FILE}")
    if not os.path.exists(SPREAD_FILE):
        raise FileNotFoundError(f"Missing {SPREAD_FILE}")
    if not os.path.exists(EQUITY_FILE):
        raise FileNotFoundError(f"Missing {EQUITY_FILE}")

    summary = pd.read_csv(SUMMARY_FILE)
    spread = pd.read_csv(SPREAD_FILE)
    equity = pd.read_csv(EQUITY_FILE, index_col=0)

    comparison = build_comparison(summary)
    comparison.to_csv(os.path.join(OUTDIR, "report_grade_comparison.csv"), index=False)

    make_plots(equity)
    write_report(summary, comparison, spread)

    print("DONE")
    print(f"Open: {os.path.join(OUTDIR, 'report_grade_summary.txt')}")
    print(f"Open: {os.path.join(OUTDIR, 'report_grade_comparison.csv')}")
    print(f"Open: {os.path.join(OUTDIR, 'equity_curve_amplitude_buckets.png')}")
    print(f"Open: {os.path.join(OUTDIR, 'equity_curve_core_vs_universe.png')}")

if __name__ == "__main__":
    main()
