#!/usr/bin/env python3
"""
Add SPY Benchmark to SCI Amplitude Portfolio Backtest

Reads:
  results_forward_amplitude/portfolio_equity_curves.csv
  results_forward_amplitude/portfolio_backtest_summary.csv
  cache_prices/SPY.csv

Creates:
  results_forward_amplitude/spy_benchmark_summary.csv
  results_forward_amplitude/portfolio_summary_with_spy.csv
  results_forward_amplitude/equity_curve_core_vs_spy.png
  results_forward_amplitude/summary_spy_benchmark.txt

Purpose:
  Compare AMPLITUDE_CORE_TOP10 directly against SPY over the same rebalance dates.
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
EQUITY_FILE = os.path.join(OUTDIR, "portfolio_equity_curves.csv")
PORTFOLIO_SUMMARY_FILE = os.path.join(OUTDIR, "portfolio_backtest_summary.csv")
SPY_CACHE = os.path.join("cache_prices", "SPY.csv")

CORE = "AMPLITUDE_CORE_TOP10"
UNIVERSE = "EQUAL_WEIGHT_UNIVERSE"
SPY = "SPY_BUY_HOLD_MATCHED"

def max_drawdown(equity):
    equity = pd.Series(equity).dropna()
    if equity.empty:
        return np.nan
    peak = equity.cummax()
    dd = equity / peak - 1.0
    return float(dd.min())

def annualized_return(total_return, n_periods, periods_per_year):
    if n_periods <= 0:
        return np.nan
    return float((1.0 + total_return) ** (periods_per_year / n_periods) - 1.0)

def summarize_return_series(name, rets, equity):
    rets = pd.Series(rets).dropna()
    equity = pd.Series(equity).dropna()

    periods_per_year = 252 / 21

    total_return = float(equity.iloc[-1] - 1.0)
    ann_ret = annualized_return(total_return, len(rets), periods_per_year)
    ann_vol = float(rets.std(ddof=1) * np.sqrt(periods_per_year))
    sharpe = ann_ret / ann_vol if ann_vol and np.isfinite(ann_vol) else np.nan

    return {
        "portfolio": name,
        "n_rebalances": len(rets),
        "total_return": total_return,
        "annualized_return": ann_ret,
        "annualized_volatility": ann_vol,
        "sharpe_like": sharpe,
        "mean_21d_return": float(rets.mean()),
        "median_21d_return": float(rets.median()),
        "std_21d_return": float(rets.std(ddof=1)),
        "win_rate": float((rets > 0).mean()),
        "max_drawdown": max_drawdown(equity),
        "best_21d": float(rets.max()),
        "worst_21d": float(rets.min()),
    }

def fmt_pct(x):
    return "NA" if pd.isna(x) else f"{100*x:.2f}%"

def fmt_num(x):
    return "NA" if pd.isna(x) else f"{x:.3f}"

def main():
    if not os.path.exists(EQUITY_FILE):
        raise FileNotFoundError(f"Missing {EQUITY_FILE}")
    if not os.path.exists(PORTFOLIO_SUMMARY_FILE):
        raise FileNotFoundError(f"Missing {PORTFOLIO_SUMMARY_FILE}")
    if not os.path.exists(SPY_CACHE):
        raise FileNotFoundError(f"Missing {SPY_CACHE}. Make sure SPY was downloaded in cache_prices.")

    equity = pd.read_csv(EQUITY_FILE, index_col=0, parse_dates=True)
    summary = pd.read_csv(PORTFOLIO_SUMMARY_FILE)

    spy_df = pd.read_csv(SPY_CACHE, parse_dates=["Date"])
    spy = pd.to_numeric(spy_df["Close"], errors="coerce")
    spy.index = pd.to_datetime(spy_df["Date"], errors="coerce")
    spy = spy.dropna().sort_index()

    dates = equity.index

    spy_rets = []
    spy_dates = []

    for d in dates:
        idx = spy.index.searchsorted(d)
        if idx >= len(spy):
            continue
        fwd_idx = idx + 21
        if fwd_idx >= len(spy):
            continue

        p0 = spy.iloc[idx]
        p1 = spy.iloc[fwd_idx]
        if p0 > 0 and np.isfinite(p0) and np.isfinite(p1):
            spy_rets.append(float(p1 / p0 - 1.0))
            spy_dates.append(d)

    spy_rets = pd.Series(spy_rets, index=pd.to_datetime(spy_dates), name=SPY)
    spy_equity = (1.0 + spy_rets).cumprod()

    spy_summary = pd.DataFrame([summarize_return_series(SPY, spy_rets, spy_equity)])
    spy_summary.to_csv(os.path.join(OUTDIR, "spy_benchmark_summary.csv"), index=False)

    combined = pd.concat([summary, spy_summary], ignore_index=True)
    combined = combined.sort_values("sharpe_like", ascending=False)
    combined.to_csv(os.path.join(OUTDIR, "portfolio_summary_with_spy.csv"), index=False)

    # Build CORE vs SPY matched returns.
    core_equity = equity[CORE].dropna()
    core_rets = core_equity.pct_change().dropna()

    matched = pd.DataFrame({
        CORE: core_rets,
        SPY: spy_rets,
    }).dropna()

    spread = matched[CORE] - matched[SPY]
    spread_equity = (1.0 + spread).cumprod()

    spread_summary = pd.DataFrame([summarize_return_series(
        f"{CORE}_minus_{SPY}",
        spread,
        spread_equity
    )])
    spread_summary.to_csv(os.path.join(OUTDIR, "core_minus_spy_spread_summary.csv"), index=False)

    # Plot CORE vs SPY
    if HAS_MPL:
        plot_df = pd.DataFrame({
            CORE: core_equity,
            SPY: spy_equity,
        }).dropna()

        plt.figure(figsize=(11, 6))
        plt.plot(plot_df.index, plot_df[CORE], label=CORE)
        plt.plot(plot_df.index, plot_df[SPY], label=SPY)
        plt.title("SCI AMPLITUDE_CORE_TOP10 vs SPY")
        plt.xlabel("Date")
        plt.ylabel("Growth of $1")
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(OUTDIR, "equity_curve_core_vs_spy.png"), dpi=200)
        plt.close()

    # Pull rows for report
    core = combined[combined["portfolio"] == CORE].iloc[0]
    spy_row = combined[combined["portfolio"] == SPY].iloc[0]
    universe = combined[combined["portfolio"] == UNIVERSE].iloc[0] if UNIVERSE in combined["portfolio"].values else None
    spread_row = spread_summary.iloc[0]

    lines = []
    lines.append("=" * 80)
    lines.append("SCI AMPLITUDE CORE vs SPY BENCHMARK")
    lines.append("=" * 80)
    lines.append("")
    lines.append("Method:")
    lines.append("- Uses same rebalance dates as the amplitude bucket portfolio test.")
    lines.append("- SPY is measured as buy-and-hold style 21-trading-day forward returns over the same dates.")
    lines.append("- CORE_TOP10 remains equal-weighted by the top decile of SCI amplitude gap.")
    lines.append("")
    lines.append("Combined portfolio summary ranked by Sharpe-like:")
    lines.append(combined.to_string(index=False))
    lines.append("")
    lines.append("=" * 80)
    lines.append("CORE_TOP10 vs SPY")
    lines.append("=" * 80)
    lines.append(f"CORE total return: {fmt_pct(core['total_return'])}")
    lines.append(f"SPY total return: {fmt_pct(spy_row['total_return'])}")
    lines.append(f"Total return difference: {fmt_pct(core['total_return'] - spy_row['total_return'])}")
    lines.append("")
    lines.append(f"CORE annualized return: {fmt_pct(core['annualized_return'])}")
    lines.append(f"SPY annualized return: {fmt_pct(spy_row['annualized_return'])}")
    lines.append(f"Annualized return difference: {fmt_pct(core['annualized_return'] - spy_row['annualized_return'])}")
    lines.append("")
    lines.append(f"CORE annualized volatility: {fmt_pct(core['annualized_volatility'])}")
    lines.append(f"SPY annualized volatility: {fmt_pct(spy_row['annualized_volatility'])}")
    lines.append(f"Volatility difference: {fmt_pct(core['annualized_volatility'] - spy_row['annualized_volatility'])}")
    lines.append("")
    lines.append(f"CORE Sharpe-like: {fmt_num(core['sharpe_like'])}")
    lines.append(f"SPY Sharpe-like: {fmt_num(spy_row['sharpe_like'])}")
    lines.append(f"Sharpe-like difference: {fmt_num(core['sharpe_like'] - spy_row['sharpe_like'])}")
    lines.append("")
    lines.append(f"CORE max drawdown: {fmt_pct(core['max_drawdown'])}")
    lines.append(f"SPY max drawdown: {fmt_pct(spy_row['max_drawdown'])}")
    lines.append(f"Drawdown improvement: {fmt_pct(abs(spy_row['max_drawdown']) - abs(core['max_drawdown']))}")
    lines.append("")
    lines.append("=" * 80)
    lines.append("CORE minus SPY spread summary")
    lines.append("=" * 80)
    lines.append(spread_summary.to_string(index=False))
    lines.append("")
    lines.append("Plain-English read:")
    lines.append("- If CORE beats SPY on total return and Sharpe-like, SCI is adding selection value beyond the market benchmark.")
    lines.append("- If CORE has higher volatility than SPY but much higher return, the signal may be growth/risk-on tilted.")
    lines.append("- If CORE-minus-SPY spread is modest, the best framing remains portfolio-quality selection, not standalone market-neutral alpha.")
    lines.append("- This still needs survivorship-bias cleanup and live forward testing.")

    report_path = os.path.join(OUTDIR, "summary_spy_benchmark.txt")
    with open(report_path, "w") as f:
        f.write("\n".join(lines))

    print("DONE")
    print(f"Open: {report_path}")
    print(f"Open: {os.path.join(OUTDIR, 'portfolio_summary_with_spy.csv')}")
    print(f"Open: {os.path.join(OUTDIR, 'equity_curve_core_vs_spy.png')}")
    print(f"Open: {os.path.join(OUTDIR, 'core_minus_spy_spread_summary.csv')}")

if __name__ == "__main__":
    main()
