#!/usr/bin/env python3
"""
Amplitude Bucket Portfolio Backtest

Reads:
  results_forward_amplitude/forward_amplitude_returns.csv

Creates:
  results_forward_amplitude/portfolio_backtest_summary.csv
  results_forward_amplitude/portfolio_equity_curves.csv
  results_forward_amplitude/summary_portfolio_backtest.txt

Purpose:
  Convert the amplitude buckets into actual equal-weight portfolio snapshots.

At each rebalance date:
  - Equal-weight each bucket
  - Use fwd_21 as the next-period return
  - Compound returns over time
  - Compare bucket equity curves
"""

import os
import numpy as np
import pandas as pd

OUTDIR = "results_forward_amplitude"
INFILE = os.path.join(OUTDIR, "forward_amplitude_returns.csv")

CORE = "AMPLITUDE_CORE_TOP10"
MID = "AMPLITUDE_MID_70_90"
TACTICAL = "AMPLITUDE_TACTICAL_40_70"
LOW = "AMPLITUDE_LOW_BOTTOM40"
SPECTRAL = "SPECTRAL_DOMINANT"

BUCKET_ORDER = [CORE, MID, TACTICAL, LOW, SPECTRAL]

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

def main():
    if not os.path.exists(INFILE):
        raise FileNotFoundError(f"Missing {INFILE}")

    df = pd.read_csv(INFILE, parse_dates=["date"])

    if "fwd_21" not in df.columns:
        raise ValueError("Need fwd_21 column in forward_amplitude_returns.csv")

    df = df[df["fwd_21"].notna()].copy()

    # Equal-weight bucket return per rebalance date.
    bucket_returns = (
        df.groupby(["date", "amplitude_bucket_v2"])["fwd_21"]
        .mean()
        .reset_index()
        .pivot(index="date", columns="amplitude_bucket_v2", values="fwd_21")
        .sort_index()
    )

    # Keep known buckets if present.
    cols = [c for c in BUCKET_ORDER if c in bucket_returns.columns]
    bucket_returns = bucket_returns[cols]

    # Also build a whole-universe equal-weight benchmark from available tickers.
    universe_return = df.groupby("date")["fwd_21"].mean().rename("EQUAL_WEIGHT_UNIVERSE")
    bucket_returns = bucket_returns.join(universe_return, how="left")

    # Compound each bucket.
    equity = (1.0 + bucket_returns.fillna(0.0)).cumprod()
    equity.to_csv(os.path.join(OUTDIR, "portfolio_equity_curves.csv"))

    rows = []
    periods_per_year = 252 / 21

    for col in bucket_returns.columns:
        rets = bucket_returns[col].dropna()
        eq = equity[col].loc[rets.index]

        if len(rets) == 0:
            continue

        total_return = float(eq.iloc[-1] - 1.0)
        ann_ret = annualized_return(total_return, len(rets), periods_per_year)
        ann_vol = float(rets.std(ddof=1) * np.sqrt(periods_per_year))
        sharpe = ann_ret / ann_vol if ann_vol and np.isfinite(ann_vol) else np.nan

        rows.append({
            "portfolio": col,
            "n_rebalances": len(rets),
            "total_return": total_return,
            "annualized_return": ann_ret,
            "annualized_volatility": ann_vol,
            "sharpe_like": sharpe,
            "mean_21d_return": float(rets.mean()),
            "median_21d_return": float(rets.median()),
            "std_21d_return": float(rets.std(ddof=1)),
            "win_rate": float((rets > 0).mean()),
            "max_drawdown": max_drawdown(eq),
            "best_21d": float(rets.max()),
            "worst_21d": float(rets.min()),
        })

    summary = pd.DataFrame(rows).sort_values("sharpe_like", ascending=False)
    summary.to_csv(os.path.join(OUTDIR, "portfolio_backtest_summary.csv"), index=False)

    # Spread equity curves
    spreads = pd.DataFrame(index=bucket_returns.index)

    if CORE in bucket_returns.columns:
        for other in [MID, TACTICAL, LOW, SPECTRAL, "EQUAL_WEIGHT_UNIVERSE"]:
            if other in bucket_returns.columns:
                spreads[f"{CORE}_minus_{other}"] = bucket_returns[CORE] - bucket_returns[other]

    spread_equity = (1.0 + spreads.fillna(0.0)).cumprod()
    spread_equity.to_csv(os.path.join(OUTDIR, "portfolio_spread_equity_curves.csv"))

    spread_rows = []
    for col in spreads.columns:
        rets = spreads[col].dropna()
        eq = spread_equity[col].loc[rets.index]
        if len(rets) == 0:
            continue

        total_return = float(eq.iloc[-1] - 1.0)
        ann_ret = annualized_return(total_return, len(rets), periods_per_year)
        ann_vol = float(rets.std(ddof=1) * np.sqrt(periods_per_year))
        sharpe = ann_ret / ann_vol if ann_vol and np.isfinite(ann_vol) else np.nan

        spread_rows.append({
            "spread_portfolio": col,
            "n_rebalances": len(rets),
            "total_return": total_return,
            "annualized_return": ann_ret,
            "annualized_volatility": ann_vol,
            "sharpe_like": sharpe,
            "mean_21d_spread": float(rets.mean()),
            "median_21d_spread": float(rets.median()),
            "win_rate": float((rets > 0).mean()),
            "max_drawdown": max_drawdown(eq),
            "best_21d_spread": float(rets.max()),
            "worst_21d_spread": float(rets.min()),
        })

    spread_summary = pd.DataFrame(spread_rows).sort_values("sharpe_like", ascending=False)
    spread_summary.to_csv(os.path.join(OUTDIR, "portfolio_spread_summary.csv"), index=False)

    # Write plain-English report.
    lines = []
    lines.append("=" * 80)
    lines.append("AMPLITUDE BUCKET PORTFOLIO BACKTEST")
    lines.append("=" * 80)
    lines.append("")
    lines.append("Method:")
    lines.append("- Uses fwd_21 returns as the next rebalance-period return.")
    lines.append("- Equal-weights every ticker inside each bucket at each rebalance date.")
    lines.append("- Compounds bucket returns over time.")
    lines.append("- This is still historical and approximate, but closer to an investable test.")
    lines.append("")
    lines.append("Portfolio summary ranked by Sharpe-like:")
    lines.append(summary.to_string(index=False))
    lines.append("")
    lines.append("=" * 80)
    lines.append("CORE spread portfolio summary")
    lines.append("=" * 80)
    if len(spread_summary):
        lines.append(spread_summary.to_string(index=False))
    else:
        lines.append("No spread summary created.")
    lines.append("")
    lines.append("Plain-English read:")
    lines.append("- If CORE has higher Sharpe-like than LOW, MID, TACTICAL, and universe, then amplitude coherence works as a quality filter.")
    lines.append("- If CORE total return is not the highest but max drawdown/volatility is better, the signal is risk-quality rather than raw alpha.")
    lines.append("- If CORE-minus-universe has positive return and tolerable drawdown, it may be closer to tradable alpha.")
    lines.append("- If CORE-minus-LOW is weak, LOW contains occasional explosive winners and should not simply be shorted.")

    report_path = os.path.join(OUTDIR, "summary_portfolio_backtest.txt")
    with open(report_path, "w") as f:
        f.write("\n".join(lines))

    print("DONE")
    print(f"Open: {report_path}")
    print(f"Open: {os.path.join(OUTDIR, 'portfolio_backtest_summary.csv')}")
    print(f"Open: {os.path.join(OUTDIR, 'portfolio_spread_summary.csv')}")
    print(f"Open: {os.path.join(OUTDIR, 'portfolio_equity_curves.csv')}")

if __name__ == "__main__":
    main()
