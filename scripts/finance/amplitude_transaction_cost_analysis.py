#!/usr/bin/env python3
"""
Amplitude Bucket Transaction Cost / Turnover Analysis

Reads:
  results_forward_amplitude/forward_amplitude_returns.csv

Creates:
  results_forward_amplitude/transaction_cost_summary.csv
  results_forward_amplitude/turnover_by_bucket.csv
  results_forward_amplitude/equity_curves_after_costs.csv
  results_forward_amplitude/summary_transaction_costs.txt

Purpose:
  Tests whether the AMPLITUDE_CORE_TOP10 result survives basic transaction costs.

Method:
  - Reconstructs equal-weight bucket holdings at each rebalance date.
  - Calculates turnover as % of portfolio weight changed.
  - Applies round-trip transaction cost assumptions:
      0 bps, 5 bps, 10 bps, 25 bps, 50 bps per traded dollar.
  - Compounds after-cost returns.
"""

import os
import numpy as np
import pandas as pd

OUTDIR = "results_forward_amplitude"
INFILE = os.path.join(OUTDIR, "forward_amplitude_returns.csv")

BUCKETS = [
    "AMPLITUDE_CORE_TOP10",
    "AMPLITUDE_MID_70_90",
    "AMPLITUDE_TACTICAL_40_70",
    "AMPLITUDE_LOW_BOTTOM40",
    "SPECTRAL_DOMINANT",
]

COST_BPS_LIST = [0, 5, 10, 25, 50]

def max_drawdown(equity):
    equity = pd.Series(equity).dropna()
    if equity.empty:
        return np.nan
    peak = equity.cummax()
    dd = equity / peak - 1
    return float(dd.min())

def annualized_return(total_return, n_periods, periods_per_year):
    if n_periods <= 0:
        return np.nan
    return float((1 + total_return) ** (periods_per_year / n_periods) - 1)

def get_weights(date_df, bucket):
    sub = date_df[date_df["amplitude_bucket_v2"] == bucket].copy()
    if sub.empty:
        return {}
    tickers = sub["ticker"].astype(str).tolist()
    w = 1.0 / len(tickers)
    return {t: w for t in tickers}

def turnover(prev_w, new_w):
    """
    One-way turnover = 0.5 * sum(abs(new_weight - old_weight)).
    If portfolio fully changes names, turnover ≈ 1.0.
    """
    names = set(prev_w.keys()) | set(new_w.keys())
    diff = sum(abs(new_w.get(t, 0.0) - prev_w.get(t, 0.0)) for t in names)
    return 0.5 * diff

def weighted_bucket_return(date_df, bucket):
    sub = date_df[(date_df["amplitude_bucket_v2"] == bucket) & (date_df["fwd_21"].notna())]
    if sub.empty:
        return np.nan
    return float(sub["fwd_21"].mean())

def main():
    if not os.path.exists(INFILE):
        raise FileNotFoundError(f"Missing {INFILE}")

    df = pd.read_csv(INFILE, parse_dates=["date"])
    df = df[df["fwd_21"].notna()].copy()
    df = df.sort_values(["date", "ticker"])

    dates = sorted(df["date"].unique())
    periods_per_year = 252 / 21

    turnover_rows = []
    return_rows = []

    prev_weights = {b: {} for b in BUCKETS}

    for date in dates:
        ddf = df[df["date"] == date]

        for bucket in BUCKETS:
            new_w = get_weights(ddf, bucket)
            raw_ret = weighted_bucket_return(ddf, bucket)

            if not new_w or not np.isfinite(raw_ret):
                continue

            to = turnover(prev_weights[bucket], new_w)
            prev_weights[bucket] = new_w

            turnover_rows.append({
                "date": date,
                "bucket": bucket,
                "n_holdings": len(new_w),
                "turnover": to,
                "raw_21d_return": raw_ret,
            })

            for cost_bps in COST_BPS_LIST:
                cost = to * (cost_bps / 10000.0)
                net_ret = raw_ret - cost

                return_rows.append({
                    "date": date,
                    "bucket": bucket,
                    "cost_bps": cost_bps,
                    "turnover": to,
                    "raw_21d_return": raw_ret,
                    "cost_drag": cost,
                    "net_21d_return": net_ret,
                })

    turnover_df = pd.DataFrame(turnover_rows)
    returns_df = pd.DataFrame(return_rows)

    turnover_df.to_csv(os.path.join(OUTDIR, "turnover_by_bucket.csv"), index=False)
    returns_df.to_csv(os.path.join(OUTDIR, "returns_after_transaction_costs.csv"), index=False)

    # Equity curves after costs
    equity_rows = []
    summary_rows = []

    for bucket in BUCKETS:
        for cost_bps in COST_BPS_LIST:
            sub = returns_df[
                (returns_df["bucket"] == bucket) &
                (returns_df["cost_bps"] == cost_bps)
            ].sort_values("date")

            if sub.empty:
                continue

            eq = (1 + sub["net_21d_return"]).cumprod()
            total_return = float(eq.iloc[-1] - 1)
            ann_ret = annualized_return(total_return, len(sub), periods_per_year)
            ann_vol = float(sub["net_21d_return"].std(ddof=1) * np.sqrt(periods_per_year))
            sharpe = ann_ret / ann_vol if ann_vol > 0 else np.nan

            summary_rows.append({
                "bucket": bucket,
                "cost_bps": cost_bps,
                "n_rebalances": len(sub),
                "avg_turnover": sub["turnover"].mean(),
                "median_turnover": sub["turnover"].median(),
                "avg_cost_drag_per_rebalance": sub["cost_drag"].mean(),
                "total_return_after_cost": total_return,
                "annualized_return_after_cost": ann_ret,
                "annualized_volatility_after_cost": ann_vol,
                "sharpe_like_after_cost": sharpe,
                "win_rate_after_cost": float((sub["net_21d_return"] > 0).mean()),
                "max_drawdown_after_cost": max_drawdown(eq),
            })

            for d, val in zip(sub["date"], eq):
                equity_rows.append({
                    "date": d,
                    "bucket": bucket,
                    "cost_bps": cost_bps,
                    "equity": val,
                })

    summary = pd.DataFrame(summary_rows)
    equity = pd.DataFrame(equity_rows)

    summary.to_csv(os.path.join(OUTDIR, "transaction_cost_summary.csv"), index=False)
    equity.to_csv(os.path.join(OUTDIR, "equity_curves_after_costs.csv"), index=False)

    # CORE comparison by cost
    core_summary = summary[summary["bucket"] == "AMPLITUDE_CORE_TOP10"].copy()

    lines = []
    lines.append("=" * 80)
    lines.append("SCI AMPLITUDE TRANSACTION COST / TURNOVER ANALYSIS")
    lines.append("=" * 80)
    lines.append("")
    lines.append("Method:")
    lines.append("- Reconstructed equal-weight holdings for each amplitude bucket at each rebalance date.")
    lines.append("- Estimated one-way turnover as 0.5 * sum(abs(new weight - old weight)).")
    lines.append("- Applied transaction costs from 0 to 50 bps per traded dollar.")
    lines.append("")
    lines.append("Average turnover by bucket:")
    avg_turnover = turnover_df.groupby("bucket").agg(
        n_rebalances=("date", "count"),
        avg_holdings=("n_holdings", "mean"),
        avg_turnover=("turnover", "mean"),
        median_turnover=("turnover", "median"),
        avg_raw_21d_return=("raw_21d_return", "mean"),
    ).reset_index()
    lines.append(avg_turnover.to_string(index=False))
    lines.append("")
    lines.append("=" * 80)
    lines.append("Transaction cost summary:")
    lines.append("=" * 80)
    lines.append(summary.to_string(index=False))
    lines.append("")
    lines.append("=" * 80)
    lines.append("CORE_TOP10 after-cost sensitivity:")
    lines.append("=" * 80)
    lines.append(core_summary.to_string(index=False))
    lines.append("")
    lines.append("Plain-English read:")
    lines.append("- If CORE_TOP10 remains above the universe/other buckets at 10–25 bps, the result is more robust.")
    lines.append("- If performance collapses by 25–50 bps, the strategy is turnover-sensitive.")
    lines.append("- Since this uses 21-trading-day rebalances, transaction cost assumptions matter.")
    lines.append("- This still excludes taxes, slippage, borrow constraints, and survivorship-bias cleanup.")

    report_path = os.path.join(OUTDIR, "summary_transaction_costs.txt")
    with open(report_path, "w") as f:
        f.write("\n".join(lines))

    print("DONE")
    print(f"Open: {report_path}")
    print(f"Open: {os.path.join(OUTDIR, 'transaction_cost_summary.csv')}")
    print(f"Open: {os.path.join(OUTDIR, 'turnover_by_bucket.csv')}")

if __name__ == "__main__":
    main()
