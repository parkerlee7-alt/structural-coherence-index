#!/usr/bin/env python3
"""
SCI Amplitude Factor-Control Analysis

Purpose:
  Tests whether SCI amplitude bucket performance is just momentum, volatility,
  drawdown, or recent return exposure.

Reads:
  results_forward_amplitude/forward_amplitude_returns.csv
  cache_prices/*.csv

Creates:
  results_forward_amplitude/factor_control_summary.txt
  results_forward_amplitude/factor_control_observations.csv
  results_forward_amplitude/factor_control_regression.csv
  results_forward_amplitude/within_momentum_bucket_stats.csv
  results_forward_amplitude/within_volatility_bucket_stats.csv
  results_forward_amplitude/core_vs_noncore_factor_profile.csv

Main questions:
  1. Is CORE_TOP10 just high momentum?
  2. Is CORE_TOP10 just high volatility?
  3. Does gap_percentile still relate to forward returns after controls?
  4. Does CORE outperform inside similar momentum groups?
  5. Does CORE outperform inside similar volatility groups?
"""

import os
import numpy as np
import pandas as pd

try:
    import statsmodels.api as sm
    HAS_STATSMODELS = True
except Exception:
    HAS_STATSMODELS = False

OUTDIR = "results_forward_amplitude"
RETURNS_FILE = os.path.join(OUTDIR, "forward_amplitude_returns.csv")
CACHE_DIR = "cache_prices"

CORE = "AMPLITUDE_CORE_TOP10"

def load_price_series(ticker):
    safe = ticker.replace("/", "_").replace(":", "_")
    path = os.path.join(CACHE_DIR, safe + ".csv")
    if not os.path.exists(path):
        return None

    try:
        df = pd.read_csv(path, parse_dates=["Date"])
        s = pd.to_numeric(df["Close"], errors="coerce")
        s.index = pd.to_datetime(df["Date"], errors="coerce")
        s = s.dropna().sort_index()
        return s
    except Exception:
        return None

def trailing_features(prices, date):
    """
    Features calculated strictly before the rebalance date.
    """
    idx = prices.index.searchsorted(date)

    if idx < 253:
        return None

    hist = prices.iloc[:idx].copy()

    if len(hist) < 253:
        return None

    p_now = hist.iloc[-1]
    p_21 = hist.iloc[-22] if len(hist) >= 22 else np.nan
    p_63 = hist.iloc[-64] if len(hist) >= 64 else np.nan
    p_126 = hist.iloc[-127] if len(hist) >= 127 else np.nan
    p_252 = hist.iloc[-253] if len(hist) >= 253 else np.nan

    r = np.log(hist).diff().dropna()
    r252 = r.iloc[-252:] if len(r) >= 252 else r

    roll_max = hist.iloc[-252:].cummax()
    dd = hist.iloc[-252:] / roll_max - 1.0

    out = {
        "mom_21": p_now / p_21 - 1.0 if p_21 > 0 else np.nan,
        "mom_63": p_now / p_63 - 1.0 if p_63 > 0 else np.nan,
        "mom_126": p_now / p_126 - 1.0 if p_126 > 0 else np.nan,
        "mom_252": p_now / p_252 - 1.0 if p_252 > 0 else np.nan,
        "vol_252": float(r252.std(ddof=1) * np.sqrt(252)) if len(r252) > 20 else np.nan,
        "drawdown_252": float(dd.min()) if len(dd) > 20 else np.nan,
        "realized_skew_252": float(r252.skew()) if len(r252) > 20 else np.nan,
        "realized_kurt_252": float(r252.kurt()) if len(r252) > 20 else np.nan,
    }

    return out

def assign_date_percentiles(df, cols):
    df = df.copy()
    for c in cols:
        df[c + "_pct"] = df.groupby("date")[c].rank(pct=True)
    return df

def quintile_label(x):
    if not np.isfinite(x):
        return "NO_DATA"
    if x >= 0.80:
        return "Q5_TOP"
    if x >= 0.60:
        return "Q4"
    if x >= 0.40:
        return "Q3"
    if x >= 0.20:
        return "Q2"
    return "Q1_BOTTOM"

def summarize_group(df, group_cols, ret_col):
    rows = []
    for keys, g in df.groupby(group_cols):
        if not isinstance(keys, tuple):
            keys = (keys,)

        rets = g[ret_col].dropna()
        if len(rets) == 0:
            continue

        row = {}
        for k, v in zip(group_cols, keys):
            row[k] = v

        row.update({
            "n": len(rets),
            "mean_return": rets.mean(),
            "median_return": rets.median(),
            "std_return": rets.std(ddof=1),
            "sharpe_like": rets.mean() / (rets.std(ddof=1) + 1e-12),
            "win_rate": (rets > 0).mean(),
            "avg_gap_pct": g["gap_percentile"].mean(),
            "avg_mom_252": g["mom_252"].mean(),
            "avg_vol_252": g["vol_252"].mean(),
        })
        rows.append(row)

    return pd.DataFrame(rows)

def run_regression(df, ret_col):
    if not HAS_STATSMODELS:
        return pd.DataFrame([{
            "forward_horizon": ret_col,
            "note": "statsmodels not installed. Run: pip3 install statsmodels"
        }])

    needed = [
        ret_col,
        "gap_percentile",
        "is_core",
        "mom_63_pct",
        "mom_252_pct",
        "vol_252_pct",
        "drawdown_252_pct",
    ]

    sub = df[needed].replace([np.inf, -np.inf], np.nan).dropna().copy()
    if len(sub) < 100:
        return pd.DataFrame([{
            "forward_horizon": ret_col,
            "note": "not enough observations"
        }])

    y = sub[ret_col]
    X = sub[[
        "gap_percentile",
        "is_core",
        "mom_63_pct",
        "mom_252_pct",
        "vol_252_pct",
        "drawdown_252_pct",
    ]]
    X = sm.add_constant(X)

    model = sm.OLS(y, X).fit(cov_type="HC3")

    rows = []
    for name in model.params.index:
        rows.append({
            "forward_horizon": ret_col,
            "term": name,
            "coef": model.params[name],
            "t_stat": model.tvalues[name],
            "p_value": model.pvalues[name],
            "n_obs": int(model.nobs),
            "r_squared": model.rsquared,
        })

    return pd.DataFrame(rows)

def make_core_profile(df):
    rows = []

    for label, g in [
        ("CORE_TOP10", df[df["is_core"] == 1]),
        ("NON_CORE", df[df["is_core"] == 0]),
    ]:
        rows.append({
            "group": label,
            "n": len(g),
            "avg_gap_percentile": g["gap_percentile"].mean(),
            "avg_mom_21": g["mom_21"].mean(),
            "avg_mom_63": g["mom_63"].mean(),
            "avg_mom_126": g["mom_126"].mean(),
            "avg_mom_252": g["mom_252"].mean(),
            "avg_vol_252": g["vol_252"].mean(),
            "avg_drawdown_252": g["drawdown_252"].mean(),
            "avg_fwd_21": g["fwd_21"].mean(),
            "avg_fwd_63": g["fwd_63"].mean(),
            "avg_fwd_126": g["fwd_126"].mean(),
            "avg_fwd_252": g["fwd_252"].mean(),
            "win_rate_21": (g["fwd_21"] > 0).mean(),
            "win_rate_252": (g["fwd_252"] > 0).mean(),
        })

    return pd.DataFrame(rows)

def main():
    if not os.path.exists(RETURNS_FILE):
        raise FileNotFoundError(f"Missing {RETURNS_FILE}")

    print("=" * 80)
    print("SCI AMPLITUDE FACTOR-CONTROL ANALYSIS")
    print("=" * 80)

    df = pd.read_csv(RETURNS_FILE, parse_dates=["date"])
    df = df.replace([np.inf, -np.inf], np.nan)

    # Use all rows with a valid gap.
    df = df[df["w500_gap"].notna()].copy()
    df["is_core"] = (df["amplitude_bucket_v2"] == CORE).astype(int)

    tickers = sorted(df["ticker"].astype(str).unique())
    price_cache = {}

    rows = []
    total = len(df)

    print(f"Rows to feature: {total}")
    print(f"Unique tickers: {len(tickers)}")

    for i, row in enumerate(df.itertuples(index=False), 1):
        ticker = str(row.ticker)
        date = row.date

        if ticker not in price_cache:
            price_cache[ticker] = load_price_series(ticker)

        prices = price_cache[ticker]
        if prices is None:
            continue

        feats = trailing_features(prices, date)
        if feats is None:
            continue

        base = row._asdict()
        base.update(feats)
        rows.append(base)

        if i % 10000 == 0:
            print(f"  processed {i}/{total}")

    out = pd.DataFrame(rows)
    if out.empty:
        raise SystemExit("No feature rows created.")

    factor_cols = [
        "mom_21", "mom_63", "mom_126", "mom_252",
        "vol_252", "drawdown_252",
        "realized_skew_252", "realized_kurt_252",
    ]

    out = assign_date_percentiles(out, factor_cols)

    out["momentum_252_quintile"] = out["mom_252_pct"].apply(quintile_label)
    out["volatility_252_quintile"] = out["vol_252_pct"].apply(quintile_label)
    out["drawdown_252_quintile"] = out["drawdown_252_pct"].apply(quintile_label)

    out_path = os.path.join(OUTDIR, "factor_control_observations.csv")
    out.to_csv(out_path, index=False)

    # Regression tests
    reg_tables = []
    for ret_col in ["fwd_21", "fwd_63", "fwd_126", "fwd_252"]:
        if ret_col in out.columns:
            reg_tables.append(run_regression(out, ret_col))

    regs = pd.concat(reg_tables, ignore_index=True)
    regs.to_csv(os.path.join(OUTDIR, "factor_control_regression.csv"), index=False)

    # Within-bucket stats
    within_mom = summarize_group(out, ["momentum_252_quintile", "amplitude_bucket_v2"], "fwd_252")
    within_mom.to_csv(os.path.join(OUTDIR, "within_momentum_bucket_stats.csv"), index=False)

    within_vol = summarize_group(out, ["volatility_252_quintile", "amplitude_bucket_v2"], "fwd_252")
    within_vol.to_csv(os.path.join(OUTDIR, "within_volatility_bucket_stats.csv"), index=False)

    profile = make_core_profile(out)
    profile.to_csv(os.path.join(OUTDIR, "core_vs_noncore_factor_profile.csv"), index=False)

    # Quick controlled CORE-vs-noncore within quintiles.
    controlled_rows = []
    for factor_bucket in ["momentum_252_quintile", "volatility_252_quintile", "drawdown_252_quintile"]:
        for q, g in out.groupby(factor_bucket):
            core = g[g["is_core"] == 1]["fwd_252"].dropna()
            non = g[g["is_core"] == 0]["fwd_252"].dropna()
            if len(core) < 10 or len(non) < 10:
                continue

            controlled_rows.append({
                "control_bucket_type": factor_bucket,
                "quintile": q,
                "core_n": len(core),
                "noncore_n": len(non),
                "core_mean_252": core.mean(),
                "noncore_mean_252": non.mean(),
                "core_minus_noncore_mean_252": core.mean() - non.mean(),
                "core_median_252": core.median(),
                "noncore_median_252": non.median(),
                "core_minus_noncore_median_252": core.median() - non.median(),
                "core_sharpe_like_252": core.mean() / (core.std(ddof=1) + 1e-12),
                "noncore_sharpe_like_252": non.mean() / (non.std(ddof=1) + 1e-12),
                "core_win_rate_252": (core > 0).mean(),
                "noncore_win_rate_252": (non > 0).mean(),
            })

    controlled = pd.DataFrame(controlled_rows)
    controlled.to_csv(os.path.join(OUTDIR, "controlled_core_vs_noncore_by_quintile.csv"), index=False)

    # Report
    lines = []
    lines.append("=" * 80)
    lines.append("SCI AMPLITUDE FACTOR-CONTROL ANALYSIS")
    lines.append("=" * 80)
    lines.append("")
    lines.append(f"Original rows: {len(df)}")
    lines.append(f"Feature rows: {len(out)}")
    lines.append(f"Unique tickers with features: {out['ticker'].nunique()}")
    lines.append("")

    lines.append("CORE vs NON-CORE factor profile:")
    lines.append(profile.to_string(index=False))
    lines.append("")

    lines.append("=" * 80)
    lines.append("Regression: forward returns vs SCI + simple controls")
    lines.append("=" * 80)
    lines.append(regs.to_string(index=False))
    lines.append("")

    lines.append("=" * 80)
    lines.append("CORE vs NON-CORE within factor quintiles, 252-day forward returns")
    lines.append("=" * 80)
    lines.append(controlled.to_string(index=False))
    lines.append("")

    lines.append("=" * 80)
    lines.append("Within momentum quintile bucket stats, 252-day forward returns")
    lines.append("=" * 80)
    lines.append(within_mom.to_string(index=False))
    lines.append("")

    lines.append("=" * 80)
    lines.append("Within volatility quintile bucket stats, 252-day forward returns")
    lines.append("=" * 80)
    lines.append(within_vol.to_string(index=False))
    lines.append("")

    lines.append("Plain-English read:")
    lines.append("- If is_core or gap_percentile stays positive after controls, SCI is not just momentum/volatility.")
    lines.append("- If CORE beats non-core within the same momentum quintiles, it is not just a momentum effect.")
    lines.append("- If CORE beats non-core within the same volatility quintiles, it is not just a volatility effect.")
    lines.append("- If the effect weakens, then SCI may be partly acting as a regime/momentum/volatility quality filter, which is still useful but less novel.")

    report_path = os.path.join(OUTDIR, "factor_control_summary.txt")
    with open(report_path, "w") as f:
        f.write("\n".join(lines))

    print("DONE")
    print(f"Open: {report_path}")
    print(f"Open: {os.path.join(OUTDIR, 'factor_control_regression.csv')}")
    print(f"Open: {os.path.join(OUTDIR, 'controlled_core_vs_noncore_by_quintile.csv')}")

if __name__ == "__main__":
    main()
