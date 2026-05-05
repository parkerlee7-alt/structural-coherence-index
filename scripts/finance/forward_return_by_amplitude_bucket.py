#!/usr/bin/env python3
"""
Forward Return Test by GARCH-SCI Amplitude Buckets

This tests the money question:

Do tickers with high historical SCI envelope gap behave differently forward?

At each rebalance date:
  1. Use only past prices.
  2. Compute 500-day SCI gap.
  3. Rank all tickers by gap percentile.
  4. Assign percentile-based amplitude bucket.
  5. Measure forward returns over 21 / 63 / 126 / 252 trading days.

Outputs:
  results_forward_amplitude/
    forward_amplitude_returns.csv
    bucket_forward_stats.csv
    summary_report.txt
"""

from __future__ import annotations

import argparse
import os
import warnings
from datetime import datetime, timedelta
from typing import List, Dict

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

try:
    import yfinance as yf
except Exception:
    yf = None

try:
    from scipy.signal import hilbert
except Exception:
    hilbert = None


# ----------------------------
# Basic helpers
# ----------------------------

def read_tickers(path: str) -> List[str]:
    with open(path, "r") as f:
        raw = [x.strip().upper() for x in f if x.strip() and not x.strip().startswith("#")]

    out = []
    seen = set()
    for t in raw:
        t = t.replace(".", "-")
        if t not in seen:
            seen.add(t)
            out.append(t)
    return out


def cache_path(cache_dir: str, ticker: str) -> str:
    safe = ticker.replace("/", "_").replace(":", "_")
    return os.path.join(cache_dir, safe + ".csv")


def load_prices(ticker: str, start: datetime, end: datetime, cache_dir: str, force_refresh: bool=False) -> pd.Series:
    os.makedirs(cache_dir, exist_ok=True)
    cp = cache_path(cache_dir, ticker)

    if not force_refresh and os.path.exists(cp):
        try:
            df = pd.read_csv(cp, parse_dates=["Date"])
            s = pd.to_numeric(df["Close"], errors="coerce")
            s.index = pd.to_datetime(df["Date"], errors="coerce")
            s = s.dropna().sort_index()
            s = s.loc[(s.index >= start) & (s.index <= end)]
            if len(s) > 600:
                s.name = ticker
                return s
        except Exception:
            pass

    if yf is None:
        return pd.Series(dtype=float)

    try:
        df = yf.download(
            ticker,
            start=start.date(),
            end=end.date(),
            progress=False,
            auto_adjust=True,
            threads=False,
        )

        if df is None or len(df) == 0:
            return pd.Series(dtype=float)

        if isinstance(df.columns, pd.MultiIndex):
            df.columns = [c[0] for c in df.columns]

        if "Close" in df.columns:
            s = df["Close"]
        elif "Adj Close" in df.columns:
            s = df["Adj Close"]
        else:
            return pd.Series(dtype=float)

        if isinstance(s, pd.DataFrame):
            s = s.iloc[:, 0]

        s = pd.to_numeric(s, errors="coerce").dropna()
        s.index = pd.to_datetime(s.index)
        s = s.sort_index()
        s.name = ticker

        pd.DataFrame({"Date": s.index, "Close": s.values}).to_csv(cp, index=False)

        return s.loc[(s.index >= start) & (s.index <= end)]

    except Exception as e:
        print(f"  [WARN] {ticker}: {e}")
        return pd.Series(dtype=float)


# ----------------------------
# SCI gap math
# ----------------------------

def log_returns(prices: np.ndarray) -> np.ndarray:
    prices = np.asarray(prices, dtype=float)
    prices = prices[np.isfinite(prices)]
    if len(prices) < 3:
        return np.array([])
    r = np.diff(np.log(prices))
    return r[np.isfinite(r)]


def moving_average(x: np.ndarray, w: int) -> np.ndarray:
    if w <= 1 or len(x) < w:
        return x
    return np.convolve(x, np.ones(w) / w, mode="same")


def envelope(x: np.ndarray, smooth: int) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if len(x) == 0:
        return x

    if hilbert is not None:
        e = np.abs(hilbert(x))
    else:
        e = np.abs(x)

    return moving_average(e, smooth)


def mean_acf(x: np.ndarray, max_lag: int) -> float:
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if len(x) < max_lag + 5:
        return np.nan

    x = x - np.mean(x)
    denom = np.sum(x * x)
    if denom <= 1e-12:
        return np.nan

    vals = []
    for lag in range(1, max_lag + 1):
        vals.append(np.sum(x[:-lag] * x[lag:]) / denom)

    return float(np.mean(vals))


def phase_randomized_surrogate(x: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    n = len(x)

    if n < 8:
        return x.copy()

    X = np.fft.rfft(x)
    mag = np.abs(X)

    phases = rng.uniform(0.0, 2.0 * np.pi, size=X.shape)
    phases[0] = 0.0

    if n % 2 == 0 and len(phases) > 1:
        phases[-1] = 0.0

    xs = np.fft.irfft(mag * np.exp(1j * phases), n=n)
    xs = xs - np.mean(xs)
    sd = np.std(xs)
    if sd > 1e-12:
        xs = xs / sd
    return xs


def sci_gap_from_price_window(prices_window: np.ndarray, smooth: int, max_lag: int, n_surrogates: int, rng: np.random.Generator) -> Dict[str, float]:
    r = log_returns(prices_window)
    if len(r) < max_lag + 20:
        return {"c_obs": np.nan, "c_surr": np.nan, "gap": np.nan, "z": np.nan, "SCI": np.nan}

    r = r - np.mean(r)
    sd = np.std(r)
    if sd > 1e-12:
        r = r / sd

    env = envelope(r, smooth)
    c_obs = mean_acf(env, max_lag)

    sur = []
    for _ in range(n_surrogates):
        rs = phase_randomized_surrogate(r, rng)
        cs = mean_acf(envelope(rs, smooth), max_lag)
        if np.isfinite(cs):
            sur.append(cs)

    if not np.isfinite(c_obs) or len(sur) < max(5, n_surrogates // 3):
        return {"c_obs": c_obs, "c_surr": np.nan, "gap": np.nan, "z": np.nan, "SCI": np.nan}

    mu = float(np.mean(sur))
    sd_s = float(np.std(sur, ddof=1) + 1e-12)
    gap = float(c_obs - mu)
    z = float(np.clip(gap / sd_s, -6, 6))
    sci = float(1.0 / (1.0 + np.exp(-0.9 * z)))

    return {"c_obs": float(c_obs), "c_surr": mu, "gap": gap, "z": z, "SCI": sci}


def bucket_from_gap_percentile(gap: float, pct: float) -> str:
    if not np.isfinite(gap) or not np.isfinite(pct):
        return "NO_DATA"
    if gap < 0:
        return "SPECTRAL_DOMINANT"
    if pct >= 0.90:
        return "AMPLITUDE_CORE_TOP10"
    if pct >= 0.70:
        return "AMPLITUDE_MID_70_90"
    if pct >= 0.40:
        return "AMPLITUDE_TACTICAL_40_70"
    return "AMPLITUDE_LOW_BOTTOM40"


# ----------------------------
# Forward return engine
# ----------------------------

def forward_return(prices: pd.Series, date: pd.Timestamp, fwd_days: int) -> float:
    idx = prices.index.searchsorted(date)
    if idx >= len(prices):
        return np.nan

    fwd_idx = idx + fwd_days
    if fwd_idx >= len(prices):
        return np.nan

    p0 = prices.iloc[idx]
    p1 = prices.iloc[fwd_idx]

    if p0 <= 0 or not np.isfinite(p0) or not np.isfinite(p1):
        return np.nan

    return float(p1 / p0 - 1.0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tickers", default="tickers.txt")
    ap.add_argument("--cache", default="cache_prices")
    ap.add_argument("--out", default="results_forward_amplitude")
    ap.add_argument("--start", default="2015-01-01")
    ap.add_argument("--end", default=None)
    ap.add_argument("--window", type=int, default=500)
    ap.add_argument("--rebalance", type=int, default=21)
    ap.add_argument("--surrogates", type=int, default=20)
    ap.add_argument("--smooth", type=int, default=7)
    ap.add_argument("--lag", type=int, default=10)
    ap.add_argument("--max", type=int, default=None)
    ap.add_argument("--force-refresh", action="store_true")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)

    tickers = read_tickers(args.tickers)
    if args.max:
        tickers = tickers[:args.max]

    start = datetime.strptime(args.start, "%Y-%m-%d")
    end = datetime.strptime(args.end, "%Y-%m-%d") if args.end else datetime.today() + timedelta(days=1)

    print("=" * 80)
    print("FORWARD RETURN TEST BY AMPLITUDE BUCKET")
    print("=" * 80)
    print(f"Tickers: {len(tickers)}")
    print(f"Window: {args.window}")
    print(f"Rebalance: {args.rebalance}")
    print(f"Surrogates: {args.surrogates}")
    print(f"Date range: {start.date()} to {end.date()}")
    print("=" * 80)

    # Load all prices first
    price_map = {}
    for i, t in enumerate(tickers, 1):
        print(f"Loading [{i}/{len(tickers)}] {t}")
        s = load_prices(t, start, end, args.cache, args.force_refresh)
        if len(s) >= args.window + 253:
            price_map[t] = s
        else:
            print(f"  skipped: only {len(s)} prices")

    if not price_map:
        raise SystemExit("No usable price data.")

    # Build common rebalance calendar from SPY if available, else first ticker
    base_ticker = "SPY" if "SPY" in price_map else list(price_map.keys())[0]
    base_dates = price_map[base_ticker].index

    start_i = args.window
    end_i = len(base_dates) - 253

    rebalance_dates = base_dates[start_i:end_i:args.rebalance]

    print(f"Usable tickers: {len(price_map)}")
    print(f"Rebalance dates: {len(rebalance_dates)}")

    all_rows = []
    rng_master = np.random.default_rng(1337)

    for d_i, date in enumerate(rebalance_dates, 1):
        print(f"\nDate [{d_i}/{len(rebalance_dates)}] {date.date()}")

        date_rows = []

        for t_i, (ticker, prices) in enumerate(price_map.items(), 1):
            idx = prices.index.searchsorted(date)
            if idx < args.window:
                continue
            if idx + 252 >= len(prices):
                continue

            win = prices.iloc[idx - args.window:idx].values
            rng = np.random.default_rng(int(rng_master.integers(0, 2**31 - 1)))

            m = sci_gap_from_price_window(
                win,
                smooth=args.smooth,
                max_lag=args.lag,
                n_surrogates=args.surrogates,
                rng=rng,
            )

            if not np.isfinite(m["gap"]):
                continue

            row = {
                "date": date,
                "ticker": ticker,
                "w500_gap": m["gap"],
                "w500_SCI": m["SCI"],
                "c_obs": m["c_obs"],
                "c_surr": m["c_surr"],
                "fwd_21": forward_return(prices, date, 21),
                "fwd_63": forward_return(prices, date, 63),
                "fwd_126": forward_return(prices, date, 126),
                "fwd_252": forward_return(prices, date, 252),
            }
            date_rows.append(row)

        if len(date_rows) < 20:
            print("  too few valid tickers")
            continue

        ddf = pd.DataFrame(date_rows)
        ddf["gap_percentile"] = ddf["w500_gap"].rank(pct=True)
        ddf["amplitude_bucket_v2"] = ddf.apply(
            lambda r: bucket_from_gap_percentile(r["w500_gap"], r["gap_percentile"]),
            axis=1,
        )

        print(ddf["amplitude_bucket_v2"].value_counts().to_string())

        all_rows.append(ddf)

        # Save checkpoint every 5 dates
        if d_i % 5 == 0 and all_rows:
            tmp = pd.concat(all_rows, ignore_index=True)
            tmp.to_csv(os.path.join(args.out, "forward_amplitude_returns_checkpoint.csv"), index=False)

    if not all_rows:
        raise SystemExit("No rows generated.")

    df = pd.concat(all_rows, ignore_index=True)
    df.to_csv(os.path.join(args.out, "forward_amplitude_returns.csv"), index=False)

    # Bucket stats
    stats_rows = []
    for horizon in [21, 63, 126, 252]:
        col = f"fwd_{horizon}"
        sub = df[df[col].notna()].copy()

        for bucket, g in sub.groupby("amplitude_bucket_v2"):
            rets = g[col].dropna()
            if len(rets) == 0:
                continue
            stats_rows.append({
                "horizon": horizon,
                "bucket": bucket,
                "n": len(rets),
                "mean_return": rets.mean(),
                "median_return": rets.median(),
                "std_return": rets.std(ddof=1),
                "sharpe_like": rets.mean() / (rets.std(ddof=1) + 1e-12),
                "win_rate": (rets > 0).mean(),
                "avg_gap": g["w500_gap"].mean(),
                "avg_sci": g["w500_SCI"].mean(),
            })

    stats = pd.DataFrame(stats_rows)
    stats.to_csv(os.path.join(args.out, "bucket_forward_stats.csv"), index=False)

    # Summary report
    lines = []
    lines.append("=" * 80)
    lines.append("FORWARD RETURN BY AMPLITUDE BUCKET SUMMARY")
    lines.append("=" * 80)
    lines.append("")
    lines.append(f"Tickers loaded: {len(price_map)}")
    lines.append(f"Observations: {len(df)}")
    lines.append(f"Rebalance dates: {df['date'].nunique()}")
    lines.append(f"Surrogates: {args.surrogates}")
    lines.append("")
    lines.append("Bucket counts:")
    lines.append(df["amplitude_bucket_v2"].value_counts().to_string())
    lines.append("")
    lines.append("Forward return stats:")
    lines.append(stats.to_string(index=False))
    lines.append("")
    lines.append("Interpretation:")
    lines.append("This is a historical point-in-time test. Buckets are assigned using only past data.")
    lines.append("Do not treat high amplitude bucket as automatically bullish until the forward-return table says so.")

    with open(os.path.join(args.out, "summary_report.txt"), "w") as f:
        f.write("\n".join(lines))

    print("\nDONE")
    print(f"Saved: {os.path.join(args.out, 'summary_report.txt')}")
    print(f"Saved: {os.path.join(args.out, 'bucket_forward_stats.csv')}")
    print(f"Saved: {os.path.join(args.out, 'forward_amplitude_returns.csv')}")


if __name__ == "__main__":
    main()
