#!/usr/bin/env python3
"""
Live SCI Watchlist Generator

Purpose:
  Creates today's live forward-test snapshot.

It:
  1. Loads tickers.txt
  2. Downloads/uses latest cached prices
  3. Computes current 500-day SCI gap
  4. Ranks all tickers cross-sectionally
  5. Assigns percentile buckets
  6. Saves the current CORE_TOP10 watchlist

Outputs:
  live_watchlists/
    live_sci_snapshot_YYYY-MM-DD.csv
    live_core_top10_YYYY-MM-DD.csv
    live_summary_YYYY-MM-DD.txt

This is for live forward testing, not backtest optimization.
"""

import os
import argparse
from datetime import datetime, timedelta

import numpy as np
import pandas as pd

try:
    import yfinance as yf
except Exception:
    yf = None

from scipy.signal import hilbert


# ----------------------------
# Config
# ----------------------------

WINDOW = 500
SURROGATES = 40
SMOOTH = 7
LAG = 10
OUTDIR = "live_watchlists"
CACHE_DIR = "cache_prices"


# ----------------------------
# Helpers
# ----------------------------

def read_tickers(path):
    with open(path, "r") as f:
        raw = [x.strip().upper() for x in f if x.strip() and not x.strip().startswith("#")]

    tickers = []
    seen = set()

    for t in raw:
        t = t.replace(".", "-")
        if t not in seen:
            seen.add(t)
            tickers.append(t)

    return tickers


def cache_path(ticker):
    safe = ticker.replace("/", "_").replace(":", "_")
    return os.path.join(CACHE_DIR, safe + ".csv")


def load_prices(ticker, start, end, force_refresh=False):
    os.makedirs(CACHE_DIR, exist_ok=True)
    path = cache_path(ticker)

    if not force_refresh and os.path.exists(path):
        try:
            df = pd.read_csv(path, parse_dates=["Date"])
            s = pd.to_numeric(df["Close"], errors="coerce")
            s.index = pd.to_datetime(df["Date"], errors="coerce")
            s = s.dropna().sort_index()
            s = s.loc[(s.index >= start) & (s.index <= end)]

            if len(s) >= WINDOW + 1:
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

        pd.DataFrame({"Date": s.index, "Close": s.values}).to_csv(path, index=False)

        return s.loc[(s.index >= start) & (s.index <= end)]

    except Exception as e:
        print(f"  [WARN] {ticker}: {e}")
        return pd.Series(dtype=float)


def log_returns(prices):
    prices = np.asarray(prices, dtype=float)
    prices = prices[np.isfinite(prices)]
    if len(prices) < 3:
        return np.array([])
    r = np.diff(np.log(prices))
    return r[np.isfinite(r)]


def moving_average(x, w):
    if w <= 1 or len(x) < w:
        return x
    return np.convolve(x, np.ones(w) / w, mode="same")


def mean_acf(x, max_lag):
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


def envelope(x):
    e = np.abs(hilbert(x))
    return moving_average(e, SMOOTH)


def phase_randomized_surrogate(x, rng):
    x = np.asarray(x, dtype=float)
    n = len(x)

    X = np.fft.rfft(x)
    mag = np.abs(X)

    phases = rng.uniform(0, 2 * np.pi, size=X.shape)
    phases[0] = 0.0

    if n % 2 == 0 and len(phases) > 1:
        phases[-1] = 0.0

    xs = np.fft.irfft(mag * np.exp(1j * phases), n=n)
    xs = xs - np.mean(xs)

    sd = np.std(xs)
    if sd > 1e-12:
        xs = xs / sd

    return xs


def sci_gap_from_prices(prices, rng):
    r = log_returns(prices)

    if len(r) < WINDOW - 5:
        return None

    r = r[-WINDOW:]
    r = r - np.mean(r)

    sd = np.std(r)
    if sd > 1e-12:
        r = r / sd

    env = envelope(r)
    c_obs = mean_acf(env, LAG)

    if not np.isfinite(c_obs):
        return None

    sur = []
    for _ in range(SURROGATES):
        rs = phase_randomized_surrogate(r, rng)
        cs = mean_acf(envelope(rs), LAG)
        if np.isfinite(cs):
            sur.append(cs)

    if len(sur) < 10:
        return None

    sur = np.array(sur)
    c_surr = float(np.mean(sur))
    c_surr_std = float(np.std(sur, ddof=1) + 1e-12)
    gap = float(c_obs - c_surr)
    z = float(np.clip(gap / c_surr_std, -6, 6))
    sci = float(1 / (1 + np.exp(-0.9 * z)))

    return {
        "c_obs": c_obs,
        "c_surr_mean": c_surr,
        "c_surr_std": c_surr_std,
        "w500_gap": gap,
        "w500_z": z,
        "w500_SCI": sci,
    }


def bucket_from_gap_percentile(gap, pct):
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


def trailing_features(prices):
    if len(prices) < 253:
        return {}

    p_now = prices.iloc[-1]
    p_21 = prices.iloc[-22] if len(prices) >= 22 else np.nan
    p_63 = prices.iloc[-64] if len(prices) >= 64 else np.nan
    p_126 = prices.iloc[-127] if len(prices) >= 127 else np.nan
    p_252 = prices.iloc[-253] if len(prices) >= 253 else np.nan

    r = np.log(prices).diff().dropna()
    r252 = r.iloc[-252:]

    roll_max = prices.iloc[-252:].cummax()
    dd = prices.iloc[-252:] / roll_max - 1.0

    return {
        "price_today": float(p_now),
        "mom_21": float(p_now / p_21 - 1) if p_21 > 0 else np.nan,
        "mom_63": float(p_now / p_63 - 1) if p_63 > 0 else np.nan,
        "mom_126": float(p_now / p_126 - 1) if p_126 > 0 else np.nan,
        "mom_252": float(p_now / p_252 - 1) if p_252 > 0 else np.nan,
        "vol_252": float(r252.std(ddof=1) * np.sqrt(252)) if len(r252) > 20 else np.nan,
        "drawdown_252": float(dd.min()) if len(dd) > 20 else np.nan,
        "last_price_date": str(prices.index[-1].date()),
    }


# ----------------------------
# Main
# ----------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tickers", default="tickers.txt")
    parser.add_argument("--max", type=int, default=None)
    parser.add_argument("--force-refresh", action="store_true")
    args = parser.parse_args()

    if yf is None:
        raise SystemExit("Missing yfinance. Install with: pip3 install yfinance")

    os.makedirs(OUTDIR, exist_ok=True)

    tickers = read_tickers(args.tickers)
    if args.max:
        tickers = tickers[:args.max]

    today = datetime.today().date()
    end = datetime.today() + timedelta(days=1)
    start = datetime.today() - timedelta(days=365 * 5)

    print("=" * 80)
    print("LIVE SCI WATCHLIST GENERATOR")
    print("=" * 80)
    print(f"Tickers: {len(tickers)}")
    print(f"Window: {WINDOW}")
    print(f"Surrogates: {SURROGATES}")
    print(f"Date: {today}")
    print("=" * 80)

    rows = []

    for i, ticker in enumerate(tickers, 1):
        print(f"[{i}/{len(tickers)}] {ticker}")

        prices = load_prices(
            ticker=ticker,
            start=start,
            end=end,
            force_refresh=args.force_refresh,
        )

        if len(prices) < WINDOW + 1:
            print(f"  skipped: only {len(prices)} prices")
            continue

        rng = np.random.default_rng(10_000 + i)
        metrics = sci_gap_from_prices(prices.values, rng)

        if metrics is None:
            print("  skipped: no metrics")
            continue

        row = {
            "snapshot_date": str(today),
            "ticker": ticker,
            **metrics,
            **trailing_features(prices),
        }

        rows.append(row)

        print(
            f"  gap={row['w500_gap']:.4f} "
            f"SCI={row['w500_SCI']:.3f} "
            f"price={row['price_today']:.2f}"
        )

    df = pd.DataFrame(rows)

    if df.empty:
        raise SystemExit("No rows created.")

    df["gap_percentile"] = df["w500_gap"].rank(pct=True)
    df["sci_percentile"] = df["w500_SCI"].rank(pct=True)
    df["amplitude_bucket_v2"] = df.apply(
        lambda r: bucket_from_gap_percentile(r["w500_gap"], r["gap_percentile"]),
        axis=1,
    )

    df = df.sort_values("gap_percentile", ascending=False)

    snapshot_path = os.path.join(OUTDIR, f"live_sci_snapshot_{today}.csv")
    core_path = os.path.join(OUTDIR, f"live_core_top10_{today}.csv")
    summary_path = os.path.join(OUTDIR, f"live_summary_{today}.txt")

    df.to_csv(snapshot_path, index=False)

    core = df[df["amplitude_bucket_v2"] == "AMPLITUDE_CORE_TOP10"].copy()
    core.to_csv(core_path, index=False)

    lines = []
    lines.append("=" * 80)
    lines.append("LIVE SCI WATCHLIST SNAPSHOT")
    lines.append("=" * 80)
    lines.append("")
    lines.append(f"Snapshot date: {today}")
    lines.append(f"Tickers analyzed: {len(df)}")
    lines.append(f"CORE_TOP10 count: {len(core)}")
    lines.append("")
    lines.append("Bucket counts:")
    lines.append(df["amplitude_bucket_v2"].value_counts().to_string())
    lines.append("")
    lines.append("Top 50 CORE candidates:")
    show_cols = [
        "ticker", "w500_gap", "gap_percentile", "w500_SCI",
        "price_today", "mom_63", "mom_252", "vol_252", "drawdown_252",
        "amplitude_bucket_v2"
    ]
    lines.append(core.head(50)[show_cols].to_string(index=False))
    lines.append("")
    lines.append("Forward-test instructions:")
    lines.append("- Do not edit this file after creation.")
    lines.append("- In 21, 63, 126, and 252 trading days, compare price_today to future prices.")
    lines.append("- This creates a clean live test that cannot be accused of being backtest-only.")

    with open(summary_path, "w") as f:
        f.write("\n".join(lines))

    print("")
    print("DONE")
    print(f"Saved: {snapshot_path}")
    print(f"Saved: {core_path}")
    print(f"Saved: {summary_path}")


if __name__ == "__main__":
    main()
