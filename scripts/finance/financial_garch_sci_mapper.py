#!/usr/bin/env python3
"""
Financial GARCH-SCI Mapper
==========================

Purpose
-------
Runs the revised SCI / GARCH-style amplitude-persistence test on a ticker universe.

This script asks:

    Do financial instruments with stronger volatility clustering also show
    larger SCI envelope-coherence gaps?

This is NOT primarily a forward-return predictor.
This is a theory-validation / structural-screening script.

It computes, per ticker:

    - log returns
    - squared-return autocorrelation
    - absolute-return autocorrelation
    - volatility persistence proxy
    - SCI gap and SCI score over 125 / 250 / 500 day windows
    - optional fitted GARCH(1,1) alpha + beta if the `arch` package is installed

Outputs
-------
results_garch_finance/
    financial_garch_sci_results.csv
    summary_report.txt
    gap_vs_vol_persistence.png
    sci_vs_vol_persistence.png
    gap_vs_garch_persistence.png        # only useful if arch installed
    top_gap_tickers.csv
    top_vol_persistence_tickers.csv

Install
-------
Required:
    pip3 install numpy scipy pandas matplotlib yfinance

Optional real GARCH fitting:
    pip3 install arch

Run
---
From inside your project folder:

    python3 financial_garch_sci_mapper.py --tickers tickers.txt

Fast test:
    python3 financial_garch_sci_mapper.py --tickers tickers.txt --max 25 --fast

Full universe:
    python3 financial_garch_sci_mapper.py --tickers tickers.txt --fast

Fresh price download:
    python3 financial_garch_sci_mapper.py --tickers tickers.txt --fast --force-refresh
"""

from __future__ import annotations

import argparse
import os
import warnings
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Optional, List, Dict

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

try:
    import yfinance as yf
    HAS_YF = True
except Exception:
    HAS_YF = False

try:
    from scipy.signal import hilbert
    from scipy.stats import pearsonr, spearmanr
    HAS_SCIPY = True
except Exception:
    HAS_SCIPY = False

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    HAS_MPL = True
except Exception:
    HAS_MPL = False

try:
    from arch import arch_model
    HAS_ARCH = True
except Exception:
    HAS_ARCH = False


# ---------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------

@dataclass
class Config:
    # windows
    windows: tuple[int, int, int] = (125, 250, 500)

    # SCI operator
    envelope_smooth: int = 7
    acf_max_lag: int = 10
    n_surrogates: int = 40
    rng_seed: int = 1337
    z_clip: float = 6.0
    logistic_k: float = 0.9

    # data
    start: str = "2015-01-01"
    end: Optional[str] = None
    cache_dir: str = "cache_prices"

    # output
    out_dir: str = "results_garch_finance"


# ---------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------

def read_tickers(path: str) -> List[str]:
    with open(path, "r") as f:
        raw = [line.strip().upper() for line in f if line.strip()]

    # Remove comments and duplicates while preserving order.
    tickers = []
    seen = set()
    for t in raw:
        if t.startswith("#"):
            continue
        t = t.replace(".", "-")  # yfinance uses BRK-B not BRK.B
        if t not in seen:
            seen.add(t)
            tickers.append(t)

    return tickers


def cache_path(cache_dir: str, ticker: str) -> str:
    safe = ticker.replace("/", "_").replace(":", "_")
    return os.path.join(cache_dir, f"{safe}.csv")


def load_prices(
    ticker: str,
    start: datetime,
    end: datetime,
    cache_dir: str,
    force_refresh: bool = False,
) -> pd.Series:
    os.makedirs(cache_dir, exist_ok=True)
    cp = cache_path(cache_dir, ticker)

    if not force_refresh and os.path.exists(cp):
        try:
            df = pd.read_csv(cp, parse_dates=["Date"])
            if "Close" in df.columns:
                s = pd.to_numeric(df["Close"], errors="coerce")
                s.index = pd.to_datetime(df["Date"], errors="coerce")
                s = s.dropna().sort_index()
                s = s.loc[(s.index >= start) & (s.index <= end)]
                if len(s) > 100:
                    s.name = ticker
                    return s
        except Exception:
            pass

    if not HAS_YF:
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


# ---------------------------------------------------------------------
# Returns and diagnostics
# ---------------------------------------------------------------------

def compute_log_returns(prices: pd.Series) -> pd.Series:
    prices = prices.dropna()
    r = np.log(prices).diff().dropna()
    r = r.replace([np.inf, -np.inf], np.nan).dropna()
    return r


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


def realized_volatility(r: pd.Series) -> float:
    if len(r) < 20:
        return np.nan
    return float(np.std(r, ddof=1) * np.sqrt(252))


def max_drawdown(prices: pd.Series) -> float:
    if len(prices) < 20:
        return np.nan
    roll_max = prices.cummax()
    dd = prices / roll_max - 1.0
    return float(dd.min())


def vol_persistence_proxy(r: pd.Series, max_lag: int = 20) -> float:
    """
    A GARCH-like persistence proxy without fitting a GARCH model.

    GARCH means volatility clusters.
    If |returns| and returns^2 are autocorrelated, volatility is clustering.

    This proxy averages:
        mean ACF(abs returns)
        mean ACF(squared returns)

    Higher = more volatility clustering.
    """
    arr = r.values
    abs_acf = mean_acf(np.abs(arr), max_lag)
    sq_acf = mean_acf(arr ** 2, max_lag)

    vals = [v for v in [abs_acf, sq_acf] if np.isfinite(v)]
    if not vals:
        return np.nan

    return float(np.mean(vals))


def fit_garch_optional(r: pd.Series) -> Dict[str, float]:
    """
    Optional true GARCH(1,1) fit using the `arch` package.

    Returns alpha, beta, alpha+beta if available.
    Uses percent returns because arch expects data on a reasonable scale.
    """
    if not HAS_ARCH:
        return {
            "garch_alpha": np.nan,
            "garch_beta": np.nan,
            "garch_persistence": np.nan,
            "garch_ok": False,
        }

    try:
        y = 100.0 * r.dropna()
        if len(y) < 500:
            raise ValueError("not enough returns for GARCH fit")

        model = arch_model(y, mean="Constant", vol="GARCH", p=1, q=1, dist="normal")
        res = model.fit(disp="off", show_warning=False)

        params = res.params
        alpha = float(params.get("alpha[1]", np.nan))
        beta = float(params.get("beta[1]", np.nan))

        return {
            "garch_alpha": alpha,
            "garch_beta": beta,
            "garch_persistence": alpha + beta if np.isfinite(alpha) and np.isfinite(beta) else np.nan,
            "garch_ok": True,
        }

    except Exception:
        return {
            "garch_alpha": np.nan,
            "garch_beta": np.nan,
            "garch_persistence": np.nan,
            "garch_ok": False,
        }


# ---------------------------------------------------------------------
# SCI math
# ---------------------------------------------------------------------

def moving_average(x: np.ndarray, w: int) -> np.ndarray:
    if w <= 1 or len(x) < w:
        return x
    return np.convolve(x, np.ones(w) / w, mode="same")


def envelope_series(x: np.ndarray, smooth: int) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]

    if len(x) == 0:
        return x

    if HAS_SCIPY:
        env = np.abs(hilbert(x))
    else:
        env = np.abs(x)

    return moving_average(env, smooth)


def acf_coherence(env: np.ndarray, max_lag: int) -> float:
    return mean_acf(env, max_lag)


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


def sci_gap_from_returns(r: np.ndarray, cfg: Config, rng: np.random.Generator) -> Dict[str, float]:
    """
    Compute raw SCI components from a return window.
    """
    r = np.asarray(r, dtype=float)
    r = r[np.isfinite(r)]

    if len(r) < cfg.acf_max_lag + 20:
        return {
            "c_obs": np.nan,
            "c_surr_mean": np.nan,
            "c_surr_std": np.nan,
            "gap": np.nan,
            "z": np.nan,
            "SCI": np.nan,
        }

    # Standardize returns so scale does not dominate.
    r = r - np.mean(r)
    sd = np.std(r)
    if sd > 1e-12:
        r = r / sd

    env = envelope_series(r, cfg.envelope_smooth)
    c_obs = acf_coherence(env, cfg.acf_max_lag)

    if not np.isfinite(c_obs):
        return {
            "c_obs": np.nan,
            "c_surr_mean": np.nan,
            "c_surr_std": np.nan,
            "gap": np.nan,
            "z": np.nan,
            "SCI": np.nan,
        }

    sur = []
    for _ in range(cfg.n_surrogates):
        rs = phase_randomized_surrogate(r, rng)
        env_s = envelope_series(rs, cfg.envelope_smooth)
        cs = acf_coherence(env_s, cfg.acf_max_lag)
        if np.isfinite(cs):
            sur.append(cs)

    if len(sur) < max(8, cfg.n_surrogates // 3):
        return {
            "c_obs": c_obs,
            "c_surr_mean": np.nan,
            "c_surr_std": np.nan,
            "gap": np.nan,
            "z": np.nan,
            "SCI": np.nan,
        }

    sur = np.array(sur, dtype=float)
    mu = float(np.mean(sur))
    sdev = float(np.std(sur, ddof=1) + 1e-12)
    gap = float(c_obs - mu)
    z = float(np.clip(gap / sdev, -cfg.z_clip, cfg.z_clip))
    sci = float(1.0 / (1.0 + np.exp(-cfg.logistic_k * z)))

    return {
        "c_obs": float(c_obs),
        "c_surr_mean": mu,
        "c_surr_std": sdev,
        "gap": gap,
        "z": z,
        "SCI": sci,
    }


def classify_gap(gap_500: float, sci_500: float) -> str:
    """
    Simple structural label for this GARCH-SCI experiment.
    Not the same as your older investment CORE/MID/TACTICAL buckets.
    """
    if not np.isfinite(gap_500) or not np.isfinite(sci_500):
        return "NO_DATA"
    if gap_500 >= 0.08 and sci_500 >= 0.75:
        return "AMPLITUDE_CORE"
    if gap_500 >= 0.04 and sci_500 >= 0.65:
        return "AMPLITUDE_MID"
    if gap_500 >= 0.02 and sci_500 >= 0.55:
        return "AMPLITUDE_TACTICAL"
    return "AMPLITUDE_LOW"


# ---------------------------------------------------------------------
# Main per-ticker analysis
# ---------------------------------------------------------------------

def analyze_ticker(
    ticker: str,
    prices: pd.Series,
    cfg: Config,
    fit_garch: bool,
    idx: int,
) -> Dict[str, object]:
    rng = np.random.default_rng(cfg.rng_seed + idx)

    r = compute_log_returns(prices)

    row: Dict[str, object] = {
        "ticker": ticker,
        "n_prices": len(prices),
        "n_returns": len(r),
        "start_date": str(prices.index.min().date()) if len(prices) else "",
        "end_date": str(prices.index.max().date()) if len(prices) else "",
        "annualized_vol": realized_volatility(r),
        "max_drawdown": max_drawdown(prices),
        "mean_daily_return": float(r.mean()) if len(r) else np.nan,
        "abs_return_acf_20": mean_acf(np.abs(r.values), 20) if len(r) else np.nan,
        "squared_return_acf_20": mean_acf(r.values ** 2, 20) if len(r) else np.nan,
        "vol_persistence_proxy": vol_persistence_proxy(r, 20) if len(r) else np.nan,
    }

    # Optional true GARCH fit.
    if fit_garch:
        row.update(fit_garch_optional(r))
    else:
        row.update({
            "garch_alpha": np.nan,
            "garch_beta": np.nan,
            "garch_persistence": np.nan,
            "garch_ok": False,
        })

    # SCI gap on latest windows.
    for w in cfg.windows:
        if len(r) >= w:
            rw = r.iloc[-w:].values
            metrics = sci_gap_from_returns(rw, cfg, rng)
        else:
            metrics = {
                "c_obs": np.nan,
                "c_surr_mean": np.nan,
                "c_surr_std": np.nan,
                "gap": np.nan,
                "z": np.nan,
                "SCI": np.nan,
            }

        prefix = f"w{w}"
        row[f"{prefix}_c_obs"] = metrics["c_obs"]
        row[f"{prefix}_c_surr_mean"] = metrics["c_surr_mean"]
        row[f"{prefix}_c_surr_std"] = metrics["c_surr_std"]
        row[f"{prefix}_gap"] = metrics["gap"]
        row[f"{prefix}_z"] = metrics["z"]
        row[f"{prefix}_SCI"] = metrics["SCI"]

    row["amplitude_bucket"] = classify_gap(row.get("w500_gap", np.nan), row.get("w500_SCI", np.nan))

    return row


# ---------------------------------------------------------------------
# Plots and report
# ---------------------------------------------------------------------

def safe_corr(df: pd.DataFrame, x: str, y: str) -> str:
    sub = df[[x, y]].replace([np.inf, -np.inf], np.nan).dropna()
    if len(sub) < 5:
        return f"{x} vs {y}: not enough data"

    pr, pp = pearsonr(sub[x], sub[y])
    sr, sp = spearmanr(sub[x], sub[y])

    return (
        f"{x} vs {y}: "
        f"Pearson r={pr:.3f}, p={pp:.4g}; "
        f"Spearman r={sr:.3f}, p={sp:.4g}; "
        f"N={len(sub)}"
    )


def make_scatter(df: pd.DataFrame, x: str, y: str, title: str, out_path: str) -> None:
    if not HAS_MPL:
        return

    sub = df[[x, y, "ticker"]].replace([np.inf, -np.inf], np.nan).dropna()
    if len(sub) < 5:
        return

    plt.figure(figsize=(9, 5))
    plt.scatter(sub[x], sub[y], alpha=0.65)
    plt.xlabel(x)
    plt.ylabel(y)
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def write_report(df: pd.DataFrame, cfg: Config, out_dir: str) -> None:
    report_path = os.path.join(out_dir, "summary_report.txt")

    valid = df[df["w500_gap"].notna()].copy()

    lines = []
    lines.append("=" * 80)
    lines.append("FINANCIAL GARCH-SCI SUMMARY")
    lines.append("=" * 80)
    lines.append("")
    lines.append(f"Tickers analyzed: {len(df)}")
    lines.append(f"Tickers with valid 500-day SCI gap: {len(valid)}")
    lines.append(f"Surrogates per window: {cfg.n_surrogates}")
    lines.append(f"Windows: {cfg.windows}")
    lines.append(f"Optional arch GARCH installed: {HAS_ARCH}")
    lines.append("")

    lines.append("Main theory correlations:")
    lines.append(safe_corr(df, "vol_persistence_proxy", "w500_gap"))
    lines.append(safe_corr(df, "vol_persistence_proxy", "w500_SCI"))
    lines.append(safe_corr(df, "squared_return_acf_20", "w500_gap"))
    lines.append(safe_corr(df, "abs_return_acf_20", "w500_gap"))

    if df["garch_persistence"].notna().sum() >= 5:
        lines.append("")
        lines.append("Optional fitted GARCH correlations:")
        lines.append(safe_corr(df, "garch_persistence", "w500_gap"))
        lines.append(safe_corr(df, "garch_persistence", "w500_SCI"))

    lines.append("")
    lines.append("Amplitude bucket counts:")
    if "amplitude_bucket" in df.columns:
        counts = df["amplitude_bucket"].value_counts(dropna=False)
        lines.append(counts.to_string())

    lines.append("")
    lines.append("Top 25 tickers by 500-day SCI gap:")
    cols = [
        "ticker", "w500_gap", "w500_SCI", "vol_persistence_proxy",
        "squared_return_acf_20", "abs_return_acf_20",
        "annualized_vol", "max_drawdown", "amplitude_bucket",
        "garch_persistence"
    ]
    use_cols = [c for c in cols if c in df.columns]
    top_gap = df.sort_values("w500_gap", ascending=False)[use_cols].head(25)
    lines.append(top_gap.to_string(index=False))

    lines.append("")
    lines.append("Top 25 tickers by volatility persistence proxy:")
    top_vol = df.sort_values("vol_persistence_proxy", ascending=False)[use_cols].head(25)
    lines.append(top_vol.to_string(index=False))

    lines.append("")
    lines.append("Interpretation:")
    lines.append(
        "If vol_persistence_proxy and/or fitted GARCH alpha+beta correlate positively "
        "with w500_gap, that supports the revised SCI theory in financial data: "
        "assets with clustered amplitude/volatility dynamics tend to show envelope "
        "coherence beyond phase-randomized spectral surrogates."
    )
    lines.append("")
    lines.append(
        "Important: this script is not yet testing forward returns. It is testing whether "
        "financial return series show the same GARCH-like amplitude persistence mechanism "
        "observed in the synthetic GARCH experiment."
    )

    with open(report_path, "w") as f:
        f.write("\n".join(lines))


# ---------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run financial GARCH-SCI mapping on ticker universe.")
    p.add_argument("--tickers", default="tickers.txt", help="Path to ticker list.")
    p.add_argument("--out", default="results_garch_finance", help="Output folder.")
    p.add_argument("--cache", default="cache_prices", help="Price cache folder.")
    p.add_argument("--start", default="2015-01-01", help="Start date YYYY-MM-DD.")
    p.add_argument("--end", default=None, help="End date YYYY-MM-DD. Default today.")
    p.add_argument("--max", type=int, default=None, help="Max tickers for test run.")
    p.add_argument("--force-refresh", action="store_true", help="Force fresh yfinance download.")
    p.add_argument("--fast", action="store_true", help="Use fewer surrogates for full-universe speed.")
    p.add_argument("--fit-garch", action="store_true", help="Fit true GARCH(1,1) if arch is installed.")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    if not HAS_YF:
        raise SystemExit("Missing yfinance. Install with: pip3 install yfinance")

    if not HAS_SCIPY:
        raise SystemExit("Missing scipy. Install with: pip3 install scipy")

    cfg = Config(
        start=args.start,
        end=args.end,
        cache_dir=args.cache,
        out_dir=args.out,
    )

    if args.fast:
        cfg.n_surrogates = 20

    os.makedirs(cfg.out_dir, exist_ok=True)

    tickers = read_tickers(args.tickers)
    if args.max:
        tickers = tickers[:args.max]

    start = datetime.strptime(cfg.start, "%Y-%m-%d")
    end = datetime.strptime(cfg.end, "%Y-%m-%d") if cfg.end else datetime.today() + timedelta(days=1)

    print("=" * 80)
    print("FINANCIAL GARCH-SCI MAPPER")
    print("=" * 80)
    print(f"Tickers: {len(tickers)}")
    print(f"Date range: {start.date()} to {end.date()}")
    print(f"Surrogates: {cfg.n_surrogates}")
    print(f"Fit true GARCH: {args.fit_garch} | arch installed: {HAS_ARCH}")
    print(f"Output: {cfg.out_dir}")
    print("=" * 80)

    rows = []

    for i, ticker in enumerate(tickers, start=1):
        print(f"[{i}/{len(tickers)}] {ticker}")

        prices = load_prices(
            ticker=ticker,
            start=start,
            end=end,
            cache_dir=cfg.cache_dir,
            force_refresh=args.force_refresh,
        )

        if len(prices) < 600:
            print(f"  skipped: only {len(prices)} price points")
            continue

        try:
            row = analyze_ticker(
                ticker=ticker,
                prices=prices,
                cfg=cfg,
                fit_garch=args.fit_garch,
                idx=i,
            )
            rows.append(row)

            print(
                f"  w500_gap={row['w500_gap']:.4f} "
                f"w500_SCI={row['w500_SCI']:.3f} "
                f"vol_persist={row['vol_persistence_proxy']:.4f} "
                f"bucket={row['amplitude_bucket']}"
            )

        except Exception as e:
            print(f"  [ERROR] {ticker}: {e}")

    df = pd.DataFrame(rows)

    results_csv = os.path.join(cfg.out_dir, "financial_garch_sci_results.csv")
    df.to_csv(results_csv, index=False)

    if len(df):
        df.sort_values("w500_gap", ascending=False).head(100).to_csv(
            os.path.join(cfg.out_dir, "top_gap_tickers.csv"), index=False
        )
        df.sort_values("vol_persistence_proxy", ascending=False).head(100).to_csv(
            os.path.join(cfg.out_dir, "top_vol_persistence_tickers.csv"), index=False
        )

        make_scatter(
            df,
            "vol_persistence_proxy",
            "w500_gap",
            "500-day SCI gap vs volatility persistence proxy",
            os.path.join(cfg.out_dir, "gap_vs_vol_persistence.png"),
        )

        make_scatter(
            df,
            "vol_persistence_proxy",
            "w500_SCI",
            "500-day SCI score vs volatility persistence proxy",
            os.path.join(cfg.out_dir, "sci_vs_vol_persistence.png"),
        )

        if df["garch_persistence"].notna().sum() >= 5:
            make_scatter(
                df,
                "garch_persistence",
                "w500_gap",
                "500-day SCI gap vs fitted GARCH alpha+beta",
                os.path.join(cfg.out_dir, "gap_vs_garch_persistence.png"),
            )

        write_report(df, cfg, cfg.out_dir)

    print("")
    print("DONE")
    print(f"Saved: {results_csv}")
    print(f"Report: {os.path.join(cfg.out_dir, 'summary_report.txt')}")


if __name__ == "__main__":
    main()
