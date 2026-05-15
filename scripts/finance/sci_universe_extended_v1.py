#!/usr/bin/env python3
"""
sci_universe_extended_v1.py
Extend the rolling SCI universe back to 2010 and repeat the CORE-fraction ACF.

Goal: test whether the ~9-month quasi-periodic oscillation found in the 56-month
(2021-2026) CORE fraction ACF holds across multiple macro regimes:
    2010-2015  QE / low-volatility era
    2015-2018  Normalization
    2018-2019  Rate tightening
    2020       COVID shock
    2021-2022  Post-COVID / inflation
    2022-2026  Current

Approach:
    1. Bulk-download historical prices (2010-01-01 → today) for all 1,339 instruments
       using yfinance batch download (single HTTP session).
    2. Compute monthly 90-day rolling SCI at each of ~196 monthly rebalance dates.
    3. At each date compute CORE / TACTICAL / INELIGIBLE fractions (instruments with data).
    4. Run ACF analysis (lags 1-24) on the extended CORE fraction series.
    5. Check whether lag-9 ACF ≥ +0.40 is stable across all six macro regime sub-periods.

Locked parameters (same as Triadic Law v1):
    W=7, L=10, S=40, k=0.9, seed=42, window=90 trading days
Survivorship-bias note:
    The 1,339 instruments are those present in the universe as of 2026.
    Pre-2015 computation is therefore survivorship-biased toward firms that survived.
    This is noted but does not affect the cycle-detection goal.
"""

import sys, os, warnings
from pathlib import Path
import multiprocessing as mp

sys.path.insert(0, "/Users/parkerlee/Desktop/If Im Right/SCI_Project")
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats
import yfinance as yf

from sci_score_v3 import sci_score_v3

# ── Configuration ─────────────────────────────────────────────────────────────
ROOT        = Path("/Users/parkerlee/Desktop/If Im Right/SCI_Project")
SM_PATH     = ROOT / "results/triadic_law_v1/sci_matrix.csv"
CACHE_DIR   = ROOT / "pulse_garch_finance/cache_prices_extended"
OUT         = ROOT / "results/sci_universe_extended_v1"
os.makedirs(CACHE_DIR, exist_ok=True)
os.makedirs(OUT / "plots", exist_ok=True)

W, L, S, K, SEED = 7, 10, 40, 0.9, 42
WINDOW  = 90     # trading days
MIN_OBS = 63     # minimum valid days for SCI (63 ≈ 3 months)
START   = "2010-01-01"
TODAY   = pd.Timestamp.today().strftime("%Y-%m-%d")
BATCH   = 100    # tickers per yfinance download batch
WORKERS = min(8, mp.cpu_count())

# Monthly rebalance dates: last trading day of each month, 2010-01 through present
MONTHLY_DATES = pd.date_range("2010-01-01", TODAY, freq="BME")  # Business Month End

MACRO_REGIMES = [
    ("QE / Low-Vol",        "2010-01-01", "2014-12-31"),
    ("Normalization",       "2015-01-01", "2017-12-31"),
    ("Rate Tightening",     "2018-01-01", "2019-12-31"),
    ("COVID",               "2020-01-01", "2020-12-31"),
    ("Post-COVID / Infl.",  "2021-01-01", "2022-12-31"),
    ("Current",             "2023-01-01", TODAY),
]

# ── Step 1: Load ticker universe ───────────────────────────────────────────────
print("Loading ticker universe…")
sm      = pd.read_csv(SM_PATH, index_col=0)
TICKERS = list(sm.columns)
print(f"  {len(TICKERS)} tickers")

# ── Step 2: Bulk download prices (batch by BATCH tickers) ─────────────────────
def download_batch(batch_tickers, start, end):
    """Download a batch of tickers; return dict ticker→pd.Series of Close prices."""
    try:
        raw = yf.download(
            batch_tickers, start=start, end=end,
            auto_adjust=True, progress=False, threads=True,
            group_by="ticker"
        )
        result = {}
        if isinstance(raw.columns, pd.MultiIndex):
            for t in batch_tickers:
                try:
                    s = raw[t]["Close"].dropna()
                    if len(s) > 0:
                        result[t] = s
                except Exception:
                    pass
        else:
            # Single ticker returned
            if len(batch_tickers) == 1:
                s = raw["Close"].dropna()
                if len(s) > 0:
                    result[batch_tickers[0]] = s
        return result
    except Exception as e:
        return {}

print(f"\nDownloading prices from {START} to {TODAY} in batches of {BATCH}…")
prices_all = {}
batches    = [TICKERS[i:i+BATCH] for i in range(0, len(TICKERS), BATCH)]
for bi, batch in enumerate(batches):
    cached = [t for t in batch if (CACHE_DIR / f"{t}.csv").exists()]
    to_dl  = [t for t in batch if t not in [c for c in cached]]

    # Load cached
    for t in cached:
        try:
            df = pd.read_csv(CACHE_DIR / f"{t}.csv", index_col=0, parse_dates=True)
            col = "Close" if "Close" in df.columns else df.columns[0]
            prices_all[t] = df[col].dropna().sort_index()
        except Exception:
            to_dl.append(t)

    # Download missing
    if to_dl:
        batch_data = download_batch(to_dl, START, TODAY)
        for t, series in batch_data.items():
            prices_all[t] = series
            pd.DataFrame({"Close": series}).to_csv(CACHE_DIR / f"{t}.csv")
        print(f"  Batch {bi+1:3d}/{len(batches)}  downloaded {len(batch_data):3d}  "
              f"cached {len(cached):3d}  total_loaded={len(prices_all):4d}")
    else:
        print(f"  Batch {bi+1:3d}/{len(batches)}  all cached ({len(cached)} tickers)")

print(f"\n  Prices loaded: {len(prices_all)} tickers")

# ── Step 3: Pre-compute returns ────────────────────────────────────────────────
print("Computing returns…")
returns_all = {}
for t, prices in prices_all.items():
    ret = prices.pct_change(fill_method=None).dropna()
    if len(ret) >= MIN_OBS:
        returns_all[t] = ret
print(f"  {len(returns_all)} tickers have sufficient history")

# ── Step 4: Monthly SCI computation ───────────────────────────────────────────
def sci_for_date(date, returns_dict, window, min_obs):
    """Compute SCI for all instruments at a single monthly date."""
    scores   = {}
    buckets  = {}
    for t, ret in returns_dict.items():
        mask = ret.index <= date
        seg  = ret[mask].iloc[-window:]
        if len(seg) < min_obs:
            continue
        vals = seg.values.astype(float)
        if not np.all(np.isfinite(vals)) or np.std(vals) < 1e-10:
            continue
        r = sci_score_v3(vals, w=W, L=L, S=S, k=K, seed=SEED)
        scores[t]  = r["SCI"]
        buckets[t] = r["bucket"]
    return date, scores, buckets

print(f"\nComputing rolling SCI for {len(MONTHLY_DATES)} dates × {len(returns_all)} tickers…")
print(f"  (using {WORKERS} workers via fork, this will take several minutes)")

# Parallelize over TICKERS (not dates) to avoid pickling the full returns dict.
# Each worker gets one ticker's return series and computes SCI at all monthly dates.
def worker_ticker(args):
    """For one ticker: compute SCI at all monthly dates. Returns list of (date, SCI, bucket)."""
    ticker, ret_values, ret_index_ns, monthly_dates_ns, window, min_obs = args
    ret_index = pd.DatetimeIndex(ret_index_ns)
    ret = pd.Series(ret_values, index=ret_index)
    out = []
    for date_ns in monthly_dates_ns:
        date = pd.Timestamp(date_ns)
        mask = ret.index <= date
        seg  = ret[mask].iloc[-window:]
        if len(seg) < min_obs:
            continue
        vals = seg.values.astype(float)
        if not np.all(np.isfinite(vals)) or np.std(vals) < 1e-10:
            continue
        r = sci_score_v3(vals, w=W, L=L, S=S, k=K, seed=SEED)
        out.append((date_ns, r["SCI"], r["bucket"]))
    return ticker, out

monthly_dates_ns = MONTHLY_DATES.view("int64")

work_args = [
    (t, returns_all[t].values, returns_all[t].index.view("int64"),
     monthly_dates_ns, WINDOW, MIN_OBS)
    for t in returns_all
]

ctx = mp.get_context("fork")   # fork avoids spawn re-import on macOS
with ctx.Pool(WORKERS) as pool:
    ticker_results = pool.map(worker_ticker, work_args)
    pool.close(); pool.join()

# Pivot: collect per-date scores and buckets
print("  Done. Pivoting results…")
date_scores  = {}   # date_ns → {ticker: SCI}
date_buckets = {}   # date_ns → {ticker: bucket}
for ticker, out in ticker_results:
    for date_ns, sci, bucket in out:
        date_scores.setdefault(date_ns, {})[ticker]  = sci
        date_buckets.setdefault(date_ns, {})[ticker] = bucket

# Use date_scores/date_buckets like the old results list
results = [(pd.Timestamp(d), date_scores[d], date_buckets[d]) for d in sorted(date_scores)]
print(f"  {len(results)} date snapshots built")

print("  Done.")

# ── Step 5: Build CORE fraction time series ────────────────────────────────────
print("\nBuilding bucket fraction time series…")
rows = []
for date, scores, buckets in results:
    if len(scores) < 50:   # skip dates with very few instruments
        continue
    n     = len(buckets)
    core  = sum(1 for b in buckets.values() if b == "CORE")     / n
    tact  = sum(1 for b in buckets.values() if b == "TACTICAL") / n
    ielig = sum(1 for b in buckets.values() if b == "INELIGIBLE") / n
    rows.append({"date": date, "core": core, "tactical": tact,
                 "ineligible": ielig, "n_instruments": n})

bf_ext = pd.DataFrame(rows).set_index("date").sort_index()
bf_ext.to_csv(OUT / "bucket_fractions_extended.csv")
print(f"  {len(bf_ext)} monthly dates with data")
print(f"  Date range: {bf_ext.index[0].date()} → {bf_ext.index[-1].date()}")
print(f"  CORE fraction: min={bf_ext['core'].min():.3f}, "
      f"max={bf_ext['core'].max():.3f}, mean={bf_ext['core'].mean():.3f}")

# ── Step 6: ACF analysis ───────────────────────────────────────────────────────
core_full  = bf_ext["core"].values
dates_full = bf_ext.index

print(f"\n── ACF of CORE fraction (N={len(core_full)} months) ─────────────────────")
acf_full = {}
for lag in range(1, 25):
    if lag >= len(core_full):
        break
    r, p = stats.pearsonr(core_full[:-lag], core_full[lag:])
    acf_full[lag] = (r, p)
    marker = " ◄◄ LAG-9 PEAK" if lag == 9 else (" ◄ sig" if p < 0.05 else "")
    bar = "█" * int(abs(r) * 25)
    print(f"  lag {lag:2d}: {'+'if r>0 else '-'}{bar:25s} r={r:+.4f}  p={p:.5f}{marker}")

# ── Step 7: Per-regime ACF ────────────────────────────────────────────────────
print(f"\n── Lag-9 ACF by macro regime ────────────────────────────────────────────")
regime_acf = []
for name, s, e in MACRO_REGIMES:
    mask = (dates_full >= pd.Timestamp(s)) & (dates_full <= pd.Timestamp(e))
    sub  = core_full[mask]
    if len(sub) < 12:   # need at least 12 months for lag-9 ACF
        print(f"  {name:25s}  n={len(sub):3d}  (too short)")
        regime_acf.append({"regime": name, "n": len(sub), "r9": np.nan, "p9": np.nan})
        continue
    lag = 9
    if lag < len(sub):
        r, p = stats.pearsonr(sub[:-lag], sub[lag:])
    else:
        r, p = np.nan, np.nan
    sig = " ***" if (not np.isnan(p) and p < 0.001) else \
          " **"  if (not np.isnan(p) and p < 0.01)  else \
          " *"   if (not np.isnan(p) and p < 0.05)  else ""
    print(f"  {name:25s}  n={len(sub):3d}  lag-9 r={r:+.4f}  p={p:.5f}{sig}")
    regime_acf.append({"regime": name, "n": len(sub), "r9": r, "p9": p})

regime_df = pd.DataFrame(regime_acf)
regime_df.to_csv(OUT / "regime_lag9_acf.csv", index=False)

# ── Step 8: Plots ─────────────────────────────────────────────────────────────
fig, axes = plt.subplots(3, 1, figsize=(15, 14),
                          gridspec_kw={"height_ratios": [3, 2, 2]})
plt.subplots_adjust(hspace=0.4)

# Panel 1: CORE fraction time series with regime shading
ax = axes[0]
ax.plot(dates_full, core_full, color="#1a6e3c", lw=1.2, label="CORE fraction")
ax.fill_between(dates_full, core_full, alpha=0.12, color="#1a6e3c")
ax.axhline(core_full.mean(), color="#1a6e3c", ls=":", lw=1, alpha=0.7)

regime_colors = ["#EAF4EA", "#F0F4FF", "#FFF0F0", "#FF000022", "#F5F0FF", "#FFF8E0"]
for (name, s, e), col in zip(MACRO_REGIMES, regime_colors):
    s0 = max(pd.Timestamp(s), dates_full[0])
    e0 = min(pd.Timestamp(e), dates_full[-1])
    if s0 < e0:
        ax.axvspan(s0, e0, alpha=0.18, color=col.rstrip("0") if len(col)>7 else col, label=name)

ax.set_ylabel("CORE fraction", fontsize=11)
ax.set_title(
    f"Universe-Wide CORE Fraction (2010–2026) — N={len(core_full)} Monthly Dates\n"
    f"Survivorship note: 1,339 instruments are those present as of 2026",
    fontsize=11, fontweight="bold")
ax.legend(fontsize=7, ncol=4, loc="upper left")
ax.set_xlim(dates_full[0], dates_full[-1])
ax.tick_params(axis="x", labelrotation=30, labelsize=8)

# Panel 2: Full ACF (lags 1-24)
ax2 = axes[1]
lags_plot = list(acf_full.keys())
r_vals    = [acf_full[l][0] for l in lags_plot]
p_vals    = [acf_full[l][1] for l in lags_plot]
colors_bar = ["#1a6e3c" if r > 0 else "#c0392b" for r in r_vals]
ax2.bar(lags_plot, r_vals, color=colors_bar, alpha=0.7, edgecolor="white")
ax2.axhline(0,   color="black", lw=0.8)
ax2.axhline( 1.96/np.sqrt(len(core_full)-1), color="#7f8c8d", ls="--", lw=1,
             label="95% CI (±1.96/√N)")
ax2.axhline(-1.96/np.sqrt(len(core_full)-1), color="#7f8c8d", ls="--", lw=1)
ax2.axvline(9, color="#e74c3c", ls="--", lw=1.5, alpha=0.7, label="Lag 9 (~9-month cycle)")
ax2.set_xticks(lags_plot)
ax2.set_xlabel("Lag (months)", fontsize=10)
ax2.set_ylabel("ACF", fontsize=10)
ax2.set_title(f"Full ACF of CORE Fraction (N={len(core_full)} months, lags 1–24)",
              fontsize=11, fontweight="bold")
ax2.legend(fontsize=9)
# Annotate sig lags
for l, r, p in zip(lags_plot, r_vals, p_vals):
    if p < 0.01:
        ax2.text(l, r + (0.02 if r > 0 else -0.04), f"r={r:.2f}", ha="center",
                 fontsize=7, color="#1a1a2e")

# Panel 3: Lag-9 ACF per regime
ax3 = axes[2]
valid = regime_df.dropna(subset=["r9"])
col3  = ["#1a6e3c" if r > 0 else "#c0392b" for r in valid["r9"]]
bars  = ax3.bar(range(len(valid)), valid["r9"], color=col3, alpha=0.7, edgecolor="white")
ax3.set_xticks(range(len(valid)))
ax3.set_xticklabels(valid["regime"], rotation=20, ha="right", fontsize=9)
ax3.axhline(0, color="black", lw=0.8)
ax3.axhline(0.40, color="#e74c3c", ls="--", lw=1.2, label="r=+0.40 threshold")
ax3.set_ylabel("Lag-9 ACF (r)", fontsize=10)
ax3.set_title("Lag-9 CORE Fraction ACF by Macro Regime", fontsize=11, fontweight="bold")
ax3.legend(fontsize=9)
for i, (_, row) in enumerate(valid.iterrows()):
    sig = "***" if row["p9"] < 0.001 else "**" if row["p9"] < 0.01 else "*" if row["p9"] < 0.05 else ""
    ax3.text(i, row["r9"] + (0.02 if row["r9"] > 0 else -0.05),
             f"r={row['r9']:.3f}{sig}\nn={row['n']:.0f}", ha="center", fontsize=8)

fig.savefig(OUT / "plots/core_fraction_acf_extended.png", dpi=150, bbox_inches="tight")
plt.close(fig)

# ── Step 9: Summary ────────────────────────────────────────────────────────────
r9_full, p9_full = acf_full.get(9, (np.nan, np.nan))
r1_full, p1_full = acf_full.get(1, (np.nan, np.nan))

summary = f"""SCI Universe Extended — CORE Fraction ACF (2010–2026)
======================================================
Script: sci_universe_extended_v1.py
Date:   2026-05-15

Universe: {len(TICKERS)} tickers (same as Triadic Law v1)
  Data range: {START} to {TODAY}
  Survivorship bias: YES — tickers are those present in 2026 universe

Monthly dates computed: {len(bf_ext)}
  {bf_ext.index[0].date()} → {bf_ext.index[-1].date()}
  Mean instruments per date: {bf_ext['n_instruments'].mean():.0f}

CORE fraction statistics:
  min={bf_ext['core'].min():.4f}  max={bf_ext['core'].max():.4f}
  mean={bf_ext['core'].mean():.4f}  std={bf_ext['core'].std():.4f}

Full-series ACF (key lags):
  Lag  1: r={acf_full[1][0]:+.4f}  p={acf_full[1][1]:.5f}
  Lag  4: r={acf_full[4][0]:+.4f}  p={acf_full[4][1]:.5f}
  Lag  5: r={acf_full[5][0]:+.4f}  p={acf_full[5][1]:.5f}
  Lag  9: r={r9_full:+.4f}  p={p9_full:.6f}
  Lag 18: r={acf_full.get(18,(np.nan,np.nan))[0]:+.4f}  p={acf_full.get(18,(np.nan,np.nan))[1]:.5f}

Regime lag-9 ACF:
"""
for _, row in regime_df.iterrows():
    sig = "***" if (not pd.isna(row['p9']) and row['p9']<0.001) else \
          "**"  if (not pd.isna(row['p9']) and row['p9']<0.01)  else \
          "*"   if (not pd.isna(row['p9']) and row['p9']<0.05)  else "ns"
    r_str = f"{row['r9']:+.4f}" if not pd.isna(row['r9']) else "  N/A "
    p_str = f"{row['p9']:.5f}"  if not pd.isna(row['p9']) else "  N/A "
    summary += f"  {row['regime']:25s}  n={row['n']:3.0f}  r={r_str}  p={p_str}  {sig}\n"

verdict = "CONFIRMED" if (not np.isnan(r9_full) and r9_full >= 0.30 and p9_full < 0.01) else \
          "PARTIAL"   if (not np.isnan(r9_full) and r9_full >= 0.15 and p9_full < 0.05) else \
          "NOT CONFIRMED"

summary += f"""
Verdict: {verdict}
Prediction: lag-9 ACF ≥ +0.30 and p < 0.01 in full series AND present in
            ≥4/6 macro regimes.
"""
print("\n" + summary)
with open(OUT / "summary.txt", "w") as f:
    f.write(summary)

print(f"\n✓ All outputs saved to {OUT}/")
