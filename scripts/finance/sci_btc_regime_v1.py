#!/usr/bin/env python3
"""
sci_btc_regime_v1.py
Bitcoin Rolling SCI — Regime Transition Test

Hypothesis: BTC SCI rises toward CORE (>0.75) during institutionally dense periods
and falls during retail-dominated or crisis periods.

Locked parameters (finance standard): W=7, L=10, S=40, k=0.9, seed=42
Rolling window: 63 trading days (~90 calendar days)
"""

import sys, os
sys.path.insert(0, "/Users/parkerlee/Desktop/If Im Right/SCI_Project")

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from scipy import stats
import yfinance as yf
from datetime import datetime

from sci_score_v3 import sci_score_v3

# ── Parameters ────────────────────────────────────────────────────────────────
ROLL_DAYS  = 63           # ~90 calendar days of trading
W, L, S, K, SEED = 7, 10, 40, 0.9, 42
START_DATE = "2017-01-01"
END_DATE   = datetime.today().strftime("%Y-%m-%d")
OUT        = "/Users/parkerlee/Desktop/If Im Right/SCI_Project/results/btc_regime_v1"
os.makedirs(f"{OUT}/plots", exist_ok=True)

# ── Regime period definitions (based on public-record events) ─────────────────
# Institutional: CME/ETF launches, verified large corporate purchases
# Retail/Crisis: bear markets, leverage unwinds, regulatory shocks
INST_PERIODS = [
    ("2017-12-17", "2018-06-30",  "CME/CBOE Futures Era"),
    ("2020-10-01", "2021-04-30",  "Corporate Treasury Wave"),
    ("2021-10-01", "2021-11-10",  "ProShares ETF / ATH"),
    ("2024-01-11", END_DATE,      "Spot ETF Era"),
]
RETAIL_PERIODS = [
    ("2017-01-01", "2017-12-16",  "Pre-Futures Retail"),
    ("2018-07-01", "2020-09-30",  "Prolonged Bear / Retail"),
    ("2021-05-01", "2021-09-30",  "China Ban / Mid-Cycle"),
    ("2021-11-11", "2022-01-31",  "Post-ATH Deleveraging"),
    ("2022-05-01", "2022-07-31",  "Luna/3AC Collapse"),
    ("2022-11-01", "2023-12-31",  "FTX Collapse / Crypto Winter"),
]
# Events for vertical markers
KEY_EVENTS = [
    ("2017-12-17", "CME launch"),
    ("2020-08-11", "MSTR 1st buy"),
    ("2021-02-08", "Tesla $1.5B"),
    ("2021-10-19", "ProShares ETF"),
    ("2021-11-10", "ATH $69K"),
    ("2022-05-12", "Luna collapse"),
    ("2022-11-11", "FTX collapse"),
    ("2024-01-11", "Spot ETF"),
    ("2025-01-23", "Strategic Reserve EO"),
]

# ── Download price data ────────────────────────────────────────────────────────
print("Downloading BTC-USD…")
btc_raw = yf.download("BTC-USD", start=START_DATE, end=END_DATE, progress=False)
price   = btc_raw["Close"].squeeze().dropna()
returns = price.pct_change().dropna()

print("Downloading GBTC (institutional proxy)…")
gbtc_raw   = yf.download("GBTC", start=START_DATE, end=END_DATE, progress=False)
gbtc_price = gbtc_raw["Close"].squeeze().dropna()

print(f"BTC returns: {len(returns)} days  ({returns.index[0].date()} → {returns.index[-1].date()})")

# ── Rolling SCI computation ────────────────────────────────────────────────────
print(f"Computing {ROLL_DAYS}-day rolling SCI…")
records = []
ret_arr = returns.values
ret_idx = returns.index

for i in range(ROLL_DAYS, len(ret_arr)):
    window = ret_arr[i - ROLL_DAYS : i]
    if np.std(window) < 1e-10:
        continue
    r = sci_score_v3(window, w=W, L=L, S=S, k=K, seed=SEED)
    records.append({
        "date":   ret_idx[i],
        "SCI":    r["SCI"],
        "z":      r["z"],
        "gap":    r["gap"],
        "c_obs":  r["c_obs"],
        "c_surr": r["c_surr_mean"],
    })

roll = pd.DataFrame(records).set_index("date")
print(f"  → {len(roll)} windows computed")

# ── Regime labels ──────────────────────────────────────────────────────────────
def label_period(dt):
    dt = pd.Timestamp(dt)
    for s, e, _ in INST_PERIODS:
        if pd.Timestamp(s) <= dt <= pd.Timestamp(e):
            return "Institutional"
    for s, e, _ in RETAIL_PERIODS:
        if pd.Timestamp(s) <= dt <= pd.Timestamp(e):
            return "Retail"
    return "Transition"

roll["regime"] = roll.index.map(label_period)

# ── Statistical tests ──────────────────────────────────────────────────────────
inst   = roll.loc[roll["regime"] == "Institutional", "SCI"]
retail = roll.loc[roll["regime"] == "Retail",        "SCI"]

t_stat, p_ttest = stats.ttest_ind(inst, retail)
u_stat, p_mw    = stats.mannwhitneyu(inst, retail, alternative="greater")

pooled_sd = np.sqrt(
    ((len(inst)-1)*inst.std()**2 + (len(retail)-1)*retail.std()**2)
    / (len(inst)+len(retail)-2)
)
cohen_d = (inst.mean() - retail.mean()) / pooled_sd

# Per-period breakdown
period_records = []
for regime, periods in [("Institutional", INST_PERIODS), ("Retail", RETAIL_PERIODS)]:
    for s, e, name in periods:
        mask    = (roll.index >= pd.Timestamp(s)) & (roll.index <= pd.Timestamp(e))
        subset  = roll.loc[mask, "SCI"]
        if len(subset) == 0:
            continue
        period_records.append({
            "Regime":    regime,
            "Label":     name,
            "Start":     s,
            "End":       e,
            "N":         len(subset),
            "Mean_SCI":  round(subset.mean(), 4),
            "SD_SCI":    round(subset.std(), 4),
            "Median_SCI":round(subset.median(), 4),
            "Max_SCI":   round(subset.max(), 4),
            "Pct_CORE":  round((subset > 0.75).mean() * 100, 1),
        })

period_df = pd.DataFrame(period_records)

# ── Write summary ──────────────────────────────────────────────────────────────
summary = f"""Bitcoin Rolling SCI — Regime Transition Test
==============================================
Script: sci_btc_regime_v1.py
Date:   {datetime.today().strftime('%Y-%m-%d')}

Data range: {roll.index[0].date()} to {roll.index[-1].date()}
Windows:    {len(roll)} ({ROLL_DAYS}-trading-day rolling, ~90 calendar days)
SCI params: W={W}, L={L}, S={S}, k={K}, seed={SEED}

Period Counts
─────────────
Institutional windows: {len(inst)}
Retail/crisis windows: {len(retail)}
Transition windows:    {len(roll) - len(inst) - len(retail)}

Grand Means
───────────
Institutional mean SCI : {inst.mean():.4f}  (SD={inst.std():.4f}, median={inst.median():.4f})
Retail/crisis mean SCI : {retail.mean():.4f}  (SD={retail.std():.4f}, median={retail.median():.4f})
Δ (inst − retail)      : {inst.mean() - retail.mean():+.4f}

Statistical Tests
─────────────────
Independent t-test:        t = {t_stat:.4f},  p = {p_ttest:.6f}
Mann-Whitney U (inst>ret): U = {u_stat:.0f},  p = {p_mw:.6f}
Cohen's d:                 {cohen_d:.4f}  ({'large' if abs(cohen_d)>=0.8 else 'medium' if abs(cohen_d)>=0.5 else 'small'} effect)

CORE Threshold (SCI > 0.75)
────────────────────────────
Institutional windows ≥ CORE: {(inst > 0.75).sum()} / {len(inst)}  ({100*(inst>0.75).mean():.1f}%)
Retail windows ≥ CORE:        {(retail > 0.75).sum()} / {len(retail)}  ({100*(retail>0.75).mean():.1f}%)

Per-Period Breakdown
────────────────────
"""
for _, row in period_df.iterrows():
    summary += (f"  [{row['Regime']:16s}] {row['Label']:35s} "
                f"n={row['N']:4d}  mean={row['Mean_SCI']:.4f}  "
                f"CORE%={row['Pct_CORE']:.1f}%\n")

print("\n" + summary)

with open(f"{OUT}/statistical_tests.txt", "w") as f:
    f.write(summary)
roll.to_csv(f"{OUT}/rolling_sci.csv")
period_df.to_csv(f"{OUT}/period_breakdown.csv", index=False)

# ── Plot 1: Main rolling SCI + regime shading ──────────────────────────────────
fig, axes = plt.subplots(3, 1, figsize=(15, 13),
                         gridspec_kw={"height_ratios": [3, 1.5, 1.5]})
plt.subplots_adjust(hspace=0.35)

ax = axes[0]
ax.plot(roll.index, roll["SCI"], color="#1a1a2e", lw=1.1, zorder=5, label="90-day SCI")
ax.fill_between(roll.index, roll["SCI"], alpha=0.12, color="#1a1a2e")
ax.axhline(0.75, color="#c0392b", ls="--", lw=1.2, alpha=0.85, label="CORE threshold (0.75)")
ax.axhline(0.50, color="#7f8c8d", ls=":",  lw=0.9, alpha=0.6,  label="Neutral (0.50)")
ax.axhline(0.096, color="#8e44ad", ls="--", lw=1.0, alpha=0.7, label="BTC universe SCI (0.096)")

# Shade regimes
first_inst = first_retail = True
for s, e, _ in INST_PERIODS:
    s0, e0 = pd.Timestamp(s), min(pd.Timestamp(e), roll.index[-1])
    if s0 > roll.index[-1]: continue
    lbl = "Institutional" if first_inst else "_"
    ax.axvspan(s0, e0, alpha=0.15, color="#27ae60", label=lbl, zorder=1)
    first_inst = False
for s, e, _ in RETAIL_PERIODS:
    s0, e0 = pd.Timestamp(s), min(pd.Timestamp(e), roll.index[-1])
    if s0 > roll.index[-1]: continue
    lbl = "Retail / Crisis" if first_retail else "_"
    ax.axvspan(s0, e0, alpha=0.10, color="#e74c3c", label=lbl, zorder=1)
    first_retail = False

# Event markers
for dt_str, lbl in KEY_EVENTS:
    dt = pd.Timestamp(dt_str)
    if not (roll.index[0] <= dt <= roll.index[-1]):
        continue
    ax.axvline(dt, color="#8e44ad", ls="--", lw=0.8, alpha=0.65, zorder=4)
    ypos = 0.02 + (0.0 if "collapse" in lbl.lower() or "Crypto" in lbl else 0.0)
    ax.text(dt, ypos, lbl, rotation=90, fontsize=6.5, va="bottom",
            color="#5d3a8e", alpha=0.9, zorder=6)

ax.set_xlim(roll.index[0], roll.index[-1])
ax.set_ylim(-0.02, 1.02)
ax.set_ylabel("SCI", fontsize=11)
ax.set_title("Rolling 90-Day SCI — Bitcoin (BTC-USD)", fontsize=13, fontweight="bold", pad=8)
ax.legend(loc="lower right", fontsize=8, ncol=2)
ax.tick_params(axis="x", labelrotation=30, labelsize=8)

# Panel 2: BTC price
ax2 = axes[1]
ax2.semilogy(price.index, price.values, color="#2980b9", lw=1.0)
ax2.set_ylabel("BTC/USD (log)", fontsize=10)
ax2.set_title("Bitcoin Price", fontsize=10)
ax2.set_xlim(roll.index[0], roll.index[-1])
ax2.tick_params(axis="x", labelrotation=30, labelsize=8)
for s, e, _ in INST_PERIODS:
    s0, e0 = pd.Timestamp(s), min(pd.Timestamp(e), roll.index[-1])
    if s0 > roll.index[-1]: continue
    ax2.axvspan(s0, e0, alpha=0.10, color="#27ae60")
for s, e, _ in RETAIL_PERIODS:
    s0, e0 = pd.Timestamp(s), min(pd.Timestamp(e), roll.index[-1])
    if s0 > roll.index[-1]: continue
    ax2.axvspan(s0, e0, alpha=0.07, color="#e74c3c")

# Panel 3: GBTC
ax3 = axes[2]
gbtc_aligned = gbtc_price.reindex(roll.index, method="ffill").dropna()
ax3.plot(gbtc_aligned.index, gbtc_aligned.values, color="#e67e22", lw=1.0)
ax3.set_ylabel("GBTC Price (USD)", fontsize=10)
ax3.set_title("GBTC — Institutional Demand Proxy", fontsize=10)
ax3.set_xlim(roll.index[0], roll.index[-1])
ax3.tick_params(axis="x", labelrotation=30, labelsize=8)
for s, e, _ in INST_PERIODS:
    s0, e0 = pd.Timestamp(s), min(pd.Timestamp(e), roll.index[-1])
    if s0 > roll.index[-1]: continue
    ax3.axvspan(s0, e0, alpha=0.10, color="#27ae60")
for s, e, _ in RETAIL_PERIODS:
    s0, e0 = pd.Timestamp(s), min(pd.Timestamp(e), roll.index[-1])
    if s0 > roll.index[-1]: continue
    ax3.axvspan(s0, e0, alpha=0.07, color="#e74c3c")

fig.savefig(f"{OUT}/plots/btc_rolling_sci.png", dpi=150, bbox_inches="tight")
plt.close(fig)

# ── Plot 2: Regime boxplot ─────────────────────────────────────────────────────
fig2, axes2 = plt.subplots(1, 2, figsize=(11, 5))

# Boxplot
ax_b = axes2[0]
trans = roll.loc[roll["regime"] == "Transition", "SCI"]
data_groups = [inst.values, retail.values, trans.values]
labels_bp   = ["Institutional", "Retail/Crisis", "Transition"]
colors_bp   = ["#27ae60",       "#e74c3c",       "#7f8c8d"]
bp = ax_b.boxplot(data_groups, labels=labels_bp, patch_artist=True,
                  widths=0.5,
                  medianprops=dict(color="black", lw=2.5),
                  whiskerprops=dict(lw=1.2),
                  capprops=dict(lw=1.2))
for patch, color in zip(bp["boxes"], colors_bp):
    patch.set_facecolor(color)
    patch.set_alpha(0.55)
ax_b.axhline(0.75, color="#c0392b", ls="--", lw=1.2, label="CORE (0.75)")
ax_b.axhline(0.096, color="#8e44ad", ls="--", lw=1.0, label="Universe BTC (0.096)")
ax_b.set_ylabel("90-day Rolling SCI", fontsize=11)
ax_b.set_title(f"SCI by Regime\nt={t_stat:.2f}, p={p_ttest:.4f}, d={cohen_d:.3f}", fontsize=10)
ax_b.legend(fontsize=8)
ax_b.set_ylim(0, 1)

# Per-period bar chart
ax_p = axes2[1]
inst_df   = period_df[period_df["Regime"] == "Institutional"]
retail_df = period_df[period_df["Regime"] == "Retail"]
all_df = pd.concat([inst_df, retail_df]).sort_values("Start")
colors_bar = ["#27ae60" if r == "Institutional" else "#e74c3c"
              for r in all_df["Regime"]]
bars = ax_p.barh(range(len(all_df)), all_df["Mean_SCI"], color=colors_bar, alpha=0.7)
ax_p.set_yticks(range(len(all_df)))
ax_p.set_yticklabels(all_df["Label"], fontsize=7)
ax_p.axvline(0.75, color="#c0392b", ls="--", lw=1.0, label="CORE (0.75)")
ax_p.axvline(inst.mean(), color="#27ae60", ls=":", lw=1.2,
             label=f"Inst mean ({inst.mean():.3f})")
ax_p.axvline(retail.mean(), color="#e74c3c", ls=":", lw=1.2,
             label=f"Retail mean ({retail.mean():.3f})")
ax_p.set_xlabel("Mean SCI", fontsize=10)
ax_p.set_title("Mean SCI by Specific Period", fontsize=10)
ax_p.set_xlim(0, 1)
ax_p.legend(fontsize=8)

# Add n= labels
for i, (_, row) in enumerate(all_df.iterrows()):
    ax_p.text(row["Mean_SCI"] + 0.01, i, f"n={row['N']}",
              va="center", fontsize=6.5, color="#333")

fig2.tight_layout()
fig2.savefig(f"{OUT}/plots/btc_sci_regime_boxplot.png", dpi=150, bbox_inches="tight")
plt.close(fig2)

# ── Plot 3: SCI vs GBTC scatter ────────────────────────────────────────────────
fig3, ax_s = plt.subplots(figsize=(7, 5))
gbtc_s = gbtc_price.reindex(roll.index, method="ffill").dropna()
merged = roll[["SCI", "regime"]].join(gbtc_s.rename("GBTC"), how="inner")
for regime, color in [("Institutional","#27ae60"),("Retail","#e74c3c"),("Transition","#7f8c8d")]:
    sub = merged[merged["regime"] == regime]
    ax_s.scatter(sub["GBTC"], sub["SCI"], c=color, alpha=0.35, s=8, label=regime)
r, p = stats.pearsonr(merged["GBTC"].values, merged["SCI"].values)
ax_s.set_xlabel("GBTC Price (USD)", fontsize=10)
ax_s.set_ylabel("90-day Rolling SCI", fontsize=10)
ax_s.set_title(f"BTC SCI vs GBTC Price\nr={r:.3f}, p={p:.4f}", fontsize=11)
ax_s.axhline(0.75, color="#c0392b", ls="--", lw=1, alpha=0.7)
ax_s.legend(fontsize=9)
fig3.tight_layout()
fig3.savefig(f"{OUT}/plots/sci_vs_gbtc.png", dpi=150, bbox_inches="tight")
plt.close(fig3)

print(f"\n✓ All outputs saved to {OUT}/")
print(f"  rolling_sci.csv        ({len(roll)} rows)")
print(f"  period_breakdown.csv   ({len(period_df)} rows)")
print(f"  statistical_tests.txt")
print(f"  plots/btc_rolling_sci.png")
print(f"  plots/btc_sci_regime_boxplot.png")
print(f"  plots/sci_vs_gbtc.png")
