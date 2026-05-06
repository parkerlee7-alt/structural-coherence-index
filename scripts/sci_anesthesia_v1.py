#!/usr/bin/env python3
"""
SCI Anesthesia v1
=================
Applies the Structural Coherence Index to propofol sedation EEG.

Dataset: OpenNeuro ds005620 (Bajwa et al.)
  "A Repeated Awakening Study Exploring the Capacity of Complexity Measures
  to Capture Dreaming During Propofol Sedation"
  21 subjects, 5000 Hz, 65 channels (standard 10-20), CC-BY-4.0
  States per subject:
    task-awake_acq-EC  — wakefulness, eyes closed (300 s)
    task-sed_acq-rest  — propofol sedation between awakenings (300 s × 3 runs)

Pre-registration: ../anesthesia/SCI_Anesthesia_PreRegistration_v1.md
  Written 2026-05-05, before any data was loaded.

Primary comparison: Awake-EC vs Sedated (task-sed run-1)
Pre-registered predictions:
  P1: SCI(Awake) > SCI(Sedated)
  P3: SCI(Sedated) in range 0.65-0.85 (Stage 3-4 analogue)
  P4: Paired t-test Awake vs Sedated, p < 0.05

Parameters: LOCKED — S=40, k=0.9, seed=42 (from sci_score_v3.py)
  W and L scaled by sample rate after downsampling to TARGET_FS=250 Hz:
  smooth_sec=0.10 s → W=round(0.10×250)=25, lag_sec=0.10 s → L=25
  This matches sci_eeg_full_analysis.py (smooth_sec=0.10, lag_sec=0.10 × 256 Hz → W=L=26)
  Raw data downsampled 5000 Hz → 250 Hz before processing.
Channel:    Fz (frontal midline, consistent with sleep-EDF approach)
Bandpass:   1-40 Hz
Epoch size: 10 s (2,500 samples at 250 Hz)

Usage
-----
  python3 sci_anesthesia_v1.py \\
      --data-dir ~/Desktop/SCI_Project/anesthesia/ds005620 \\
      --out ~/Desktop/SCI_Project/results/anesthesia_full_v1

Author: Parker J. Lee
Patent: US Provisional 63/904,444
"""

from __future__ import annotations

import argparse
import os
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.signal import butter, filtfilt
from scipy.stats import ttest_rel, wilcoxon, friedmanchisquare

warnings.filterwarnings("ignore")

# ── import canonical SCI function ──────────────────────────────────────────────
_SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(_SCRIPT_DIR.parent))   # adds SCI_Project/ to path
from sci_score_v3 import sci_score_v3         # noqa: E402

# ── constants ──────────────────────────────────────────────────────────────────
PRIMARY_CH   = "Fz"          # frontal midline (Ch52 in this dataset)
FALLBACK_CHS = ["FCz", "Cz", "F3", "F4"]   # if Fz absent, try these
BANDPASS_LO  = 1.0
BANDPASS_HI  = 40.0
EPOCH_SEC    = 10            # seconds per epoch
TARGET_FS    = 250.0         # downsample to this before SCI (matches sleep-EDF scale)
SMOOTH_SEC   = 0.10          # envelope smoothing window in seconds
LAG_SEC      = 0.10          # ACF max lag in seconds
# W and L computed from TARGET_FS at runtime:
#   W = round(SMOOTH_SEC * TARGET_FS) = 25
#   L = round(LAG_SEC   * TARGET_FS) = 25
STATE_AWAKE  = "awake"
STATE_SED    = "sedated"


# ── signal helpers ─────────────────────────────────────────────────────────────

def bandpass(data: np.ndarray, lo: float, hi: float, fs: float, order: int = 4) -> np.ndarray:
    nyq = fs / 2
    b, a = butter(order, [max(lo / nyq, 0.001), min(hi / nyq, 0.999)], btype="band")
    return filtfilt(b, a, data)


def pick_channel(ch_names: list[str]) -> str:
    for ch in [PRIMARY_CH] + FALLBACK_CHS:
        if ch in ch_names:
            return ch
    # last resort: first non-EOG/EMG channel
    for ch in ch_names:
        if not any(x in ch.upper() for x in ("EOG", "EMG", "STIM")):
            return ch
    return ch_names[0]


# ── per-epoch computation ──────────────────────────────────────────────────────

def process_epoch(segment: np.ndarray, fs: float, state: str, subject: str, epoch_idx: int) -> dict:
    filt = bandpass(segment, BANDPASS_LO, BANDPASS_HI, fs)
    w = round(SMOOTH_SEC * fs)
    L = round(LAG_SEC   * fs)
    r = sci_score_v3(filt, w=w, L=L)
    return {
        "subject":    subject,
        "state":      state,
        "epoch":      epoch_idx,
        "c_obs":      r["c_obs"],
        "c_surr_mean": r["c_surr_mean"],
        "c_surr_std": r["c_surr_std"],
        "gap":        r["gap"],
        "z":          r["z"],
        "SCI":        r["SCI"],
        "bucket":     r["bucket"],
    }


# ── subject loader ─────────────────────────────────────────────────────────────

def load_recording(vhdr_path: Path):
    """Load BrainVision recording, downsample to TARGET_FS, return (data_1d, fs, ch_name)."""
    import mne
    raw = mne.io.read_raw_brainvision(str(vhdr_path), preload=True, verbose=False)
    # Downsample to TARGET_FS so W=round(SMOOTH_SEC*fs) matches sleep-EDF scale
    if raw.info["sfreq"] != TARGET_FS:
        raw.resample(TARGET_FS, npad="auto", verbose=False)
    fs = raw.info["sfreq"]   # now TARGET_FS
    ch = pick_channel(raw.ch_names)
    idx = raw.ch_names.index(ch)
    data = raw.get_data(picks=[idx])[0]   # shape (n_samples,)
    return data, fs, ch


def process_recording(vhdr_path: Path, state: str, subject: str) -> list[dict]:
    """Cut a recording into EPOCH_SEC epochs, compute SCI for each."""
    try:
        data, fs, ch = load_recording(vhdr_path)
    except Exception as e:
        print(f"  [WARN] Could not load {vhdr_path.name}: {e}")
        return []

    epoch_len = int(EPOCH_SEC * fs)
    n_epochs  = len(data) // epoch_len
    records   = []
    for i in range(n_epochs):
        seg = data[i * epoch_len : (i + 1) * epoch_len]
        rec = process_epoch(seg, fs, state, subject, i)
        records.append(rec)

    print(f"    {state}: {n_epochs} epochs (ch={ch})")
    return records


# ── main ───────────────────────────────────────────────────────────────────────

def find_subjects(data_dir: Path) -> list[str]:
    return sorted(d.name for d in data_dir.iterdir()
                  if d.is_dir() and d.name.startswith("sub-"))


def find_vhdr(subj_dir: Path, task: str, acq: str, run: str | None = None) -> Path | None:
    eeg_dir = subj_dir / "eeg"
    if not eeg_dir.exists():
        return None
    for f in eeg_dir.glob("*.vhdr"):
        name = f.stem
        if f"task-{task}" not in name:
            continue
        if acq and f"acq-{acq}" not in name:
            continue
        if run and f"run-{run}" not in name:
            continue
        return f
    return None


def cohen_d_paired(a: np.ndarray, b: np.ndarray) -> float:
    diff = a - b
    if np.std(diff) < 1e-12:
        return float("nan")
    return float(np.mean(diff) / np.std(diff, ddof=1))


def run_stats(df_subj: pd.DataFrame, out_path: Path) -> None:
    """Run pre-registered statistical tests on subject-level means."""
    states = sorted(df_subj["state"].unique())
    lines  = []

    lines.append("=" * 70)
    lines.append("SCI ANESTHESIA v1 — STATISTICAL TESTS")
    lines.append("Pre-registration: SCI_Anesthesia_PreRegistration_v1.md")
    lines.append("=" * 70)
    lines.append("")
    lines.append("Unit of analysis: per-subject mean SCI (correct — epochs not independent)")
    lines.append(f"States present: {states}")
    lines.append(f"N subjects:     {df_subj['subject'].nunique()}")
    lines.append("")

    # Grand means
    lines.append("-" * 70)
    lines.append("GRAND MEANS BY STATE")
    lines.append("-" * 70)
    for s in states:
        sub = df_subj[df_subj["state"] == s]["mean_SCI"]
        lines.append(f"  {s:12s}  mean={sub.mean():.3f}  sd={sub.std(ddof=1):.3f}  n={len(sub)}")
    lines.append("")

    # Primary test: Awake vs Sedated
    lines.append("-" * 70)
    lines.append("PRIMARY TEST: Awake vs Sedated (paired t-test, pre-registered)")
    lines.append("-" * 70)
    pivot = df_subj.pivot(index="subject", columns="state", values="mean_SCI")
    if STATE_AWAKE in pivot.columns and STATE_SED in pivot.columns:
        merged = pivot[[STATE_AWAKE, STATE_SED]].dropna()
        a = merged[STATE_AWAKE].values
        b = merged[STATE_SED].values
        t, p = ttest_rel(a, b)
        d = cohen_d_paired(a, b)
        n = len(merged)
        lines.append(f"  Awake mean SCI = {a.mean():.3f}  Sedated mean SCI = {b.mean():.3f}")
        lines.append(f"  diff = {(a-b).mean():+.3f}")
        lines.append(f"  t({n-1}) = {t:.3f},  p = {p:.4f},  Cohen's d = {d:.3f},  n = {n}")
        lines.append("")
        # P1 check
        p1 = (a.mean() > b.mean())
        lines.append(f"  P1 (SCI_Awake > SCI_Sedated): {'CONFIRMED' if p1 else 'NOT CONFIRMED'}")
        # P3 check
        p3 = (0.65 <= b.mean() <= 0.85)
        lines.append(f"  P3 (SCI_Sedated in [0.65, 0.85]): {'CONFIRMED' if p3 else 'NOT CONFIRMED'}")
        lines.append(f"       SCI_Sedated = {b.mean():.3f}")
        # P4 check
        p4 = (p < 0.05)
        lines.append(f"  P4 (p < 0.05): {'CONFIRMED' if p4 else 'NOT CONFIRMED'}")
        lines.append("")
        # Wilcoxon non-parametric
        try:
            w_stat, w_p = wilcoxon(a, b)
            lines.append(f"  Wilcoxon signed-rank: W={w_stat:.1f},  p={w_p:.4f}")
        except Exception:
            lines.append("  Wilcoxon: could not compute (n too small)")
    else:
        lines.append(f"  Missing states. Available: {list(pivot.columns)}")
    lines.append("")

    # Friedman test if >2 states
    if len(states) > 2:
        lines.append("-" * 70)
        lines.append("FRIEDMAN TEST (all states)")
        lines.append("-" * 70)
        groups = [df_subj[df_subj["state"] == s]["mean_SCI"].values for s in states]
        min_n = min(len(g) for g in groups)
        groups = [g[:min_n] for g in groups]
        try:
            stat, p_f = friedmanchisquare(*groups)
            lines.append(f"  χ²({len(states)-1}) = {stat:.3f},  p = {p_f:.4f}")
        except Exception as e:
            lines.append(f"  Could not compute: {e}")
        lines.append("")

    lines.append("=" * 70)
    text = "\n".join(lines)
    print(text)
    (out_path / "statistical_tests.txt").write_text(text)


def make_plot(df_subj: pd.DataFrame, out_path: Path) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        states = sorted(df_subj["state"].unique(),
                        key=lambda s: (s != STATE_AWAKE, s))  # awake first
        fig, ax = plt.subplots(figsize=(6, 5))
        data   = [df_subj[df_subj["state"] == s]["mean_SCI"].values for s in states]
        colors = ["steelblue", "tomato", "goldenrod", "mediumseagreen"][:len(states)]

        bp = ax.boxplot(data, patch_artist=True, notch=False, widths=0.5)
        for patch, c in zip(bp["boxes"], colors):
            patch.set_facecolor(c)
            patch.set_alpha(0.7)

        # Overlay individual subject lines
        for i in range(len(data[0])):
            pts = [d[i] if i < len(d) else np.nan for d in data]
            ax.plot(range(1, len(states) + 1), pts, "k-o",
                    alpha=0.2, markersize=3, linewidth=0.7)

        ax.set_xticks(range(1, len(states) + 1))
        ax.set_xticklabels([s.replace("_", "\n") for s in states], fontsize=9)
        ax.set_ylabel("SCI (mean per subject)")
        ax.set_title("SCI by Propofol State\n(ds005620, locked params W=12/L=12/S=40/k=0.9)")
        ax.axhline(0.75, color="green", linestyle="--", alpha=0.4, linewidth=0.8, label="CORE ≥0.75")
        ax.axhline(0.55, color="red",   linestyle="--", alpha=0.4, linewidth=0.8, label="INELIGIBLE <0.55")
        ax.legend(fontsize=8)
        ax.set_ylim(0, 1.05)

        plots_dir = out_path / "plots"
        plots_dir.mkdir(exist_ok=True)
        fig.tight_layout()
        fig.savefig(plots_dir / "sci_by_state.png", dpi=150)
        plt.close(fig)
        print(f"  Plot saved: {plots_dir / 'sci_by_state.png'}")
    except Exception as e:
        print(f"  [WARN] Plot failed: {e}")


def main():
    parser = argparse.ArgumentParser(description="SCI Anesthesia v1")
    parser.add_argument("--data-dir", required=True,
                        help="Root folder of ds005620 (contains sub-XXXX/ subdirs)")
    parser.add_argument("--out",      required=True,
                        help="Output directory for CSVs and plots")
    parser.add_argument("--subjects", nargs="*", default=None,
                        help="Subject IDs to process (default: all found)")
    args = parser.parse_args()

    data_dir = Path(args.data_dir).expanduser()
    out_dir  = Path(args.out).expanduser()
    out_dir.mkdir(parents=True, exist_ok=True)

    all_subjects = find_subjects(data_dir)
    subjects = args.subjects if args.subjects else all_subjects
    print(f"Found {len(all_subjects)} subjects, processing {len(subjects)}: {subjects}")

    all_records = []

    for subj in subjects:
        subj_dir = data_dir / subj
        print(f"\n[{subj}]")

        # Awake (eyes closed)
        vhdr_awake = find_vhdr(subj_dir, task="awake", acq="EC")
        if vhdr_awake:
            recs = process_recording(vhdr_awake, STATE_AWAKE, subj)
            all_records.extend(recs)
        else:
            print(f"  [WARN] No awake-EC file found for {subj}")

        # Sedated (run-1, first sedation block)
        vhdr_sed = find_vhdr(subj_dir, task="sed", acq="rest", run="1")
        if vhdr_sed:
            recs = process_recording(vhdr_sed, STATE_SED, subj)
            all_records.extend(recs)
        else:
            print(f"  [WARN] No sed-run-1 file found for {subj}")

    if not all_records:
        print("No records computed — check data paths.")
        return

    df = pd.DataFrame(all_records)

    # Save epoch-level results
    epoch_csv = out_dir / "epoch_metrics.csv"
    df.to_csv(epoch_csv, index=False)
    print(f"\nSaved: {epoch_csv}  ({len(df)} rows)")

    # Per-subject per-state means
    df_subj = (
        df.groupby(["subject", "state"])
        .agg(
            mean_SCI   = ("SCI",  "mean"),
            mean_gap   = ("gap",  "mean"),
            mean_z     = ("z",    "mean"),
            n_epochs   = ("SCI",  "count"),
            pct_CORE   = ("bucket", lambda x: (x == "CORE").mean() * 100),
        )
        .reset_index()
    )
    subj_csv = out_dir / "subject_state_means.csv"
    df_subj.to_csv(subj_csv, index=False)
    print(f"Saved: {subj_csv}")

    # Grand summary
    df_summary = (
        df.groupby("state")
        .agg(
            n_epochs   = ("SCI", "count"),
            mean_SCI   = ("SCI", "mean"),
            sd_SCI     = ("SCI", lambda x: x.std(ddof=1)),
            mean_gap   = ("gap", "mean"),
            sd_gap     = ("gap", lambda x: x.std(ddof=1)),
            pct_CORE   = ("bucket", lambda x: (x == "CORE").mean() * 100),
        )
        .reset_index()
    )
    print("\nSUMMARY BY STATE (epoch level)")
    print(df_summary.to_string(index=False))
    summary_csv = out_dir / "summary_by_state.csv"
    df_summary.to_csv(summary_csv, index=False)
    print(f"\nSaved: {summary_csv}")

    # Statistics (on subject means)
    run_stats(df_subj, out_dir)

    # Plot
    make_plot(df_subj, out_dir)

    print(f"\nDone. Results in {out_dir}")


if __name__ == "__main__":
    main()
