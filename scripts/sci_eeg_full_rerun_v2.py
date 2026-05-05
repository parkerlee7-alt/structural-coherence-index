#!/usr/bin/env python3
"""
SCI EEG Full Re-Run v2
======================

Purpose
-------
Updated version of the CHB-MIT EEG seizure validation script.

Modern SCI claim:
  SCI measures excess Hilbert-envelope temporal coherence relative to a
  phase-randomized surrogate baseline that preserves the power spectrum.

Main update from older sci_eeg.py
---------------------------------
The old script saved mostly the final SCI score. This version saves the full
diagnostic stack for every segment:

  c_obs
  c_surr_mean
  c_surr_std
  gap = c_obs - c_surr_mean
  z
  sci
  bucket

It also adds:
  - per-subject summary
  - per-file summary
  - bucket summary
  - seizure vs non-seizure summary
  - optional simple ROC/AUC if scikit-learn is installed
  - plots
  - clearer report language

Dataset
-------
Designed for the CHB-MIT Scalp EEG Database from PhysioNet.

Expected folder layout examples:
  chbmit/chb01/chb01_03.edf
  chbmit/chb01/chb01-summary.txt
  chbmit/chb02/chb02_16.edf
  chbmit/chb02/chb02-summary.txt

Usage
-----
From your EEG project folder:

  python3 sci_eeg_full_rerun_v2.py --data ./chbmit --out results_eeg_v2

Fast smoke test:

  python3 sci_eeg_full_rerun_v2.py \
    --data ./chbmit \
    --subjects chb01 \
    --max-files 4 \
    --fast \
    --out test_eeg_v2

More channels:

  python3 sci_eeg_full_rerun_v2.py \
    --data ./chbmit \
    --subjects chb01,chb02 \
    --aggregate-channels 4 \
    --out results_eeg_v2_ch4

Outputs
-------
results_eeg_v2/
  segment_metrics.csv
  bucket_stats.csv
  subject_summary.csv
  file_summary.csv
  label_summary.csv
  eeg_score_tests.csv
  summary_report.txt
  plots/

Dependencies
------------
  pip3 install numpy pandas scipy mne matplotlib
Optional:
  pip3 install scikit-learn
"""

from __future__ import annotations

import argparse
import os
import re
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

try:
    import mne
    HAS_MNE = True
except Exception:
    HAS_MNE = False

try:
    from scipy.signal import hilbert, butter, filtfilt
    from scipy.stats import ttest_ind, mannwhitneyu
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
    from sklearn.metrics import roc_auc_score, average_precision_score
    HAS_SKLEARN = True
except Exception:
    HAS_SKLEARN = False


# =============================================================================
# CONFIG
# =============================================================================

@dataclass
class SCIConfig:
    # EEG segmentation
    target_fs: int = 256
    segment_sec: float = 10.0
    step_sec: float = 5.0
    channel: Optional[str] = None
    aggregate_channels: int = 1

    # EEG filtering
    bandpass_low: float = 0.5
    bandpass_high: float = 40.0

    # SCI operator
    envelope_smooth_sec: float = 0.10
    acf_lag_sec: float = 0.10
    n_surrogates: int = 60
    rng_seed: int = 1337
    z_clip: float = 6.0
    logistic_k: float = 0.9

    # Buckets
    sci_core_thresh: float = 0.75
    sci_mid_thresh: float = 0.65
    sci_tactical_thresh: float = 0.55

    # Runtime
    max_segments_per_file: Optional[int] = None


# =============================================================================
# CORE SCI MATH
# =============================================================================

def bandpass_filter(signal: np.ndarray, fs: int, low: float, high: float) -> np.ndarray:
    if not HAS_SCIPY:
        return signal

    x = np.asarray(signal, dtype=float)
    x = x[np.isfinite(x)]
    if len(x) < 20:
        return x

    nyq = fs / 2.0
    high = min(high, nyq * 0.95)
    low = max(low, 0.01)

    if low >= high:
        return x

    b, a = butter(4, [low / nyq, high / nyq], btype="band")
    return filtfilt(b, a, x)


def moving_average(x: np.ndarray, w: int) -> np.ndarray:
    if w <= 1 or len(x) < w:
        return x
    return np.convolve(x, np.ones(w) / w, mode="same")


def hilbert_envelope(x: np.ndarray, smooth: int) -> np.ndarray:
    if not HAS_SCIPY:
        return np.abs(x)
    env = np.abs(hilbert(x))
    return moving_average(env, smooth)


def acf_coherence(env: np.ndarray, max_lag: int) -> float:
    env = np.asarray(env, dtype=float)
    env = env[np.isfinite(env)]

    if len(env) < max_lag + 5:
        return np.nan

    y = env - np.mean(env)
    denom = np.sum(y * y)

    if denom <= 1e-12:
        return np.nan

    vals = []
    max_lag = min(max_lag, len(y) - 2)
    for lag in range(1, max_lag + 1):
        vals.append(np.sum(y[:-lag] * y[lag:]) / denom)

    return float(np.mean(vals)) if vals else np.nan


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

    return np.asarray(np.fft.irfft(mag * np.exp(1j * phases), n=n), dtype=float)


def empty_metrics() -> Dict[str, float]:
    return {
        "c_obs": np.nan,
        "c_surr_mean": np.nan,
        "c_surr_std": np.nan,
        "gap": np.nan,
        "z": np.nan,
        "sci": np.nan,
    }


def sci_metrics_for_channel(segment: np.ndarray, fs: int, cfg: SCIConfig, rng: np.random.Generator) -> Dict[str, float]:
    x = np.asarray(segment, dtype=float)
    x = x[np.isfinite(x)]

    if len(x) < int(fs * 2) or np.std(x) <= 1e-12:
        return empty_metrics()

    try:
        xf = bandpass_filter(x, fs, cfg.bandpass_low, cfg.bandpass_high)
        xf = xf - np.mean(xf)

        smooth = max(2, int(cfg.envelope_smooth_sec * fs))
        max_lag = max(2, int(cfg.acf_lag_sec * fs))

        env = hilbert_envelope(xf, smooth)
        c_obs = acf_coherence(env, max_lag)
    except Exception:
        return empty_metrics()

    if not np.isfinite(c_obs):
        return empty_metrics()

    surr = []
    for _ in range(cfg.n_surrogates):
        try:
            xs = phase_randomized_surrogate(xf, rng)
            es = hilbert_envelope(xs, smooth)
            cs = acf_coherence(es, max_lag)
            if np.isfinite(cs):
                surr.append(cs)
        except Exception:
            continue

    if len(surr) < max(8, cfg.n_surrogates // 3):
        return empty_metrics()

    mu = float(np.mean(surr))
    sd = float(np.std(surr, ddof=1))
    gap = float(c_obs - mu)

    if sd <= 1e-12:
        z = np.nan
        sci = np.nan
    else:
        z = float(np.clip(gap / sd, -cfg.z_clip, cfg.z_clip))
        sci = float(np.clip(1.0 / (1.0 + np.exp(-cfg.logistic_k * z)), 0.0, 1.0))

    return {
        "c_obs": float(c_obs),
        "c_surr_mean": mu,
        "c_surr_std": sd,
        "gap": gap,
        "z": z,
        "sci": sci,
    }


def aggregate_channel_metrics(metrics: List[Dict[str, float]]) -> Dict[str, float]:
    if not metrics:
        return empty_metrics()

    df = pd.DataFrame(metrics)
    out = {}
    for col in ["c_obs", "c_surr_mean", "c_surr_std", "gap", "z", "sci"]:
        vals = pd.to_numeric(df[col], errors="coerce").dropna()
        out[col] = float(vals.mean()) if len(vals) else np.nan
    return out


def classify_sci(score: float, cfg: SCIConfig) -> str:
    if not np.isfinite(score):
        return "INELIGIBLE"
    if score >= cfg.sci_core_thresh:
        return "CORE"
    if score >= cfg.sci_mid_thresh:
        return "CORE_MID"
    if score >= cfg.sci_tactical_thresh:
        return "TACTICAL"
    return "INELIGIBLE"


# =============================================================================
# CHB-MIT PARSING
# =============================================================================

def discover_subject_dirs(data_dir: str, subjects: Optional[List[str]]) -> List[str]:
    root = Path(data_dir).expanduser()
    if subjects:
        return [str(root / s) for s in subjects]

    dirs = []
    for p in sorted(root.iterdir()):
        if p.is_dir() and re.match(r"chb\d+", p.name.lower()):
            dirs.append(str(p))
    return dirs


def parse_summary_file(summary_path: str) -> Dict[str, List[Tuple[int, int]]]:
    """
    Parse CHB-MIT *-summary.txt into:
      {edf_filename: [(start_sec, end_sec), ...]}
    """
    seizures: Dict[str, List[Tuple[int, int]]] = {}

    if not os.path.exists(summary_path):
        return seizures

    current_file: Optional[str] = None
    pending_start: Optional[int] = None

    with open(summary_path, "r", errors="ignore") as f:
        for raw in f:
            line = raw.strip()

            m_file = re.search(r"File Name:\s*(\S+)", line, flags=re.I)
            if m_file:
                current_file = m_file.group(1)
                seizures.setdefault(current_file, [])
                pending_start = None
                continue

            m_start = re.search(r"Seizure(?:\s+\d+)?\s+Start Time:\s*(\d+)\s*seconds", line, flags=re.I)
            if m_start and current_file:
                pending_start = int(m_start.group(1))
                continue

            m_end = re.search(r"Seizure(?:\s+\d+)?\s+End Time:\s*(\d+)\s*seconds", line, flags=re.I)
            if m_end and current_file and pending_start is not None:
                end = int(m_end.group(1))
                if end > pending_start:
                    seizures.setdefault(current_file, []).append((pending_start, end))
                pending_start = None

    return seizures


def choose_channels(raw: "mne.io.BaseRaw", cfg: SCIConfig) -> List[int]:
    names = raw.ch_names

    if cfg.channel:
        target = cfg.channel.lower()
        for i, ch in enumerate(names):
            if ch.lower() == target:
                return [i]
        for i, ch in enumerate(names):
            if target in ch.lower():
                return [i]
        print(f"  [WARN] channel '{cfg.channel}' not found; using first EEG channels")

    picks = []
    for i, ch in enumerate(names):
        cl = ch.lower()
        if any(bad in cl for bad in ["ecg", "ekg", "vns", "mark", "status", "event"]):
            continue
        picks.append(i)

    if not picks:
        picks = list(range(min(len(names), cfg.aggregate_channels)))

    return picks[:max(1, cfg.aggregate_channels)]


def segment_overlaps_seizure(start_sec: float, end_sec: float, intervals: List[Tuple[int, int]]) -> bool:
    return any(start_sec < e and end_sec > s for s, e in intervals)


def process_edf(edf_path: str, seizure_intervals: List[Tuple[int, int]], cfg: SCIConfig,
                rng: np.random.Generator) -> List[Dict]:
    results: List[Dict] = []

    if not HAS_MNE:
        return results

    try:
        raw = mne.io.read_raw_edf(edf_path, preload=True, verbose=False)
    except Exception as e:
        print(f"load error: {e}")
        return results

    try:
        if abs(float(raw.info["sfreq"]) - cfg.target_fs) > 1:
            raw.resample(cfg.target_fs, npad="auto", verbose=False)
    except Exception:
        pass

    fs = int(round(float(raw.info["sfreq"])))
    picks = choose_channels(raw, cfg)
    data = raw.get_data(picks=picks)
    n_samples = data.shape[1]

    seg_len = int(cfg.segment_sec * fs)
    step = int(cfg.step_sec * fs)

    if n_samples < seg_len:
        return results

    subject = os.path.basename(os.path.dirname(edf_path))
    file_name = os.path.basename(edf_path)
    channel_names = [raw.ch_names[i] for i in picks]

    n_done = 0

    for start in range(0, n_samples - seg_len + 1, step):
        if cfg.max_segments_per_file is not None and n_done >= cfg.max_segments_per_file:
            break

        end = start + seg_len
        start_sec = start / fs
        end_sec = end / fs

        channel_metrics = []
        for ci in range(data.shape[0]):
            segment = data[ci, start:end]
            if np.sum(~np.isfinite(segment)) > seg_len * 0.1:
                continue
            if np.std(segment) < 1e-12:
                continue

            local_rng = np.random.default_rng(rng.integers(0, 2**32 - 1))
            m = sci_metrics_for_channel(segment, fs, cfg, local_rng)
            if np.isfinite(m["sci"]):
                channel_metrics.append(m)

        if not channel_metrics:
            continue

        m = aggregate_channel_metrics(channel_metrics)
        bucket = classify_sci(m["sci"], cfg)
        is_seizure = segment_overlaps_seizure(start_sec, end_sec, seizure_intervals)

        results.append({
            "subject": subject,
            "file": file_name,
            "channels_used": ",".join(channel_names),
            "n_channels_used": len(channel_metrics),
            "start_sec": round(start_sec, 2),
            "end_sec": round(end_sec, 2),
            **m,
            "bucket": bucket,
            "clinical_label": "seizure" if is_seizure else "non_seizure",
            "is_seizure": 1 if is_seizure else 0,
            "is_non_seizure": 0 if is_seizure else 1,
        })

        n_done += 1

    return results


def process_subject(subject_dir: str, cfg: SCIConfig, rng: np.random.Generator,
                    max_files_remaining: Optional[int] = None) -> List[Dict]:
    subject = os.path.basename(subject_dir)

    summary_candidates = [
        os.path.join(subject_dir, f"{subject}-summary.txt"),
        os.path.join(subject_dir, f"{subject.lower()}-summary.txt"),
        os.path.join(subject_dir, "summary.txt"),
    ]
    summary_path = next((p for p in summary_candidates if os.path.exists(p)), "")
    seizure_map = parse_summary_file(summary_path) if summary_path else {}

    edfs = sorted([f for f in os.listdir(subject_dir) if f.lower().endswith(".edf")])
    if max_files_remaining is not None:
        edfs = edfs[:max_files_remaining]

    results: List[Dict] = []

    for edf in edfs:
        intervals = seizure_map.get(edf, [])
        path = os.path.join(subject_dir, edf)

        print(f"    {edf:<18} seizures:{len(intervals)} ... ", end="")
        r = process_edf(path, intervals, cfg, rng)
        results.extend(r)

        if r:
            nsz = sum(x["is_seizure"] for x in r)
            print(f"{len(r)} segments, seizure segs:{nsz}")
        else:
            print("0 segments")

    return results


# =============================================================================
# STATS / REPORTING
# =============================================================================

def bucket_stats(df: pd.DataFrame) -> pd.DataFrame:
    rows = []

    for bucket in ["CORE", "CORE_MID", "TACTICAL", "INELIGIBLE", "ALL"]:
        sub = df if bucket == "ALL" else df[df["bucket"] == bucket]
        if len(sub) == 0:
            continue

        rows.append({
            "bucket": bucket,
            "n": len(sub),
            "pct_seizure": sub["is_seizure"].mean() * 100.0,
            "pct_non_seizure": sub["is_non_seizure"].mean() * 100.0,
            "mean_sci": sub["sci"].mean(),
            "median_sci": sub["sci"].median(),
            "mean_gap": sub["gap"].mean(),
            "median_gap": sub["gap"].median(),
            "mean_z": sub["z"].mean(),
            "pct_positive_gap": sub["gap"].gt(0).mean() * 100.0,
        })

    return pd.DataFrame(rows)


def label_summary(df: pd.DataFrame) -> pd.DataFrame:
    return (
        df.groupby("clinical_label")
        .agg(
            n=("sci", "size"),
            mean_sci=("sci", "mean"),
            median_sci=("sci", "median"),
            mean_gap=("gap", "mean"),
            median_gap=("gap", "median"),
            mean_z=("z", "mean"),
            pct_positive_gap=("gap", lambda s: s.gt(0).mean() * 100.0),
            pct_core=("bucket", lambda s: s.eq("CORE").mean() * 100.0),
        )
        .reset_index()
    )


def subject_summary(df: pd.DataFrame) -> pd.DataFrame:
    return (
        df.groupby("subject")
        .agg(
            n_segments=("sci", "size"),
            n_files=("file", "nunique"),
            seizure_segments=("is_seizure", "sum"),
            pct_seizure=("is_seizure", lambda s: s.mean() * 100.0),
            mean_sci=("sci", "mean"),
            mean_gap=("gap", "mean"),
            pct_positive_gap=("gap", lambda s: s.gt(0).mean() * 100.0),
            pct_core=("bucket", lambda s: s.eq("CORE").mean() * 100.0),
        )
        .reset_index()
    )


def file_summary(df: pd.DataFrame) -> pd.DataFrame:
    return (
        df.groupby(["subject", "file"])
        .agg(
            n_segments=("sci", "size"),
            seizure_segments=("is_seizure", "sum"),
            pct_seizure=("is_seizure", lambda s: s.mean() * 100.0),
            mean_sci=("sci", "mean"),
            mean_gap=("gap", "mean"),
            median_gap=("gap", "median"),
            mean_z=("z", "mean"),
            pct_positive_gap=("gap", lambda s: s.gt(0).mean() * 100.0),
            pct_core=("bucket", lambda s: s.eq("CORE").mean() * 100.0),
        )
        .reset_index()
    )


def score_tests(df: pd.DataFrame) -> pd.DataFrame:
    tests = []
    metrics = ["sci", "gap", "z", "c_obs", "c_surr_mean"]

    seizure = df[df["is_seizure"] == 1]
    non = df[df["is_seizure"] == 0]

    for metric in metrics:
        a = seizure[metric].dropna()
        b = non[metric].dropna()

        row = {
            "metric": metric,
            "contrast": "seizure vs non_seizure",
            "mean_seizure": a.mean() if len(a) else np.nan,
            "mean_non_seizure": b.mean() if len(b) else np.nan,
            "diff": (a.mean() - b.mean()) if len(a) and len(b) else np.nan,
            "n_seizure": len(a),
            "n_non_seizure": len(b),
            "welch_p": np.nan,
            "mannwhitney_p": np.nan,
            "note": "",
        }

        if len(a) >= 2 and len(b) >= 2:
            try:
                row["welch_p"] = ttest_ind(a, b, equal_var=False).pvalue
                row["mannwhitney_p"] = mannwhitneyu(a, b, alternative="two-sided").pvalue
            except Exception as e:
                row["note"] = str(e)
        else:
            row["note"] = "too few seizure or non-seizure segments"

        tests.append(row)

    return pd.DataFrame(tests)


def auc_summary(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    if not HAS_SKLEARN:
        return pd.DataFrame([{"score": "NA", "roc_auc": np.nan, "average_precision": np.nan, "note": "scikit-learn not installed"}])

    y = df["is_seizure"].values.astype(int)

    for score in ["sci", "gap", "z", "c_obs"]:
        sub = df[[score, "is_seizure"]].dropna()
        if sub["is_seizure"].nunique() < 2:
            rows.append({"score": score, "roc_auc": np.nan, "average_precision": np.nan, "note": "only one class present"})
            continue

        try:
            rows.append({
                "score": score,
                "roc_auc": roc_auc_score(sub["is_seizure"].values, sub[score].values),
                "average_precision": average_precision_score(sub["is_seizure"].values, sub[score].values),
                "note": "",
            })
        except Exception as e:
            rows.append({"score": score, "roc_auc": np.nan, "average_precision": np.nan, "note": str(e)})

    return pd.DataFrame(rows)


def make_plots(df: pd.DataFrame, bstats: pd.DataFrame, out_dir: Path):
    if not HAS_MPL or df.empty:
        return

    plot_dir = out_dir / "plots"
    plot_dir.mkdir(exist_ok=True)

    # Gap by clinical label
    labels = []
    data = []
    for label, sub in df.groupby("clinical_label"):
        vals = sub["gap"].dropna().values
        if len(vals):
            labels.append(label)
            data.append(vals)

    if data:
        plt.figure(figsize=(8, 5))
        plt.boxplot(data, labels=labels, showfliers=False)
        plt.ylabel("gap = c_obs - mean(c_surr)")
        plt.title("EEG SCI gap by clinical label")
        plt.tight_layout()
        plt.savefig(plot_dir / "gap_by_clinical_label.png", dpi=200)
        plt.close()

    # SCI by clinical label
    labels = []
    data = []
    for label, sub in df.groupby("clinical_label"):
        vals = sub["sci"].dropna().values
        if len(vals):
            labels.append(label)
            data.append(vals)

    if data:
        plt.figure(figsize=(8, 5))
        plt.boxplot(data, labels=labels, showfliers=False)
        plt.ylabel("SCI")
        plt.title("EEG SCI by clinical label")
        plt.tight_layout()
        plt.savefig(plot_dir / "sci_by_clinical_label.png", dpi=200)
        plt.close()

    # Percent seizure by bucket
    if not bstats.empty:
        sub = bstats[bstats["bucket"] != "ALL"].copy()
        if not sub.empty:
            plt.figure(figsize=(9, 5))
            plt.bar(sub["bucket"], sub["pct_seizure"])
            plt.ylabel("% seizure segments")
            plt.title("Seizure enrichment by SCI bucket")
            plt.tight_layout()
            plt.savefig(plot_dir / "pct_seizure_by_bucket.png", dpi=200)
            plt.close()

    # Time examples: first few files with seizures
    seizure_files = (
        df.groupby(["subject", "file"])["is_seizure"]
        .sum()
        .reset_index()
        .query("is_seizure > 0")
        .head(5)
    )

    for _, row in seizure_files.iterrows():
        sub = df[(df["subject"] == row["subject"]) & (df["file"] == row["file"])].sort_values("start_sec")
        if len(sub) < 2:
            continue

        plt.figure(figsize=(12, 5))
        plt.plot(sub["start_sec"], sub["gap"], label="gap")
        seizure_mask = sub["is_seizure"].astype(bool).values
        if seizure_mask.any():
            plt.scatter(sub.loc[seizure_mask, "start_sec"], sub.loc[seizure_mask, "gap"], marker="o", label="seizure segment")
        plt.xlabel("Start time (sec)")
        plt.ylabel("Gap")
        plt.title(f"{row['subject']} {row['file']}: SCI gap over time")
        plt.legend()
        plt.tight_layout()
        safe = f"{row['subject']}_{row['file']}".replace(".", "_")
        plt.savefig(plot_dir / f"{safe}_gap_timeline.png", dpi=200)
        plt.close()


def write_report(df: pd.DataFrame, bstats: pd.DataFrame, lsum: pd.DataFrame,
                 tests: pd.DataFrame, aucs: pd.DataFrame, out_dir: Path,
                 n_subjects: int, n_files: int, cfg: SCIConfig):
    lines = []
    lines.append("=" * 88)
    lines.append("SCI EEG FULL RE-RUN V2")
    lines.append("CHB-MIT Scalp EEG Seizure Database")
    lines.append("=" * 88)
    lines.append("")
    lines.append("Modern claim tested:")
    lines.append("  SCI measures excess Hilbert-envelope temporal coherence relative to")
    lines.append("  a phase-randomized surrogate baseline that preserves the power spectrum.")
    lines.append("")
    lines.append("Parameters:")
    lines.append(f"  target_fs={cfg.target_fs}")
    lines.append(f"  segment_sec={cfg.segment_sec}")
    lines.append(f"  step_sec={cfg.step_sec}")
    lines.append(f"  bandpass={cfg.bandpass_low}-{cfg.bandpass_high} Hz")
    lines.append(f"  envelope_smooth_sec={cfg.envelope_smooth_sec}")
    lines.append(f"  acf_lag_sec={cfg.acf_lag_sec}")
    lines.append(f"  surrogates={cfg.n_surrogates}")
    lines.append(f"  aggregate_channels={cfg.aggregate_channels}")
    lines.append(f"  channel={cfg.channel or 'auto/first EEG channels'}")
    lines.append("")
    lines.append(f"Subjects analyzed: {n_subjects}")
    lines.append(f"EDF files analyzed: {n_files}")
    lines.append(f"Total segments: {len(df):,}")
    lines.append(f"Seizure segments: {int(df['is_seizure'].sum()):,}")
    lines.append(f"Non-seizure segments: {int(df['is_non_seizure'].sum()):,}")
    lines.append("")

    lines.append("-" * 88)
    lines.append("SUMMARY BY CLINICAL LABEL")
    lines.append("-" * 88)
    if not lsum.empty:
        lines.append(lsum.to_string(index=False, float_format=lambda x: f"{x:0.4f}"))
    lines.append("")

    lines.append("-" * 88)
    lines.append("SUMMARY BY SCI BUCKET")
    lines.append("-" * 88)
    if not bstats.empty:
        lines.append(bstats.to_string(index=False, float_format=lambda x: f"{x:0.4f}"))
    lines.append("")

    lines.append("-" * 88)
    lines.append("SEIZURE VS NON-SEIZURE SCORE TESTS")
    lines.append("-" * 88)
    if not tests.empty:
        lines.append(tests.to_string(index=False, float_format=lambda x: f"{x:0.6f}"))
    lines.append("")

    lines.append("-" * 88)
    lines.append("AUC SUMMARY")
    lines.append("-" * 88)
    if not aucs.empty:
        lines.append(aucs.to_string(index=False, float_format=lambda x: f"{x:0.6f}"))
    lines.append("")

    lines.append("-" * 88)
    lines.append("INTERPRETATION GUIDE")
    lines.append("-" * 88)
    lines.append("  c_obs        = envelope autocorrelation of the filtered EEG segment")
    lines.append("  c_surr_mean  = mean envelope autocorrelation of phase-randomized surrogates")
    lines.append("  gap          = c_obs - c_surr_mean")
    lines.append("  z            = gap / c_surr_std")
    lines.append("  SCI          = logistic(z), clipped before mapping")
    lines.append("")
    lines.append("A positive result would mean seizure-labeled segments show higher gap/SCI")
    lines.append("than non-seizure segments, or that CORE buckets are enriched for seizure")
    lines.append("segments. This is a screening/structure result, not a medical diagnostic.")
    lines.append("")
    lines.append("Required before any strong claim:")
    lines.append("  - more subjects")
    lines.append("  - patient-level train/test split")
    lines.append("  - channel sensitivity checks")
    lines.append("  - false-positive rate per hour")
    lines.append("  - comparison to conventional EEG seizure features")
    lines.append("  - clinical review")
    lines.append("")

    path = out_dir / "summary_report.txt"
    path.write_text("\n".join(lines))
    print("\n" + "\n".join(lines))
    print(f"\nSaved report: {path}")


# =============================================================================
# CLI
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="SCI EEG Full Re-Run v2 — CHB-MIT")
    parser.add_argument("--data", required=True, help="Path to CHB-MIT root folder containing chb01, chb02, ...")
    parser.add_argument("--out", default="results_eeg_v2", help="Output directory")
    parser.add_argument("--subjects", default=None, help="Comma-separated subjects, e.g. chb01,chb02")
    parser.add_argument("--max-subjects", type=int, default=None)
    parser.add_argument("--max-files", type=int, default=None)
    parser.add_argument("--max-segments-per-file", type=int, default=None)

    parser.add_argument("--channel", default=None, help="Specific EEG channel name, e.g. FP1-F7")
    parser.add_argument("--aggregate-channels", type=int, default=1)

    parser.add_argument("--target-fs", type=int, default=256)
    parser.add_argument("--segment-sec", type=float, default=10.0)
    parser.add_argument("--step-sec", type=float, default=5.0)
    parser.add_argument("--bandpass-low", type=float, default=0.5)
    parser.add_argument("--bandpass-high", type=float, default=40.0)
    parser.add_argument("--smooth-sec", type=float, default=0.10)
    parser.add_argument("--lag-sec", type=float, default=0.10)
    parser.add_argument("--surrogates", type=int, default=60)
    parser.add_argument("--fast", action="store_true", help="Fast mode: 20 surrogates and max 200 segments/file unless specified")

    args = parser.parse_args()

    if not HAS_MNE:
        print("ERROR: mne not installed. Run: pip3 install mne")
        return
    if not HAS_SCIPY:
        print("ERROR: scipy not installed. Run: pip3 install scipy")
        return

    cfg = SCIConfig(
        target_fs=args.target_fs,
        segment_sec=args.segment_sec,
        step_sec=args.step_sec,
        channel=args.channel,
        aggregate_channels=args.aggregate_channels,
        bandpass_low=args.bandpass_low,
        bandpass_high=args.bandpass_high,
        envelope_smooth_sec=args.smooth_sec,
        acf_lag_sec=args.lag_sec,
        n_surrogates=args.surrogates,
        max_segments_per_file=args.max_segments_per_file,
    )

    if args.fast:
        cfg.n_surrogates = 20
        if cfg.max_segments_per_file is None:
            cfg.max_segments_per_file = 200
        print("[FAST MODE] surrogates=20, max_segments_per_file=200")

    out_dir = Path(args.out).expanduser()
    out_dir.mkdir(parents=True, exist_ok=True)

    subjects = [s.strip() for s in args.subjects.split(",")] if args.subjects else None
    subject_dirs = discover_subject_dirs(args.data, subjects)

    if args.max_subjects is not None:
        subject_dirs = subject_dirs[:args.max_subjects]

    rng = np.random.default_rng(cfg.rng_seed)
    all_results: List[Dict] = []
    files_processed = 0

    print("\nSCI EEG Full Re-Run v2 — CHB-MIT")
    print(f"Subjects: {len(subject_dirs)} | Segment: {cfg.segment_sec}s | Step: {cfg.step_sec}s")
    print(f"Surrogates: {cfg.n_surrogates} | Bandpass: {cfg.bandpass_low}-{cfg.bandpass_high} Hz")
    print(f"Channel: {cfg.channel or 'auto/first EEG channel(s)'} | Aggregate channels: {cfg.aggregate_channels}")
    print("-" * 80)

    for si, subject_dir in enumerate(subject_dirs, 1):
        if args.max_files is not None and files_processed >= args.max_files:
            break

        if not os.path.isdir(subject_dir):
            print(f"[WARN] Missing subject dir: {subject_dir}")
            continue

        subject = os.path.basename(subject_dir)
        edfs_total = len([f for f in os.listdir(subject_dir) if f.lower().endswith(".edf")])
        remaining = None if args.max_files is None else max(0, args.max_files - files_processed)
        files_to_process = edfs_total if remaining is None else min(edfs_total, remaining)
        files_processed += files_to_process

        print(f"[{si}/{len(subject_dirs)}] Subject {subject} — processing {files_to_process}/{edfs_total} EDF files")
        results = process_subject(subject_dir, cfg, rng, max_files_remaining=files_to_process)
        all_results.extend(results)

    if not all_results:
        print("\nNo results generated. Check data path, dependencies, and CHB-MIT folder structure.")
        return

    df = pd.DataFrame(all_results)
    df.to_csv(out_dir / "segment_metrics.csv", index=False)
    print(f"\nSaved {len(df):,} segments → {out_dir / 'segment_metrics.csv'}")

    bstats = bucket_stats(df)
    lsum = label_summary(df)
    ssum = subject_summary(df)
    fsum = file_summary(df)
    tests = score_tests(df)
    aucs = auc_summary(df)

    bstats.to_csv(out_dir / "bucket_stats.csv", index=False)
    lsum.to_csv(out_dir / "label_summary.csv", index=False)
    ssum.to_csv(out_dir / "subject_summary.csv", index=False)
    fsum.to_csv(out_dir / "file_summary.csv", index=False)
    tests.to_csv(out_dir / "eeg_score_tests.csv", index=False)
    aucs.to_csv(out_dir / "auc_summary.csv", index=False)

    make_plots(df, bstats, out_dir)

    write_report(
        df=df,
        bstats=bstats,
        lsum=lsum,
        tests=tests,
        aucs=aucs,
        out_dir=out_dir,
        n_subjects=df["subject"].nunique(),
        n_files=df["file"].nunique(),
        cfg=cfg,
    )

    print("\nDone.")
    print(f"Open this first: {out_dir / 'summary_report.txt'}")
    print(f"Plots: {out_dir / 'plots'}")


if __name__ == "__main__":
    main()
