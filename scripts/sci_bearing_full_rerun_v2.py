#!/usr/bin/env python3
"""
SCI Bearing Full Re-Run v2
==========================

Purpose
-------
Re-runs the current SCI bearing validation in a more modern way than the older
sci_bearing.py script.

Main update:
  The script no longer saves only the final SCI score. It saves the full modern
  diagnostic stack:

      c_obs
      c_surr_mean
      c_surr_std
      gap = c_obs - c_surr_mean
      z
      sci
      bucket

This matches the newer claim:

  SCI measures excess Hilbert-envelope temporal coherence relative to a
  phase-randomized surrogate baseline that preserves the power spectrum.

Supported datasets
------------------
1. CWRU seeded faults, .mat files
2. MFPT seeded faults, flexible .mat/.csv/.txt numeric scan
3. IMS run-to-failure, flexible text/csv numeric scan
4. FEMTO / PRONOSTIA run-to-failure, flexible csv/txt numeric scan
5. AUTO mode for any folder of numeric files

This script is intentionally tolerant of different folder layouts. It attempts
to infer labels from file/folder names. You should review labels after the run.

Example usage
-------------
From your SCI SENSE folder:

  python3 sci_bearing_full_rerun_v2.py \
    --cwru cwru_data \
    --mfpt mfpt_data \
    --ims ims_data \
    --femto ieee-phm-2012-data-challenge-dataset-master \
    --out results_bearing_full_v2

Fast smoke test:

  python3 sci_bearing_full_rerun_v2.py --cwru cwru_data --out test_v2 --fast --max-files 10

Outputs
-------
results_bearing_full_v2/
  all_segment_metrics.csv
  summary_by_dataset_label.csv
  summary_by_dataset_fault_type.csv
  ims_femto_degradation_timeseries.csv
  summary_report.txt
  plots/*.png

Dependencies
------------
  pip3 install numpy pandas scipy matplotlib
"""

from __future__ import annotations

import argparse
import os
import re
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Dict, List, Tuple

import numpy as np
import pandas as pd
from scipy.signal import butter, filtfilt, hilbert
from scipy.io import loadmat

warnings.filterwarnings("ignore")

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    HAS_MPL = True
except Exception:
    HAS_MPL = False


# =============================================================================
# CONFIG
# =============================================================================

@dataclass
class SCIConfig:
    fs: int = 12000
    segment_sec: float = 0.5
    step_sec: float = 0.5
    bandpass_low: float = 500.0
    bandpass_high: float = 5000.0
    envelope_smooth: int = 12
    acf_max_lag: int = 12
    n_surrogates: int = 40
    z_clip: float = 6.0
    logistic_k: float = 0.9
    core_thresh: float = 0.75
    mid_thresh: float = 0.65
    tactical_thresh: float = 0.55
    seed: int = 42
    max_segments_per_file: Optional[int] = None


CWRU_FILES = {
    # Healthy baseline
    "97.mat":  ("healthy", 0.000, "Healthy 0HP", 0),
    "98.mat":  ("healthy", 0.000, "Healthy 1HP", 1),
    "99.mat":  ("healthy", 0.000, "Healthy 2HP", 2),
    "100.mat": ("healthy", 0.000, "Healthy 3HP", 3),

    # Inner race
    "105.mat": ("inner_race", 0.007, "Inner 0.007 0HP", 0),
    "106.mat": ("inner_race", 0.007, "Inner 0.007 1HP", 1),
    "107.mat": ("inner_race", 0.007, "Inner 0.007 2HP", 2),
    "108.mat": ("inner_race", 0.007, "Inner 0.007 3HP", 3),
    "169.mat": ("inner_race", 0.014, "Inner 0.014 0HP", 0),
    "170.mat": ("inner_race", 0.014, "Inner 0.014 1HP", 1),
    "171.mat": ("inner_race", 0.014, "Inner 0.014 2HP", 2),
    "172.mat": ("inner_race", 0.014, "Inner 0.014 3HP", 3),
    "209.mat": ("inner_race", 0.021, "Inner 0.021 0HP", 0),
    "210.mat": ("inner_race", 0.021, "Inner 0.021 1HP", 1),
    "211.mat": ("inner_race", 0.021, "Inner 0.021 2HP", 2),
    "212.mat": ("inner_race", 0.021, "Inner 0.021 3HP", 3),

    # Outer race
    "130.mat": ("outer_race", 0.007, "Outer 0.007 0HP", 0),
    "131.mat": ("outer_race", 0.007, "Outer 0.007 1HP", 1),
    "132.mat": ("outer_race", 0.007, "Outer 0.007 2HP", 2),
    "133.mat": ("outer_race", 0.007, "Outer 0.007 3HP", 3),
    "197.mat": ("outer_race", 0.014, "Outer 0.014 0HP", 0),
    "198.mat": ("outer_race", 0.014, "Outer 0.014 1HP", 1),
    "199.mat": ("outer_race", 0.014, "Outer 0.014 2HP", 2),
    "200.mat": ("outer_race", 0.014, "Outer 0.014 3HP", 3),
    "234.mat": ("outer_race", 0.021, "Outer 0.021 0HP", 0),
    "235.mat": ("outer_race", 0.021, "Outer 0.021 1HP", 1),
    "236.mat": ("outer_race", 0.021, "Outer 0.021 2HP", 2),
    "237.mat": ("outer_race", 0.021, "Outer 0.021 3HP", 3),

    # Ball
    "118.mat": ("ball", 0.007, "Ball 0.007 0HP", 0),
    "119.mat": ("ball", 0.007, "Ball 0.007 1HP", 1),
    "120.mat": ("ball", 0.007, "Ball 0.007 2HP", 2),
    "185.mat": ("ball", 0.014, "Ball 0.014 0HP", 0),
    "186.mat": ("ball", 0.014, "Ball 0.014 1HP", 1),
    "187.mat": ("ball", 0.014, "Ball 0.014 2HP", 2),
    "222.mat": ("ball", 0.021, "Ball 0.021 0HP", 0),
    "223.mat": ("ball", 0.021, "Ball 0.021 1HP", 1),
    "224.mat": ("ball", 0.021, "Ball 0.021 2HP", 2),
}


# =============================================================================
# CORE SCI MATH
# =============================================================================

def safe_bandpass(x: np.ndarray, fs: int, low: float, high: float) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if len(x) < 50:
        return x

    nyq = fs / 2.0
    lo = max(low / nyq, 0.001)
    hi = min(high / nyq, 0.999)

    if hi <= lo:
        # Fall back to highpass-ish safe range if fs is lower than expected.
        lo = 0.01
        hi = 0.95

    b, a = butter(4, [lo, hi], btype="band")
    return filtfilt(b, a, x)


def moving_average(x: np.ndarray, w: int) -> np.ndarray:
    if w <= 1 or len(x) < w:
        return x
    return np.convolve(x, np.ones(w) / w, mode="same")


def envelope(x: np.ndarray, smooth: int) -> np.ndarray:
    return moving_average(np.abs(hilbert(x)), smooth)


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
    for lag in range(1, max_lag + 1):
        vals.append(np.sum(y[:-lag] * y[lag:]) / denom)

    return float(np.mean(vals))


def phase_randomized_surrogate(x: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    n = len(x)

    X = np.fft.rfft(x)
    mag = np.abs(X)

    phases = rng.uniform(0, 2 * np.pi, size=X.shape)
    phases[0] = 0.0
    if n % 2 == 0 and len(phases) > 1:
        phases[-1] = 0.0

    return np.fft.irfft(mag * np.exp(1j * phases), n=n)


def sci_metrics(segment: np.ndarray, cfg: SCIConfig, rng: np.random.Generator) -> Dict[str, float]:
    """
    Returns the full modern diagnostic stack:
      c_obs, c_surr_mean, c_surr_std, gap, z, sci
    """
    x = np.asarray(segment, dtype=float)
    x = x[np.isfinite(x)]
    if len(x) < cfg.acf_max_lag + 50 or np.std(x) <= 1e-12:
        return empty_metrics()

    try:
        xf = safe_bandpass(x, cfg.fs, cfg.bandpass_low, cfg.bandpass_high)
        env_obs = envelope(xf, cfg.envelope_smooth)
        c_obs = acf_coherence(env_obs, cfg.acf_max_lag)
    except Exception:
        return empty_metrics()

    if not np.isfinite(c_obs):
        return empty_metrics()

    sur_scores = []
    for _ in range(cfg.n_surrogates):
        try:
            xs = phase_randomized_surrogate(xf, rng)
            cs = acf_coherence(envelope(xs, cfg.envelope_smooth), cfg.acf_max_lag)
            if np.isfinite(cs):
                sur_scores.append(cs)
        except Exception:
            continue

    if len(sur_scores) < max(5, cfg.n_surrogates // 3):
        return empty_metrics()

    mu = float(np.mean(sur_scores))
    sd = float(np.std(sur_scores, ddof=1))
    gap = float(c_obs - mu)

    if sd <= 1e-12:
        z = np.nan
        sci = np.nan
    else:
        z = float(np.clip(gap / sd, -cfg.z_clip, cfg.z_clip))
        sci = float(1.0 / (1.0 + np.exp(-cfg.logistic_k * z)))

    return {
        "c_obs": float(c_obs),
        "c_surr_mean": mu,
        "c_surr_std": sd,
        "gap": gap,
        "z": z,
        "sci": sci,
    }


def empty_metrics() -> Dict[str, float]:
    return {
        "c_obs": np.nan,
        "c_surr_mean": np.nan,
        "c_surr_std": np.nan,
        "gap": np.nan,
        "z": np.nan,
        "sci": np.nan,
    }


def classify_sci(sci: float, cfg: SCIConfig) -> str:
    if not np.isfinite(sci):
        return "INELIGIBLE"
    if sci >= cfg.core_thresh:
        return "CORE"
    if sci >= cfg.mid_thresh:
        return "CORE_MID"
    if sci >= cfg.tactical_thresh:
        return "TACTICAL"
    return "INELIGIBLE"


# =============================================================================
# LOADERS
# =============================================================================

def load_mat_numeric_arrays(path: Path) -> List[Tuple[str, np.ndarray]]:
    out = []
    mat = loadmat(str(path))
    for key, val in mat.items():
        if key.startswith("_"):
            continue
        arr = np.asarray(val).squeeze()
        if arr.ndim == 1 and arr.size > 500:
            out.append((key, arr.astype(float)))
        elif arr.ndim == 2 and min(arr.shape) <= 16 and max(arr.shape) > 500:
            # Multi-channel matrix; each row/column can be a channel.
            if arr.shape[0] <= arr.shape[1]:
                for i in range(arr.shape[0]):
                    out.append((f"{key}_ch{i+1}", arr[i, :].astype(float)))
            else:
                for i in range(arr.shape[1]):
                    out.append((f"{key}_ch{i+1}", arr[:, i].astype(float)))
    return out


def load_text_or_csv_numeric_arrays(path: Path) -> List[Tuple[str, np.ndarray]]:
    """
    Flexible numeric loader.
    Handles comma, tab, semicolon, or whitespace files with one or many columns.
    """
    try:
        df = pd.read_csv(path, sep=None, engine="python", header=None, comment="#")
    except Exception:
        try:
            df = pd.read_csv(path, delim_whitespace=True, header=None, comment="#")
        except Exception:
            return []

    arrays = []
    for col in df.columns:
        s = pd.to_numeric(df[col], errors="coerce").dropna()
        if len(s) > 500 and np.std(s.values) > 1e-12:
            arrays.append((f"col{col}", s.values.astype(float)))

    # If no columns worked, try raw numpy.
    if not arrays:
        try:
            arr = np.loadtxt(path)
            arr = np.asarray(arr)
            if arr.ndim == 1 and arr.size > 500:
                arrays.append(("signal", arr.astype(float)))
            elif arr.ndim == 2:
                for j in range(arr.shape[1]):
                    col = arr[:, j]
                    if len(col) > 500 and np.std(col) > 1e-12:
                        arrays.append((f"col{j}", col.astype(float)))
        except Exception:
            pass

    return arrays


def load_numeric_arrays(path: Path) -> List[Tuple[str, np.ndarray]]:
    suffix = path.suffix.lower()
    if suffix == ".mat":
        return load_mat_numeric_arrays(path)
    if suffix in [".csv", ".txt", ".dat", ".asc"]:
        return load_text_or_csv_numeric_arrays(path)
    return []


def infer_fault_metadata(path: Path, dataset: str) -> Dict[str, object]:
    text = str(path).lower()
    name = path.name.lower()

    fault_type = "unknown"
    fault_size = np.nan
    load_hp = np.nan
    label = path.stem

    if dataset == "CWRU" and path.name in CWRU_FILES:
        ft, fs, lab, hp = CWRU_FILES[path.name]
        return {
            "fault_type": ft,
            "fault_size": fs,
            "load_hp": hp,
            "label": lab,
        }

    # General inference
    if any(k in text for k in ["healthy", "normal", "baseline", "good"]):
        fault_type = "healthy"
    elif any(k in text for k in ["inner", "ir", "bpfi"]):
        fault_type = "inner_race"
    elif any(k in text for k in ["outer", "or", "bpfo"]):
        fault_type = "outer_race"
    elif any(k in text for k in ["ball", "roller", "bsf"]):
        fault_type = "ball_or_roller"
    elif any(k in text for k in ["cage", "ftf"]):
        fault_type = "cage"
    elif any(k in text for k in ["fault", "fail", "defect"]):
        fault_type = "fault"

    # Fault size patterns like 0.007, 007, 7mil, etc.
    m = re.search(r"0\.(007|014|021|028|040)", text)
    if m:
        fault_size = float("0." + m.group(1))
    else:
        m2 = re.search(r"(?<!\d)(007|014|021|028|040)(?!\d)", text)
        if m2:
            fault_size = float("0." + m2.group(1))

    # Load HP
    hp = re.search(r"([0-3])\s*hp", text)
    if hp:
        load_hp = int(hp.group(1))

    return {
        "fault_type": fault_type,
        "fault_size": fault_size,
        "load_hp": load_hp,
        "label": label,
    }


# =============================================================================
# PROCESSING
# =============================================================================

def segment_signal(x: np.ndarray, cfg: SCIConfig):
    seg_len = int(cfg.segment_sec * cfg.fs)
    step = int(cfg.step_sec * cfg.fs)
    if seg_len <= 0 or step <= 0:
        raise ValueError("segment_sec and step_sec must produce positive sample counts.")

    n = len(x)
    count = 0
    for start in range(0, n - seg_len + 1, step):
        if cfg.max_segments_per_file is not None and count >= cfg.max_segments_per_file:
            break
        yield start, x[start:start + seg_len]
        count += 1


def process_one_file(path: Path, dataset: str, cfg: SCIConfig, rng: np.random.Generator) -> List[Dict[str, object]]:
    rows = []
    meta = infer_fault_metadata(path, dataset)
    arrays = load_numeric_arrays(path)

    if not arrays:
        return rows

    for channel_name, sig in arrays:
        sig = np.asarray(sig, dtype=float)
        sig = sig[np.isfinite(sig)]

        if len(sig) < int(cfg.segment_sec * cfg.fs) + 10:
            continue

        for start, seg in segment_signal(sig, cfg):
            m = sci_metrics(seg, cfg, rng)
            sci = m["sci"]
            row = {
                "dataset": dataset,
                "file": str(path),
                "filename": path.name,
                "channel": channel_name,
                "start_sample": start,
                "fault_type": meta["fault_type"],
                "fault_size": meta["fault_size"],
                "load_hp": meta["load_hp"],
                "label": meta["label"],
                **m,
                "bucket": classify_sci(sci, cfg),
            }
            rows.append(row)

    return rows


def collect_files(root: Optional[str], dataset: str, max_files: Optional[int] = None) -> List[Path]:
    if root is None:
        return []
    root_path = Path(root).expanduser()
    if not root_path.exists():
        print(f"[WARN] {dataset}: path does not exist: {root_path}")
        return []

    exts = {".mat", ".csv", ".txt", ".dat", ".asc"}
    files = [p for p in root_path.rglob("*") if p.is_file() and p.suffix.lower() in exts]

    # For CWRU, prefer known files if present.
    if dataset == "CWRU":
        known = []
        for fname in CWRU_FILES:
            p = root_path / fname
            if p.exists():
                known.append(p)
        if known:
            files = known

    files = sorted(files)
    if max_files is not None:
        files = files[:max_files]
    return files


def summarize(df: pd.DataFrame, out_dir: Path) -> Tuple[pd.DataFrame, pd.DataFrame]:
    if df.empty:
        return pd.DataFrame(), pd.DataFrame()

    def agg(g):
        n = len(g)
        return pd.Series({
            "n_segments": n,
            "mean_sci": g["sci"].mean(),
            "median_sci": g["sci"].median(),
            "mean_gap": g["gap"].mean(),
            "median_gap": g["gap"].median(),
            "mean_c_obs": g["c_obs"].mean(),
            "mean_c_surr": g["c_surr_mean"].mean(),
            "mean_z": g["z"].mean(),
            "pct_CORE": (g["bucket"].eq("CORE").mean() * 100.0),
            "pct_CORE_OR_MID": (g["bucket"].isin(["CORE", "CORE_MID"]).mean() * 100.0),
            "pct_positive_gap": (g["gap"].gt(0).mean() * 100.0),
        })

    by_label = (
        df.groupby(["dataset", "label", "fault_type", "fault_size", "load_hp"], dropna=False)
        .apply(agg)
        .reset_index()
        .sort_values(["dataset", "fault_type", "fault_size", "load_hp", "label"])
    )

    by_type = (
        df.groupby(["dataset", "fault_type"], dropna=False)
        .apply(agg)
        .reset_index()
        .sort_values(["dataset", "fault_type"])
    )

    by_label.to_csv(out_dir / "summary_by_dataset_label.csv", index=False)
    by_type.to_csv(out_dir / "summary_by_dataset_fault_type.csv", index=False)

    # Run-to-failure style file-level timeseries for IMS/FEMTO/general datasets.
    time_df = (
        df.groupby(["dataset", "filename", "file"], dropna=False)
        .agg(
            n_segments=("sci", "size"),
            mean_sci=("sci", "mean"),
            median_sci=("sci", "median"),
            mean_gap=("gap", "mean"),
            median_gap=("gap", "median"),
            pct_core=("bucket", lambda s: (s == "CORE").mean() * 100.0),
        )
        .reset_index()
        .sort_values(["dataset", "file"])
    )
    time_df.to_csv(out_dir / "ims_femto_degradation_timeseries.csv", index=False)

    return by_label, by_type


# =============================================================================
# REPORTING / PLOTS
# =============================================================================

def write_report(df: pd.DataFrame, by_label: pd.DataFrame, by_type: pd.DataFrame, out_dir: Path, cfg: SCIConfig):
    lines = []
    lines.append("=" * 88)
    lines.append("SCI BEARING FULL RE-RUN V2")
    lines.append("=" * 88)
    lines.append("")
    lines.append("Modern claim tested:")
    lines.append("  SCI measures excess Hilbert-envelope temporal coherence relative to")
    lines.append("  a phase-randomized surrogate baseline that preserves the power spectrum.")
    lines.append("")
    lines.append("Parameters:")
    lines.append(f"  fs={cfg.fs}, segment_sec={cfg.segment_sec}, step_sec={cfg.step_sec}")
    lines.append(f"  bandpass={cfg.bandpass_low}-{cfg.bandpass_high} Hz")
    lines.append(f"  envelope_smooth={cfg.envelope_smooth}, acf_max_lag={cfg.acf_max_lag}")
    lines.append(f"  surrogates={cfg.n_surrogates}, logistic_k={cfg.logistic_k}, seed={cfg.seed}")
    lines.append("")
    lines.append(f"Total segments analyzed: {len(df):,}")
    lines.append(f"Datasets: {', '.join(sorted(df['dataset'].dropna().unique())) if not df.empty else 'None'}")
    lines.append("")

    if not by_type.empty:
        lines.append("-" * 88)
        lines.append("SUMMARY BY DATASET / FAULT TYPE")
        lines.append("-" * 88)
        cols = ["dataset", "fault_type", "n_segments", "mean_sci", "mean_gap", "pct_CORE", "pct_positive_gap"]
        lines.append(by_type[cols].to_string(index=False, float_format=lambda x: f"{x:0.4f}"))

    lines.append("")
    lines.append("-" * 88)
    lines.append("INTERPRETATION GUIDE")
    lines.append("-" * 88)
    lines.append("  c_obs        = envelope autocorrelation of the real filtered signal")
    lines.append("  c_surr_mean  = mean envelope autocorrelation of phase-randomized surrogates")
    lines.append("  gap          = c_obs - c_surr_mean")
    lines.append("  z            = gap / c_surr_std")
    lines.append("  SCI          = logistic(z), clipped to +/-6 before mapping")
    lines.append("")
    lines.append("Positive result is no longer only 'monotonic fault severity'.")
    lines.append("The modern diagnostic is whether fault/degradation states show a")
    lines.append("positive gap and elevated SCI relative to healthy/baseline states.")
    lines.append("")
    lines.append("For CWRU, compare healthy vs race faults.")
    lines.append("For IMS/FEMTO, inspect ims_femto_degradation_timeseries.csv for trajectory.")
    lines.append("For MFPT, compare normal/baseline files vs fault files.")
    lines.append("")

    path = out_dir / "summary_report.txt"
    path.write_text("\n".join(lines))
    print("\n".join(lines))
    print(f"\nSaved report: {path}")


def make_plots(df: pd.DataFrame, by_label: pd.DataFrame, out_dir: Path):
    if not HAS_MPL or df.empty:
        return

    plot_dir = out_dir / "plots"
    plot_dir.mkdir(exist_ok=True)

    # Mean gap by label.
    if not by_label.empty:
        tmp = by_label.copy()
        tmp["display"] = tmp["dataset"].astype(str) + " | " + tmp["label"].astype(str)
        tmp = tmp.sort_values("mean_gap").tail(40)

        plt.figure(figsize=(12, max(6, 0.25 * len(tmp))))
        plt.barh(tmp["display"], tmp["mean_gap"])
        plt.xlabel("Mean gap: c_obs - mean(c_surr)")
        plt.title("Top mean SCI gaps by dataset/label")
        plt.tight_layout()
        plt.savefig(plot_dir / "top_mean_gap_by_label.png", dpi=200)
        plt.close()

    # SCI distribution by dataset/fault type.
    sub = df.dropna(subset=["sci"]).copy()
    if len(sub) > 0:
        groups = [g["sci"].values for _, g in sub.groupby(["dataset", "fault_type"])]
        labels = [f"{k[0]}-{k[1]}" for k, _ in sub.groupby(["dataset", "fault_type"])]

        if len(groups) <= 30:
            plt.figure(figsize=(max(10, 0.5 * len(groups)), 6))
            plt.boxplot(groups, labels=labels, showfliers=False)
            plt.xticks(rotation=75, ha="right")
            plt.ylabel("SCI")
            plt.title("SCI distribution by dataset/fault type")
            plt.tight_layout()
            plt.savefig(plot_dir / "sci_distribution_by_fault_type.png", dpi=200)
            plt.close()

    # Run-to-failure timeseries for IMS/FEMTO.
    ts_path = out_dir / "ims_femto_degradation_timeseries.csv"
    if ts_path.exists():
        ts = pd.read_csv(ts_path)
        for dataset in ts["dataset"].unique():
            if dataset not in ["IMS", "FEMTO"]:
                continue
            d = ts[ts["dataset"] == dataset].copy().reset_index(drop=True)
            if len(d) < 5:
                continue

            plt.figure(figsize=(12, 5))
            plt.plot(np.arange(len(d)), d["mean_gap"], marker=".", linewidth=1)
            plt.xlabel("File order")
            plt.ylabel("Mean gap")
            plt.title(f"{dataset}: run-to-failure / file-order mean gap")
            plt.tight_layout()
            plt.savefig(plot_dir / f"{dataset.lower()}_mean_gap_timeseries.png", dpi=200)
            plt.close()


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Full SCI bearing re-run with modern gap diagnostics.")
    parser.add_argument("--cwru", type=str, default=None, help="Path to CWRU .mat folder")
    parser.add_argument("--mfpt", type=str, default=None, help="Path to MFPT folder")
    parser.add_argument("--ims", type=str, default=None, help="Path to IMS folder")
    parser.add_argument("--femto", type=str, default=None, help="Path to FEMTO / PRONOSTIA folder")
    parser.add_argument("--auto", type=str, default=None, help="Path to any extra folder of numeric files")
    parser.add_argument("--out", type=str, default="results_bearing_full_v2")

    parser.add_argument("--fs", type=int, default=12000, help="Sampling rate. Use 12000 for CWRU, 20000 for IMS, etc.")
    parser.add_argument("--segment-sec", type=float, default=0.5)
    parser.add_argument("--step-sec", type=float, default=0.5)
    parser.add_argument("--bandpass-low", type=float, default=500.0)
    parser.add_argument("--bandpass-high", type=float, default=5000.0)
    parser.add_argument("--smooth", type=int, default=12)
    parser.add_argument("--lag", type=int, default=12)
    parser.add_argument("--surrogates", type=int, default=40)
    parser.add_argument("--fast", action="store_true", help="Use 15 surrogates and max 20 segments/file")
    parser.add_argument("--max-files", type=int, default=None)
    parser.add_argument("--max-segments-per-file", type=int, default=None)

    args = parser.parse_args()

    cfg = SCIConfig(
        fs=args.fs,
        segment_sec=args.segment_sec,
        step_sec=args.step_sec,
        bandpass_low=args.bandpass_low,
        bandpass_high=args.bandpass_high,
        envelope_smooth=args.smooth,
        acf_max_lag=args.lag,
        n_surrogates=args.surrogates,
        max_segments_per_file=args.max_segments_per_file,
    )

    if args.fast:
        cfg.n_surrogates = 15
        if cfg.max_segments_per_file is None:
            cfg.max_segments_per_file = 20
        print("[FAST MODE] surrogates=15, max_segments_per_file=20")

    out_dir = Path(args.out).expanduser()
    out_dir.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(cfg.seed)

    dataset_roots = [
        ("CWRU", args.cwru),
        ("MFPT", args.mfpt),
        ("IMS", args.ims),
        ("FEMTO", args.femto),
        ("AUTO", args.auto),
    ]

    all_rows = []

    for dataset, root in dataset_roots:
        files = collect_files(root, dataset, max_files=args.max_files)
        if not files:
            continue

        print(f"\n[{dataset}] Found {len(files)} files")
        for i, path in enumerate(files, 1):
            print(f"  {i}/{len(files)} {path.name}", end=" ... ")
            try:
                rows = process_one_file(path, dataset, cfg, rng)
                all_rows.extend(rows)
                print(f"{len(rows)} segments")
            except Exception as e:
                print(f"ERROR: {e}")

    if not all_rows:
        print("\nNo rows produced. Check folder paths, file formats, and sampling rate.")
        return

    df = pd.DataFrame(all_rows)
    df.to_csv(out_dir / "all_segment_metrics.csv", index=False)

    by_label, by_type = summarize(df, out_dir)
    write_report(df, by_label, by_type, out_dir, cfg)
    make_plots(df, by_label, out_dir)

    print("\nDone.")
    print(f"Main output: {out_dir / 'all_segment_metrics.csv'}")
    print(f"Summary:     {out_dir / 'summary_report.txt'}")


if __name__ == "__main__":
    main()
