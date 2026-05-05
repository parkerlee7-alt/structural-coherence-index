#!/usr/bin/env python3
"""
SCI Cells Full Re-Run v2
========================

Purpose
-------
Consolidates the scattered cellular SCI scripts into one cleaner rerun pipeline.

Modern SCI claim:
  SCI measures excess Hilbert-envelope temporal coherence relative to a
  phase-randomized surrogate baseline that preserves the power spectrum.

This script saves the full diagnostic stack:
  c_obs
  c_surr_mean
  c_surr_std
  gap = c_obs - c_surr_mean
  z
  sci

Supported data styles
---------------------
A) Cell-death text time-series dataset
   Expected format:
     first row = time points
     remaining rows = one single-cell trace per row

B) Cell Tracking Challenge style image sequences
   Huh7 / HeLa style folder:
     root/01/*.tif
     root/02/*.tif
     optional root/01_GT/TRA/*.tif or root/01_GT/SEG/*.tif
     optional root/02_GT/TRA/*.tif or root/02_GT/SEG/*.tif

   It will prefer TRA masks if available, then SEG masks, then fall back to
   simple global image intensity signals only.

Outputs
-------
out_cells_full_v2/
  all_cell_sci_metrics.csv
  all_cell_timeseries.csv
  sliding_window_sci.csv
  summary_by_dataset_condition.csv
  summary_by_dataset_signal_type.csv
  file_level_summary.csv
  file_level_condition_tests.csv
  peak_window_by_cell.csv
  summary_report.txt
  plots/

Example usage
-------------
From your sci_cells folder:

  python3 sci_cells_full_rerun_v2.py \
    --cell-death cell_death/Data_Huh7 \
    --huh7 Fluo-C2DL-Huh7/Fluo-C2DL-Huh7 \
    --hela Fluo-N2DL-HeLa/Fluo-N2DL-HeLa \
    --out out_cells_full_v2

Fast test:

  python3 sci_cells_full_rerun_v2.py \
    --cell-death cell_death/Data_Huh7 \
    --out test_cells_v2 \
    --fast \
    --max-files 10

Dependencies
------------
  pip3 install numpy pandas scipy matplotlib tifffile scikit-image
"""

from __future__ import annotations

import argparse
import os
import glob
import re
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
from scipy.signal import hilbert
from scipy.stats import ttest_ind, mannwhitneyu

warnings.filterwarnings("ignore")

try:
    import tifffile as tiff
    HAS_TIFF = True
except Exception:
    HAS_TIFF = False

try:
    from skimage.filters import threshold_otsu
    from skimage.measure import label as sk_label
    HAS_SKIMAGE = True
except Exception:
    HAS_SKIMAGE = False

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
    n_surrogates: int = 40
    smooth: int = 6
    acf_lag: int = 6
    logistic_k: float = 0.9
    z_clip: float = 6.0
    seed: int = 42
    min_points: int = 20
    sliding_window: int = 12
    sliding_step: int = 2
    max_cells_per_file: Optional[int] = None


# =============================================================================
# CORE SCI MATH
# =============================================================================

def acf_coherence(env: np.ndarray, max_lag: int) -> float:
    env = np.asarray(env, dtype=float)
    env = env[np.isfinite(env)]

    if len(env) < max_lag + 3:
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


def moving_average(x: np.ndarray, w: int) -> np.ndarray:
    if w <= 1 or len(x) < w:
        return x
    return np.convolve(x, np.ones(w) / w, mode="same")


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


def sci_metrics(signal: np.ndarray, cfg: SCIConfig, rng: np.random.Generator) -> Dict[str, float]:
    x = np.asarray(signal, dtype=float)
    x = x[np.isfinite(x)]

    if len(x) < max(cfg.min_points, cfg.smooth + cfg.acf_lag + 5) or np.std(x) <= 1e-12:
        return empty_metrics()

    x = x - np.mean(x)

    try:
        env = moving_average(np.abs(hilbert(x)), cfg.smooth)
        c_obs = acf_coherence(env, cfg.acf_lag)
    except Exception:
        return empty_metrics()

    if not np.isfinite(c_obs):
        return empty_metrics()

    surr_scores = []
    for _ in range(cfg.n_surrogates):
        try:
            xs = phase_randomized_surrogate(x, rng)
            es = moving_average(np.abs(hilbert(xs)), cfg.smooth)
            cs = acf_coherence(es, cfg.acf_lag)
            if np.isfinite(cs):
                surr_scores.append(cs)
        except Exception:
            continue

    if len(surr_scores) < max(5, cfg.n_surrogates // 3):
        return empty_metrics()

    mu = float(np.mean(surr_scores))
    sd = float(np.std(surr_scores, ddof=1))
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


def shuffled_control(x: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    y = np.asarray(x, dtype=float).copy()
    rng.shuffle(y)
    return y


# =============================================================================
# CONDITION / LABEL HELPERS
# =============================================================================

def infer_condition_from_path(path: str) -> str:
    lower = str(path).lower()

    # Prefer explicit experimental groups.
    if re.search(r"\bctrl\b|control|untreated", lower):
        return "ctrl"
    if "np100" in lower or "np_100" in lower or "100" in lower and "np" in lower:
        return "NP100"
    if "np25" in lower or "np_25" in lower or "25" in lower and "np" in lower:
        return "NP25"
    if "stauro" in lower or "staurosporine" in lower:
        return "staurosporine"
    if "treated" in lower:
        return "treated"

    return "unknown"


def clean_dataset_name(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9_]+", "_", name).strip("_")


# =============================================================================
# CELL-DEATH TEXT TIMESERIES LOADER
# =============================================================================

def read_cell_death_timeseries_file(path: Path) -> Tuple[np.ndarray, np.ndarray]:
    """
    Expected format:
      first row = time points
      remaining rows = one single-cell trace per row
    """
    for sep in [",", ";", "\t", None]:
        try:
            df = pd.read_csv(path, header=None, sep=sep, engine="python")
            if df.shape[0] >= 2 and df.shape[1] >= 8:
                arr = df.apply(pd.to_numeric, errors="coerce").values
                time = arr[0, :]
                cells = arr[1:, :]
                return time, cells
        except Exception:
            continue

    raise ValueError(f"Could not parse {path}")


def load_cell_death_folder(root: Optional[str], cfg: SCIConfig) -> Tuple[List[dict], List[dict]]:
    if root is None:
        return [], []

    root_path = Path(root).expanduser()
    if not root_path.exists():
        print(f"[WARN] cell-death path does not exist: {root_path}")
        return [], []

    files = sorted(root_path.rglob("*.txt"))
    if not files:
        files = sorted(root_path.rglob("*.csv"))

    print(f"\n[CELL_DEATH] Found {len(files)} text/csv files")

    rows = []
    series_rows = []
    rng = np.random.default_rng(cfg.seed)

    for file_i, path in enumerate(files, 1):
        if file_i % 25 == 0 or file_i == 1:
            print(f"  Processing file {file_i}/{len(files)}: {path.name}")

        try:
            time, cells = read_cell_death_timeseries_file(path)
        except Exception:
            continue

        condition = infer_condition_from_path(str(path))
        rel_file = str(path.relative_to(root_path)) if path.is_relative_to(root_path) else str(path)

        count = 0
        for cell_idx, x in enumerate(cells):
            x = np.asarray(x, dtype=float)
            valid = np.isfinite(x)

            if valid.sum() < cfg.min_points or np.nanstd(x) <= 1e-12:
                continue

            if cfg.max_cells_per_file is not None and count >= cfg.max_cells_per_file:
                break

            count += 1
            sample_id = f"CELLDEATH_file{file_i:04d}_cell{cell_idx:05d}"

            controls = [
                ("real", x),
                ("shuffled_control", shuffled_control(x, np.random.default_rng(cfg.seed + cell_idx))),
                ("phase_randomized_control", phase_randomized_surrogate(
                    np.nan_to_num(x - np.nanmean(x)), np.random.default_rng(cfg.seed + cell_idx)
                )),
            ]

            for control_type, sig in controls:
                local_rng = np.random.default_rng(cfg.seed + file_i * 100000 + cell_idx)
                m = sci_metrics(sig, cfg, local_rng)
                rows.append({
                    "dataset": "CELL_DEATH",
                    "source": "text_timeseries",
                    "file": rel_file,
                    "condition": condition,
                    "sequence": np.nan,
                    "signal_type": "single_cell_fluorescence",
                    "roi_id": f"cell_{cell_idx:05d}",
                    "sample_id": sample_id,
                    "control_type": control_type,
                    "n_points": int(valid.sum()),
                    "start_t": np.nan,
                    "end_t": np.nan,
                    **m,
                })

            for t, v in zip(time, x):
                if np.isfinite(v):
                    series_rows.append({
                        "dataset": "CELL_DEATH",
                        "file": rel_file,
                        "condition": condition,
                        "sequence": np.nan,
                        "signal_type": "single_cell_fluorescence",
                        "roi_id": f"cell_{cell_idx:05d}",
                        "sample_id": sample_id,
                        "time": float(t) if np.isfinite(t) else np.nan,
                        "value": float(v),
                    })

    return rows, series_rows


# =============================================================================
# IMAGE SEQUENCE LOADER
# =============================================================================

def find_sequence_dirs(root: Path) -> List[Path]:
    seqs = []
    for name in ["01", "02"]:
        p = root / name
        if p.is_dir():
            tif_files = list(p.glob("*.tif")) + list(p.glob("*.tiff"))
            if tif_files:
                seqs.append(p)
    if seqs:
        return sorted(seqs)

    # Fallback: any directory with tif files.
    for p in root.rglob("*"):
        if p.is_dir():
            tif_files = list(p.glob("*.tif")) + list(p.glob("*.tiff"))
            if len(tif_files) >= 5:
                seqs.append(p)
    return sorted(set(seqs))


def load_frames(seq_dir: Path) -> Tuple[List[Path], List[np.ndarray]]:
    files = sorted(list(seq_dir.glob("*.tif")) + list(seq_dir.glob("*.tiff")))
    frames = [tiff.imread(str(f)).astype(float) for f in files]
    return files, frames


def load_masks(root: Path, seq_name: str) -> Tuple[Optional[str], Optional[List[np.ndarray]], List[Path]]:
    """
    Prefer tracking masks, then segmentation masks.
    """
    for mask_type in ["TRA", "SEG"]:
        mask_dir = root / f"{seq_name}_GT" / mask_type
        if mask_dir.is_dir():
            mask_files = sorted(list(mask_dir.glob("*.tif")) + list(mask_dir.glob("*.tiff")))
            if mask_files:
                masks = [tiff.imread(str(f)) for f in mask_files]
                return mask_type, masks, mask_files
    return None, None, []


def extract_global_image_signals(frames: List[np.ndarray]) -> List[dict]:
    arrs = {
        "global_mean_intensity": np.array([np.mean(f) for f in frames]),
        "global_std_intensity": np.array([np.std(f) for f in frames]),
        "global_median_intensity": np.array([np.median(f) for f in frames]),
    }
    return [
        {
            "roi_id": "GLOBAL",
            "signal_type": k,
            "series": v,
            "times": list(range(len(v))),
            "mean_area": np.nan,
        }
        for k, v in arrs.items()
    ]


def extract_mask_label_series(frames: List[np.ndarray], masks: List[np.ndarray], seq_name: str,
                              mask_type: str, min_points: int, max_cells: Optional[int]) -> List[dict]:
    n = min(len(frames), len(masks))
    frames = frames[:n]
    masks = masks[:n]

    label_ids = set()
    for m in masks:
        ids = np.unique(m)
        ids = ids[ids > 0]
        label_ids.update(ids.tolist())

    records = []
    for label_id in sorted(label_ids):
        vals = []
        times = []
        areas = []

        for t_i, (frame, mask) in enumerate(zip(frames, masks)):
            region = mask == label_id
            if np.any(region):
                vals.append(float(np.mean(frame[region])))
                times.append(t_i)
                areas.append(int(np.sum(region)))

        if len(vals) >= min_points:
            records.append({
                "roi_id": f"{mask_type.lower()}_{int(label_id):05d}",
                "signal_type": f"{mask_type.lower()}_mask_mean_intensity",
                "series": np.array(vals, dtype=float),
                "times": times,
                "mean_area": float(np.mean(areas)) if areas else np.nan,
            })

        if max_cells is not None and len(records) >= max_cells:
            break

    return records


def extract_otsu_fixed_roi_series(frames: List[np.ndarray], min_points: int, max_cells: Optional[int]) -> List[dict]:
    if not HAS_SKIMAGE or len(frames) < min_points:
        return []

    first = frames[0]
    try:
        thresh = threshold_otsu(first)
        mask = first > thresh
        lab = sk_label(mask)
    except Exception:
        return []

    ids = np.unique(lab)
    ids = ids[ids > 0]

    records = []
    for label_id in ids:
        coords = np.argwhere(lab == label_id)
        if len(coords) < 20:
            continue

        vals = []
        for f in frames:
            vals.append(float(np.mean(f[coords[:, 0], coords[:, 1]])))

        records.append({
            "roi_id": f"fixed_roi_{int(label_id):05d}",
            "signal_type": "fixed_otsu_roi_mean_intensity",
            "series": np.array(vals, dtype=float),
            "times": list(range(len(vals))),
            "mean_area": float(len(coords)),
        })

        if max_cells is not None and len(records) >= max_cells:
            break

    return records


def load_image_dataset(root: Optional[str], dataset_name: str, cfg: SCIConfig) -> Tuple[List[dict], List[dict]]:
    if root is None:
        return [], []

    if not HAS_TIFF:
        print("[WARN] tifffile not installed; image dataset skipped.")
        return [], []

    root_path = Path(root).expanduser()
    if not root_path.exists():
        print(f"[WARN] {dataset_name} path does not exist: {root_path}")
        return [], []

    seq_dirs = find_sequence_dirs(root_path)
    print(f"\n[{dataset_name}] Found {len(seq_dirs)} sequence folders")

    rows = []
    series_rows = []

    for seq_dir in seq_dirs:
        seq_name = seq_dir.name
        try:
            frame_files, frames = load_frames(seq_dir)
        except Exception as e:
            print(f"  {seq_name}: frame load failed: {e}")
            continue

        if len(frames) < cfg.min_points:
            print(f"  {seq_name}: too few frames ({len(frames)}), skipping")
            continue

        mask_type, masks, mask_files = load_masks(root_path, seq_name)

        print(f"  Sequence {seq_name}: frames={len(frames)}, masks={mask_type if mask_type else 'none'}")

        records = []
        records.extend(extract_global_image_signals(frames))

        if masks is not None:
            records.extend(extract_mask_label_series(
                frames, masks, seq_name, mask_type, cfg.min_points, cfg.max_cells_per_file
            ))
        else:
            # Only fallback to fixed ROIs if no masks.
            records.extend(extract_otsu_fixed_roi_series(
                frames, cfg.min_points, cfg.max_cells_per_file
            ))

        for rec_i, rec in enumerate(records):
            x = np.asarray(rec["series"], dtype=float)
            if len(x) < cfg.min_points or np.nanstd(x) <= 1e-12:
                continue

            sample_id = f"{clean_dataset_name(dataset_name)}_{seq_name}_{rec['roi_id']}"

            controls = [
                ("real", x),
                ("shuffled_control", shuffled_control(x, np.random.default_rng(cfg.seed + rec_i))),
                ("phase_randomized_control", phase_randomized_surrogate(
                    np.nan_to_num(x - np.nanmean(x)), np.random.default_rng(cfg.seed + rec_i)
                )),
            ]

            for control_type, sig in controls:
                local_rng = np.random.default_rng(cfg.seed + rec_i * 1000 + len(rows))
                m = sci_metrics(sig, cfg, local_rng)
                rows.append({
                    "dataset": dataset_name,
                    "source": "image_sequence",
                    "file": str(seq_dir.relative_to(root_path)) if seq_dir.is_relative_to(root_path) else str(seq_dir),
                    "condition": infer_condition_from_path(str(seq_dir)),
                    "sequence": seq_name,
                    "signal_type": rec["signal_type"],
                    "roi_id": rec["roi_id"],
                    "sample_id": sample_id,
                    "control_type": control_type,
                    "n_points": len(x),
                    "mean_area": rec.get("mean_area", np.nan),
                    "start_t": min(rec["times"]) if rec.get("times") else np.nan,
                    "end_t": max(rec["times"]) if rec.get("times") else np.nan,
                    **m,
                })

            for t_i, v in zip(rec["times"], x):
                if np.isfinite(v):
                    series_rows.append({
                        "dataset": dataset_name,
                        "file": str(seq_dir.relative_to(root_path)) if seq_dir.is_relative_to(root_path) else str(seq_dir),
                        "condition": infer_condition_from_path(str(seq_dir)),
                        "sequence": seq_name,
                        "signal_type": rec["signal_type"],
                        "roi_id": rec["roi_id"],
                        "sample_id": sample_id,
                        "time": float(t_i),
                        "value": float(v),
                    })

    return rows, series_rows


# =============================================================================
# SLIDING WINDOW / PEAK ANALYSIS
# =============================================================================

def compute_sliding_windows(series_df: pd.DataFrame, cfg: SCIConfig) -> pd.DataFrame:
    rows = []

    if series_df.empty:
        return pd.DataFrame()

    for sample_id, sub in series_df.groupby("sample_id"):
        sub = sub.sort_values("time")
        vals = sub["value"].values.astype(float)
        times = sub["time"].values.astype(float)

        if len(vals) < cfg.sliding_window:
            continue

        base = sub.iloc[0].to_dict()

        for start in range(0, len(vals) - cfg.sliding_window + 1, cfg.sliding_step):
            end = start + cfg.sliding_window
            segment = vals[start:end]

            local_rng = np.random.default_rng(cfg.seed + start + abs(hash(sample_id)) % 100000)
            m = sci_metrics(segment, cfg, local_rng)

            rows.append({
                "dataset": base["dataset"],
                "file": base["file"],
                "condition": base["condition"],
                "sequence": base.get("sequence", np.nan),
                "signal_type": base["signal_type"],
                "roi_id": base["roi_id"],
                "sample_id": sample_id,
                "start_idx": start,
                "end_idx": end - 1,
                "center_idx": start + cfg.sliding_window // 2,
                "start_time": times[start],
                "end_time": times[end - 1],
                "center_time": times[start + cfg.sliding_window // 2],
                "window_n": cfg.sliding_window,
                **m,
            })

    return pd.DataFrame(rows)


def peak_windows(sliding: pd.DataFrame) -> pd.DataFrame:
    if sliding.empty or "sci" not in sliding.columns:
        return pd.DataFrame()

    good = sliding[np.isfinite(sliding["sci"])].copy()
    if good.empty:
        return pd.DataFrame()

    idx = good.groupby("sample_id")["sci"].idxmax()
    return good.loc[idx].reset_index(drop=True)


# =============================================================================
# SUMMARY / STATS
# =============================================================================

def summarize_metrics(df: pd.DataFrame, group_cols: List[str]) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame()

    real = df[df["control_type"].eq("real")].copy() if "control_type" in df.columns else df.copy()

    def agg(g):
        return pd.Series({
            "n": len(g),
            "mean_sci": g["sci"].mean(),
            "median_sci": g["sci"].median(),
            "mean_gap": g["gap"].mean(),
            "median_gap": g["gap"].median(),
            "mean_z": g["z"].mean(),
            "pct_positive_gap": g["gap"].gt(0).mean() * 100.0,
            "mean_c_obs": g["c_obs"].mean(),
            "mean_c_surr": g["c_surr_mean"].mean(),
        })

    return real.groupby(group_cols, dropna=False).apply(agg).reset_index()


def file_level_summary(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame()

    real = df[df["control_type"].eq("real")].copy()

    return (
        real.groupby(["dataset", "condition", "file"], dropna=False)
        .agg(
            n_samples=("sample_id", "nunique"),
            mean_sci=("sci", "mean"),
            median_sci=("sci", "median"),
            mean_gap=("gap", "mean"),
            median_gap=("gap", "median"),
            mean_z=("z", "mean"),
            pct_positive_gap=("gap", lambda x: x.gt(0).mean() * 100.0),
        )
        .reset_index()
    )


def run_file_level_tests(file_df: pd.DataFrame) -> pd.DataFrame:
    if file_df.empty:
        return pd.DataFrame()

    tests = []
    metrics = ["mean_sci", "mean_gap", "mean_z", "pct_positive_gap"]

    # Compare all condition pairs inside each dataset.
    for dataset, d in file_df.groupby("dataset"):
        conditions = sorted([c for c in d["condition"].dropna().unique() if c != "unknown"])

        # Default useful order.
        pairs = []
        for a, b in [("NP25", "ctrl"), ("NP100", "ctrl"), ("NP100", "NP25"), ("treated", "ctrl"), ("staurosporine", "ctrl")]:
            if a in conditions and b in conditions:
                pairs.append((a, b))

        # If no known pairs, compare all pairs.
        if not pairs:
            for i in range(len(conditions)):
                for j in range(i + 1, len(conditions)):
                    pairs.append((conditions[i], conditions[j]))

        for metric in metrics:
            for a_cond, b_cond in pairs:
                a = d[d["condition"] == a_cond][metric].dropna()
                b = d[d["condition"] == b_cond][metric].dropna()

                row = {
                    "dataset": dataset,
                    "metric": metric,
                    "contrast": f"{a_cond} vs {b_cond}",
                    "mean_a": a.mean() if len(a) else np.nan,
                    "mean_b": b.mean() if len(b) else np.nan,
                    "diff": (a.mean() - b.mean()) if len(a) and len(b) else np.nan,
                    "n_a_files": len(a),
                    "n_b_files": len(b),
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
                    row["note"] = "too few files"

                tests.append(row)

    return pd.DataFrame(tests)


# =============================================================================
# PLOTS / REPORT
# =============================================================================

def make_plots(metrics: pd.DataFrame, sliding: pd.DataFrame, file_df: pd.DataFrame, out_dir: Path):
    if not HAS_MPL:
        return

    plot_dir = out_dir / "plots"
    plot_dir.mkdir(exist_ok=True)

    real = metrics[metrics["control_type"].eq("real")].copy() if not metrics.empty else pd.DataFrame()

    if not real.empty:
        # Gap by dataset/signal type.
        summary = (
            real.groupby(["dataset", "signal_type"])["gap"]
            .mean()
            .reset_index()
            .sort_values("gap")
        )
        if not summary.empty:
            summary["label"] = summary["dataset"] + " | " + summary["signal_type"]
            plt.figure(figsize=(12, max(5, 0.3 * len(summary))))
            plt.barh(summary["label"], summary["gap"])
            plt.xlabel("Mean gap: c_obs - mean(c_surr)")
            plt.title("Cellular SCI mean gap by dataset / signal type")
            plt.tight_layout()
            plt.savefig(plot_dir / "mean_gap_by_dataset_signal_type.png", dpi=200)
            plt.close()

        # Real vs controls.
        ctl = (
            metrics.groupby(["control_type"])["gap"]
            .mean()
            .reset_index()
            .sort_values("gap")
        )
        if not ctl.empty:
            plt.figure(figsize=(8, 5))
            plt.bar(ctl["control_type"], ctl["gap"])
            plt.ylabel("Mean gap")
            plt.title("Real vs shuffled / phase-randomized controls")
            plt.xticks(rotation=30, ha="right")
            plt.tight_layout()
            plt.savefig(plot_dir / "real_vs_controls_mean_gap.png", dpi=200)
            plt.close()

    if not sliding.empty:
        # Sliding trajectories by condition, file-level-ish mean.
        for dataset, d in sliding.groupby("dataset"):
            tmp = (
                d.groupby(["condition", "center_idx"])["gap"]
                .mean()
                .reset_index()
            )
            if tmp.empty:
                continue
            plt.figure(figsize=(10, 5))
            for cond, g in tmp.groupby("condition"):
                plt.plot(g["center_idx"], g["gap"], label=str(cond))
            plt.xlabel("Window center index")
            plt.ylabel("Mean gap")
            plt.title(f"{dataset}: sliding-window SCI gap trajectory")
            plt.legend()
            plt.tight_layout()
            plt.savefig(plot_dir / f"{clean_dataset_name(dataset)}_sliding_gap_by_condition.png", dpi=200)
            plt.close()

    if not file_df.empty:
        for metric in ["mean_gap", "mean_sci"]:
            for dataset, d in file_df.groupby("dataset"):
                if d["condition"].nunique() < 2:
                    continue
                groups = [g[metric].dropna().values for _, g in d.groupby("condition")]
                labels = [str(k) for k, _ in d.groupby("condition")]
                if not groups:
                    continue
                plt.figure(figsize=(8, 5))
                plt.boxplot(groups, labels=labels, showfliers=False)
                plt.ylabel(metric)
                plt.title(f"{dataset}: file-level {metric} by condition")
                plt.xticks(rotation=30, ha="right")
                plt.tight_layout()
                plt.savefig(plot_dir / f"{clean_dataset_name(dataset)}_file_level_{metric}_by_condition.png", dpi=200)
                plt.close()


def write_report(out_dir: Path, cfg: SCIConfig, metrics: pd.DataFrame,
                 by_condition: pd.DataFrame, by_signal: pd.DataFrame,
                 file_df: pd.DataFrame, tests: pd.DataFrame):
    lines = []
    lines.append("=" * 90)
    lines.append("SCI CELLS FULL RE-RUN V2")
    lines.append("=" * 90)
    lines.append("")
    lines.append("Modern claim tested:")
    lines.append("  SCI measures excess Hilbert-envelope temporal coherence relative to")
    lines.append("  a phase-randomized surrogate baseline that preserves the power spectrum.")
    lines.append("")
    lines.append("Parameters:")
    lines.append(f"  surrogates={cfg.n_surrogates}")
    lines.append(f"  envelope_smooth={cfg.smooth}")
    lines.append(f"  acf_lag={cfg.acf_lag}")
    lines.append(f"  logistic_k={cfg.logistic_k}")
    lines.append(f"  seed={cfg.seed}")
    lines.append(f"  min_points={cfg.min_points}")
    lines.append(f"  sliding_window={cfg.sliding_window}")
    lines.append(f"  sliding_step={cfg.sliding_step}")
    lines.append("")
    lines.append(f"Total SCI rows: {len(metrics):,}")
    if not metrics.empty:
        lines.append(f"Datasets: {', '.join(sorted(metrics['dataset'].dropna().unique()))}")
    lines.append("")

    if not by_condition.empty:
        lines.append("-" * 90)
        lines.append("SUMMARY BY DATASET / CONDITION")
        lines.append("-" * 90)
        cols = ["dataset", "condition", "n", "mean_sci", "mean_gap", "mean_z", "pct_positive_gap"]
        lines.append(by_condition[cols].to_string(index=False, float_format=lambda x: f"{x:0.4f}"))
        lines.append("")

    if not by_signal.empty:
        lines.append("-" * 90)
        lines.append("SUMMARY BY DATASET / SIGNAL TYPE")
        lines.append("-" * 90)
        cols = ["dataset", "signal_type", "n", "mean_sci", "mean_gap", "mean_z", "pct_positive_gap"]
        lines.append(by_signal[cols].to_string(index=False, float_format=lambda x: f"{x:0.4f}"))
        lines.append("")

    if not file_df.empty:
        lines.append("-" * 90)
        lines.append("FILE-LEVEL SUMMARY")
        lines.append("-" * 90)
        lines.append(file_df.head(40).to_string(index=False, float_format=lambda x: f"{x:0.4f}"))
        lines.append("")

    if not tests.empty:
        lines.append("-" * 90)
        lines.append("FILE-LEVEL CONDITION TESTS")
        lines.append("-" * 90)
        lines.append(tests.head(60).to_string(index=False, float_format=lambda x: f"{x:0.6f}"))
        lines.append("")

    lines.append("-" * 90)
    lines.append("INTERPRETATION GUIDE")
    lines.append("-" * 90)
    lines.append("  c_obs        = envelope autocorrelation of the real signal")
    lines.append("  c_surr_mean  = mean envelope autocorrelation of phase-randomized surrogates")
    lines.append("  gap          = c_obs - c_surr_mean")
    lines.append("  z            = gap / c_surr_std")
    lines.append("  SCI          = logistic(z), clipped before mapping")
    lines.append("")
    lines.append("Most defensible biological result is file-level or experiment-level, not")
    lines.append("treating every cell as an independent replicate. Use file_level_summary.csv")
    lines.append("and file_level_condition_tests.csv before making strong claims.")
    lines.append("")

    (out_dir / "summary_report.txt").write_text("\n".join(lines))
    print("\n".join(lines))


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Consolidated cellular SCI rerun pipeline.")
    parser.add_argument("--cell-death", type=str, default=None, help="Path to cell-death text timeseries root")
    parser.add_argument("--huh7", type=str, default=None, help="Path to Fluo-C2DL-Huh7 root")
    parser.add_argument("--hela", type=str, default=None, help="Path to Fluo-N2DL-HeLa root")
    parser.add_argument("--image-root", type=str, default=None, help="Extra generic image dataset root")
    parser.add_argument("--out", type=str, default="out_cells_full_v2")

    parser.add_argument("--surrogates", type=int, default=40)
    parser.add_argument("--smooth", type=int, default=6)
    parser.add_argument("--lag", type=int, default=6)
    parser.add_argument("--min-points", type=int, default=20)
    parser.add_argument("--sliding-window", type=int, default=12)
    parser.add_argument("--sliding-step", type=int, default=2)
    parser.add_argument("--max-cells-per-file", type=int, default=None)
    parser.add_argument("--fast", action="store_true", help="Use fewer surrogates and fewer cells per file")

    args = parser.parse_args()

    cfg = SCIConfig(
        n_surrogates=args.surrogates,
        smooth=args.smooth,
        acf_lag=args.lag,
        min_points=args.min_points,
        sliding_window=args.sliding_window,
        sliding_step=args.sliding_step,
        max_cells_per_file=args.max_cells_per_file,
    )

    if args.fast:
        cfg.n_surrogates = 15
        if cfg.max_cells_per_file is None:
            cfg.max_cells_per_file = 50
        print("[FAST MODE] surrogates=15, max_cells_per_file=50")

    out_dir = Path(args.out).expanduser()
    out_dir.mkdir(parents=True, exist_ok=True)

    all_metric_rows = []
    all_series_rows = []

    # Cell-death text dataset.
    rows, series = load_cell_death_folder(args.cell_death, cfg)
    all_metric_rows.extend(rows)
    all_series_rows.extend(series)

    # Image datasets.
    for dataset_name, root in [
        ("HUH7", args.huh7),
        ("HELA", args.hela),
        ("IMAGE_EXTRA", args.image_root),
    ]:
        rows, series = load_image_dataset(root, dataset_name, cfg)
        all_metric_rows.extend(rows)
        all_series_rows.extend(series)

    metrics = pd.DataFrame(all_metric_rows)
    series_df = pd.DataFrame(all_series_rows)

    if metrics.empty:
        print("\nNo metrics produced. Check input folder paths and formats.")
        return

    metrics.to_csv(out_dir / "all_cell_sci_metrics.csv", index=False)
    series_df.to_csv(out_dir / "all_cell_timeseries.csv", index=False)

    print(f"\nSaved: {out_dir / 'all_cell_sci_metrics.csv'}")
    print(f"Saved: {out_dir / 'all_cell_timeseries.csv'}")

    # Sliding windows only on real time series.
    sliding = compute_sliding_windows(series_df, cfg)
    sliding.to_csv(out_dir / "sliding_window_sci.csv", index=False)
    print(f"Saved: {out_dir / 'sliding_window_sci.csv'}")

    peaks = peak_windows(sliding)
    peaks.to_csv(out_dir / "peak_window_by_cell.csv", index=False)
    print(f"Saved: {out_dir / 'peak_window_by_cell.csv'}")

    by_condition = summarize_metrics(metrics, ["dataset", "condition"])
    by_signal = summarize_metrics(metrics, ["dataset", "signal_type"])

    by_condition.to_csv(out_dir / "summary_by_dataset_condition.csv", index=False)
    by_signal.to_csv(out_dir / "summary_by_dataset_signal_type.csv", index=False)

    file_df = file_level_summary(metrics)
    file_df.to_csv(out_dir / "file_level_summary.csv", index=False)

    tests = run_file_level_tests(file_df)
    tests.to_csv(out_dir / "file_level_condition_tests.csv", index=False)

    make_plots(metrics, sliding, file_df, out_dir)
    write_report(out_dir, cfg, metrics, by_condition, by_signal, file_df, tests)

    print("\nDone.")
    print(f"Open this first: {out_dir / 'summary_report.txt'}")


if __name__ == "__main__":
    main()
