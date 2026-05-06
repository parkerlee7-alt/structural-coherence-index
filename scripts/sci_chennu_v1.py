"""
SCI analysis of the Chennu et al. (2014/2016) propofol sedation dataset.

Dataset: ~/Desktop/SCI_Project/Sedation-RestingState/
Format:  EEGLAB v7.3 HDF5 .set + float32 binary .fdt
         250 Hz, 91 channels, 10-sec epochs (2500 samples), variable trial count
         4 files per subject in chronological order = baseline/mild/moderate/recovery

Pre-registration: anesthesia/SCI_Anesthesia_PreRegistration_v1.md
Locked params:    W=25, L=25 (0.10 s × 250 Hz), S=40, k=0.9, seed=42

Usage:
    python3 scripts/sci_chennu_v1.py \
        --data-dir Sedation-RestingState \
        --out results/chennu_v1
"""

import sys, os, argparse, re
from pathlib import Path
from collections import defaultdict

import numpy as np
import pandas as pd
import scipy.signal as sg
from scipy.stats import ttest_rel, wilcoxon, friedmanchisquare
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import h5py

# ── import SCI ──────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from sci_score_v3 import sci_score_v3

# ── locked parameters ────────────────────────────────────────────────────────
TARGET_FS   = 250.0
SMOOTH_SEC  = 0.10        # W = round(0.10 × 250) = 25
LAG_SEC     = 0.10        # L = 25
S           = 40
K           = 0.9
SEED        = 42
BANDPASS_LO = 1.0
BANDPASS_HI = 40.0
EPOCH_SEC   = 10          # seconds per epoch
FZ_IDX      = 8           # Fz channel index (confirmed via h5py probe)

STATES = ["baseline", "mild", "moderate", "recovery"]


# ── helpers ──────────────────────────────────────────────────────────────────

def bandpass(x, lo, hi, fs, order=4):
    nyq = fs / 2.0
    b, a = sg.butter(order, [lo / nyq, hi / nyq], btype="band")
    return sg.filtfilt(b, a, x)


def read_hdf5_string(h, ref):
    """Dereference an HDF5 object reference to a string (MATLAB char array)."""
    arr = h[ref][:]
    return "".join(chr(int(v)) for v in arr.flatten())


def load_set_metadata(set_path):
    """Return (srate, nbchan, pnts_per_epoch, n_trials, fz_index) from .set."""
    with h5py.File(set_path, "r") as h:
        eeg = h["EEG"]
        srate  = int(np.array(eeg["srate"]).flatten()[0])
        nbchan = int(np.array(eeg["nbchan"]).flatten()[0])
        pnts   = int(np.array(eeg["pnts"]).flatten()[0])
        trials = int(np.array(eeg["trials"]).flatten()[0])

        # find Fz index
        labels = eeg["chanlocs"]["labels"]
        fz_idx = FZ_IDX  # fallback
        for i in range(nbchan):
            try:
                name = read_hdf5_string(h, labels[i, 0])
                if name.upper() == "FZ":
                    fz_idx = i
                    break
            except Exception:
                pass

    return srate, nbchan, pnts, trials, fz_idx


def load_epochs(set_path):
    """
    Load the Fz channel for all epochs from a Chennu .set/.fdt file pair.

    Returns (epochs_array, n_trials) where epochs_array has shape (n_trials, pnts).
    """
    set_path = Path(set_path)
    fdt_path = set_path.with_suffix(".fdt")

    srate, nbchan, pnts, n_trials, fz_idx = load_set_metadata(set_path)

    # MATLAB v7.3 .fdt: float32, shape [nbchan, pnts, trials] Fortran order
    raw = np.fromfile(fdt_path, dtype=np.float32)
    expected = nbchan * pnts * n_trials
    if raw.size != expected:
        # fallback: try C order reshape to [nbchan, total_time]
        total_tp = raw.size // nbchan
        data_2d = raw.reshape((nbchan, total_tp), order="C")
        fz = data_2d[fz_idx]
        # cut into epochs
        ep_len = pnts
        n_eps = total_tp // ep_len
        epochs = np.stack([fz[i * ep_len:(i + 1) * ep_len] for i in range(n_eps)])
        return epochs, n_eps

    data = raw.reshape((nbchan, pnts, n_trials), order="F")
    epochs = data[fz_idx, :, :].T  # (n_trials, pnts)
    return epochs, n_trials


def cohen_d(a, b):
    diff = np.array(a) - np.array(b)
    return diff.mean() / (diff.std(ddof=1) + 1e-12)


def process_file(set_path, subject, state):
    """Compute per-epoch SCI metrics for one .set file."""
    epochs, n_trials = load_epochs(set_path)
    fs = TARGET_FS
    w  = round(SMOOTH_SEC * fs)
    L  = round(LAG_SEC   * fs)

    records = []
    for i, epoch in enumerate(epochs):
        filt = bandpass(epoch.astype(np.float64), BANDPASS_LO, BANDPASS_HI, fs)
        r = sci_score_v3(filt, w=w, L=L, S=S, k=K, seed=SEED)
        records.append({
            "subject": subject,
            "state":   state,
            "epoch":   i,
            "c_obs":        r["c_obs"],
            "c_surr_mean":  r["c_surr_mean"],
            "c_surr_std":   r["c_surr_std"],
            "gap":          r["gap"],
            "z":            r["z"],
            "SCI":          r["SCI"],
            "bucket":       r["bucket"],
        })
    return records


def find_subject_files(data_dir):
    """
    Return dict: subject_id → [file1, file2, file3, file4] sorted chronologically.
    Subject ID = first 2 chars of filename.
    """
    data_dir = Path(data_dir)
    set_files = sorted(data_dir.glob("*.set"))

    grouped = defaultdict(list)
    for f in set_files:
        subj = f.name[:2]
        grouped[subj].append(f)

    # Sort each subject's files by name (which encodes date+time → chronological)
    result = {}
    for subj, files in sorted(grouped.items()):
        files_sorted = sorted(files)
        if len(files_sorted) != 4:
            print(f"  WARNING: subject {subj} has {len(files_sorted)} files, expected 4 — skipping")
            continue
        result[subj] = files_sorted
    return result


# ── statistical helpers ──────────────────────────────────────────────────────

def run_stats(subj_df, out_dir):
    states = STATES
    lines  = []

    def h(s):
        lines.append(s)
        print(s)

    h("=" * 60)
    h("SCI CHENNU ANESTHESIA — STATISTICAL TESTS")
    h("=" * 60)
    h(f"N subjects: {subj_df['subject'].nunique()}")
    h("")

    # grand means
    h("Grand means ± SD by state:")
    for st in states:
        d = subj_df[subj_df["state"] == st]["SCI"]
        h(f"  {st:12s}  SCI={d.mean():.4f} ± {d.std(ddof=1):.4f}  "
          f"gap={subj_df[subj_df['state']==st]['gap'].mean():.4f}")
    h("")

    # Friedman
    groups = [subj_df[subj_df["state"] == st]["SCI"].values for st in states]
    if all(len(g) == len(groups[0]) for g in groups):
        stat, p = friedmanchisquare(*groups)
        h(f"Friedman test across all 4 states: χ²={stat:.3f}, p={p:.4f}")
    h("")

    # Paired t-tests + Wilcoxon for all pairs (Bonferroni k=6)
    pairs = [
        ("baseline",  "mild"),
        ("baseline",  "moderate"),
        ("baseline",  "recovery"),
        ("mild",      "moderate"),
        ("mild",      "recovery"),
        ("moderate",  "recovery"),
    ]
    k = len(pairs)
    h(f"Pairwise comparisons (Bonferroni α=0.05/{k}={0.05/k:.4f}):")
    primary_done = False
    for s1, s2 in pairs:
        a = subj_df[subj_df["state"] == s1]["SCI"].values
        b = subj_df[subj_df["state"] == s2]["SCI"].values
        t, pt = ttest_rel(a, b)
        try:
            _, pw = wilcoxon(a, b)
        except Exception:
            pw = float("nan")
        d = cohen_d(a, b)
        tag = " *** PRIMARY ***" if (s1, s2) == ("baseline", "moderate") else ""
        h(f"  {s1} vs {s2}:{tag}")
        h(f"    t={t:.3f} p={pt:.4f}  Wilcoxon p={pw:.4f}  Cohen d={d:.3f}")
        if (s1, s2) == ("baseline", "moderate"):
            primary_done = True
    h("")

    # P2: recovery vs baseline
    rec = subj_df[subj_df["state"] == "recovery"]["SCI"].values
    bas = subj_df[subj_df["state"] == "baseline"]["SCI"].values
    mod = subj_df[subj_df["state"] == "moderate"]["SCI"].values
    diff_rec = abs(rec - bas).mean()
    diff_mod = abs(mod - bas).mean()
    h(f"P2 check — |SCI(Recovery)−SCI(Baseline)| = {diff_rec:.4f}")
    h(f"           |SCI(Moderate)−SCI(Baseline)| = {diff_mod:.4f}")
    h(f"           Recovery closer to Baseline: {diff_rec < diff_mod}")
    h("")

    # P3: moderate SCI range
    mod_mean = mod.mean()
    h(f"P3 check — SCI(Moderate) = {mod_mean:.4f}  (target 0.65–0.85)")
    h(f"           In range: {0.65 <= mod_mean <= 0.85}")
    h("")

    # monotonic ordering check
    means = {st: subj_df[subj_df["state"] == st]["SCI"].mean() for st in states}
    mono = (means["baseline"] > means["mild"] > means["moderate"])
    h(f"P1 monotonic ordering (Baseline > Mild > Moderate):")
    for st in states:
        h(f"  {st}: {means[st]:.4f}")
    h(f"  Monotonic: {mono}")
    h("")

    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "statistical_tests.txt", "w") as fh:
        fh.write("\n".join(lines))


# ── plotting ─────────────────────────────────────────────────────────────────

def make_plot(subj_df, out_dir):
    state_order = STATES
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # SCI
    sns.boxplot(data=subj_df, x="state", y="SCI", order=state_order, ax=axes[0])
    sns.stripplot(data=subj_df, x="state", y="SCI", order=state_order,
                  color="black", alpha=0.5, size=4, ax=axes[0])
    axes[0].set_title("SCI by Sedation State (Chennu et al.)")
    axes[0].set_ylabel("SCI")
    axes[0].set_xlabel("")

    # gap
    sns.boxplot(data=subj_df, x="state", y="gap", order=state_order, ax=axes[1])
    sns.stripplot(data=subj_df, x="state", y="gap", order=state_order,
                  color="black", alpha=0.5, size=4, ax=axes[1])
    axes[1].set_title("SCI gap by Sedation State (Chennu et al.)")
    axes[1].set_ylabel("gap (c_obs − c_surr_mean)")
    axes[1].set_xlabel("")

    plt.tight_layout()
    plot_dir = out_dir / "plots"
    plot_dir.mkdir(exist_ok=True)
    fig.savefig(plot_dir / "sci_by_state.png", dpi=150)
    plt.close(fig)
    print(f"  Plot saved → {plot_dir / 'sci_by_state.png'}")


# ── main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="SCI Chennu propofol sedation analysis")
    parser.add_argument("--data-dir", default="Sedation-RestingState",
                        help="Path to dataset directory")
    parser.add_argument("--out", default="results/chennu_v1",
                        help="Output directory")
    parser.add_argument("--subjects", nargs="*",
                        help="Subset of subject IDs to process (default: all)")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    out_dir  = Path(args.out)

    subject_files = find_subject_files(data_dir)
    if args.subjects:
        subject_files = {k: v for k, v in subject_files.items()
                         if k in args.subjects}

    print(f"Processing {len(subject_files)} subjects from {data_dir}")

    all_records = []
    for subj, files in subject_files.items():
        print(f"\nSubject {subj}:")
        for state, set_path in zip(STATES, files):
            print(f"  {state}: {set_path.name}")
            records = process_file(set_path, subj, state)
            all_records.extend(records)
            sci_vals = [r["SCI"] for r in records]
            print(f"    epochs={len(records)}  mean_SCI={np.mean(sci_vals):.4f}")

    if not all_records:
        print("No records produced — check data directory.")
        sys.exit(1)

    epoch_df = pd.DataFrame(all_records)

    # per-subject per-state means
    subj_df = (epoch_df.groupby(["subject", "state"])
               [["c_obs", "c_surr_mean", "c_surr_std", "gap", "z", "SCI"]]
               .mean()
               .reset_index())

    # state-level summary
    summary = (subj_df.groupby("state")
               [["SCI", "gap", "z"]]
               .agg(["mean", "std"])
               .round(4))

    out_dir.mkdir(parents=True, exist_ok=True)
    epoch_df.to_csv(out_dir / "epoch_metrics.csv", index=False)
    subj_df.to_csv(out_dir / "subject_state_means.csv", index=False)
    summary.to_csv(out_dir / "summary_by_state.csv")
    print(f"\nSaved CSVs to {out_dir}/")

    run_stats(subj_df, out_dir)
    make_plot(subj_df, out_dir)

    print("\nDone.")


if __name__ == "__main__":
    main()
