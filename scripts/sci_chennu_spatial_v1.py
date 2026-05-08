"""
SCI multi-channel spatial analysis — Chennu et al. propofol sedation dataset.

Runs SCI on all 91 channels for all subjects and states.
Produces topographic maps of SCI per state and the moderate-baseline difference.

Usage:
    python3 scripts/sci_chennu_spatial_v1.py \
        --data-dir Sedation-RestingState \
        --out results/chennu_spatial_v1
"""

import sys, os, argparse, warnings
from pathlib import Path
from collections import defaultdict
import multiprocessing as mp

import numpy as np
import pandas as pd
import scipy.signal as sg
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import h5py
import mne
mne.set_log_level("ERROR")

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from sci_score_v3 import sci_score_v3

# ── locked parameters ────────────────────────────────────────────────────────
TARGET_FS  = 250.0
SMOOTH_SEC = 0.10
LAG_SEC    = 0.10
S          = 40
K          = 0.9
SEED       = 42
BPLO       = 1.0
BPHI       = 40.0

STATES = ["baseline", "mild", "moderate", "recovery"]


# ── helpers ──────────────────────────────────────────────────────────────────

def bandpass(x, lo, hi, fs, order=4):
    nyq = fs / 2.0
    b, a = sg.butter(order, [lo / nyq, hi / nyq], btype="band")
    return sg.filtfilt(b, a, x)


def read_hdf5_string(h, ref):
    return "".join(chr(int(v)) for v in h[ref][:].flatten())


def read_hdf5_scalar(h, ref):
    return float(h[ref][:].flatten()[0])


def load_channel_info(set_path):
    """Return (names, xyz) for all channels from a .set file."""
    with h5py.File(set_path, "r") as h:
        eeg = h["EEG"]
        cl  = eeg["chanlocs"]
        n   = int(np.array(eeg["nbchan"]).flatten()[0])
        names = []
        xyz   = []
        for i in range(n):
            name = read_hdf5_string(h, cl["labels"][i, 0])
            x    = read_hdf5_scalar(h, cl["X"][i, 0])
            y    = read_hdf5_scalar(h, cl["Y"][i, 0])
            z    = read_hdf5_scalar(h, cl["Z"][i, 0])
            names.append(name)
            xyz.append([x, y, z])
    return names, np.array(xyz)


def load_all_epochs(set_path):
    """
    Return data array of shape (n_trials, nbchan, pnts) for a .set/.fdt pair.
    """
    set_path = Path(set_path)
    fdt_path = set_path.with_suffix(".fdt")

    with h5py.File(set_path, "r") as h:
        eeg    = h["EEG"]
        nbchan = int(np.array(eeg["nbchan"]).flatten()[0])
        pnts   = int(np.array(eeg["pnts"]).flatten()[0])
        trials = int(np.array(eeg["trials"]).flatten()[0])

    raw = np.fromfile(fdt_path, dtype=np.float32)
    expected = nbchan * pnts * trials

    if raw.size == expected:
        # [nbchan, pnts, trials] Fortran order
        data = raw.reshape((nbchan, pnts, trials), order="F")
        # → (trials, nbchan, pnts)
        return data.transpose(2, 0, 1)
    else:
        # fallback: C order [nbchan, total_time]
        total_tp = raw.size // nbchan
        data_2d  = raw.reshape((nbchan, total_tp), order="C")
        n_eps = total_tp // pnts
        return data_2d[:, :n_eps * pnts].reshape(nbchan, n_eps, pnts).transpose(1, 0, 2)


def process_subject(args):
    """Worker: process one subject, all states, all channels. Returns list of records."""
    subj, files = args
    w  = round(SMOOTH_SEC * TARGET_FS)
    L  = round(LAG_SEC    * TARGET_FS)
    records = []

    for state, set_path in zip(STATES, files):
        try:
            data = load_all_epochs(set_path)   # (trials, nbchan, pnts)
        except Exception as e:
            print(f"  ERROR {subj} {state}: {e}")
            continue

        n_trials, nbchan, pnts = data.shape
        # per-channel, per-epoch SCI — collect mean SCI per channel
        ch_sci  = np.zeros(nbchan)
        ch_gap  = np.zeros(nbchan)

        for ch in range(nbchan):
            sci_vals = []
            gap_vals = []
            for ep in range(n_trials):
                seg  = data[ep, ch, :].astype(np.float64)
                filt = bandpass(seg, BPLO, BPHI, TARGET_FS)
                r    = sci_score_v3(filt, w=w, L=L, S=S, k=K, seed=SEED)
                sci_vals.append(r["SCI"])
                gap_vals.append(r["gap"])
            ch_sci[ch] = np.mean(sci_vals)
            ch_gap[ch] = np.mean(gap_vals)

        for ch in range(nbchan):
            records.append({
                "subject": subj,
                "state":   state,
                "ch_idx":  ch,
                "SCI":     ch_sci[ch],
                "gap":     ch_gap[ch],
            })

        print(f"  {subj} {state} done — mean SCI across channels: "
              f"{ch_sci.mean():.4f}")

    return records


def find_subject_files(data_dir):
    data_dir = Path(data_dir)
    set_files = sorted(data_dir.glob("*.set"))
    grouped = defaultdict(list)
    for f in set_files:
        grouped[f.name[:2]].append(f)
    result = {}
    for subj, files in sorted(grouped.items()):
        if len(sorted(files)) == 4:
            result[subj] = sorted(files)
    return result


# ── topographic plotting ─────────────────────────────────────────────────────

def make_mne_info(ch_names, xyz_cm):
    """Build an MNE Info object from channel names and XYZ positions (in cm)."""
    ch_pos = {name: pos / 100.0 for name, pos in zip(ch_names, xyz_cm)}
    montage = mne.channels.make_dig_montage(ch_pos=ch_pos, coord_frame="head")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        info = mne.create_info(ch_names=list(ch_names), sfreq=250.0, ch_types="eeg")
        info.set_montage(montage, on_missing="ignore")
    return info


def plot_topomaps(grand, ch_names, xyz, out_dir):
    """
    grand: dict state → array (n_ch,) of mean SCI across subjects
    """
    info = make_mne_info(ch_names, xyz)

    states     = STATES
    diff_mod   = grand["moderate"] - grand["baseline"]
    diff_rec   = grand["recovery"] - grand["baseline"]

    # ── Figure 1: SCI per state (2×2) ────────────────────────────────────────
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.flatten()
    vmin = min(grand[s].min() for s in states)
    vmax = max(grand[s].max() for s in states)

    for ax, state in zip(axes, states):
        im, cn = mne.viz.plot_topomap(
            grand[state], info, axes=ax, show=False,
            vlim=(vmin, vmax), cmap="RdYlBu_r",
            contours=6, sensors=True, sphere="auto"
        )
        ax.set_title(f"{state.capitalize()}\nmean SCI={grand[state].mean():.3f}", fontsize=11)
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    fig.suptitle("SCI by Sedation State — Grand Average (N=20)\nChennu et al. propofol dataset",
                 fontsize=13, fontweight="bold")
    plt.tight_layout()
    fig.savefig(out_dir / "topomap_sci_by_state.png", dpi=150)
    plt.close(fig)
    print(f"  Saved topomap_sci_by_state.png")

    # ── Figure 2: difference maps ─────────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    vlim_d = max(abs(diff_mod).max(), abs(diff_rec).max())
    for ax, diff, label in zip(
        axes,
        [diff_mod, diff_rec],
        ["Moderate − Baseline", "Recovery − Baseline"]
    ):
        im, cn = mne.viz.plot_topomap(
            diff, info, axes=ax, show=False,
            vlim=(-vlim_d, vlim_d), cmap="RdBu_r",
            contours=6, sensors=True, sphere="auto"
        )
        ax.set_title(f"ΔSCI: {label}\n"
                     f"mean Δ={diff.mean():.3f}  max Δ={diff.max():.3f}",
                     fontsize=11)
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="ΔSCI")

    fig.suptitle("SCI Difference Maps (N=20)\nRed = SCI increase with sedation",
                 fontsize=13, fontweight="bold")
    plt.tight_layout()
    fig.savefig(out_dir / "topomap_sci_difference.png", dpi=150)
    plt.close(fig)
    print(f"  Saved topomap_sci_difference.png")

    # ── Figure 3: top-20 channels by moderate−baseline effect ────────────────
    n_top = 20
    order = np.argsort(diff_mod)[::-1]
    top_idx   = order[:n_top]
    top_names = [ch_names[i] for i in top_idx]
    top_diff  = diff_mod[top_idx]

    fig, ax = plt.subplots(figsize=(10, 5))
    colors = ["tomato" if d > 0 else "steelblue" for d in top_diff]
    ax.bar(range(n_top), top_diff, color=colors)
    ax.set_xticks(range(n_top))
    ax.set_xticklabels(top_names, rotation=45, ha="right")
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_ylabel("ΔSCI (Moderate − Baseline)")
    ax.set_title(f"Top {n_top} Channels by SCI Increase at Moderate Sedation\n"
                 f"(Grand average, N=20 subjects)")
    plt.tight_layout()
    fig.savefig(out_dir / "top_channels_moderate_vs_baseline.png", dpi=150)
    plt.close(fig)
    print(f"  Saved top_channels_moderate_vs_baseline.png")


# ── main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="Sedation-RestingState")
    parser.add_argument("--out",      default="results/chennu_spatial_v1")
    parser.add_argument("--subjects", nargs="*")
    parser.add_argument("--workers",  type=int,
                        default=min(8, mp.cpu_count()))
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    out_dir  = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    subject_files = find_subject_files(data_dir)
    if args.subjects:
        subject_files = {k: v for k, v in subject_files.items()
                         if k in args.subjects}

    # channel info from first file
    first_set = next(iter(subject_files.values()))[0]
    ch_names, xyz = load_channel_info(first_set)
    n_ch = len(ch_names)
    print(f"Channels: {n_ch}  |  Subjects: {len(subject_files)}  |  Workers: {args.workers}")

    # ── run parallel ──────────────────────────────────────────────────────────
    work = list(subject_files.items())
    with mp.Pool(processes=args.workers) as pool:
        results = pool.map(process_subject, work)

    all_records = [r for sublist in results for r in sublist]
    df = pd.DataFrame(all_records)
    df.to_csv(out_dir / "channel_epoch_means.csv", index=False)
    print(f"\nSaved channel_epoch_means.csv  ({len(df)} rows)")

    # ── grand average per channel per state ───────────────────────────────────
    grand_df = (df.groupby(["state", "ch_idx"])
                [["SCI", "gap"]]
                .mean()
                .reset_index())

    # save channel labels into grand_df
    grand_df["ch_name"] = grand_df["ch_idx"].apply(lambda i: ch_names[i])
    grand_df.to_csv(out_dir / "channel_state_means.csv", index=False)

    # dict: state → array(n_ch)
    grand = {}
    for state in STATES:
        sub = grand_df[grand_df["state"] == state].sort_values("ch_idx")
        grand[state] = sub["SCI"].values

    # ── summary table ─────────────────────────────────────────────────────────
    diff = grand["moderate"] - grand["baseline"]
    top5 = np.argsort(diff)[::-1][:5]
    print("\nTop 5 channels by ΔSCI (Moderate − Baseline):")
    for i in top5:
        print(f"  {ch_names[i]:8s}  Δ={diff[i]:+.4f}  "
              f"baseline={grand['baseline'][i]:.4f}  "
              f"moderate={grand['moderate'][i]:.4f}")

    # ── topomaps ──────────────────────────────────────────────────────────────
    print("\nGenerating topomaps...")
    plot_topomaps(grand, ch_names, xyz, out_dir)

    print(f"\nDone. Outputs in {out_dir}/")


if __name__ == "__main__":
    main()
