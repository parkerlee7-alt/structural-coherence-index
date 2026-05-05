#!/usr/bin/env python3
import os
import numpy as np
import pandas as pd

INFILE = "results_garch_finance/financial_garch_sci_results.csv"
OUTDIR = "results_garch_finance"
OUTFILE = os.path.join(OUTDIR, "financial_garch_sci_results_v2_ranked.csv")

def pct_rank(s):
    return s.rank(pct=True, method="average")

def bucket_from_row(row):
    gap = row.get("w500_gap", np.nan)
    gp = row.get("gap_percentile", np.nan)

    if not np.isfinite(gap) or not np.isfinite(gp):
        return "NO_DATA"

    if gap < 0:
        return "SPECTRAL_DOMINANT"

    if gp >= 0.90:
        return "AMPLITUDE_CORE_TOP10"

    if gp >= 0.70:
        return "AMPLITUDE_MID_70_90"

    if gp >= 0.40:
        return "AMPLITUDE_TACTICAL_40_70"

    return "AMPLITUDE_LOW_BOTTOM40"

def main():
    if not os.path.exists(INFILE):
        raise FileNotFoundError(f"Could not find {INFILE}")

    df = pd.read_csv(INFILE)

    # Percentile ranks
    df["gap_percentile"] = pct_rank(df["w500_gap"])
    df["sci_percentile"] = pct_rank(df["w500_SCI"])
    df["vol_persistence_percentile"] = pct_rank(df["vol_persistence_proxy"])

    if "garch_persistence" in df.columns:
        df["garch_persistence_percentile"] = pct_rank(df["garch_persistence"])
    else:
        df["garch_persistence_percentile"] = np.nan

    # Combined score: mostly SCI gap, with some volatility/GARCH persistence.
    # This keeps SCI as the main thing, but rewards true volatility persistence.
    df["combined_amplitude_score"] = (
        0.60 * df["gap_percentile"].fillna(0)
        + 0.20 * df["vol_persistence_percentile"].fillna(0)
        + 0.20 * df["garch_persistence_percentile"].fillna(0)
    )

    df["combined_amplitude_percentile"] = pct_rank(df["combined_amplitude_score"])

    # New percentile-based bucket
    df["amplitude_bucket_v2"] = df.apply(bucket_from_row, axis=1)

    # Useful ranking deltas
    df["gap_minus_vol_proxy_pct"] = df["gap_percentile"] - df["vol_persistence_percentile"]
    df["gap_minus_garch_pct"] = df["gap_percentile"] - df["garch_persistence_percentile"]

    # Save full ranked file
    df.to_csv(OUTFILE, index=False)

    # Save bucket counts
    counts = df["amplitude_bucket_v2"].value_counts().reset_index()
    counts.columns = ["bucket", "count"]
    counts.to_csv(os.path.join(OUTDIR, "bucket_counts_v2.csv"), index=False)

    # Save top lists
    df.sort_values("gap_percentile", ascending=False).head(150).to_csv(
        os.path.join(OUTDIR, "top_150_by_gap_percentile.csv"), index=False
    )

    df.sort_values("combined_amplitude_score", ascending=False).head(150).to_csv(
        os.path.join(OUTDIR, "top_150_by_combined_amplitude_score.csv"), index=False
    )

    df.sort_values("gap_minus_vol_proxy_pct", ascending=False).head(100).to_csv(
        os.path.join(OUTDIR, "top_100_sci_gap_above_simple_vol_proxy.csv"), index=False
    )

    df.sort_values("gap_percentile", ascending=True).head(100).to_csv(
        os.path.join(OUTDIR, "bottom_100_by_gap_percentile.csv"), index=False
    )

    # Plain English report
    report = []
    report.append("=" * 80)
    report.append("GARCH-SCI POSTPROCESS V2")
    report.append("=" * 80)
    report.append("")
    report.append(f"Rows analyzed: {len(df)}")
    report.append("")
    report.append("New percentile bucket counts:")
    report.append(counts.to_string(index=False))
    report.append("")
    report.append("Top 25 by raw SCI gap percentile:")
    show_cols = [
        "ticker", "w500_gap", "gap_percentile", "w500_SCI",
        "vol_persistence_proxy", "garch_persistence",
        "combined_amplitude_score", "amplitude_bucket_v2",
        "annualized_vol", "max_drawdown"
    ]
    show_cols = [c for c in show_cols if c in df.columns]
    report.append(
        df.sort_values("gap_percentile", ascending=False)
        .head(25)[show_cols]
        .to_string(index=False)
    )
    report.append("")
    report.append("Top 25 by combined amplitude score:")
    report.append(
        df.sort_values("combined_amplitude_score", ascending=False)
        .head(25)[show_cols]
        .to_string(index=False)
    )
    report.append("")
    report.append("Interpretation:")
    report.append(
        "V2 buckets are percentile-based. This is better for finance because most liquid assets "
        "show some amplitude-envelope excess. The useful question is not yes/no; it is relative rank."
    )

    with open(os.path.join(OUTDIR, "summary_report_v2_ranked.txt"), "w") as f:
        f.write("\n".join(report))

    print("DONE")
    print(f"Saved: {OUTFILE}")
    print(f"Saved: {os.path.join(OUTDIR, 'summary_report_v2_ranked.txt')}")
    print(f"Saved: {os.path.join(OUTDIR, 'bucket_counts_v2.csv')}")

if __name__ == "__main__":
    main()
