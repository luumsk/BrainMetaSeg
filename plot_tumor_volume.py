#!/usr/bin/env python3
"""Plot tumor volume over time from tumor_tracking.py's output CSV.

Reads the long-format CSV produced by `tumor_tracking.py` (or
`scripts/run_tumor_tracking.sh`) -- one row per tracked tumor per timepoint,
with at least: patient_id, tumor_id, date, volume_cm3 -- and draws a
RANO-BM-style volume-over-time chart per tracked tumor:

* the volume trend line (mL) across all timepoints for that tumor_id
* the running nadir (lowest volume seen so far) marked and labeled
* a progression-threshold reference line (nadir x (1 + --progression-threshold),
  default 40% above nadir) with the first timepoint that crosses it flagged

This mirrors the nadir/progression logic clinicians use to call disease
progression (RANO-BM/RECIST-style: compare each follow-up to the lowest
volume observed since baseline, not to baseline itself).

Usage
-----
    python plot_tumor_volume.py --input-csv tumor_volumes.csv

If the CSV contains more than one tumor_id, one plot (and one text summary)
is written per tumor_id. Restrict to specific tracks with --tumor-id.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Optional

import matplotlib
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

matplotlib.use("Agg")

logger = logging.getLogger("plot_tumor_volume")

# ---- palette (validated: see dataviz skill reference) ---------------------
SURFACE = "#fcfcfb"
INK_PRIMARY = "#0b0b0b"
INK_SECONDARY = "#52514e"
INK_MUTED = "#898781"
GRIDLINE = "#e1e0d9"
AXIS_LINE = "#c3c2b7"
VOLUME_COLOR = "#2a78d6"  # categorical slot 1 (blue)
CRITICAL = "#d03b3b"  # status: critical (progression)


def load_tracking_csv(input_csv: Path) -> pd.DataFrame:
    df = pd.read_csv(input_csv, parse_dates=["date"])
    required = {"tumor_id", "date", "volume_cm3"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(
            f"{input_csv} is missing required column(s) {sorted(missing)}. "
            "This script expects the CSV produced by tumor_tracking.py, not a raw data-quality report."
        )
    return df


def compute_nadir_progression(
    dates: np.ndarray, volumes: np.ndarray, progression_threshold: float
) -> dict:
    """Running nadir + every progression *onset* (RANO-BM/RECIST-style).

    A timepoint is "above threshold" when its volume is >= (1 + threshold)
    times the lowest volume observed up to and including that timepoint (the
    running nadir). Volume can cross back under the threshold later (e.g. the
    tumor shrinks again) and then cross it again after a later, lower nadir --
    each such crossing is a distinct clinical episode, so every rising edge
    (not just the first ever) is reported as its own progression event,
    paired with the nadir it was measured against.
    """
    running_nadir = np.minimum.accumulate(volumes)
    threshold_line = running_nadir * (1.0 + progression_threshold)
    above_threshold = volumes >= threshold_line

    episodes = []
    for i in range(len(volumes)):
        is_onset = above_threshold[i] and (i == 0 or not above_threshold[i - 1])
        if is_onset:
            nadir_idx = int(np.argmin(volumes[: i + 1]))
            episodes.append({"nadir_idx": nadir_idx, "progression_idx": i})

    overall_nadir_idx = int(np.argmin(volumes))

    return {
        "running_nadir": running_nadir,
        "threshold_line": threshold_line,
        "overall_nadir_idx": overall_nadir_idx,
        "episodes": episodes,
    }


def format_summary_text(
    tumor_id: str, dates: np.ndarray, volumes: np.ndarray, result: dict, progression_threshold: float
) -> str:
    lines = [f"TUMOR VOLUME SUMMARY -- {tumor_id}", "=" * 60, ""]
    lines.append(f"{'Date':12s} {'Volume (mL)':>12s} {'vs. nadir':>12s}")
    lines.append("-" * 40)
    for date, volume, nadir in zip(dates, volumes, result["running_nadir"]):
        pct_vs_nadir = (volume / nadir - 1.0) * 100.0 if nadir > 0 else float("nan")
        date_str = pd.Timestamp(date).strftime("%Y-%m-%d")
        lines.append(f"{date_str:12s} {volume:12.3f} {pct_vs_nadir:11.1f}%")
    lines.append("")

    overall_nadir_idx = result["overall_nadir_idx"]
    overall_nadir_date = pd.Timestamp(dates[overall_nadir_idx]).strftime("%Y-%m-%d")
    lines.append(f"Overall nadir: {overall_nadir_date} = {volumes[overall_nadir_idx]:.3f} mL")
    lines.append("")

    episodes = result["episodes"]
    if episodes:
        lines.append(f"Progression episode(s) (>= {progression_threshold * 100:.0f}% above nadir at the time):")
        for episode in episodes:
            nadir_idx = episode["nadir_idx"]
            progression_idx = episode["progression_idx"]
            nadir_date = pd.Timestamp(dates[nadir_idx]).strftime("%Y-%m-%d")
            prog_date = pd.Timestamp(dates[progression_idx]).strftime("%Y-%m-%d")
            pct = (volumes[progression_idx] / volumes[nadir_idx] - 1.0) * 100.0
            lines.append(
                f"  - nadir {nadir_date} ({volumes[nadir_idx]:.3f} mL) -> "
                f"progression {prog_date} ({volumes[progression_idx]:.3f} mL, {pct:+.1f}%)"
            )
    else:
        latest_pct = (volumes[-1] / volumes[overall_nadir_idx] - 1.0) * 100.0
        lines.append(
            f"No timepoint has reached the {progression_threshold * 100:.0f}% progression threshold. "
            f"Latest value is {latest_pct:+.1f}% vs. overall nadir."
        )

    return "\n".join(lines)


def _annotate_marker(ax, x, y, text: str, arrow_color: str, y_top: float, offset: int = 34) -> None:
    """Place a leader-lined label above or below its point, whichever has room.

    Points near the bottom of the y-range get labeled above (and vice versa)
    so labels never run into the x-axis tick labels or off the top of the figure.
    """
    place_above = y < y_top * 0.5
    ax.annotate(
        text,
        xy=(x, y),
        xytext=(0, offset if place_above else -offset),
        textcoords="offset points",
        ha="center",
        va="bottom" if place_above else "top",
        color=INK_SECONDARY,
        fontsize=8.5,
        linespacing=1.6,
        arrowprops={"arrowstyle": "-", "color": arrow_color, "linewidth": 1.2},
    )


def plot_tumor_volume(
    tumor_id: str,
    df: pd.DataFrame,
    progression_threshold: float,
    output_path: Path,
    dpi: int,
) -> str:
    df = df.sort_values("date").reset_index(drop=True)
    dates = df["date"].to_numpy()
    volumes = df["volume_cm3"].to_numpy(dtype=float)

    result = compute_nadir_progression(dates, volumes, progression_threshold)
    threshold_line = result["threshold_line"]
    overall_nadir_idx = result["overall_nadir_idx"]
    episodes = result["episodes"]

    fig, ax = plt.subplots(figsize=(10, 5.5), dpi=dpi)
    fig.patch.set_facecolor(SURFACE)
    ax.set_facecolor(SURFACE)

    # Reference threshold line (drawn first, sits behind the data line)
    ax.plot(
        dates,
        threshold_line,
        color=INK_SECONDARY,
        linewidth=1.5,
        linestyle=(0, (4, 3)),
        alpha=0.9,
        zorder=2,
    )
    ax.annotate(
        f"nadir +{progression_threshold * 100:.0f}%",
        xy=(dates[-1], threshold_line[-1]),
        xytext=(8, 0),
        textcoords="offset points",
        va="center",
        ha="left",
        color=INK_SECONDARY,
        fontsize=9,
    )

    # Main volume trend line
    ax.plot(
        dates,
        volumes,
        color=VOLUME_COLOR,
        linewidth=2,
        marker="o",
        markersize=6,
        markerfacecolor=VOLUME_COLOR,
        markeredgecolor=SURFACE,
        markeredgewidth=1.5,
        solid_capstyle="round",
        solid_joinstyle="round",
        zorder=3,
    )
    ax.annotate(
        f"{volumes[-1]:.2f} mL",
        xy=(dates[-1], volumes[-1]),
        xytext=(8, 0),
        textcoords="offset points",
        va="center",
        ha="left",
        color=INK_PRIMARY,
        fontsize=9,
        fontweight="bold",
    )

    y_top = float(max(volumes.max(), threshold_line.max())) * 1.05

    # Nadir marker(s) + label(s): the overall nadir, plus any earlier nadir
    # that preceded a progression episode (each episode is measured against
    # the nadir current at that point, which may not be the overall nadir).
    labeled_nadir_idxs = {episode["nadir_idx"] for episode in episodes} | {overall_nadir_idx}
    for nadir_idx in sorted(labeled_nadir_idxs):
        ax.plot(
            dates[nadir_idx],
            volumes[nadir_idx],
            marker="o",
            markersize=10,
            markerfacecolor="none",
            markeredgecolor=INK_PRIMARY,
            markeredgewidth=2,
            zorder=4,
        )
        label = "Nadir" if nadir_idx == overall_nadir_idx else "Local nadir"
        _annotate_marker(
            ax,
            dates[nadir_idx],
            volumes[nadir_idx],
            f"{label}\n{volumes[nadir_idx]:.2f} mL\n{pd.Timestamp(dates[nadir_idx]).strftime('%Y-%m')}",
            INK_MUTED,
            y_top,
        )

    for episode in episodes:
        nadir_idx, progression_idx = episode["nadir_idx"], episode["progression_idx"]
        pct = (volumes[progression_idx] / volumes[nadir_idx] - 1.0) * 100.0
        ax.plot(
            dates[progression_idx],
            volumes[progression_idx],
            marker="o",
            markersize=10,
            markerfacecolor=CRITICAL,
            markeredgecolor=SURFACE,
            markeredgewidth=1.5,
            zorder=5,
        )
        _annotate_marker(
            ax,
            dates[progression_idx],
            volumes[progression_idx],
            f"Progression\n{pd.Timestamp(dates[progression_idx]).strftime('%Y-%m')}  ({pct:+.0f}%)",
            CRITICAL,
            y_top,
        )

    if episodes:
        last = episodes[-1]
        pct = (volumes[last["progression_idx"]] / volumes[last["nadir_idx"]] - 1.0) * 100.0
        status_line = (
            f"{len(episodes)} progression episode(s); most recent "
            f"{pd.Timestamp(dates[last['progression_idx']]).strftime('%Y-%m-%d')} ({pct:+.0f}% vs. nadir)"
        )
    else:
        status_line = "No progression from nadir detected"

    fig.suptitle(f"Tumor volume over time — {tumor_id}", color=INK_PRIMARY, fontsize=13, x=0.02, ha="left", y=0.98)
    ax.set_title(status_line, color=INK_SECONDARY, fontsize=9.5, loc="left", pad=12)

    ax.set_ylabel("Tumor volume (mL)", color=INK_SECONDARY, fontsize=10)
    ax.set_ylim(bottom=0)

    ax.yaxis.grid(True, color=GRIDLINE, linewidth=1, zorder=0)
    ax.xaxis.grid(False)
    ax.set_axisbelow(True)

    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    for spine in ("left", "bottom"):
        ax.spines[spine].set_color(AXIS_LINE)

    ax.tick_params(colors=INK_MUTED, labelsize=9)
    locator = mdates.AutoDateLocator()
    ax.xaxis.set_major_locator(locator)
    ax.xaxis.set_major_formatter(mdates.ConciseDateFormatter(locator))

    # Leave room on the right for the end-of-line labels.
    x_min, x_max = mdates.date2num(dates[0]), mdates.date2num(dates[-1])
    ax.set_xlim(x_min - (x_max - x_min) * 0.02, x_max + (x_max - x_min) * 0.14)

    fig.tight_layout(rect=(0, 0, 1, 0.94))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=dpi, facecolor=fig.get_facecolor())
    plt.close(fig)

    return format_summary_text(tumor_id, dates, volumes, result, progression_threshold)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Plot tumor volume over time (with nadir/progression annotations) from tumor_tracking.py output.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--input-csv", required=True, type=Path, help="CSV produced by tumor_tracking.py (e.g. tumor_volumes.csv)")
    parser.add_argument(
        "--tumor-id",
        nargs="*",
        default=None,
        help="Restrict to these tumor_id value(s) (default: plot every tumor_id found in the CSV)",
    )
    parser.add_argument(
        "--progression-threshold",
        type=float,
        default=0.40,
        help="Fraction above the running nadir considered progression (0.40 = RANO-BM-style 40%% rule)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output image path (default: <input-csv stem>_plot.png; a tumor_id suffix is added when plotting more than one track)",
    )
    parser.add_argument("--dpi", type=int, default=150)
    parser.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    return parser


def main(argv: Optional[list[str]] = None) -> None:
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    logging.basicConfig(level=args.log_level, format="%(asctime)s %(levelname)s %(message)s")

    df = load_tracking_csv(args.input_csv)

    tumor_ids = args.tumor_id if args.tumor_id else sorted(df["tumor_id"].unique())
    missing_ids = set(tumor_ids) - set(df["tumor_id"].unique())
    if missing_ids:
        raise ValueError(f"tumor_id(s) not found in {args.input_csv}: {sorted(missing_ids)}")

    output_base = args.output or args.input_csv.with_name(args.input_csv.stem + "_plot.png")
    multiple = len(tumor_ids) > 1

    for tumor_id in tumor_ids:
        group = df[df["tumor_id"] == tumor_id]
        if multiple:
            output_path = output_base.with_name(f"{output_base.stem}_{tumor_id}{output_base.suffix}")
        else:
            output_path = output_base

        summary = plot_tumor_volume(
            tumor_id=tumor_id,
            df=group,
            progression_threshold=args.progression_threshold,
            output_path=output_path,
            dpi=args.dpi,
        )

        summary_path = output_path.with_suffix(".txt")
        summary_path.write_text(summary)

        logger.info("Wrote %s and %s", output_path, summary_path)
        print(summary)
        print()


if __name__ == "__main__":
    main()
