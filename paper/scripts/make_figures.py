#!/usr/bin/env python3
"""Draw the result plots of the paper in the figures4papers house style.

Style source: https://github.com/ChenLiu-1996/figures4papers (CC BY-NC 4.0),
checked out under paper/third_party/figures4papers.  That repository ships a
specification (scientific-figure-making/references/api.md) and not an
importable module, so this file implements that specification: the PALETTE
and DEFAULT_COLORS constants, FigureStyle, apply_publication_style,
create_subplots, finalize_figure, and the bar, trend and scatter helpers.

House style kept as specified
    open frame (top and right spines off), frameless legends, no grid,
    black bar edges, hatches for print-safe separation, values written on
    the bars, blue for the proposed method, red for contrasts, neutral grey
    for reference categories, a dedicated legend panel for shared legends.

Two deliberate deviations, both required by the paper
    the font is a Times face (Liberation Serif, metric-compatible with Times
    New Roman) instead of Helvetica, to match the text of an IEEE paper;
    figures are drawn at their final size with 9 pt text, so that every figure
    has the same font size after inclusion without scaling.

All data come from paper/generated/*.dat, written by build_data.py.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.ticker import FixedLocator, NullLocator  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "generated"
OUT = ROOT / "figures" / "plots"

# --------------------------------------------------------------------------
# figures4papers API (references/api.md)
# --------------------------------------------------------------------------
PALETTE = {
    "blue_main": "#0F4D92",
    "blue_secondary": "#3775BA",
    "green_1": "#DDF3DE", "green_2": "#AADCA9", "green_3": "#8BCF8B",
    "red_1": "#F6CFCB", "red_2": "#E9A6A1", "red_strong": "#B64342",
    "neutral": "#CFCECE", "highlight": "#FFD700",
    "teal": "#42949E", "violet": "#9A4D8E",
}
DARK_NEUTRAL = "#4D4D4D"
DEFAULT_COLORS = [PALETTE[name] for name in ("blue_main", "green_3", "red_strong", "teal", "violet", "neutral")]


@dataclass(frozen=True)
class FigureStyle:
    font_size: int = 16
    axes_linewidth: float = 2.5
    use_tex: bool = False
    font_family: tuple[str, ...] = ("DejaVu Sans", "Helvetica", "Arial", "sans-serif")


def apply_publication_style(style: FigureStyle | None = None) -> FigureStyle:
    style = style or FigureStyle()
    plt.rcParams.update({
        "font.family": list(style.font_family),
        "font.size": style.font_size,
        "axes.labelsize": style.font_size,
        "axes.titlesize": style.font_size,
        "xtick.labelsize": style.font_size,
        "ytick.labelsize": style.font_size,
        "legend.fontsize": style.font_size,
        "axes.spines.right": False,
        "axes.spines.top": False,
        "axes.linewidth": style.axes_linewidth,
        "xtick.major.width": style.axes_linewidth,
        "ytick.major.width": style.axes_linewidth,
        "xtick.minor.width": style.axes_linewidth * 0.7,
        "ytick.minor.width": style.axes_linewidth * 0.7,
        "xtick.major.size": 3.5,
        "ytick.major.size": 3.5,
        "legend.frameon": False,
        "text.usetex": style.use_tex,
        "mathtext.fontset": "stix",
        "svg.fonttype": "none",
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "hatch.linewidth": 0.6,
        "axes.grid": False,
    })
    return style


def create_subplots(nrows: int = 1, ncols: int = 1, figsize=None, **kwargs):
    fig, axes = plt.subplots(nrows, ncols, figsize=figsize, **kwargs)
    return fig, np.atleast_1d(axes).ravel()


def finalize_figure(fig, out_path, formats=None, dpi: int = 300, close: bool = True, pad: float = 0.05, **kwargs):
    out_path = Path(out_path)
    supported = {"pdf", "svg", "eps", "png", "jpg", "jpeg", "tif", "tiff"}
    if formats is None:
        formats = [out_path.suffix.lstrip(".")] if out_path.suffix else ["pdf", "svg", "eps"]
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if kwargs.pop("tight", True):
        fig.tight_layout(pad=pad)
    saved = []
    for fmt in formats:
        if fmt not in supported:
            raise ValueError(f"unsupported format: {fmt}")
        target = out_path.with_suffix("." + fmt)
        fig.savefig(target, dpi=dpi, **kwargs)
        saved.append(target)
    if close:
        plt.close(fig)
    return saved


def annotate_bars(ax, bars, fmt: str = "{:.2f}", fontsize: int = 10, padding: float = 3):
    for bar in bars:
        height = bar.get_height()
        ax.annotate(fmt.format(height), (bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, padding), textcoords="offset points", ha="center", va="bottom", fontsize=fontsize)


def make_grouped_bar(ax, categories, series, labels, ylabel: str = "Value", colors=None, annotate: bool = False,
                     edge_width: float = 0.9, fmt: str = "{:.0f}", fontsize: int = 9):
    series = [np.asarray(values, dtype=float) for values in series]
    if any(values.ndim != 1 or len(values) != len(categories) for values in series):
        raise ValueError("every series must be one-dimensional and match the categories")
    colors = colors or DEFAULT_COLORS
    width = 0.8 / len(series)
    positions = np.arange(len(categories))
    bars = None
    for index, (values, label) in enumerate(zip(series, labels)):
        offset = (index - (len(series) - 1) / 2) * width
        bars = ax.bar(positions + offset, values, width, label=label, color=colors[index % len(colors)],
                      edgecolor="black", linewidth=edge_width)
        if annotate:
            annotate_bars(ax, bars, fmt=fmt, fontsize=fontsize, padding=2)
    ax.set_xticks(positions)
    ax.set_xticklabels(categories)
    ax.set_ylabel(ylabel)
    return bars


def make_trend(ax, x, y_series, labels, colors=None, ylabel=None, xlabel=None, show_shadow: bool = True,
               markers=None, linewidth: float = 1.8, markersize: float = 5.0):
    x = np.asarray(x, dtype=float)
    colors = colors or DEFAULT_COLORS
    markers = markers or ["o"] * len(y_series)
    for index, (values, label) in enumerate(zip(y_series, labels)):
        values = np.asarray(values, dtype=float)
        if values.shape != x.shape:
            raise ValueError("each series must have the same length as x")
        ax.plot(x, values, label=label, color=colors[index % len(colors)], linewidth=linewidth,
                marker=markers[index % len(markers)], markersize=markersize,
                markeredgecolor="black", markeredgewidth=0.5, clip_on=False)
        if show_shadow:
            ax.fill_between(x, values, alpha=0.08, color=colors[index % len(colors)])
    if ylabel:
        ax.set_ylabel(ylabel)
    if xlabel:
        ax.set_xlabel(xlabel)


def make_scatter(ax, x, y, label=None, color=None, size: float = 50, alpha: float = 0.7, **kwargs):
    x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    if x.shape != y.shape or x.ndim != 1:
        raise ValueError("x and y must be one-dimensional arrays of the same length")
    return ax.scatter(x, y, s=size, alpha=alpha, label=label, color=color, **kwargs)


# --------------------------------------------------------------------------
# Paper constants
# --------------------------------------------------------------------------
FONT = 9                       # pt, identical in every figure and diagram
COLUMN = 3.45                  # inches, one IEEE column
STYLE = FigureStyle(font_size=FONT, axes_linewidth=1.0, font_family=("Liberation Serif",))
OURS = PALETTE["blue_main"]
CONTRAST = PALETTE["red_strong"]
REFERENCE = PALETTE["neutral"]
SYSTEM = re.search(r"\\newcommand\{\\sys\}\{([^}]*)\}", (ROOT / "main.tex").read_text()).group(1)


def read_table(name: str) -> tuple[list[str], list[list[str]]]:
    lines = [line for line in (DATA / name).read_text().splitlines() if line.strip()]
    header = lines[0].split()
    rows = []
    for line in lines[1:]:
        match = re.match(r"\{([^}]*)\}\s+(.*)", line)
        rows.append([match.group(1)] + match.group(2).split() if match else line.split())
    return header, rows


def column(rows, index) -> np.ndarray:
    return np.array([float(row[index]) for row in rows])


# --------------------------------------------------------------------------
# Latency per statement
# --------------------------------------------------------------------------
def per_query() -> None:
    handles = [
        Line2D([], [], linestyle="none", marker="o", markersize=4.5, markerfacecolor="none",
               markeredgecolor=DARK_NEUTRAL, markeredgewidth=1.0, label="Original, finished"),
        Line2D([], [], linestyle="none", marker="^", markersize=5.5, markerfacecolor="none",
               markeredgecolor=CONTRAST, markeredgewidth=1.0, label="Original, cut off at 300 s"),
        Line2D([], [], linestyle="none", marker="o", markersize=4.5, markerfacecolor=OURS,
               markeredgecolor="black", markeredgewidth=0.4, label=f"{SYSTEM} rewrite (no mark: original kept)"),
    ]
    for key, xmax, ticks in (("dev", 31, [1, 5, 10, 15, 20, 25, 30]),
                             ("heldAll", 134, [1, 20, 40, 60, 80, 100, 120])):
        fig, axes = create_subplots(figsize=(COLUMN, 2.08))
        ax = axes[0]
        _, finished = read_table(f"perquery_{key}_finished.dat")
        _, cut_off = read_table(f"perquery_{key}_censored.dat")
        _, ours = read_table(f"perquery_{key}_ours.dat")
        make_scatter(ax, column(finished, 0), column(finished, 1), size=16, alpha=1.0, color="none",
                     edgecolors=DARK_NEUTRAL, linewidths=0.9, marker="o", zorder=2)
        make_scatter(ax, column(cut_off, 0), column(cut_off, 1), size=24, alpha=1.0, color="none",
                     edgecolors=CONTRAST, linewidths=0.9, marker="^", zorder=2)
        make_scatter(ax, column(ours, 0), column(ours, 1), size=18, alpha=1.0, color=OURS,
                     edgecolors="black", linewidths=0.4, marker="o", zorder=3)
        ax.set_yscale("log")
        ax.set_ylim(6e-4, 700)
        ax.yaxis.set_major_locator(FixedLocator([1e-3, 1e-2, 1e-1, 1, 10, 100]))
        ax.yaxis.set_minor_locator(NullLocator())
        ax.set_xlim(0, xmax)
        ax.set_xticks(ticks)
        ax.set_xlabel("Statements ordered by original latency")
        ax.set_ylabel("Latency (s)")
        finalize_figure(fig, OUT / f"perquery_{key}.pdf", pad=0.3)
    fig, axes = create_subplots(figsize=(2 * COLUMN, 0.28))
    axes[0].set_axis_off()
    axes[0].legend(handles=handles, loc="center", ncol=3, handletextpad=0.3, columnspacing=1.8, borderaxespad=0)
    finalize_figure(fig, OUT / "perquery_legend.pdf", pad=0.05)


# --------------------------------------------------------------------------
# Paired outcomes against the replay arms
# --------------------------------------------------------------------------
def paired() -> None:
    header, rows = read_table("paired.dat")
    classes = [("faster", f"{SYSTEM} faster", OURS, None), ("within", "Within 10 %", PALETTE["green_2"], None),
               ("same", "Same statement", REFERENCE, None), ("slower", f"{SYSTEM} slower", CONTRAST, None),
               ("undetermined", "Undetermined", "white", "////")]
    fig, axes = create_subplots(figsize=(COLUMN, 3.28))
    ax = axes[0]
    BAR = 0.62
    positions, labels, headers = [], [], []
    y = 0.0
    for index in range(0, len(rows), 2):
        cohort = rows[index][0].rsplit(", ", 1)[0]
        headers.append((y, cohort))
        y += 1.02
        for row in rows[index:index + 2]:
            positions.append(y)
            labels.append(row[0].rsplit(", ", 1)[1])
            y += 1.0
        y += 0.5
    left = np.zeros(len(rows))
    for name, label, color, hatch in classes:
        values = column(rows, header.index(name))
        ax.barh(positions, values, left=left, height=BAR, color=color, edgecolor="black", linewidth=0.8,
                hatch=hatch, label=label)
        left += values
    for position, cohort in headers:
        ax.annotate(cohort, xy=(0, position + 1.02 - BAR / 2), xytext=(0, 2.5), textcoords="offset points",
                    ha="left", va="bottom", fontsize=FONT, fontstyle="italic")
    ax.set_yticks(positions)
    ax.set_yticklabels(labels)
    ax.set_ylim(max(positions) + 0.6, -0.6)
    ax.set_xlim(0, 100)
    ax.set_xticks([0, 20, 40, 60, 80, 100])
    ax.set_xlabel("Share of statements (%)")
    ax.tick_params(axis="y", length=0)
    fig.legend(loc="upper center", ncol=3, handlelength=1.3, handletextpad=0.4, columnspacing=1.0,
               bbox_to_anchor=(0.5, 1.0))
    fig.tight_layout(pad=0.3, rect=(0, 0, 1, 0.87))
    finalize_figure(fig, OUT / "paired.pdf", tight=False)


# --------------------------------------------------------------------------
# The same statements on two databases
# --------------------------------------------------------------------------
def skew_scale() -> None:
    fig, axes = create_subplots(1, 2, figsize=(COLUMN, 2.35), sharey=True)
    panels = (("skew.dat", ["Uniform", "Zipf"], "(a) Calcite,\n30 statements"),
              ("scale.dat", ["SF 10", "SF 50"], "(b) TPC-H and DSB,\n20 statements"))
    for ax, (name, categories, title) in zip(axes, panels):
        _, rows = read_table(name)
        original, original_gap = column(rows, 1), column(rows, 2)
        ours, ours_gap = column(rows, 3), column(rows, 4)
        make_grouped_bar(ax, categories, [original, ours], ["Original", SYSTEM],
                         ylabel="", colors=[REFERENCE, OURS], annotate=True)
        width = 0.4
        for offset, values, gaps in ((-width / 2, original, original_gap), (width / 2, ours, ours_gap)):
            for index, (value, gap) in enumerate(zip(values, gaps)):
                if gap > 0.05:
                    ax.plot([index + offset, index + offset], [value - gap, value], color="black", linewidth=1.2,
                            solid_capstyle="butt", zorder=4)
                    ax.plot([index + offset - 0.07, index + offset + 0.07], [value - gap, value - gap],
                            color="black", linewidth=1.2, zorder=4)
        ax.set_ylim(0, 330)
        ax.yaxis.set_major_locator(FixedLocator([0, 100, 200, 300]))
        ax.set_xlabel(title, labelpad=6)
        ax.tick_params(axis="x", length=0)
    axes[0].set_ylabel("Mean latency (s)")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=2, handlelength=1.3, handletextpad=0.4,
               columnspacing=1.6, bbox_to_anchor=(0.5, 1.0))
    fig.tight_layout(pad=0.3, w_pad=0.6, rect=(0, 0, 1, 0.90))
    finalize_figure(fig, OUT / "skew_scale.pdf", tight=False)


# --------------------------------------------------------------------------
# Budget
# --------------------------------------------------------------------------
def budget() -> None:
    header, rows = read_table("budget.dat")
    budgets = [row[0] for row in rows]
    x = np.arange(len(budgets))
    series = [("dev", "Dev", OURS, "o"), ("tpchHeld", "TPC-H held-out", CONTRAST, "s"),
              ("calciteHeld", "Calcite held-out", PALETTE["teal"], "^"), ("zipf", "Zipf", PALETTE["violet"], "D"),
              ("sfFifty", "SF 50", DARK_NEUTRAL, "X")]
    fig, axes = create_subplots(1, 2, figsize=(COLUMN, 2.35))
    for ax, suffix, ylabel, title, limits, ticks in (
            (axes[0], "", "Mean latency ratio", "(a)", (0.3, 1.0), [0.4, 0.6, 0.8, 1.0]),
            (axes[1], "Imp", "Improved statements (%)", "(b)", (0, 80), [0, 20, 40, 60, 80])):
        make_trend(ax, x, [column(rows, header.index(key + suffix)) for key, *_ in series],
                   [label for _, label, *_ in series], colors=[color for *_, color, _ in series],
                   markers=[marker for *_, marker in series], ylabel=ylabel, show_shadow=False,
                   linewidth=1.5, markersize=4.5)
        ax.set_ylim(*limits)
        ax.yaxis.set_major_locator(FixedLocator(ticks))
        ax.set_xlim(-0.3, len(budgets) - 0.7)
        ax.set_xticks(x)
        ax.set_xticklabels(budgets)
        ax.set_xlabel(f"{title} Budget $M$")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=3, handlelength=1.8, handletextpad=0.4,
               columnspacing=1.0, bbox_to_anchor=(0.5, 1.0))
    fig.tight_layout(pad=0.3, w_pad=0.8, rect=(0, 0, 1, 0.83))
    finalize_figure(fig, OUT / "budget.pdf", tight=False)


# --------------------------------------------------------------------------
# Source of the returned rewrite
# --------------------------------------------------------------------------
def sources() -> None:
    header, rows = read_table("sources.dat")
    classes = [("macro", "LLM proposal", DEFAULT_COLORS[0], None, "white"),
               ("rule", "Rule rewrite", DEFAULT_COLORS[1], "///", "black"),
               ("quite", "QUITE replay", DEFAULT_COLORS[2], "\\\\\\", "white"),
               ("hint", "Hint only", DEFAULT_COLORS[3], "...", "white"),
               ("rewritehint", "Rewrite with hint", DEFAULT_COLORS[4], "xxx", "white")]
    fig, axes = create_subplots(figsize=(COLUMN, 2.18))
    ax = axes[0]
    positions = np.arange(len(rows))
    left = np.zeros(len(rows))
    for name, label, color, hatch, text_color in classes:
        values = column(rows, header.index(name))
        ax.barh(positions, values, left=left, height=0.66, color=color, edgecolor="black", linewidth=0.8,
                hatch=hatch, label=label)
        for position, start, value in zip(positions, left, values):
            if value >= 1:
                ax.text(start + value / 2, position, f"{value:.0f}", ha="center", va="center", fontsize=FONT,
                        color=text_color,
                        bbox=dict(boxstyle="round,pad=0.12", facecolor=color, edgecolor="none"))
        left += values
    ax.set_yticks(positions)
    ax.set_yticklabels([row[0] for row in rows])
    ax.invert_yaxis()
    ax.set_xlim(0, 24)
    ax.set_xticks([0, 5, 10, 15, 20])
    ax.set_xlabel("Improved statements")
    ax.tick_params(axis="y", length=0)
    fig.legend(loc="upper center", ncol=3, handlelength=1.3, handletextpad=0.4, columnspacing=1.0,
               bbox_to_anchor=(0.5, 1.0))
    fig.tight_layout(pad=0.3, rect=(0, 0, 1, 0.84))
    finalize_figure(fig, OUT / "sources.pdf", tight=False)


# --------------------------------------------------------------------------
# Candidate funnel per cohort
# --------------------------------------------------------------------------
def funnel() -> None:
    header, rows = read_table("funnel.dat")
    stages = [("proposed", "Proposed", REFERENCE), ("admitted", "Admitted by the gate", PALETTE["blue_secondary"]),
              ("returned", "Returned", OURS)]
    fig, axes = create_subplots(figsize=(COLUMN, 2.62))
    ax = axes[0]
    height = 0.25
    positions = np.arange(len(rows)) * 1.08
    for index, (name, label, color) in enumerate(stages):
        values = column(rows, header.index(name))
        bars = ax.barh(positions + (index - 1) * height, values, height, color=color, edgecolor="black",
                       linewidth=0.8, label=label)
        for bar, value in zip(bars, values):
            ax.annotate(f"{value:.0f}", (value, bar.get_y() + bar.get_height() / 2), xytext=(3, 0),
                        textcoords="offset points", ha="left", va="center", fontsize=FONT - 1.5)
    ax.set_yticks(positions)
    ax.set_yticklabels([row[0] for row in rows])
    ax.invert_yaxis()
    ax.set_xlim(0, 300)
    ax.xaxis.set_major_locator(FixedLocator([0, 100, 200, 300]))
    ax.set_xlabel("Candidates")
    ax.tick_params(axis="y", length=0)
    handles, legend_labels = ax.get_legend_handles_labels()
    fig.legend(handles, legend_labels, loc="upper center", ncol=3, handlelength=1.3, handletextpad=0.4,
               columnspacing=1.0, bbox_to_anchor=(0.5, 1.0))
    fig.tight_layout(pad=0.3, rect=(0, 0, 1, 0.90))
    finalize_figure(fig, OUT / "funnel.pdf", tight=False)


# --------------------------------------------------------------------------
# Executions after which selection has paid for itself
# --------------------------------------------------------------------------
def breakeven() -> None:
    _, rows = read_table("breakeven.dat")
    runs = np.sort(column(rows, 1))
    share = 100.0 * np.arange(1, len(runs) + 1) / len(runs)
    fig, axes = create_subplots(figsize=(COLUMN, 1.95))
    ax = axes[0]
    ax.step(np.concatenate(([runs[0]], runs)), np.concatenate(([0.0], share)), where="post",
            color=OURS, linewidth=1.8)
    median = float(re.search(r"\\BreakevenMedian\}\{([0-9.]+)\}", (DATA / "numbers.tex").read_text()).group(1))
    ax.plot([median, median], [0, 50], color=DARK_NEUTRAL, linewidth=1.0, linestyle=(0, (4, 3)))
    ax.plot([0.1, median], [50, 50], color=DARK_NEUTRAL, linewidth=1.0, linestyle=(0, (4, 3)))
    ax.annotate(f"median {median:.1f}", (median, 50), xytext=(6, -11), textcoords="offset points", fontsize=FONT)
    ax.set_xscale("log")
    ax.set_xlim(0.1, 100)
    ax.xaxis.set_major_locator(FixedLocator([0.1, 1, 10, 100]))
    ax.set_xticklabels(["0.1", "1", "10", "100"])
    ax.xaxis.set_minor_locator(NullLocator())
    ax.set_ylim(0, 100)
    ax.yaxis.set_major_locator(FixedLocator([0, 25, 50, 75, 100]))
    ax.set_xlabel("Executions until the selection cost is recovered")
    ax.set_ylabel("Improved pairs (%)")
    finalize_figure(fig, OUT / "breakeven.pdf", pad=0.3)


# --------------------------------------------------------------------------
# Outcome of every arm against the original, all cohorts
# --------------------------------------------------------------------------
def outcomes() -> None:
    header, rows = read_table("outcomes.dat")
    classes = [("faster", "Faster than the original", OURS, None),
               ("within", "Within 10 %", PALETTE["green_2"], None),
               ("same", "Original returned", REFERENCE, None),
               ("slower", "Slower", CONTRAST, None),
               ("undetermined", "Undetermined", "white", "////")]
    cohorts = []
    for row in rows:
        cohort = row[0].rsplit(", ", 1)[0]
        if cohort not in cohorts:
            cohorts.append(cohort)
    fig, axes = create_subplots(1, len(cohorts), figsize=(2 * COLUMN + 0.1, 1.82), sharey=True)
    for ax, cohort in zip(axes, cohorts):
        subset = [row for row in rows if row[0].rsplit(", ", 1)[0] == cohort]
        names = [row[0].rsplit(", ", 1)[1].replace("ours", SYSTEM) for row in subset]
        positions = np.arange(len(subset))
        left = np.zeros(len(subset))
        for name, label, color, hatch in classes:
            values = column(subset, header.index(name))
            ax.barh(positions, values, left=left, height=0.7, color=color, edgecolor="black", linewidth=0.8,
                    hatch=hatch, label=label)
            left += values
        ax.set_yticks(positions)
        ax.set_yticklabels(names)
        ax.set_ylim(len(subset) - 0.45, -0.55)
        ax.set_xlim(0, 100)
        ax.set_xticks([0, 50, 100])
        ax.set_title(cohort, pad=4)
        ax.tick_params(axis="y", length=0)
    axes[len(axes) // 2].set_xlabel("Share of statements (%)")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=5, handlelength=1.3, handletextpad=0.4,
               columnspacing=1.4, bbox_to_anchor=(0.5, 1.0))
    fig.tight_layout(pad=0.3, w_pad=0.9, rect=(0, 0, 1, 0.88))
    finalize_figure(fig, OUT / "outcomes.pdf", tight=False)


# --------------------------------------------------------------------------
# Validation session against the campaign logs
# --------------------------------------------------------------------------
def validation_scatter() -> None:
    _, originals = read_table("validation_scatter_original.dat")
    _, rewrites = read_table("validation_scatter_rewrite.dat")
    if not originals:
        return
    values = np.concatenate([column(rows, index) for rows in (originals, rewrites) if rows for index in (0, 1)])
    low = 10.0 ** np.floor(np.log10(max(values.min(), 1e-3)))
    fig, axes = create_subplots(figsize=(COLUMN, 2.40))
    ax = axes[0]
    ax.plot([low, 600], [low, 600], color=DARK_NEUTRAL, linewidth=1.0, linestyle=(0, (4, 3)), zorder=1)
    make_scatter(ax, column(originals, 0), column(originals, 1), size=24, alpha=1.0, color="none",
                 edgecolors=CONTRAST, linewidths=1.0, marker="s", zorder=3, label="Original statement")
    if rewrites:
        make_scatter(ax, column(rewrites, 0), column(rewrites, 1), size=20, alpha=1.0, color=OURS,
                     edgecolors="black", linewidths=0.4, marker="o", zorder=2, label="Returned rewrite")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(low, 600)
    ax.set_ylim(low, 600)
    ax.xaxis.set_minor_locator(NullLocator())
    ax.yaxis.set_minor_locator(NullLocator())
    ax.set_aspect("equal")
    ax.set_xlabel("Latency in the campaign logs (s)")
    ax.set_ylabel("Latency in the session (s)")
    handles, labels = ax.get_legend_handles_labels()
    handles.append(Line2D([], [], color=DARK_NEUTRAL, linewidth=1.0, linestyle=(0, (4, 3))))
    labels.append("Equal latency")
    fig.legend(handles, labels, loc="upper center", ncol=3, handlelength=1.6, handletextpad=0.3,
               columnspacing=0.9, bbox_to_anchor=(0.5, 1.0))
    fig.tight_layout(pad=0.3, rect=(0, 0, 1, 0.90))
    finalize_figure(fig, OUT / "validation_scatter.pdf", tight=False)


def main() -> None:
    apply_publication_style(STYLE)
    funnel()
    breakeven()
    outcomes()
    validation_scatter()
    per_query()
    paired()
    skew_scale()
    budget()
    sources()
    print("wrote", ", ".join(sorted(path.name for path in OUT.glob("*.pdf"))))


if __name__ == "__main__":
    main()
