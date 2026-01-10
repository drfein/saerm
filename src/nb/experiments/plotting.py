"""
Consistent plotting utilities for bias evaluation experiments.

All plots use:
- Serif font family (Palatino-style)
- Error bars computed as 95% confidence intervals (1.96 × SE)
- Consistent color scheme across experiments
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np


@dataclass
class PlotStyle:
    """Consistent plot styling for all experiments."""
    
    # Colors
    baseline_color: str = "#5a7d9a"  # Steel blue
    nulled_color: str = "#8fbc8f"    # Sage green
    ideal_color: str = "#cc0000"     # Red for reference lines
    
    # Position bias specific colors
    position_colors: List[str] = field(default_factory=lambda: [
        "#E07A5F",  # Terracotta
        "#F4D35E",  # Golden yellow
        "#3D85C6",  # Steel blue
        "#81B29A",  # Sage
    ])
    
    # Bar colors for grouped comparisons
    bar_colors: List[str] = field(default_factory=lambda: [
        "#2b6cb0",  # Blue
        "#c53030",  # Red
    ])
    
    # Figure settings
    figsize: Tuple[float, float] = (8, 5)
    dpi: int = 150
    facecolor: str = "white"
    
    # Font settings
    font_family: str = "serif"
    font_size: int = 11
    title_size: int = 14
    label_size: int = 12
    tick_size: int = 10
    legend_size: int = 10
    
    # Spine settings
    hide_top_spine: bool = True
    hide_right_spine: bool = True
    
    def apply(self) -> None:
        """Apply style settings to matplotlib."""
        plt.rcParams.update({
            "font.family": self.font_family,
            "font.size": self.font_size,
            "axes.titlesize": self.title_size,
            "axes.labelsize": self.label_size,
            "xtick.labelsize": self.tick_size,
            "ytick.labelsize": self.tick_size,
            "legend.fontsize": self.legend_size,
            "figure.facecolor": self.facecolor,
            "axes.facecolor": self.facecolor,
            "axes.spines.top": not self.hide_top_spine,
            "axes.spines.right": not self.hide_right_spine,
            "axes.grid": True,
            "grid.alpha": 0.3,
            "grid.linestyle": "--",
        })


# Default style instance
DEFAULT_STYLE = PlotStyle()


# Multiplier for 95% confidence interval (1.96 standard errors)
CI_95_MULTIPLIER = 1.96


def binomial_ci95(p: float, n: int) -> float:
    """Compute 95% confidence interval half-width for a proportion.
    
    CI_95 = 1.96 * sqrt(p * (1-p) / n)
    
    Args:
        p: Proportion (0-1)
        n: Sample size
        
    Returns:
        Half-width of 95% confidence interval
    """
    if n <= 0:
        return 0.0
    se = math.sqrt(max(p * (1.0 - p), 0.0) / n)
    return CI_95_MULTIPLIER * se


def create_comparison_plot(
    baseline_metrics: Dict[str, float],
    nulled_metrics: Dict[str, float],
    metric_labels: List[Tuple[str, str]],
    output_path: Path,
    title: str,
    ylabel: str = "Proportion",
    ylim: Tuple[float, float] = (0.0, 1.05),
    n_examples: Optional[int] = None,
    null_alpha: Optional[float] = None,
    style: PlotStyle = DEFAULT_STYLE,
    show_values: bool = True,
    reference_line: Optional[float] = None,
    reference_label: str = "Ideal",
) -> None:
    """Create grouped bar plot comparing baseline vs nulled metrics.
    
    Args:
        baseline_metrics: Dictionary with baseline metric values
        nulled_metrics: Dictionary with nulled metric values
        metric_labels: List of (key, display_label) tuples
        output_path: Path to save the plot
        title: Plot title
        ylabel: Y-axis label
        ylim: Y-axis limits
        n_examples: Number of examples (for error bars)
        null_alpha: Nulling strength (for title)
        style: Plot style settings
        show_values: Whether to show values on bars
        reference_line: Optional horizontal reference line value
        reference_label: Label for reference line
    """
    style.apply()
    
    fig, ax = plt.subplots(figsize=style.figsize)
    
    labels = [label for _, label in metric_labels]
    keys = [key for key, _ in metric_labels]
    x = np.arange(len(labels))
    width = 0.35
    
    # Get values
    baseline_vals = [baseline_metrics.get(key, 0) for key in keys]
    nulled_vals = [nulled_metrics.get(key, 0) for key in keys]
    
    # Compute error bars if n_examples provided
    if n_examples and n_examples > 0:
        baseline_errs = [binomial_ci95(v, n_examples) for v in baseline_vals]
        nulled_errs = [binomial_ci95(v, n_examples) for v in nulled_vals]
    else:
        baseline_errs = None
        nulled_errs = None
    
    # Create bars
    bars1 = ax.bar(
        x - width / 2,
        baseline_vals,
        width,
        yerr=baseline_errs,
        label="Baseline",
        color=style.baseline_color,
        capsize=4,
        edgecolor="#333",
        linewidth=0.5,
    )
    bars2 = ax.bar(
        x + width / 2,
        nulled_vals,
        width,
        yerr=nulled_errs,
        label="With nulling",
        color=style.nulled_color,
        capsize=4,
        edgecolor="#333",
        linewidth=0.5,
    )
    
    # Reference line
    if reference_line is not None:
        ax.axhline(
            y=reference_line,
            color=style.ideal_color,
            linestyle="--",
            linewidth=1.5,
            label=reference_label,
            alpha=0.7,
        )
    
    # Value labels
    if show_values:
        for bar in bars1:
            height = bar.get_height()
            ax.annotate(
                f"{height:.2f}" if height < 1 else f"{height:.1f}",
                xy=(bar.get_x() + bar.get_width() / 2, height),
                xytext=(0, 3),
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontsize=9,
            )
        for bar in bars2:
            height = bar.get_height()
            ax.annotate(
                f"{height:.2f}" if height < 1 else f"{height:.1f}",
                xy=(bar.get_x() + bar.get_width() / 2, height),
                xytext=(0, 3),
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontsize=9,
            )
    
    # Labels and title
    ax.set_ylabel(ylabel)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylim(ylim)
    ax.legend(loc="upper right")
    
    if null_alpha is not None:
        full_title = f"{title} (alpha={null_alpha:.2f})"
    else:
        full_title = title
    ax.set_title(full_title, fontweight="bold")
    
    # Save
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=style.dpi, bbox_inches="tight", facecolor=style.facecolor)
    plt.close()


def create_position_bias_plot(
    baseline_metrics: Dict[str, Any],
    nulled_metrics: Dict[str, Any],
    output_path: Path,
    title: str,
    n_examples: Optional[int] = None,
    style: PlotStyle = DEFAULT_STYLE,
) -> None:
    """Create position bias bar plot showing accuracy when answer is at each position.
    
    Args:
        baseline_metrics: Baseline metrics dict with accuracy_when_A/B/C/D keys
        nulled_metrics: Nulled metrics dict with accuracy_when_A/B/C/D keys
        output_path: Path to save the plot
        title: Plot title
        n_examples: Number of examples (for error bars)
        style: Plot style settings
    """
    style.apply()
    
    positions = ["A", "B", "C", "D"]
    fig, ax = plt.subplots(figsize=style.figsize)
    
    x = np.arange(len(positions))
    width = 0.35
    
    # Extract accuracy per position (convert to percentage)
    baseline_accs = [baseline_metrics.get(f"accuracy_when_{p}", 0.0) * 100 for p in positions]
    nulled_accs = [nulled_metrics.get(f"accuracy_when_{p}", 0.0) * 100 for p in positions]
    
    # Get sample sizes per position for error bars
    baseline_ns = [int(baseline_metrics.get(f"n_correct_at_{p}", n_examples or 100)) for p in positions]
    nulled_ns = [int(nulled_metrics.get(f"n_correct_at_{p}", n_examples or 100)) for p in positions]
    
    # Compute error bars using sample size per position
    baseline_errs = [binomial_ci95(acc / 100, n) * 100 if n > 0 else 0 for acc, n in zip(baseline_accs, baseline_ns)]
    nulled_errs = [binomial_ci95(acc / 100, n) * 100 if n > 0 else 0 for acc, n in zip(nulled_accs, nulled_ns)]
    
    # Create bars
    bars1 = ax.bar(
        x - width / 2,
        baseline_accs,
        width,
        yerr=baseline_errs,
        label="Normal",
        color="#666666",
        capsize=4,
        edgecolor="#333",
        linewidth=1,
    )
    bars2 = ax.bar(
        x + width / 2,
        nulled_accs,
        width,
        yerr=nulled_errs,
        label="Debiased",
        color="#4a90d9",
        capsize=4,
        edgecolor="#333",
        linewidth=1,
    )
    
    # Show overall accuracy in legend
    baseline_overall = baseline_metrics.get("accuracy", 0.0) * 100
    nulled_overall = nulled_metrics.get("accuracy", 0.0) * 100
    
    # Value labels
    for bar in bars1:
        height = bar.get_height()
        ax.annotate(
            f"{height:.1f}%",
            xy=(bar.get_x() + bar.get_width() / 2, height),
            xytext=(0, 3),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=10,
        )
    for bar in bars2:
        height = bar.get_height()
        ax.annotate(
            f"{height:.1f}%",
            xy=(bar.get_x() + bar.get_width() / 2, height),
            xytext=(0, 3),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=10,
        )
    
    # Labels
    ax.set_ylabel("Accuracy (%)")
    ax.set_xlabel("Correct Answer Position")
    ax.set_xticks(x)
    ax.set_xticklabels(positions)
    max_val = max(max(baseline_accs) if baseline_accs else 50, max(nulled_accs) if nulled_accs else 50)
    ax.set_ylim(0, min(max_val + 15, 105))
    
    # Add overall accuracy to legend labels
    ax.legend(
        [f"Normal (overall: {baseline_overall:.1f}%)", f"Debiased (overall: {nulled_overall:.1f}%)"],
        loc="upper right",
    )
    ax.set_title(title, fontweight="bold")
    
    # Save
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=style.dpi, bbox_inches="tight", facecolor=style.facecolor)
    plt.close()


def create_accuracy_plot(
    baseline_metrics: Dict[str, float],
    nulled_metrics: Dict[str, float],
    output_path: Path,
    title: str,
    n_examples: Optional[int] = None,
    style: PlotStyle = DEFAULT_STYLE,
) -> None:
    """Create grouped accuracy bar chart with 95% confidence intervals."""
    create_comparison_plot(
        baseline_metrics=baseline_metrics,
        nulled_metrics=nulled_metrics or {},
        metric_labels=[("accuracy", "Accuracy")],
        output_path=output_path,
        title=title,
        ylabel="Accuracy",
        ylim=(0.0, 1.05),
        n_examples=n_examples,
        reference_line=1.0,
        reference_label="Perfect",
        style=style,
    )


def create_binary_position_plot(
    baseline: Dict[str, float],
    nulled: Dict[str, float],
    output_path: Path,
    title: str = "Binary Position Bias",
    n_examples: Optional[int] = None,
    style: PlotStyle = DEFAULT_STYLE,
) -> None:
    """Create binary position bias plot showing accuracy by prompt orientation.
    
    Shows accuracy (selecting correct answer) when:
    - Correct answer is mentioned first in prompt
    - Incorrect answer is mentioned first in prompt
    
    Args:
        baseline: Baseline metrics dict
        nulled: Nulled metrics dict
        output_path: Path to save the plot
        title: Plot title
        n_examples: Number of examples for error bars
        style: Plot style settings
    """
    style.apply()
    
    fig, ax = plt.subplots(figsize=(6, 5))
    
    labels = ["Correct\nFirst", "Incorrect\nFirst"]
    baseline_vals = [
        baseline.get("accuracy_correct_first", 0.5) * 100,
        baseline.get("accuracy_incorrect_first", 0.5) * 100,
    ]
    nulled_vals = [
        nulled.get("accuracy_correct_first", 0.5) * 100,
        nulled.get("accuracy_incorrect_first", 0.5) * 100,
    ]
    
    x = np.arange(len(labels))
    width = 0.35
    
    # Error bars
    if n_examples and n_examples > 0:
        baseline_errs = [binomial_ci95(v/100, n_examples) * 100 for v in baseline_vals]
        nulled_errs = [binomial_ci95(v/100, n_examples) * 100 for v in nulled_vals]
    else:
        baseline_errs = nulled_errs = None
    
    bars1 = ax.bar(x - width/2, baseline_vals, width, yerr=baseline_errs,
                   label="Baseline", color=style.baseline_color, capsize=4,
                   edgecolor="#333", linewidth=0.5)
    bars2 = ax.bar(x + width/2, nulled_vals, width, yerr=nulled_errs,
                   label="Nulled", color=style.nulled_color, capsize=4,
                   edgecolor="#333", linewidth=0.5)
    
    # Reference line at 50% (random chance for binary)
    ax.axhline(y=50, color="gray", linestyle="--", alpha=0.7, linewidth=1, label="Random")
    
    # Value labels
    for bar in list(bars1) + list(bars2):
        height = bar.get_height()
        ax.annotate(f"{height:.1f}%", xy=(bar.get_x() + bar.get_width()/2, height),
                   xytext=(0, 3), textcoords="offset points", ha="center", va="bottom", fontsize=10)
    
    ax.set_ylabel("Accuracy (%)", fontsize=12)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylim(0, 105)
    ax.legend(loc="upper right")
    ax.set_title(title, fontweight="bold")
    
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=style.dpi, bbox_inches="tight", facecolor=style.facecolor)
    plt.close()


def create_freeform_position_plot(
    baseline: Dict[str, float],
    nulled: Dict[str, float],
    output_path: Path,
    title: str = "Position Bias: Effect on Accuracy",
    n_examples: Optional[int] = None,
    style: PlotStyle = DEFAULT_STYLE,
) -> None:
    """Create freeform position bias plot showing accuracy by correct answer position.
    
    Clearly shows:
    - Baseline: High accuracy when correct is first, low when last (bias)
    - Nulled: Similar accuracy regardless of position (bias removed)
    
    Args:
        baseline: Baseline metrics dict
        nulled: Nulled metrics dict
        output_path: Path to save the plot
        title: Plot title
        n_examples: Number of examples for error bars
        style: Plot style settings
    """
    style.apply()
    
    fig, ax = plt.subplots(figsize=(8, 6))
    
    # Data
    labels = ["Correct Answer\nListed First", "Correct Answer\nListed Last"]
    baseline_vals = [
        baseline.get("accuracy_correct_first", 0.5) * 100,
        baseline.get("accuracy_correct_last", 0.5) * 100,
    ]
    nulled_vals = [
        nulled.get("accuracy_correct_first", 0.5) * 100,
        nulled.get("accuracy_correct_last", 0.5) * 100,
    ]
    
    x = np.arange(len(labels))
    width = 0.35
    
    # Error bars
    if n_examples and n_examples > 0:
        baseline_errs = [binomial_ci95(v/100, n_examples) * 100 for v in baseline_vals]
        nulled_errs = [binomial_ci95(v/100, n_examples) * 100 for v in nulled_vals]
    else:
        baseline_errs = nulled_errs = None
    
    # Bars
    bars1 = ax.bar(x - width/2, baseline_vals, width, yerr=baseline_errs,
                   label="Before Debiasing", color="#c44e52", capsize=5,
                   edgecolor="#333", linewidth=0.8)
    bars2 = ax.bar(x + width/2, nulled_vals, width, yerr=nulled_errs,
                   label="After Debiasing", color="#4c72b0", capsize=5,
                   edgecolor="#333", linewidth=0.8)
    
    # Reference line at random chance (depends on number of choices)
    num_choices = int(baseline.get("num_choices", 4) or 4)
    random_pct = 100.0 / max(num_choices, 1)
    ax.axhline(
        y=random_pct,
        color="gray",
        linestyle="--",
        alpha=0.7,
        linewidth=1.5,
        label=f"Random ({random_pct:.0f}%)",
    )
    
    # Value labels on bars
    for bar in list(bars1) + list(bars2):
        height = bar.get_height()
        ax.annotate(f"{height:.1f}%", xy=(bar.get_x() + bar.get_width()/2, height),
                   xytext=(0, 4), textcoords="offset points", ha="center", va="bottom", 
                   fontsize=11, fontweight="bold")
    
    ax.set_ylabel("Accuracy (%)", fontsize=13, fontweight='bold')
    ax.set_xlabel("Position of Correct Answer in List", fontsize=13, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=11)
    ax.set_ylim(0, max(baseline_vals + nulled_vals) * 1.2)
    ax.legend(loc="upper right", fontsize=10)
    ax.set_title(title, fontweight="bold", fontsize=14)
    
    # Add grid for readability
    ax.yaxis.grid(True, linestyle='--', alpha=0.3)
    ax.set_axisbelow(True)
    
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=style.dpi, bbox_inches="tight", facecolor=style.facecolor)
    plt.close()


def create_sycophancy_plot(
    baseline: Dict[str, float],
    nulled: Dict[str, float],
    output_path: Path,
    title: Optional[str] = None,
    n_examples: Optional[int] = None,
    style: PlotStyle = DEFAULT_STYLE,
) -> None:
    """Create sycophancy plot with 3 subplots: Overall, Easy, Hard.
    
    Shows accuracy under three conditions for each difficulty level:
    - No Opinion (baseline)
    - Correct Opinion (user suggests correct)
    - Incorrect Opinion (user suggests wrong)
    
    Easy/Hard split is defined by BASELINE performance:
    - Easy = questions BASELINE gets right without opinion (baseline knows answer)
    - Hard = questions BASELINE gets wrong without opinion (baseline uncertain)
    
    This same split is used for both baseline and nulled, allowing direct comparison
    of how nulled performs on questions the baseline found easy vs hard.
    
    Args:
        baseline: Baseline metrics dict
        nulled: Nulled metrics dict
        output_path: Path to save the plot
        title: Optional title
        n_examples: Number of examples for error bars
        style: Plot style settings
    """
    style.apply()
    
    # Check if we have easy/hard breakdown
    has_breakdown = "easy_n" in baseline and "hard_n" in baseline
    
    if has_breakdown:
        fig, axes = plt.subplots(1, 3, figsize=(14, 5))
    else:
        fig, axes = plt.subplots(1, 1, figsize=(7, 5))
        axes = [axes]
    
    def plot_subplot(ax, b_vals, n_vals, labels, subtitle, n_for_err=None):
        """Helper to plot a single subplot."""
        x = np.arange(len(labels))
        width = 0.35
        
        if n_for_err and n_for_err > 0:
            b_errs = [binomial_ci95(v / 100, n_for_err) * 100 for v in b_vals]
            n_errs = [binomial_ci95(v / 100, n_for_err) * 100 for v in n_vals]
        else:
            b_errs = n_errs = None
        
        bars1 = ax.bar(x - width/2, b_vals, width, yerr=b_errs, label="Baseline",
                       color=style.baseline_color, capsize=3, edgecolor="#333", linewidth=0.5)
        bars2 = ax.bar(x + width/2, n_vals, width, yerr=n_errs, label="Nulled",
                       color=style.nulled_color, capsize=3, edgecolor="#333", linewidth=0.5)
        
        ax.axhline(y=25, color="gray", linestyle="--", alpha=0.7, linewidth=1)
        
        # Value labels
        for bar in list(bars1) + list(bars2):
            height = bar.get_height()
            ax.annotate(f"{height:.0f}", xy=(bar.get_x() + bar.get_width()/2, height),
                       xytext=(0, 2), textcoords="offset points", ha="center", va="bottom", fontsize=8)
        
        ax.set_ylabel("Accuracy (%)")
        ax.set_xticks(x)
        ax.set_xticklabels(labels)
        ax.set_ylim(0, 105)
        ax.set_title(subtitle, fontweight="bold")
        ax.legend(loc="upper right", fontsize=8)
    
    # Overall plot
    overall_labels = ["No\nOpinion", "Correct\nOpinion", "Incorrect\nOpinion"]
    overall_baseline = [
        baseline["accuracy_no_opinion"] * 100,
        baseline["accuracy_correct_opinion"] * 100,
        baseline["accuracy_incorrect_opinion"] * 100,
    ]
    overall_nulled = [
        nulled["accuracy_no_opinion"] * 100,
        nulled["accuracy_correct_opinion"] * 100,
        nulled["accuracy_incorrect_opinion"] * 100,
    ]
    plot_subplot(axes[0], overall_baseline, overall_nulled, overall_labels, 
                 f"Overall (n={baseline.get('n_questions', n_examples)})", n_examples)
    
    if has_breakdown:
        # Easy questions (baseline gets right without opinion)
        # Baseline is 100% by definition, but nulled may differ on these same questions
        easy_labels = ["No\nOpinion", "Correct\nOpinion", "Incorrect\nOpinion"]
        easy_baseline = [
            baseline.get("easy_accuracy_no_opinion", 1.0) * 100,
            baseline.get("easy_accuracy_correct_opinion", 0) * 100,
            baseline.get("easy_accuracy_incorrect_opinion", 0) * 100,
        ]
        easy_nulled = [
            nulled.get("easy_accuracy_no_opinion", 1.0) * 100,
            nulled.get("easy_accuracy_correct_opinion", 0) * 100,
            nulled.get("easy_accuracy_incorrect_opinion", 0) * 100,
        ]
        easy_n = baseline.get("easy_n", 0)
        plot_subplot(axes[1], easy_baseline, easy_nulled, easy_labels,
                     f"Easy Questions (n={easy_n})\nBaseline knows answer", easy_n)
        
        # Hard questions (baseline gets wrong without opinion)
        # Baseline is 0% by definition, but nulled may differ on these same questions
        hard_labels = ["No\nOpinion", "Correct\nOpinion", "Incorrect\nOpinion"]
        hard_baseline = [
            baseline.get("hard_accuracy_no_opinion", 0.0) * 100,
            baseline.get("hard_accuracy_correct_opinion", 0) * 100,
            baseline.get("hard_accuracy_incorrect_opinion", 0) * 100,
        ]
        hard_nulled = [
            nulled.get("hard_accuracy_no_opinion", 0.0) * 100,
            nulled.get("hard_accuracy_correct_opinion", 0) * 100,
            nulled.get("hard_accuracy_incorrect_opinion", 0) * 100,
        ]
        hard_n = baseline.get("hard_n", 0)
        plot_subplot(axes[2], hard_baseline, hard_nulled, hard_labels,
                     f"Hard Questions (n={hard_n})\nBaseline uncertain", hard_n)
    
    plt.suptitle(title or "Sycophancy by Question Difficulty", fontsize=14, fontweight="bold")
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=style.dpi, bbox_inches="tight", facecolor=style.facecolor)
    plt.close()


def create_length_bias_plot(
    baseline: Dict[str, float],
    nulled: Optional[Dict[str, float]],
    output_path: Path,
    title: Optional[str] = None,
    n_examples: Optional[int] = None,
    style: PlotStyle = DEFAULT_STYLE,
) -> None:
    """Create length/verbosity bias plot.
    
    Shows error rates for:
    - Incorrect > Correct
    - Incorrect > Correct (Verbose) if available
    
    Args:
        baseline: Baseline metrics dict
        nulled: Optional nulled metrics dict
        output_path: Path to save the plot
        title: Optional title
        n_examples: Number of examples for error bars
        style: Plot style settings
    """
    style.apply()
    
    fig, ax = plt.subplots(figsize=(6, 5))
    
    labels = ["Incorrect >\nCorrect"]
    baseline_vals = [baseline.get("incorrect_beats_correct_pct", 0) * 100]

    # Optional verbose metric
    if "incorrect_beats_correct_verbose_pct" in baseline:
        labels.append("Incorrect >\nCorrect (Verbose)")
        baseline_vals.append(baseline["incorrect_beats_correct_verbose_pct"] * 100)
    
    if nulled is not None:
        nulled_vals = [nulled.get("incorrect_beats_correct_pct", 0) * 100]
        if "incorrect_beats_correct_verbose_pct" in baseline:
            nulled_vals.append(nulled.get("incorrect_beats_correct_verbose_pct", 0) * 100)
        x = np.arange(len(labels))
        width = 0.35
        
        if n_examples:
            baseline_errs = [binomial_ci95(v / 100, n_examples) * 100 for v in baseline_vals]
            nulled_errs = [binomial_ci95(v / 100, n_examples) * 100 for v in nulled_vals]
        else:
            baseline_errs = nulled_errs = None
        
        bars1 = ax.bar(
            x - width / 2, baseline_vals, width,
            yerr=baseline_errs, label="Baseline",
            color=style.bar_colors[0], capsize=4, edgecolor="black", linewidth=0.5
        )
        bars2 = ax.bar(
            x + width / 2, nulled_vals, width,
            yerr=nulled_errs, label="Nulled",
            color=style.bar_colors[1], capsize=4, edgecolor="black", linewidth=0.5
        )
        ax.legend()
    else:
        x = np.arange(len(labels))
        colors = style.bar_colors[:2]
        
        if n_examples:
            errs = [binomial_ci95(v / 100, n_examples) * 100 for v in baseline_vals]
        else:
            errs = None
        
        bars1 = ax.bar(
            x, baseline_vals, 0.45,
            yerr=errs, color=colors, capsize=4, edgecolor="black", linewidth=0.5
        )
    
    # Value labels
    for bar in ax.patches:
        height = bar.get_height()
        y_pos = max(height + 1, 3)
        ax.annotate(
            f"{height:.0f}%",
            xy=(bar.get_x() + bar.get_width() / 2, y_pos),
            ha="center", va="bottom", fontsize=12, fontweight="bold"
        )
    
    ax.set_ylabel("Error Rate (%)", fontsize=13)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylim(0, max(baseline_vals) * 1.5 + 5)
    ax.set_title(title or "Length/Verbosity Bias", fontsize=14, fontweight="bold")
    
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=style.dpi, bbox_inches="tight", facecolor=style.facecolor)
    plt.close()




