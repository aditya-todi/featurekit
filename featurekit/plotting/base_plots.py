# from typing import Any, Callable, Literal
import typing

import scipy
import numpy as np
import seaborn as sns

import pandas as pd
from pandas.core.groupby.generic import DataFrameGroupBy

import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.lines import Line2D

import featurekit.utils as utils
from featurekit.stats import (
    linear_regression_with_ci,
    linear_regression_with_lowess,
    target_mean_with_ci,
)
from .config import plotting_config


@utils.enrich_args_from_config("line_color", "text_color", config=plotting_config)
def _plot_kde_custom(
    x: pd.Series,
    ax: plt.Axes,
    line_color: typing.Any = None,
    text_color: typing.Any = None,
) -> None:
    kde = scipy.stats.gaussian_kde(x)
    kde_x = np.linspace(min(x), max(x), 100)
    kde_y = kde.evaluate(kde_x)
    ax.plot(kde_x, kde_y, color=line_color)
    ax.set_ylabel("Density", color=text_color)


@utils.enrich_args_from_config("line_color", "text_color", config=plotting_config)
def _plot_kde_wrapper(
    x: pd.Series,
    ax: plt.Axes,
    line_color: typing.Any = None,
    text_color: typing.Any = None,
) -> None:
    sns.kdeplot(x=x, ax=ax, color=line_color)
    ax.set_ylabel("Density", color=text_color)


@utils.mode_router(custom_impl=_plot_kde_custom, wrapper_impl=_plot_kde_wrapper)
def _plot_kde(
    x: pd.Series,
    ax: plt.Axes,
    line_color: typing.Any = None,
    text_color: typing.Any = None,
) -> None: ...


@utils.enrich_args_from_config(
    "bar_color", "line_color", "text_color", "alpha", config=plotting_config
)
def _plot_hist(
    x: pd.Series,
    ax: plt.Axes,
    bins: int,
    title: str,
    bar_color: typing.Any = None,
    line_color: typing.Any = None,
    text_color: typing.Any = None,
    alpha: float | None = None,
) -> None:
    ax.hist(
        x,
        bins=bins,
        color=bar_color,
        edgecolor=bar_color,
        linewidth=1.5,
        alpha=alpha,
    )
    _plot_kde(x, ax.twinx(), line_color=line_color, text_color=text_color)
    ax.set_ylabel("Frequency", color=text_color)
    ax.set_title(f"'{title}' Histogram Distribution", color=text_color)
    ax.grid(axis="y", linestyle="--", alpha=0.3)


@utils.enrich_args_from_config("text_color", config=plotting_config)
def _plot_violinplot(
    x: pd.Series,
    ax: plt.Axes,
    title: str,
    text_color: typing.Any = None,
):
    ax.violinplot(dataset=[x], vert=False, showmedians=True, showmeans=False)
    ax.set_title(f"'{title}' Violin Plot", color=text_color)
    ax.grid(axis="y", linestyle="--", alpha=0.3)


@utils.enrich_args_from_config("primary_color", "text_color", config=plotting_config)
def _plot_boxplot(
    x: pd.Series,
    ax: plt.Axes,
    title: str,
    primary_color: typing.Any = None,
    text_color: typing.Any = None,
):
    bp = ax.boxplot(x, vert=False)
    for median in bp["medians"]:
        median.set_color(primary_color)
    ax.set_yticks([])
    ax.set_title(f"'{title}' Boxplot", color=text_color)
    ax.legend(
        handles=[
            Patch(color=primary_color, label=f"Median: {x.median():.2f}"),
        ]
    )


@utils.enrich_args_from_config("line_color", "text_color", config=plotting_config)
def _plot_qqplot(
    x: pd.Series,
    ax: plt.Axes,
    title: str,
    line_color: typing.Any = None,
    text_color: typing.Any = None,
):
    scipy.stats.probplot(x, dist="norm", plot=ax)
    ax.get_lines()[1].set_color(line_color)
    ax.set_title(f"'{title}' QQ Plot", color=text_color)
    ax.grid(axis="y", linestyle="--", alpha=0.3)


@utils.enrich_args_from_config("line_color", "text_color", config=plotting_config)
def _plot_ecdf(
    x: pd.Series,
    ax: plt.Axes,
    title: str,
    line_color: typing.Any = None,
    text_color: typing.Any = None,
):
    ax.ecdf(x, color=line_color)
    ax.set_title(f"'{title}' ECDF Plot", color=text_color)
    ax.grid(axis="y", linestyle="--", alpha=0.3)


@utils.enrich_args_from_config("text_color", "alpha", config=plotting_config)
def _plot_barh(
    counts: pd.Series,
    ax: plt.Axes,
    cmap: typing.Any,
    title: str,
    text_color: typing.Any = None,
    alpha: float | None = None,
):
    labels = counts.index.astype(str)[::-1]
    values = counts.values[::-1]
    colors = [cmap(i / len(labels)) for i in range(len(labels))]
    bars = ax.barh(
        labels,
        values,
        tick_label=labels,
        color=colors,
        edgecolor=colors,
        linewidth=1.5,
        alpha=alpha,
    )

    for bar in bars:
        width = bar.get_width()
        ax.text(
            width + max(values) * 0.02,
            bar.get_y() + bar.get_height() / 2,
            f"{width}",
            ha="center",
            va="center",
            fontsize=9,
            color=text_color,
        )

    ax.set_title(f"'{title}' Category Counts", color=text_color)
    ax.set_ylabel("Count", color=text_color)


@utils.enrich_args_from_config("text_color", config=plotting_config)
def _plot_donut(
    counts: pd.Series,
    ax: plt.Axes,
    cmap: typing.Any,
    title: str,
    text_color: typing.Any = None,
):
    labels = counts.index.astype(str)
    values = counts.values
    colors = [cmap(i / len(labels)) for i in range(len(labels))]
    ax.pie(
        values,
        labels=labels,
        autopct="%1.1f%%",
        startangle=90,
        colors=colors,
        wedgeprops=dict(width=0.25),
    )
    ax.set_title(f"'{title}' Distribution (Donut)", fontsize=12, color=text_color)


@utils.enrich_args_from_config(
    "point_color",
    "line_color",
    "text_color",
    "alpha",
    "confidence_interval",
    "bootstrap_max_iterations",
    "bootstrap_regression_sample_limit",
    config=plotting_config,
)
def _plot_scatter_with_regression(
    x: pd.Series,
    y: pd.Series,
    ax: plt.Axes,
    title: str,
    point_color: typing.Any = None,
    line_color: typing.Any = None,
    text_color: typing.Any = None,
    alpha: float | None = None,
    confidence_interval: float | None = None,
    bootstrap_max_iterations: int | None = None,
    bootstrap_regression_sample_limit: int | None = None,
    logy: bool = False,
):
    x_vals = np.linspace(x.min(), x.max(), 100)
    y_vals, lower_ci, upper_ci = linear_regression_with_ci(
        x,
        y,
        x_vals=x_vals,
        confidence_interval=confidence_interval,
        bootstrap_max_iterations=bootstrap_max_iterations,
        bootstrap_regression_sample_limit=bootstrap_regression_sample_limit,
        logy=logy,
    )
    ax.scatter(x, y, s=6, color=point_color, alpha=alpha)
    ax.plot(x_vals, y_vals, color=line_color, linewidth=2)
    ax.fill_between(
        x_vals,
        lower_ci,
        upper_ci,
        color=line_color,
        alpha=0.2,
        label=f"{int(confidence_interval*100)}% CI",
    )
    ax.set_title(
        f"'{title}' Scatter Plot with Regression (CI: {confidence_interval * 100:.0f}%)",
        color=text_color,
    )
    ax.grid(axis="y", linestyle="--", alpha=0.3)


@utils.enrich_args_from_config(
    "point_color",
    "line_color",
    "text_color",
    "alpha",
    "lowess_frac",
    "lowess_kernel",
    "lowess_regression_sample_limit",
    config=plotting_config,
)
def _plot_scatter_with_lowess(
    x: pd.Series,
    y: pd.Series,
    ax: plt.Axes,
    title: str,
    point_color: typing.Any = None,
    line_color: typing.Any = None,
    text_color: typing.Any = None,
    alpha: float | None = None,
    lowess_frac: float | None = None,
    lowess_kernel: typing.Literal["tricube", "gaussian"] | None = None,
    lowess_regression_sample_limit: int | None = None,
) -> None:
    x_vals = np.linspace(x.min(), x.max(), 100)
    y_vals = linear_regression_with_lowess(
        x,
        y,
        x_vals=x_vals,
        lowess_frac=lowess_frac,
        lowess_kernel=lowess_kernel,
        lowess_regression_sample_limit=lowess_regression_sample_limit,
    )
    ax.scatter(x, y, s=6, color=point_color, alpha=alpha)
    ax.plot(
        x_vals,
        y_vals,
        color=line_color,
        linewidth=2,
    )
    ax.set_title(
        f"'{title}' Scatter Plot with LOWESS (Fraction: {lowess_frac * 100:.0f}%)",
        color=text_color,
    )
    ax.grid(axis="y", linestyle="--", alpha=0.3)


@utils.enrich_args_from_config("text_color", config=plotting_config)
def _plot_boxplot_categorical(
    grouped: DataFrameGroupBy,
    target: str,
    ax: plt.Axes,
    title: str,
    text_color: typing.Any = None,
) -> None:
    categories_sorted_by_median = utils._sort_groups_by_median(grouped, target)
    median_handles = [
        Patch(color="none", label=f"{name}: {grp[target].median():.2f}")
        for name, grp in categories_sorted_by_median
    ]
    ax.boxplot(
        x=[grp[target] for _, grp in categories_sorted_by_median],
    )
    ax.set_title(
        f"'{title}' Boxplot per category",
        color=text_color,
    )
    ax.set_xticks(np.arange(1, len(grouped) + 1))
    ax.set_xticklabels(
        [str(name) for name, _ in categories_sorted_by_median],
        rotation=45,
        ha="right",
    )
    ax.grid(axis="y", linestyle="--", alpha=0.3)
    ax.legend(
        handles=median_handles[: plotting_config.PATCH_LIMIT],
        title="Medians",
        loc="upper right",
        fontsize="x-small",
        title_fontsize="x-small",
    )


@utils.enrich_args_from_config("text_color", config=plotting_config)
def _plot_violinplot_categorical(
    grouped: DataFrameGroupBy,
    target: str,
    ax: plt.Axes,
    title: str,
    text_color: typing.Any = None,
) -> None:
    categories_sorted_by_median = utils._sort_groups_by_median(grouped, target)
    ax.violinplot(
        [grp[target] for _, grp in categories_sorted_by_median],
        showmedians=True,
        showmeans=False,
    )
    ax.set_title(f"'{title}' Violin per category", color=text_color)
    ax.set_xticks(np.arange(1, len(grouped) + 1))
    ax.set_xticklabels(
        [str(name) for name, _ in categories_sorted_by_median],
        rotation=45,
        ha="right",
    )
    ax.grid(axis="y", linestyle="--", alpha=0.3)


@utils.enrich_args_from_config(
    "point_color",
    "line_color",
    "text_color",
    "confidence_interval",
    config=plotting_config,
)
def _plot_mean_barplot_with_ci(
    grouped: DataFrameGroupBy,
    target: str,
    ax: plt.Axes,
    title: str,
    point_color: typing.Any = None,
    line_color: typing.Any = None,
    text_color: typing.Any = None,
    confidence_interval: float | None = None,
) -> None:
    target_means, target_mean_errs = target_mean_with_ci(
        grouped,
        target,
        confidence_interval=confidence_interval,
        sort_means=True,
    )
    ticks = np.arange(1, len(grouped) + 1)
    labels = target_means.index.astype(str)
    mean_handles = [
        Patch(color="none", label=f"{l}: {m}")
        for l, m in zip(labels, target_means.values)
    ]
    ax.errorbar(
        ticks,
        target_means.values,
        target_mean_errs,
        ecolor=line_color,
        elinewidth=1,
        capsize=3,
        linestyle="",
        marker="o",
        markersize=5,
        markerfacecolor=point_color,
        markeredgecolor=point_color,
    )
    ax.set_title(
        f"'{title}' Means per category (CI: {confidence_interval * 100:.0f}%)",
        color=text_color,
    )
    ax.set_xticks(ticks)
    ax.set_xticklabels(
        labels=labels,
        rotation=45,
        ha="right",
    )
    ax.grid(axis="y", linestyle="--", alpha=0.3)
    ax.legend(
        handles=mean_handles[: plotting_config.PATCH_LIMIT],
        title="Means",
        loc="upper right",
        fontsize="x-small",
        title_fontsize="x-small",
    )
