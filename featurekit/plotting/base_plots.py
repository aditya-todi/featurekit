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
    "bootstrap_iter",
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
    bootstrap_iter: int | None = None,
    bootstrap_regression_sample_limit: int | None = None,
    restrict_y: bool = False,
):
    y_min, y_max = y.min() - plotting_config.EPS, y.max() + plotting_config.EPS
    y_transformed = y
    if restrict_y:
        y_transformed = np.log((y - y_min) / (y_max - y))

    pf = np.polynomial.Polynomial.fit(x=x, y=y_transformed, deg=1)
    x_vals = np.linspace(x.min(), x.max(), 100)
    y_vals = pf(x_vals)
    bootstrap_preds = np.zeros((bootstrap_iter, len(x_vals)))

    for i in range(bootstrap_iter):
        bootstrap_ids = np.random.choice(
            len(x),
            size=min(bootstrap_regression_sample_limit, len(x)),
            replace=True,
        )
        bootstrap_x = x.iloc[bootstrap_ids]
        bootstrap_y = y_transformed.iloc[bootstrap_ids]
        bootstrap_pf = np.polynomial.Polynomial.fit(x=bootstrap_x, y=bootstrap_y, deg=1)
        bootstrap_preds[i] = bootstrap_pf(x_vals)

    if restrict_y:
        bootstrap_preds = ((y_max - y_min) / (1 + np.exp(-bootstrap_preds))) + y_min
        y_vals = ((y_max - y_min) / (1 + np.exp(-y_vals))) + y_min

    lower_ci = np.percentile(
        bootstrap_preds, (1 - confidence_interval) * 100 / 2, axis=0
    )
    upper_ci = np.percentile(
        bootstrap_preds, (1 + confidence_interval) * 100 / 2, axis=0
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


def _tricube_weight_kernel(distances: np.ndarray) -> np.ndarray:
    return (1 - (np.clip(distances) ** 3)) ** 3


def _gaussian_weight_kernel(distances: np.ndarray) -> np.ndarray:
    return np.exp(-0.5 * (distances**2))


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
    if lowess_kernel not in ["tricube", "gaussian"]:
        raise ValueError(
            f"unknown weight kernel: '{lowess_kernel}', valid values: 'tricube', 'gaussian'"
        )

    x_trunc, y_trunc = utils._truncate_simulatneous(
        x, y, lowess_regression_sample_limit
    )
    n = len(x_trunc)
    sorted_ilocs = np.argsort(x_trunc)
    x_sorted = np.array(x_trunc.iloc[sorted_ilocs])
    y_sorted = np.array(y_trunc.iloc[sorted_ilocs])

    window_size = max(5, int(lowess_frac * n))
    _weight_kernel_func = (
        _gaussian_weight_kernel
        if lowess_kernel == "gaussian"
        else _tricube_weight_kernel
    )

    x_vals = np.linspace(min(x), max(x), 100)
    y_vals = np.zeros_like(x_vals)

    for i, x_v in enumerate(x_vals):
        distances = np.abs(x_v - x_sorted)
        window_x_ids = np.argsort(distances)[:window_size]
        window_distances = distances[window_x_ids]
        window_distances = window_distances / max(window_distances.max(), 1)

        x_nearest = x_sorted[window_x_ids]
        y_nearest = y_sorted[window_x_ids]
        x_nearest = np.vstack((np.ones_like(x_nearest), x_nearest)).T
        weights = _weight_kernel_func(window_distances)

        x_nearest_weighted_t = (x_nearest * weights.reshape(-1, 1)).T
        coeffs = (np.linalg.pinv(x_nearest_weighted_t @ x_nearest)) @ (
            x_nearest_weighted_t @ y_nearest
        )
        y_vals[i] = coeffs[0] + coeffs[1] * x_v

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
    category_counts = grouped[target].count().clip(lower=2)
    t_stars = scipy.stats.t.ppf((1 + confidence_interval) / 2, df=category_counts - 1)
    target_means = grouped[target].mean()
    target_mean_errs = np.array(
        t_stars * (grouped[target].std() / np.sqrt(category_counts))
    )

    sorted_ids = np.argsort(target_means)[::-1]
    target_means_sorted = target_means.iloc[sorted_ids]
    target_mean_errs_sorted = target_mean_errs[sorted_ids]

    ticks = np.arange(1, len(grouped) + 1)
    labels = target_means_sorted.index.astype(str)
    mean_handles = [
        Patch(color="none", label=f"{l}: {m}")
        for l, m in zip(labels, target_means_sorted.values)
    ]

    ax.errorbar(
        ticks,
        target_means_sorted.values,
        target_mean_errs_sorted,
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
