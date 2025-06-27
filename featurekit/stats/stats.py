import typing

import scipy
import numpy as np

import pandas as pd
from pandas.core.groupby.generic import DataFrameGroupBy

import featurekit.utils as utils
from .config import stats_config


@utils.enrich_args_from_config(
    "confidence_interval",
    "bootstrap_max_iterations",
    "bootstrap_max_iterations",
    "bootstrap_regression_sample_limit",
    config=stats_config,
)
def linear_regression_with_ci(
    x: pd.Series,
    y: pd.Series,
    x_vals: np.ndarray | None = None,
    confidence_interval: float | None = None,
    bootstrap_max_iterations: int | None = None,
    bootstrap_regression_sample_limit: int | None = None,
    logy: bool = False,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    y_transformed = None

    if logy:
        if y.min() < 0:
            raise ValueError(
                "logy transformation is only available for non-negative target"
            )
        y_transformed = np.log(y + stats_config.EPS)

    if x_vals is None:
        x_vals = np.linspace(x.min(), x.max(), 100)

    y_target = y_transformed if y_transformed is not None else y
    ploy_fit = np.polynomial.Polynomial.fit(x=x, y=y_target, deg=1)
    y_vals = ploy_fit(x_vals)
    resampled_preds = np.zeros((bootstrap_max_iterations, len(x_vals)))

    for i in range(bootstrap_max_iterations):
        _resampled_ids = np.random.choice(
            len(x),
            size=min(bootstrap_regression_sample_limit, len(x)),
            replace=True,
        )
        _resampled_x = x.iloc[_resampled_ids]
        _resampled_y = y_target.iloc[_resampled_ids]
        _curr_poly_fit = np.polynomial.Polynomial.fit(
            x=_resampled_x, y=_resampled_y, deg=1
        )
        resampled_preds[i] = _curr_poly_fit(x_vals)

    if logy:
        resampled_preds = np.clip(
            np.exp(resampled_preds) - stats_config.EPS, a_min=0, a_max=None
        )
        y_vals = np.clip(np.exp(y_vals) - stats_config.EPS, a_min=0, a_max=None)

    lower_ci = np.percentile(
        resampled_preds, (1 - confidence_interval) * 100 / 2, axis=0
    )
    upper_ci = np.percentile(
        resampled_preds, (1 + confidence_interval) * 100 / 2, axis=0
    )

    return y_vals, lower_ci, upper_ci


def _tricube_weight_kernel(distances: np.ndarray) -> np.ndarray:
    return (1 - (np.clip(distances) ** 3)) ** 3


def _gaussian_weight_kernel(distances: np.ndarray) -> np.ndarray:
    return np.exp(-0.5 * (distances**2))


@utils.enrich_args_from_config(
    "lowess_frac",
    "lowess_kernel",
    "lowess_regression_sample_limit",
    config=stats_config,
)
def linear_regression_with_lowess(
    x: pd.Series,
    y: pd.Series,
    x_vals: np.ndarray | None = None,
    lowess_frac: float | None = None,
    lowess_kernel: typing.Literal["tricube", "gaussian"] | None = None,
    lowess_regression_sample_limit: int | None = None,
) -> np.ndarray:
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

    x_vals = x_vals if x_vals is not None else np.linspace(min(x), max(x), 100)
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

    return y_vals


@utils.enrich_args_from_config("confidence_interval", config=stats_config)
def target_mean_with_ci(
    grouped: DataFrameGroupBy,
    target: str,
    confidence_interval: float | None = None,
    sort_means: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    category_counts = grouped[target].count().clip(lower=2)
    t_critical = scipy.stats.t.ppf(
        (1 + confidence_interval) / 2, df=category_counts - 1
    )
    target_means = grouped[target].mean()
    target_mean_errs = np.array(
        t_critical * (grouped[target].std() / np.sqrt(category_counts))
    )

    if sort_means:
        sorted_ids = np.argsort(target_means)[::-1]
        return target_means.iloc[sorted_ids], target_mean_errs[sorted_ids]

    return target_means, target_mean_errs
