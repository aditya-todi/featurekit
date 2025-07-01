import typing
import functools

import numpy as np
import matplotlib.pyplot as plt

import pandas as pd
from pandas.core.groupby.generic import DataFrameGroupBy

from featurekit.config import fk_config


def _truncate_df(df: pd.DataFrame, limit: int) -> pd.DataFrame:
    if len(df) > limit:
        ilocs = np.random.choice(len(df), limit, replace=False)
        return df.iloc[ilocs]
    return df


def _truncate_simulatneous(
    x: pd.Series, y: pd.Series, limit: int
) -> typing.Tuple[pd.Series, pd.Series]:
    if len(x) > limit:
        ilocs = np.random.choice(len(x), limit, replace=False)
        x = x.iloc[ilocs]
        y = y.iloc[ilocs]
    return x, y


def _sort_groups_by_median(grouped: DataFrameGroupBy, target: str) -> list:
    return sorted(
        grouped,
        key=lambda grp: grp[1][target].median(),
        reverse=True,
    )


def _sort_groups_by_mean(grouped: DataFrameGroupBy, target: str) -> list:
    return sorted(
        grouped,
        key=lambda grp: grp[1][target].mean(),
        reverse=True,
    )


def mode_router(
    custom_impl: typing.Callable[..., typing.Any],
    wrapper_impl: typing.Callable[..., typing.Any],
) -> typing.Callable[
    [typing.Callable[..., typing.Any]], typing.Callable[..., typing.Any]
]:
    def decorator(
        fn: typing.Callable[..., typing.Any],
    ) -> typing.Callable[..., typing.Any]:
        @functools.wraps(fn)
        def wrapper(*args, **kwargs):
            if fk_config.mode == "custom":
                custom_impl(*args, **kwargs)
            else:
                wrapper_impl(*args, **kwargs)

        return wrapper

    return decorator


def enrich_args_from_config(
    *enrich_args: str, config: typing.Any
) -> typing.Callable[
    [typing.Callable[..., typing.Any]], typing.Callable[..., typing.Any]
]:
    def decorator(
        fn: typing.Callable[..., typing.Any],
    ) -> typing.Callable[..., typing.Any]:
        @functools.wraps(fn)
        def wrapper(*args, **kwargs):
            for arg in enrich_args:
                if kwargs.get(arg, None) is None:
                    kwargs[arg] = getattr(config, arg)
            return fn(*args, **kwargs)

        return wrapper

    return decorator


def _safe_get_axs(
    axs: plt.Axes,
    n: int,
    m: int,
    i: int | None = None,
    j: int | None = None,
) -> plt.Axes:
    if i is None and j is None:
        raise ValueError("either i or j is required, provided none")

    if n == 1:
        if j is None:
            return axs
        return axs[j]
    if m == 1:
        if i is None:
            return axs
        return axs[i]

    if i is None:
        return axs[j]
    if j is None:
        return axs[i]

    return axs[i][j]
