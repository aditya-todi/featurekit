import typing
import functools

import numpy as np

import pandas as pd
from pandas.core.groupby.generic import DataFrameGroupBy

from featurekit.config import fk_config


def _is_regression(target_type: str) -> bool:
    return target_type == "regression"


def _is_classification(target_type: str) -> bool:
    return target_type == "classification"


def _drop_simultaneous_na(
    x: pd.Series, y: pd.Series
) -> typing.Tuple[pd.Series, pd.Series]:
    na_idx = x.isna() | y.isna()
    return x[~na_idx], y[~na_idx]


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
