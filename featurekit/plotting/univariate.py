import random
import typing

import pandas as pd

import matplotlib.pyplot as plt
from matplotlib.patches import Patch

import featurekit.utils as utils
from featurekit.types import TargetType
from .config import plotting_config
from .base_plots import (
    _plot_hist,
    _plot_qqplot,
    _plot_violinplot,
    _plot_boxplot,
    _plot_ecdf,
    _plot_barh,
    _plot_donut,
    _plot_scatter_with_regression,
    _plot_scatter_with_lowess,
    _plot_boxplot_categorical,
    _plot_violinplot_categorical,
    _plot_target_mean_with_ci,
    _plot_correspondence_categorical,
    _plot_contingency_heatmap,
)


class UnivariateFeaturePlotter(object):
    def __init__(
        self,
        df: pd.DataFrame,
        numerical_features: list[str] | None,
        categorical_features: list[str] | None,
        target: str,
        target_type: TargetType,
    ) -> None:
        self.df = utils._truncate_df(df, plotting_config.max_samples).copy()
        self.numerical_features = numerical_features or []
        self.categorical_features = categorical_features or []
        self.target = target
        self.target_type = target_type.lower()
        self._validate_inputs()

    def _validate_inputs(self) -> None:
        all_columns = (
            self.numerical_features + self.categorical_features + [self.target]
        )
        missing_columns = [col for col in all_columns if col not in self.df.columns]

        if missing_columns:
            raise ValueError(f"columns are missing in dataframe: {missing_columns}")

    def _plot_single_numeric_feature(
        self,
        feature: str,
        axs: plt.Axes,
        color: typing.Any,
        title: str | None = None,
    ) -> None:
        x = self.df[feature].dropna()
        title = title or feature
        _plot_hist(x, axs[0], bar_color=color, title=title)
        _plot_boxplot(x, axs[1], primary_color=color, title=title)
        _plot_violinplot(x, axs[2], title=title)
        _plot_qqplot(x, axs[3], title=title)
        _plot_ecdf(x, axs[4], title=feature)

    def plot_univariate_numeric(self) -> None:
        if not self.numerical_features:
            print("No numerical features to plot")
            return

        n = len(self.numerical_features)
        ncols = 5
        nrows = n + int(self.target_type == TargetType.REGRESSION)
        figsize = (20, 4 * nrows)
        axs_start = int(self.target_type == TargetType.REGRESSION)
        fig, axs = plt.subplots(
            nrows=nrows, ncols=ncols, figsize=figsize, layout="constrained"
        )

        if self.target_type == TargetType.REGRESSION:
            color = plt.colormaps[random.choice(plotting_config.cmaps)](random.random())
            title = f"Target: '{self.target}'"
            self._plot_single_numeric_feature(
                self.target,
                utils._safe_get_axs(axs, nrows, ncols, i=0, j=None),
                color=color,
                title=title,
            )
        for i, feature in enumerate(self.numerical_features):
            color = plt.colormaps[random.choice(plotting_config.cmaps)](i / n)
            self._plot_single_numeric_feature(
                feature,
                utils._safe_get_axs(axs, nrows, ncols, i=i + axs_start, j=None),
                color=color,
            )

        plt.show()

    def _plot_single_categorical_feature(
        self,
        feature: str,
        axs: plt.Axes,
        cmap: typing.Any,
        title: str | None = None,
    ) -> None:
        title = title or feature
        counts = self.df[feature].fillna("Missing").value_counts()
        _plot_barh(counts=counts, ax=axs[0], cmap=cmap, title=title)
        _plot_donut(counts=counts, ax=axs[1], cmap=cmap, title=title)

    def plot_univariate_categorical(self) -> None:
        if not self.categorical_features:
            print("No categorical features to plot")
            return

        n = len(self.categorical_features)
        ncols = 2
        nrows = n + int(self.target_type == TargetType.CLASSIFICATION)
        figsize = (20, 4 * nrows)
        axs_start = int(self.target_type == TargetType.CLASSIFICATION)
        fig, axs = plt.subplots(
            nrows=nrows, ncols=ncols, figsize=figsize, layout="constrained"
        )

        if self.target_type == TargetType.CLASSIFICATION:
            cmap = plt.colormaps[random.choice(plotting_config.cmaps)]
            title = f"Target: '{self.target}"
            self._plot_single_categorical_feature(
                self.target,
                utils._safe_get_axs(axs, nrows, ncols, i=0, j=None),
                cmap=cmap,
                title=title,
            )
        for i, feature in enumerate(self.categorical_features):
            cmap = plt.colormaps[random.choice(plotting_config.cmaps)]
            self._plot_single_categorical_feature(
                feature,
                utils._safe_get_axs(axs, nrows, ncols, i=i + axs_start, j=None),
                cmap=cmap,
            )

        plt.show()

    def plot(self) -> None:
        self.plot_univariate_numeric()
        self.plot_univariate_categorical()


class UnivariateTargetVariationPlotter(object):
    def __init__(
        self,
        df: pd.DataFrame,
        numerical_features: list[str] | None,
        categorical_features: list[str] | None,
        target: str,
        target_type: TargetType,
    ) -> None:
        self.df = utils._truncate_df(df, plotting_config.max_samples).copy()
        self.numerical_features = numerical_features or []
        self.categorical_features = categorical_features or []
        self.target = target
        self.target_type = target_type
        self._validate_inputs()

    def _validate_inputs(self) -> None:
        all_columns = (
            self.numerical_features + self.categorical_features + [self.target]
        )
        missing_columns = [col for col in all_columns if col not in self.df.columns]

        if missing_columns:
            raise ValueError(f"columns are missing in dataframe: {missing_columns}")

    def _plot_numeric_target_numeric_feature_relationships(
        self,
        confidence_interval: float | None = None,
        bootstrap_max_iterations: int | None = None,
        bootstrap_regression_sample_limit: int | None = None,
        lowess_frac: float | None = None,
        lowess_regression_sample_limit: int | None = None,
        logy: bool = False,
    ) -> None:
        if not self.numerical_features:
            print("No numerical features to plot")
            return

        n = len(self.numerical_features)
        ncols = 2
        nrows = n
        figsize = (20, 6 * nrows)
        fig, axs = plt.subplots(
            nrows=nrows, ncols=ncols, figsize=figsize, layout="constrained"
        )

        for i, feature in enumerate(self.numerical_features):
            data = self.df[[feature, self.target]].dropna()
            x, y = data[feature], data[self.target]
            _plot_scatter_with_regression(
                x,
                y,
                # axs[0] if nrows == 1 else axs[i][0],
                utils._safe_get_axs(axs, nrows, ncols, i, 0),
                title=f"{feature} v/s {self.target}",
                confidence_interval=confidence_interval,
                bootstrap_max_iterations=bootstrap_max_iterations,
                bootstrap_regression_sample_limit=bootstrap_regression_sample_limit,
                logy=logy,
            )
            _plot_scatter_with_lowess(
                x,
                y,
                # axs[1] if nrows == 1 else axs[i][1],
                utils._safe_get_axs(axs, nrows, ncols, i, 1),
                title=f"{feature} v/s {self.target}",
                lowess_frac=lowess_frac,
                lowess_regression_sample_limit=lowess_regression_sample_limit,
            )
            utils._safe_get_axs(axs, nrows, ncols, 0, 0).set_ylabel(
                f"{self.target}", color=plotting_config.text_color
            )

        plt.show()

    def _plot_numeric_target_categorical_feature_relationships(
        self,
        confidence_interval: float | None = None,
    ) -> None:
        if not self.categorical_features:
            print("No categorical features to plot")
            return

        n = len(self.categorical_features)
        ncols = 3
        nrows = n
        figsize = (20, 6 * nrows)
        y = self.df[self.target]
        fig, axs = plt.subplots(
            nrows=nrows, ncols=ncols, figsize=figsize, layout="constrained"
        )

        for i, feature in enumerate(self.categorical_features):
            grouped = self.df[[feature, self.target]].copy()
            grouped[feature] = grouped[feature].fillna("Missing")
            grouped = grouped.groupby(feature)
            title = f"{feature} v/s {self.target}"
            _plot_boxplot_categorical(
                grouped,
                self.target,
                # axs[0] if nrows == 1 else axs[i][0],
                utils._safe_get_axs(axs, nrows, ncols, i, 0),
                title=title,
            )
            _plot_violinplot_categorical(
                grouped,
                self.target,
                # axs[1] if nrows == 1 else axs[i][1],
                utils._safe_get_axs(axs, nrows, ncols, i, 1),
                title=title,
            )
            _plot_target_mean_with_ci(
                grouped,
                self.target,
                # axs[2] if nrows == 1 else axs[i][2],
                utils._safe_get_axs(axs, nrows, ncols, i, 2),
                title=title,
                confidence_interval=confidence_interval,
            )

        plt.show()

    def _plot_numeric_target_relationships(
        self,
        confidence_interval: float | None = None,
        bootstrap_max_iterations: int | None = None,
        bootstrap_regression_sample_limit: int | None = None,
        lowess_frac: float | None = None,
        lowess_regression_sample_limit: int | None = None,
        logy: bool = False,
    ) -> None:
        self._plot_numeric_target_numeric_feature_relationships(
            confidence_interval,
            bootstrap_max_iterations,
            bootstrap_regression_sample_limit,
            lowess_frac,
            lowess_regression_sample_limit,
            logy,
        )
        self._plot_numeric_target_categorical_feature_relationships(confidence_interval)

    def _plot_categorical_target_numerical_feature_relationships(
        self, confidence_interval: float | None = None
    ):
        if not self.numerical_features:
            print("No numerical features to plot")
            return

        n = len(self.numerical_features)
        ncols = 3
        nrows = n
        figsize = (20, 6 * nrows)
        # y = self.df[self.target]
        fig, axs = plt.subplots(
            nrows=nrows, ncols=ncols, figsize=figsize, layout="constrained"
        )

        for i, feature in enumerate(self.numerical_features):
            grouped = self.df[[self.target, feature]].dropna()
            # grouped[feature] = grouped[feature].fillna("Missing")
            grouped = grouped.groupby(self.target)
            title = f"{feature} v/s {self.target}"
            _plot_boxplot_categorical(
                grouped,
                feature,
                # axs[0] if nrows == 1 else axs[i][0],
                utils._safe_get_axs(axs, nrows, ncols, i, 0),
                title=title,
            )
            _plot_violinplot_categorical(
                grouped,
                feature,
                # axs[1] if nrows == 1 else axs[i][1],
                utils._safe_get_axs(axs, nrows, ncols, i, 1),
                title=title,
            )
            _plot_target_mean_with_ci(
                grouped,
                feature,
                # axs[2] if nrows == 1 else axs[i][2],
                utils._safe_get_axs(axs, nrows, ncols, i, 2),
                title=title,
                confidence_interval=confidence_interval,
            )

        plt.show()

    def _plot_categorical_target_categorical_feature_relationships(
        self, random_state: int, confidence_interval: float | None = None
    ):
        if not self.categorical_features:
            print("No categorical features to plot")
            return

        y = self.df[self.target]
        plot_target_mean_with_ci = False
        if y.nunique() == 2:
            plot_target_mean_with_ci = True

        n = len(self.categorical_features)
        ncols = 2 + int(plot_target_mean_with_ci)
        nrows = n
        figsize = (16, 4 * nrows)
        fig, axs = plt.subplots(
            nrows=nrows, ncols=ncols, figsize=figsize, layout="constrained"
        )

        for i, feature in enumerate(self.categorical_features):
            _plot_contingency_heatmap(
                self.df,
                feature,
                self.target,
                fig,
                # axs[0] if nrows == 1 else axs[i][0],
                utils._safe_get_axs(axs, nrows, ncols, i, 0),
            )
            _plot_correspondence_categorical(
                self.df,
                feature,
                self.target,
                # axs[1] if nrows == 1 else axs[i][1],
                utils._safe_get_axs(axs, nrows, ncols, i, 1),
                random_state=random_state,
                plot_names=True,
            )
            if plot_target_mean_with_ci:
                grouped = self.df[[self.target, feature]].dropna().groupby(feature)
                title = f"{feature} v/s {self.target}"
                _plot_target_mean_with_ci(
                    grouped,
                    self.target,
                    # axs[2] if nrows == 1 else axs[i][2],
                    utils._safe_get_axs(axs, nrows, ncols, i, 2),
                    title=title,
                    confidence_interval=confidence_interval,
                )

        plt.show()

    def _plot_categorical_target_relationships(
        self, random_state: int, confidence_interval: float | None = None
    ) -> None:
        self._plot_categorical_target_numerical_feature_relationships(
            confidence_interval
        )
        self._plot_categorical_target_categorical_feature_relationships(random_state)

    def plot(
        self,
        random_state: int | None = None,
        confidence_interval: float | None = None,
        bootstrap_max_iterations: int | None = None,
        bootstrap_regression_sample_limit: int | None = None,
        lowess_frac: float | None = None,
        lowess_regression_sample_limit: int | None = None,
        logy: bool = False,
    ) -> None:
        if self.target_type == TargetType.REGRESSION:
            self._plot_numeric_target_relationships(
                confidence_interval,
                bootstrap_max_iterations,
                bootstrap_regression_sample_limit,
                lowess_frac,
                lowess_regression_sample_limit,
                logy,
            )
        elif self.target_type == TargetType.CLASSIFICATION:
            if random_state is None:
                raise ValueError("random_state is required for classification problems")
            self._plot_categorical_target_relationships(
                random_state, confidence_interval
            )
        else:
            raise ValueError(f"Invalid target type: {self.target_type}")
