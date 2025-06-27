import random
import typing

import pandas as pd

import matplotlib.pyplot as plt
from matplotlib.patches import Patch

import featurekit.utils as utils
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
    _plot_mean_barplot_with_ci,
)


class UnivariateFeaturePlotter(object):
    """
    UnivariateFeaturePlotter generates univariate exploratory data visualizations
    for numerical and categorical features along with target

    Attributes:
    ----------
    df : pd.DataFrame
        The input dataset.
    numerical_features : list[str]
        List of numerical feature column names.
    categorical_features : list[str]
        List of categorical feature column names.
    target : str
        Name of the target column.
    target_type : str
        Type of problem: 'regression' or 'classification'.
    """

    def __init__(
        self,
        df: pd.DataFrame,
        numerical_features: list[str] | None,
        categorical_features: list[str] | None,
        target: str,
        target_type: str = "regression",
        bins: int = 50,
    ) -> None:
        """
        Initializes the UnivariateFeaturePlotter.

        Parameters:
        ----------
        df : pd.DataFrame
            The input dataset.
        numerical_features : list[str]
            List of numerical feature column names.
        categorical_features : list[str]
            List of categorical feature column names.
        target : str
            Name of the target column.
        target_type : str
            Type of problem: 'regression' or 'classification'.
        """
        self.df = df.copy()
        self.numerical_features = numerical_features or []
        self.categorical_features = categorical_features or []
        self.target = target
        self.target_type = target_type.lower()
        self.bins = bins
        self._validate_inputs()

    def _validate_inputs(self) -> None:
        """
        Validates that input column names exist in the DataFrame and target_type is valid.
        """
        if self.target_type not in ["regression", "classification"]:
            raise ValueError(
                f"target_type must be either 'regression' or 'classification', {self.target_type} was provided"
            )

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
        _plot_hist(x, axs[0], bins=self.bins, bar_color=color, title=title)
        _plot_boxplot(x, axs[1], primary_color=color, title=title)
        _plot_violinplot(x, axs[2], title=title)
        _plot_qqplot(x, axs[3], title=title)
        _plot_ecdf(x, axs[4], title=feature)

    def plot_univariate_numeric(self) -> None:
        """
        Plots the following univariate distributions for numeric features:
            1. Histogram and KDE
            2. Box Plot
            3. Violin Plot
            4. ECDF (Empirical Cumulative Distribution Function) Plot
            5. QQ Plot (Quantile-Quantile Plot)
        """
        if not self.numerical_features:
            print("No numerical features to plot")
            return

        n = len(self.numerical_features)

        ncols = 5
        nrows = n + (int(utils._is_regression(self.target_type)))
        figsize = (20, 4 * nrows)
        axs_start = 0

        fig, axs = plt.subplots(
            nrows=nrows, ncols=ncols, figsize=figsize, layout="constrained"
        )

        if utils._is_regression(self.target_type):
            color = plt.colormaps[random.choice(plotting_config.cmaps)](random.random())
            title = f"Target: '{self.target}'"
            self._plot_single_numeric_feature(
                self.target,
                axs if nrows == 1 else axs[0],
                color=color,
                title=title,
            )
            axs_start += 1

        for i, feature in enumerate(self.numerical_features):
            color = plt.colormaps[random.choice(plotting_config.cmaps)](i / n)
            self._plot_single_numeric_feature(
                feature, axs if nrows == 1 else axs[i + axs_start], color=color
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
        """
        Plots univariate distributions for each categorical feature in the dataset.

        For each categorical feature, it generates two side-by-side visualizations:
        1. A sorted horizontal bar plot showing the frequency of each category.
        2. A donut chart (pie chart with a hollow center) displaying category proportions.
        """

        if not self.categorical_features:
            print("No categorical features to plot")
            return

        n = len(self.categorical_features)
        ncols = 2
        nrows = n + (int(utils._is_classification(self.target_type)))
        figsize = (20, 4 * nrows)
        axs_start = 0

        fig, axs = plt.subplots(
            nrows=nrows, ncols=ncols, figsize=figsize, layout="constrained"
        )

        if utils._is_classification(self.target_type):
            cmap = plt.colormaps[random.choice(plotting_config.cmaps)]
            title = f"Target: '{self.target}"
            self._plot_single_categorical_feature(
                self.target, axs if nrows == 1 else axs[0], cmap=cmap, title=title
            )
            axs_start += 1

        for i, feature in enumerate(self.categorical_features):
            cmap = plt.colormaps[random.choice(plotting_config.cmaps)]
            self._plot_single_categorical_feature(
                feature, axs if nrows == 1 else axs[i + axs_start], cmap=cmap
            )

        plt.show()

    def plot(self) -> None:
        self.plot_univariate_numeric()
        self.plot_univariate_categorical()


class UnivariateTargetVariationPlotter(object):
    """
    UnivariateTargetVariationPlotter generates bivariate plots to visualize
    how a numerical target variable varies with each individual feature.

    Handles both numerical and categorical features.

    Attributes:
    ----------
    df : pd.DataFrame
        The input dataset.
    numerical_features : list[str]
        List of numerical feature column names.
    categorical_features : list[str]
        List of categorical feature column names.
    target : str
        Name of the numerical target column.
    """

    def __init__(
        self,
        df: pd.DataFrame,
        numerical_features: list[str] | None,
        categorical_features: list[str] | None,
        target: str,
        target_type: str = "regression",
    ) -> None:
        self.df = df.copy()
        self.numerical_features = numerical_features or []
        self.categorical_features = categorical_features or []
        self.target = target
        self.target_type = target_type
        self._validate_inputs()

    def _validate_inputs(self) -> None:
        """
        Validates that input column names exist in the DataFrame and target_type is valid.
        """
        if self.target_type not in ["regression", "classification"]:
            raise ValueError(
                f"target_type must be either 'regression' or 'classification', {self.target_type} was provided"
            )

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
            x, y = utils._drop_simultaneous_na(self.df[feature], self.df[self.target])
            _plot_scatter_with_regression(
                x,
                y,
                axs[0] if nrows == 1 else axs[i][0],
                title=f"{feature} v/s {self.target}",
                confidence_interval=confidence_interval,
                bootstrap_max_iterations=bootstrap_max_iterations,
                bootstrap_regression_sample_limit=bootstrap_regression_sample_limit,
                logy=logy,
            )
            _plot_scatter_with_lowess(
                x,
                y,
                axs[1] if nrows == 1 else axs[i][1],
                title=f"{feature} v/s {self.target}",
                lowess_frac=lowess_frac,
                lowess_regression_sample_limit=lowess_regression_sample_limit,
            )
            (axs[0] if nrows == 1 else axs[i][0]).set_ylabel(
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
            # cmap = plt.colormaps[random.choice(self.cmaps)]
            title = f"{feature} v/s {self.target}"

            _plot_boxplot_categorical(
                grouped,
                self.target,
                axs[0] if nrows == 1 else axs[i][0],
                title=title,
            )
            _plot_violinplot_categorical(
                grouped,
                self.target,
                axs[1] if nrows == 1 else axs[i][1],
                title=title,
            )
            _plot_mean_barplot_with_ci(
                grouped,
                self.target,
                axs[2] if nrows == 1 else axs[i][2],
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
        """
        Plots how each feature relates to the numerical target.

        For each numerical feature, it generates following visualizations:
        1. Scatter Plot with regression line and confidence interval around regression line
        2. Scatter Plot with LOWESS smoother
        3. Hexbin Plot (To be Implemented)
        4. 2D KDE Plot (To be Implemented)

        For each categorical feature, it generates following visualizations:
        1. Box plots of target per category
        2. Violin plots of target per category
        3. Mean error bars with ci
        """
        self._plot_numeric_target_numeric_feature_relationships(
            confidence_interval,
            bootstrap_max_iterations,
            bootstrap_regression_sample_limit,
            lowess_frac,
            lowess_regression_sample_limit,
            logy,
        )
        self._plot_numeric_target_categorical_feature_relationships(confidence_interval)

    def _plot_categorical_target_relationships(self) -> None:
        """
        Plots how each feature relates to the numerical target.
        """
        pass

    def plot(
        self,
        confidence_interval: float | None = None,
        bootstrap_max_iterations: int | None = None,
        bootstrap_regression_sample_limit: int | None = None,
        lowess_frac: float | None = None,
        lowess_regression_sample_limit: int | None = None,
        logy: bool = False,
    ) -> None:
        """
        Wrapper method to plot bivariate visualizations based on target type.
        """
        if self.target_type == "regression":
            self._plot_numeric_target_relationships(
                confidence_interval,
                bootstrap_max_iterations,
                bootstrap_regression_sample_limit,
                lowess_frac,
                lowess_regression_sample_limit,
                logy,
            )
        else:
            self._plot_categorical_target_relationships
