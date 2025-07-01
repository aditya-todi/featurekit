import pandas as pd
import matplotlib.pyplot as plt
import random

import featurekit.types as custom_types
import featurekit.utils as utils
from .config import plotting_config
from .base_plots import (
    _plot_scatter_2d,
    _plot_hexbin_2d,
    _plot_contour,
    _plot_categorical_scatter_with_regression,
    _plot_categorical_scatter_with_lowess,
)


class BivariateTargetVariationPlotter(object):
    """
    BivariateTargetVariationPlotter generates bivariate plots to visualize
    how a target variable varies with pairs of features.

    Handles both numerical and categorical features.
    """

    def __init__(
        self,
        df: pd.DataFrame,
        numerical_features: list[str],
        categorical_features: list[str],
        target: str,
        target_type: custom_types.TargetType,
    ) -> None:
        self.df = utils._truncate_df(df, plotting_config.max_samples).copy()
        self.numerical_features = numerical_features
        self.categorical_features = categorical_features
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

    @staticmethod
    def _get_total_interactions(f1: list[str], f2: list[str] | None = None) -> int:
        if not f2:
            n = len(f1)
            return int((n * (n - 1)) / 2)
        return len(f1) * len(f2)

    def _plot_numeric_target_numeric_feature_relationships(self) -> None:
        n = self._get_total_interactions(self.numerical_features)
        ncols = 3
        nrows = n
        figsize = (5 * ncols, 5 * nrows)
        fig, axs = plt.subplots(
            nrows=nrows, ncols=ncols, figsize=figsize, layout="constrained"
        )
        hexbin_cmap = plt.colormaps[random.choice(plotting_config.cmaps)]
        scatter_cmap = plt.colormaps[random.choice(plotting_config.cmaps)]
        contour_cmap = random.choice(["Reds", "Greens", "RdBu_r"])
        curr_row = 0

        for i, f1 in enumerate(self.numerical_features):
            for f2 in self.numerical_features[i + 1 :]:
                data = self.df[[f1, f2, self.target]].dropna()
                _plot_scatter_2d(
                    data,
                    f1,
                    f2,
                    self.target,
                    fig,
                    utils._safe_get_axs(axs, nrows, ncols, curr_row, 0),
                    cmap=scatter_cmap,
                )
                _plot_hexbin_2d(
                    data,
                    f1,
                    f2,
                    self.target,
                    utils._safe_get_axs(axs, nrows, ncols, curr_row, 1),
                    cmap=hexbin_cmap,
                )
                _plot_contour(
                    data,
                    f1,
                    f2,
                    self.target,
                    utils._safe_get_axs(axs, nrows, ncols, curr_row, 2),
                    cmap=contour_cmap,
                )
                curr_row += 1

        plt.show()

    def _plot_numeric_target_numeric_and_categorical_feature_relationships(
        self,
        confidence_interval: float | None = None,
        bootstrap_max_iterations: int | None = None,
        bootstrap_regression_sample_limit: int | None = None,
        lowess_frac: float | None = None,
        lowess_regression_sample_limit: int | None = None,
        logy: bool = False,
    ) -> None:
        if not self.categorical_features or not self.numerical_features:
            print("Not enough numerical/categorical features to plot")
            return

        n = self._get_total_interactions(
            self.numerical_features, self.categorical_features
        )
        ncols = 2
        nrows = n
        figsize = (5 * ncols, 5 * nrows)
        fig, axs = plt.subplots(
            nrows=nrows, ncols=ncols, figsize=figsize, layout="constrained"
        )
        cmaps = [
            plt.colormaps[random.choice(plotting_config.cmaps)]
            for _ in range(len(self.categorical_features))
        ]
        curr_row = 0

        for i, num_feature in enumerate(self.numerical_features):
            for j, cat_feature in enumerate(self.categorical_features):
                data = self.df[[num_feature, cat_feature, self.target]].copy()
                data[cat_feature] = data[cat_feature].fillna("Missing")
                grouped = data.dropna().groupby(cat_feature)
                _plot_categorical_scatter_with_regression(
                    grouped,
                    num_feature,
                    cat_feature,
                    self.target,
                    utils._safe_get_axs(axs, nrows, ncols, curr_row, 0),
                    cmaps[j],
                    confidence_interval=confidence_interval,
                    bootstrap_max_iterations=bootstrap_max_iterations,
                    bootstrap_regression_sample_limit=bootstrap_regression_sample_limit,
                    logy=logy,
                )
                _plot_categorical_scatter_with_lowess(
                    grouped,
                    num_feature,
                    cat_feature,
                    self.target,
                    utils._safe_get_axs(axs, nrows, ncols, curr_row, 1),
                    cmaps[j],
                    lowess_frac=lowess_frac,
                    lowess_regression_sample_limit=lowess_regression_sample_limit,
                )
                
                curr_row += 1

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
        if len(self.numerical_features) < 2:
            print("Not enough numerical features to plot")
            return

        self._plot_numeric_target_numeric_feature_relationships()
        self._plot_numeric_target_numeric_and_categorical_feature_relationships(
            confidence_interval,
            bootstrap_max_iterations,
            bootstrap_regression_sample_limit,
            lowess_frac,
            lowess_regression_sample_limit,
            logy,
        )

    def _plot_categorical_target_relationships(self) -> None:
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
        if self.target_type == custom_types.TargetType.REGRESSION:
            self._plot_numeric_target_relationships(
                confidence_interval,
                bootstrap_max_iterations,
                bootstrap_regression_sample_limit,
                lowess_frac,
                lowess_regression_sample_limit,
                logy,
            )
        elif self.target_type == custom_types.TargetType.CLASSIFICATION:
            self._plot_categorical_target_relationships()
        else:
            raise ValueError(f"Invalid target type: {self.target_type}")
