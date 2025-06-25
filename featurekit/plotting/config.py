import typing

from pydantic_settings import BaseSettings, SettingsConfigDict


class PlottingConfig(BaseSettings):
    confidence_interval: float = 0.95
    bootstrap_iter: int = 1000
    bootstrap_regression_sample_limit: int = 100000
    lowess_frac: float = 0.1
    lowess_kernel: typing.Literal["tricube", "gaussian"] = "tricube"
    lowess_regression_sample_limit: int = 100000

    PATCH_LIMIT: int = 20
    EPS: float = 1e-3

    cmaps: list[str] = ["viridis", "plasma", "magma", "cividis"]
    primary_color: str = "firebrick"
    bar_color: str = "firebrick"
    point_color: str = "slateblue"
    line_color: str = "darkslategray"
    text_color: str = "darkblue"
    alpha: float = 0.6

    model_config = SettingsConfigDict(frozen=True, extra="forbid")


plotting_config = PlottingConfig()
