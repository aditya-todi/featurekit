import typing

from pydantic_settings import BaseSettings


class StatsConfig(BaseSettings):
    EPS: float = 1e-3

    confidence_interval: float = 0.95
    bootstrap_max_iterations: int = int(1e3)
    bootstrap_regression_sample_limit: int = int(2 * 1e4)
    lowess_frac: float = 0.2
    lowess_kernel: typing.Literal["tricube", "gaussian"] = "tricube"
    lowess_regression_sample_limit: int = int(2 * 1e4)


stats_config = StatsConfig()
