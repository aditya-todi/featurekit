import typing

from pydantic_settings import BaseSettings


class StatsConfig(BaseSettings):
    EPS: float = 1e-3

    confidence_interval: float = 0.95
    bootstrap_max_iterations: int = 1000
    bootstrap_regression_sample_limit: int = 100000
    lowess_frac: float = 0.1
    lowess_kernel: typing.Literal["tricube", "gaussian"] = "tricube"
    lowess_regression_sample_limit: int = 100000


stats_config = StatsConfig()
