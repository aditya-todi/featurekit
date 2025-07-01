import typing

from pydantic_settings import BaseSettings, SettingsConfigDict


class PlottingConfig(BaseSettings):
    PATCH_LIMIT: int = 20

    max_samples: int = int(1e5)
    scatter_plot_max_samples: int = int(2 * 1e4)
    cmaps: list[str] = ["viridis", "plasma", "magma", "cividis"]
    primary_color: str = "firebrick"
    bar_color: str = "firebrick"
    point_color: str = "slateblue"
    line_color: str = "darkslategray"
    text_color: str = "darkblue"
    alpha: float = 0.6
    bins: int = 50

    model_config = SettingsConfigDict(frozen=True, extra="forbid")


plotting_config = PlottingConfig()
