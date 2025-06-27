import typing

from pydantic_settings import BaseSettings, SettingsConfigDict


class PlottingConfig(BaseSettings):
    PATCH_LIMIT: int = 20

    cmaps: list[str] = ["viridis", "plasma", "magma", "cividis"]
    primary_color: str = "firebrick"
    bar_color: str = "firebrick"
    point_color: str = "slateblue"
    line_color: str = "darkslategray"
    text_color: str = "darkblue"
    alpha: float = 0.6

    model_config = SettingsConfigDict(frozen=True, extra="forbid")


plotting_config = PlottingConfig()
