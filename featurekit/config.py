import typing

from pydantic_settings import BaseSettings, SettingsConfigDict


class FeatureKitConfig(BaseSettings):
    mode: typing.Literal["custom", "wrapper"] = "custom"

    model_config = SettingsConfigDict(
        validate_assignment=True,
        revalidate_instances="always",
        extra="forbid",
    )


_default_config = FeatureKitConfig()
fk_config = FeatureKitConfig()


def get_option(opt: str) -> typing.Any:
    if not hasattr(fk_config, opt):
        raise KeyError(f"'{opt}' is not a valid config option")
    return getattr(fk_config, opt)


def set_option(opt: str, val: typing.Any) -> None:
    if not hasattr(fk_config, opt):
        raise KeyError(f"'{opt}' is not a valid config option")

    updated_fk_config = fk_config.model_copy(update={opt: val})
    FeatureKitConfig.model_validate(updated_fk_config)
    setattr(fk_config, opt, val)


def reset_default(*opts: str) -> None:
    if not opts:
        for opt, val in _default_config.model_dump().items():
            setattr(fk_config, opt, val)
    else:
        for opt in opts:
            if not hasattr(fk_config, opt):
                raise KeyError(f"'{opt}' is not a valid config option")
        for opt in opts:
            setattr(fk_config, opt, getattr(_default_config, opt))
