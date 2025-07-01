import enum


class TargetType(enum.StrEnum):
    REGRESSION = "regression"
    CLASSIFICATION = "classification"


class FeatureType(enum.StrEnum):
    NUMERICAL = "numerical"
    CATEGORICAL = "categorical"
