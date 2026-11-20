"""Explicit configuration, data, replay and persistence errors."""


class StratifiedModelsError(Exception):
    """Base exception for errors reported by the package."""


class ValidationError(StratifiedModelsError, ValueError):
    """A configuration does not describe a valid model or transformation."""


class SchemaError(ValidationError):
    """Required columns, labels or array shapes do not match the schema."""


class DataValidationError(ValidationError):
    """Observed data violates its declared numerical contract."""


class UnknownNodeError(DataValidationError):
    """An observed stratum label is absent from its declared axis."""


class ReplayError(StratifiedModelsError):
    """A source changed or failed to replay its declared data snapshot."""


class OptionalDependencyError(StratifiedModelsError, ImportError):
    """An optional integration requires an additional dependency."""


class ArtifactError(StratifiedModelsError):
    """An artifact cannot be safely encoded or decoded."""


class CoordinateError(ValidationError):
    """Coefficients and fitted prediction coordinates are incompatible."""
