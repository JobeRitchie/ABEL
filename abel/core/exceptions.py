"""Custom exceptions for ABEL services."""


class ABELError(Exception):
    """Base ABEL error."""


class ProjectError(ABELError):
    """Raised for project lifecycle issues."""


class ValidationError(ABELError):
    """Raised when input or schema validation fails."""


class DependencyError(ABELError):
    """Raised for dependency install/inspection issues."""
