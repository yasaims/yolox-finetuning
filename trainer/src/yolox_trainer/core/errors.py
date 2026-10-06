class NotFound(Exception):
    """Unknown model, dataset or job."""


class InvalidArgument(ValueError):
    """Malformed or out-of-range input."""


class FailedPrecondition(Exception):
    """The operation is not allowed in the current state."""
