class DeimApplicationError(Exception):
    """Base class for user-facing application failures."""


class AppConfigError(DeimApplicationError):
    pass


class AdapterConfigurationError(DeimApplicationError):
    pass


class CheckpointCompatibilityError(DeimApplicationError):
    pass


class InputSourceError(DeimApplicationError):
    pass


class InferenceBackendError(DeimApplicationError):
    pass


class ExportError(DeimApplicationError):
    pass
