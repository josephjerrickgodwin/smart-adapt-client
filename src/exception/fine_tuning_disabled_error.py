class FineTuningDisabledError(Exception):
    """Raised when a fine-tuning task is in progress and the current request cannot be served"""
    pass
