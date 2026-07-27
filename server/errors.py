from typing import Optional


class JinaError(Exception):
    status: int
    code: str

    def __init__(self, message: str, *, field: Optional[str] = None):
        super().__init__(message)
        self.message = message
        self.field = field


class BadRequest(JinaError):
    status = 400
    code = "bad_request"


class NotAcceptable(JinaError):
    status = 406
    code = "not_acceptable"


class PayloadTooLarge(JinaError):
    status = 413
    code = "payload_too_large"


class ModelNotLoaded(JinaError):
    status = 503
    code = "model_not_loaded"


class LicenseBlocked(JinaError):
    status = 403
    code = "license_blocked"
