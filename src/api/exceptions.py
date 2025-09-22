"""
API Exception Classes

API异常类定义和异常处理器
"""

from fastapi import Request, HTTPException
from fastapi.responses import JSONResponse
from typing import Any, Dict
import traceback
import logging


class APIException(Exception):
    """基础API异常类"""

    def __init__(
        self,
        message: str,
        status_code: int = 500,
        error_code: str = "INTERNAL_ERROR",
        details: Dict[str, Any] = None
    ):
        self.message = message
        self.status_code = status_code
        self.error_code = error_code
        self.details = details or {}
        super().__init__(self.message)


class ValidationError(APIException):
    """验证错误"""

    def __init__(self, message: str, details: Dict[str, Any] = None):
        super().__init__(
            message=message,
            status_code=400,
            error_code="VALIDATION_ERROR",
            details=details
        )


class ResourceNotFoundError(APIException):
    """资源未找到错误"""

    def __init__(self, resource: str, resource_id: str = None):
        message = f"Resource '{resource}' not found"
        if resource_id:
            message += f" with ID '{resource_id}'"

        super().__init__(
            message=message,
            status_code=404,
            error_code="RESOURCE_NOT_FOUND",
            details={"resource": resource, "resource_id": resource_id}
        )


class AuthenticationError(APIException):
    """认证错误"""

    def __init__(self, message: str = "Authentication required"):
        super().__init__(
            message=message,
            status_code=401,
            error_code="AUTHENTICATION_ERROR"
        )


class AuthorizationError(APIException):
    """授权错误"""

    def __init__(self, message: str = "Insufficient permissions"):
        super().__init__(
            message=message,
            status_code=403,
            error_code="AUTHORIZATION_ERROR"
        )


class ServiceUnavailableError(APIException):
    """服务不可用错误"""

    def __init__(self, service_name: str):
        super().__init__(
            message=f"Service '{service_name}' is currently unavailable",
            status_code=503,
            error_code="SERVICE_UNAVAILABLE",
            details={"service": service_name}
        )


class RateLimitError(APIException):
    """频率限制错误"""

    def __init__(self, message: str = "Rate limit exceeded"):
        super().__init__(
            message=message,
            status_code=429,
            error_code="RATE_LIMIT_EXCEEDED"
        )


async def api_exception_handler(request: Request, exc: APIException) -> JSONResponse:
    """API异常处理器"""

    logger = logging.getLogger(__name__)

    # 记录异常日志
    if exc.status_code >= 500:
        logger.error(
            f"API Error: {exc.error_code} - {exc.message}",
            extra={
                "path": str(request.url),
                "method": request.method,
                "status_code": exc.status_code,
                "error_code": exc.error_code,
                "details": exc.details
            }
        )
    else:
        logger.warning(
            f"API Warning: {exc.error_code} - {exc.message}",
            extra={
                "path": str(request.url),
                "method": request.method,
                "status_code": exc.status_code,
                "error_code": exc.error_code
            }
        )

    # 构建错误响应
    error_response = {
        "error": {
            "code": exc.error_code,
            "message": exc.message,
            "status_code": exc.status_code
        }
    }

    # 在开发模式下添加详细信息
    if exc.details:
        error_response["error"]["details"] = exc.details

    return JSONResponse(
        status_code=exc.status_code,
        content=error_response
    )