"""
Exception Handling Framework
异常处理框架，定义系统中所有的错误类型和异常处理机制
"""

import enum
from typing import Dict, Any, Optional, List
from dataclasses import dataclass


class ErrorSeverity(enum.Enum):
    """错误严重程度"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class ErrorCategory(enum.Enum):
    """错误分类"""
    VALIDATION = "validation"  # 验证错误
    AUTHENTICATION = "authentication"  # 认证错误
    AUTHORIZATION = "authorization"  # 授权错误
    BUSINESS_LOGIC = "business_logic"  # 业务逻辑错误
    DATA_ACCESS = "data_access"  # 数据访问错误
    EXTERNAL_SERVICE = "external_service"  # 外部服务错误
    SYSTEM = "system"  # 系统错误
    CONFIGURATION = "configuration"  # 配置错误
    PERFORMANCE = "performance"  # 性能错误
    SECURITY = "security"  # 安全错误


@dataclass
class ErrorContext:
    """错误上下文信息"""
    user_id: Optional[str] = None
    request_id: Optional[str] = None
    agent_id: Optional[str] = None
    operation: Optional[str] = None
    params: Optional[Dict[str, Any]] = None
    trace_id: Optional[str] = None


class BaseSystemException(Exception):
    """系统基础异常类"""

    def __init__(
        self,
        message: str,
        error_code: str,
        category: ErrorCategory = ErrorCategory.SYSTEM,
        severity: ErrorSeverity = ErrorSeverity.MEDIUM,
        context: Optional[ErrorContext] = None,
        cause: Optional[Exception] = None,
        recoverable: bool = True
    ):
        super().__init__(message)
        self.message = message
        self.error_code = error_code
        self.category = category
        self.severity = severity
        self.context = context or ErrorContext()
        self.cause = cause
        self.recoverable = recoverable

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典格式"""
        return {
            "error_code": self.error_code,
            "message": self.message,
            "category": self.category.value,
            "severity": self.severity.value,
            "recoverable": self.recoverable,
            "context": {
                "user_id": self.context.user_id,
                "request_id": self.context.request_id,
                "agent_id": self.context.agent_id,
                "operation": self.context.operation,
                "params": self.context.params,
                "trace_id": self.context.trace_id
            } if self.context else None,
            "cause": str(self.cause) if self.cause else None
        }


# === 验证异常 ===
class ValidationException(BaseSystemException):
    """验证异常"""

    def __init__(self, message: str, field: Optional[str] = None,
                 value: Optional[Any] = None, **kwargs):
        super().__init__(
            message=message,
            error_code="VALIDATION_ERROR",
            category=ErrorCategory.VALIDATION,
            severity=ErrorSeverity.LOW,
            **kwargs
        )
        self.field = field
        self.value = value


class InvalidParameterException(ValidationException):
    """无效参数异常"""

    def __init__(self, parameter: str, value: Any, expected: str = "", **kwargs):
        message = f"Invalid parameter '{parameter}': {value}"
        if expected:
            message += f". Expected: {expected}"
        super().__init__(
            message=message,
            field=parameter,
            value=value,
            **kwargs
        )
        self.error_code = "INVALID_PARAMETER"


class MissingParameterException(ValidationException):
    """缺失参数异常"""

    def __init__(self, parameter: str, **kwargs):
        super().__init__(
            message=f"Required parameter '{parameter}' is missing",
            field=parameter,
            **kwargs
        )
        self.error_code = "MISSING_PARAMETER"


class SchemaValidationException(ValidationException):
    """Schema验证异常"""

    def __init__(self, errors: List[str], **kwargs):
        super().__init__(
            message=f"Schema validation failed: {'; '.join(errors)}",
            **kwargs
        )
        self.error_code = "SCHEMA_VALIDATION_ERROR"
        self.validation_errors = errors


# === 业务逻辑异常 ===
class BusinessLogicException(BaseSystemException):
    """业务逻辑异常"""

    def __init__(self, message: str, **kwargs):
        super().__init__(
            message=message,
            error_code="BUSINESS_LOGIC_ERROR",
            category=ErrorCategory.BUSINESS_LOGIC,
            **kwargs
        )


class FactorNotFoundException(BusinessLogicException):
    """因子未找到异常"""

    def __init__(self, factor_id: str, **kwargs):
        super().__init__(
            message=f"Factor not found: {factor_id}",
            **kwargs
        )
        self.error_code = "FACTOR_NOT_FOUND"
        self.factor_id = factor_id


class SignalExpiredException(BusinessLogicException):
    """信号过期异常"""

    def __init__(self, signal_id: str, **kwargs):
        super().__init__(
            message=f"Signal expired: {signal_id}",
            **kwargs
        )
        self.error_code = "SIGNAL_EXPIRED"
        self.signal_id = signal_id


class InsufficientDataException(BusinessLogicException):
    """数据不足异常"""

    def __init__(self, resource: str, required: int, available: int, **kwargs):
        super().__init__(
            message=f"Insufficient data for {resource}: required {required}, available {available}",
            **kwargs
        )
        self.error_code = "INSUFFICIENT_DATA"
        self.resource = resource
        self.required = required
        self.available = available


class ConfirmationRequiredException(BusinessLogicException):
    """需要确认异常"""

    def __init__(self, operation: str, **kwargs):
        super().__init__(
            message=f"Operation requires confirmation: {operation}",
            **kwargs
        )
        self.error_code = "CONFIRMATION_REQUIRED"
        self.operation = operation


# === 数据访问异常 ===
class DataAccessException(BaseSystemException):
    """数据访问异常"""

    def __init__(self, message: str, **kwargs):
        super().__init__(
            message=message,
            error_code="DATA_ACCESS_ERROR",
            category=ErrorCategory.DATA_ACCESS,
            severity=ErrorSeverity.HIGH,
            **kwargs
        )


class DatabaseConnectionException(DataAccessException):
    """数据库连接异常"""

    def __init__(self, database: str, **kwargs):
        super().__init__(
            message=f"Failed to connect to database: {database}",
            **kwargs
        )
        self.error_code = "DATABASE_CONNECTION_ERROR"
        self.database = database


class DatabaseQueryException(DataAccessException):
    """数据库查询异常"""

    def __init__(self, query: str, **kwargs):
        super().__init__(
            message=f"Database query failed: {query[:100]}...",
            **kwargs
        )
        self.error_code = "DATABASE_QUERY_ERROR"
        self.query = query


class CacheException(DataAccessException):
    """缓存异常"""

    def __init__(self, operation: str, key: str, **kwargs):
        super().__init__(
            message=f"Cache {operation} failed for key: {key}",
            **kwargs
        )
        self.error_code = "CACHE_ERROR"
        self.operation = operation
        self.key = key


# === 外部服务异常 ===
class ExternalServiceException(BaseSystemException):
    """外部服务异常"""

    def __init__(self, service: str, message: str, **kwargs):
        super().__init__(
            message=f"External service error ({service}): {message}",
            error_code="EXTERNAL_SERVICE_ERROR",
            category=ErrorCategory.EXTERNAL_SERVICE,
            **kwargs
        )
        self.service = service


class HikyuuException(ExternalServiceException):
    """Hikyuu框架异常"""

    def __init__(self, operation: str, **kwargs):
        super().__init__(
            service="hikyuu",
            message=f"Hikyuu operation failed: {operation}",
            **kwargs
        )
        self.error_code = "HIKYUU_ERROR"
        self.operation = operation


class APIException(ExternalServiceException):
    """API调用异常"""

    def __init__(self, endpoint: str, status_code: int, response: str = "", **kwargs):
        super().__init__(
            service="api",
            message=f"API call failed: {endpoint} (HTTP {status_code})",
            **kwargs
        )
        self.error_code = "API_ERROR"
        self.endpoint = endpoint
        self.status_code = status_code
        self.response = response


# === 配置异常 ===
class ConfigurationException(BaseSystemException):
    """配置异常"""

    def __init__(self, message: str, **kwargs):
        super().__init__(
            message=message,
            error_code="CONFIGURATION_ERROR",
            category=ErrorCategory.CONFIGURATION,
            severity=ErrorSeverity.HIGH,
            **kwargs
        )


class MissingConfigException(ConfigurationException):
    """缺失配置异常"""

    def __init__(self, config_key: str, **kwargs):
        super().__init__(
            message=f"Missing required configuration: {config_key}",
            **kwargs
        )
        self.error_code = "MISSING_CONFIG"
        self.config_key = config_key


class InvalidConfigException(ConfigurationException):
    """无效配置异常"""

    def __init__(self, config_key: str, value: Any, reason: str = "", **kwargs):
        message = f"Invalid configuration '{config_key}': {value}"
        if reason:
            message += f" ({reason})"
        super().__init__(message=message, **kwargs)
        self.error_code = "INVALID_CONFIG"
        self.config_key = config_key
        self.value = value


# === 认证授权异常 ===
class AuthenticationException(BaseSystemException):
    """认证异常"""

    def __init__(self, message: str = "Authentication failed", **kwargs):
        super().__init__(
            message=message,
            error_code="AUTHENTICATION_ERROR",
            category=ErrorCategory.AUTHENTICATION,
            severity=ErrorSeverity.HIGH,
            recoverable=False,
            **kwargs
        )


class AuthorizationException(BaseSystemException):
    """授权异常"""

    def __init__(self, resource: str, action: str, **kwargs):
        super().__init__(
            message=f"Access denied to {resource} for action: {action}",
            error_code="AUTHORIZATION_ERROR",
            category=ErrorCategory.AUTHORIZATION,
            severity=ErrorSeverity.HIGH,
            recoverable=False,
            **kwargs
        )
        self.resource = resource
        self.action = action


# === 性能异常 ===
class PerformanceException(BaseSystemException):
    """性能异常"""

    def __init__(self, operation: str, threshold: float, actual: float, **kwargs):
        super().__init__(
            message=f"Performance threshold exceeded for {operation}: {actual:.2f} > {threshold:.2f}",
            error_code="PERFORMANCE_ERROR",
            category=ErrorCategory.PERFORMANCE,
            severity=ErrorSeverity.MEDIUM,
            **kwargs
        )
        self.operation = operation
        self.threshold = threshold
        self.actual = actual


class TimeoutException(PerformanceException):
    """超时异常"""

    def __init__(self, operation: str, timeout: float, **kwargs):
        super().__init__(
            operation=operation,
            threshold=timeout,
            actual=timeout,
            **kwargs
        )
        self.error_code = "TIMEOUT_ERROR"
        self.timeout = timeout


class MemoryException(PerformanceException):
    """内存异常"""

    def __init__(self, operation: str, limit_mb: float, usage_mb: float, **kwargs):
        super().__init__(
            operation=operation,
            threshold=limit_mb,
            actual=usage_mb,
            **kwargs
        )
        self.error_code = "MEMORY_ERROR"
        self.limit_mb = limit_mb
        self.usage_mb = usage_mb


# === 安全异常 ===
class SecurityException(BaseSystemException):
    """安全异常"""

    def __init__(self, message: str, **kwargs):
        super().__init__(
            message=message,
            error_code="SECURITY_ERROR",
            category=ErrorCategory.SECURITY,
            severity=ErrorSeverity.CRITICAL,
            recoverable=False,
            **kwargs
        )


class RateLimitException(SecurityException):
    """频率限制异常"""

    def __init__(self, limit: int, window: int, **kwargs):
        super().__init__(
            message=f"Rate limit exceeded: {limit} requests per {window} seconds",
            **kwargs
        )
        self.error_code = "RATE_LIMIT_ERROR"
        self.limit = limit
        self.window = window


# === 异常处理工具 ===
class ExceptionHandler:
    """异常处理器"""

    @staticmethod
    def handle_exception(exc: Exception, context: Optional[ErrorContext] = None) -> BaseSystemException:
        """将标准异常转换为系统异常"""
        if isinstance(exc, BaseSystemException):
            return exc

        # 根据异常类型进行转换
        if isinstance(exc, ValueError):
            return ValidationException(
                message=str(exc),
                context=context,
                cause=exc
            )
        elif isinstance(exc, KeyError):
            return MissingParameterException(
                parameter=str(exc).strip("'\""),
                context=context,
                cause=exc
            )
        elif isinstance(exc, ConnectionError):
            return DatabaseConnectionException(
                database="unknown",
                context=context,
                cause=exc
            )
        elif isinstance(exc, TimeoutError):
            return TimeoutException(
                operation="unknown",
                timeout=0.0,
                context=context,
                cause=exc
            )
        else:
            return BaseSystemException(
                message=f"Unexpected error: {str(exc)}",
                error_code="UNEXPECTED_ERROR",
                category=ErrorCategory.SYSTEM,
                severity=ErrorSeverity.HIGH,
                context=context,
                cause=exc,
                recoverable=False
            )

    @staticmethod
    def format_error_response(exc: BaseSystemException) -> Dict[str, Any]:
        """格式化错误响应"""
        return {
            "success": False,
            "error": exc.to_dict(),
            "timestamp": __import__('datetime').datetime.now().isoformat()
        }

    @staticmethod
    def should_retry(exc: BaseSystemException) -> bool:
        """判断是否应该重试"""
        if not exc.recoverable:
            return False

        # 根据错误类型判断
        retry_categories = [
            ErrorCategory.EXTERNAL_SERVICE,
            ErrorCategory.DATA_ACCESS,
            ErrorCategory.PERFORMANCE
        ]

        return exc.category in retry_categories

    @staticmethod
    def get_retry_delay(attempt: int) -> float:
        """获取重试延迟（指数退避）"""
        import random
        base_delay = 1.0
        max_delay = 60.0
        jitter = random.uniform(0.1, 0.3)

        delay = min(base_delay * (2 ** attempt) + jitter, max_delay)
        return delay


# === 错误码映射 ===
ERROR_CODE_MAPPING = {
    # 验证错误 400
    "VALIDATION_ERROR": 400,
    "INVALID_PARAMETER": 400,
    "MISSING_PARAMETER": 400,
    "SCHEMA_VALIDATION_ERROR": 400,

    # 认证错误 401
    "AUTHENTICATION_ERROR": 401,

    # 授权错误 403
    "AUTHORIZATION_ERROR": 403,

    # 资源未找到 404
    "FACTOR_NOT_FOUND": 404,

    # 业务逻辑错误 409
    "BUSINESS_LOGIC_ERROR": 409,
    "SIGNAL_EXPIRED": 409,
    "CONFIRMATION_REQUIRED": 409,

    # 数据不足 422
    "INSUFFICIENT_DATA": 422,

    # 频率限制 429
    "RATE_LIMIT_ERROR": 429,

    # 服务器错误 500
    "DATABASE_CONNECTION_ERROR": 500,
    "DATABASE_QUERY_ERROR": 500,
    "CACHE_ERROR": 500,
    "EXTERNAL_SERVICE_ERROR": 500,
    "HIKYUU_ERROR": 500,
    "CONFIGURATION_ERROR": 500,
    "MISSING_CONFIG": 500,
    "INVALID_CONFIG": 500,
    "SECURITY_ERROR": 500,
    "MEMORY_ERROR": 500,
    "UNEXPECTED_ERROR": 500,

    # 服务不可用 503
    "API_ERROR": 503,

    # 超时错误 504
    "TIMEOUT_ERROR": 504,
    "PERFORMANCE_ERROR": 504
}


def get_http_status_code(error_code: str) -> int:
    """根据错误码获取HTTP状态码"""
    return ERROR_CODE_MAPPING.get(error_code, 500)