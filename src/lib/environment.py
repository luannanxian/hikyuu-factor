"""
Environment Detection and Mock Data Protection

环境检测和Mock数据防护机制
"""

import os
import logging
from typing import Optional, Any
from enum import Enum


class Environment(Enum):
    """环境类型枚举"""
    DEVELOPMENT = "development"
    TESTING = "testing"
    INTEGRATION = "integration"
    DEMO = "demo"
    STAGING = "staging"
    PRODUCTION = "production"


class MockDataError(Exception):
    """Mock数据使用错误"""
    pass


class EnvironmentManager:
    """环境管理器"""

    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self._current_env = self._detect_environment()

    def _detect_environment(self) -> Environment:
        """检测当前环境"""
        env_str = os.getenv("ENVIRONMENT", "development").lower()

        try:
            return Environment(env_str)
        except ValueError:
            self.logger.warning(f"Unknown environment '{env_str}', defaulting to development")
            return Environment.DEVELOPMENT

    @property
    def current_environment(self) -> Environment:
        """获取当前环境"""
        return self._current_env

    def is_development_only(self) -> bool:
        """检查是否仅限开发环境"""
        return self._current_env == Environment.DEVELOPMENT

    def is_mock_data_allowed(self) -> bool:
        """检查是否允许使用Mock数据"""
        # 严格规范：仅开发环境允许Mock数据
        allowed_environments = {Environment.DEVELOPMENT}
        return self._current_env in allowed_environments

    def validate_no_mock_data(self, context: str = ""):
        """验证不允许使用Mock数据"""
        if not self.is_mock_data_allowed():
            error_msg = f"Mock data usage is prohibited in {self._current_env.value} environment"
            if context:
                error_msg += f" (Context: {context})"

            self.logger.critical(error_msg)
            raise MockDataError(error_msg)

    def get_real_data_requirement_message(self) -> str:
        """获取真实数据要求提示信息"""
        return (
            f"Current environment: {self._current_env.value}. "
            "Real data integration required. Please implement actual data sources."
        )


# 全局环境管理器实例
env_manager = EnvironmentManager()


def require_real_data(func):
    """装饰器：要求使用真实数据"""
    def wrapper(*args, **kwargs):
        env_manager.validate_no_mock_data(f"Function: {func.__name__}")
        return func(*args, **kwargs)
    return wrapper


def development_only(func):
    """装饰器：仅开发环境可用"""
    def wrapper(*args, **kwargs):
        if not env_manager.is_development_only():
            raise MockDataError(
                f"Function {func.__name__} is only available in development environment. "
                f"Current environment: {env_manager.current_environment.value}"
            )
        return func(*args, **kwargs)
    return wrapper


def warn_mock_data(context: str = ""):
    """警告Mock数据使用"""
    if env_manager.is_mock_data_allowed():
        logger = logging.getLogger(__name__)
        logger.warning(f"Using mock data in development environment. Context: {context}")
    else:
        env_manager.validate_no_mock_data(context)