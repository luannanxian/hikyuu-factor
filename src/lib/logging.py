"""
Logging Configuration and Utilities
日志配置和工具类，提供结构化日志记录和统一的日志管理
"""

import logging
import logging.config
import logging.handlers
import json
import sys
import os
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional, Union, List
from dataclasses import dataclass, field
import threading
import traceback
from contextlib import contextmanager
import uuid


@dataclass
class LogConfig:
    """日志配置"""
    level: str = "INFO"
    format_style: str = "structured"  # structured, simple, detailed
    enable_console: bool = True
    enable_file: bool = True
    file_path: str = "logs/application.log"
    max_file_size: int = 100 * 1024 * 1024  # 100MB
    backup_count: int = 5
    enable_rotation: bool = True
    enable_async: bool = False
    buffer_size: int = 1000
    flush_interval: float = 1.0
    custom_fields: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            "level": self.level,
            "format_style": self.format_style,
            "enable_console": self.enable_console,
            "enable_file": self.enable_file,
            "file_path": self.file_path,
            "max_file_size": self.max_file_size,
            "backup_count": self.backup_count,
            "enable_rotation": self.enable_rotation,
            "enable_async": self.enable_async,
            "buffer_size": self.buffer_size,
            "flush_interval": self.flush_interval,
            "custom_fields": self.custom_fields
        }


class StructuredFormatter(logging.Formatter):
    """结构化日志格式化器"""

    def __init__(self, include_extra: bool = True):
        super().__init__()
        self.include_extra = include_extra

    def format(self, record: logging.LogRecord) -> str:
        """格式化日志记录"""
        # 基础字段
        log_data = {
            "timestamp": datetime.fromtimestamp(record.created).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno,
            "thread_id": record.thread,
            "thread_name": record.threadName,
            "process_id": record.process
        }

        # 添加异常信息
        if record.exc_info:
            log_data["exception"] = {
                "type": record.exc_info[0].__name__ if record.exc_info[0] else None,
                "message": str(record.exc_info[1]) if record.exc_info[1] else None,
                "traceback": self.formatException(record.exc_info)
            }

        # 添加额外字段
        if self.include_extra:
            extra_fields = {}
            for key, value in record.__dict__.items():
                if key not in {
                    'name', 'msg', 'args', 'levelname', 'levelno', 'pathname',
                    'filename', 'module', 'lineno', 'funcName', 'created',
                    'msecs', 'relativeCreated', 'thread', 'threadName',
                    'processName', 'process', 'message', 'exc_info', 'exc_text',
                    'stack_info', 'getMessage'
                }:
                    try:
                        # 确保值可以JSON序列化
                        json.dumps(value)
                        extra_fields[key] = value
                    except (TypeError, ValueError):
                        extra_fields[key] = str(value)

            if extra_fields:
                log_data["extra"] = extra_fields

        return json.dumps(log_data, ensure_ascii=False)


class ColoredFormatter(logging.Formatter):
    """彩色日志格式化器（用于控制台）"""

    # ANSI颜色代码
    COLORS = {
        'DEBUG': '\033[36m',      # 青色
        'INFO': '\033[32m',       # 绿色
        'WARNING': '\033[33m',    # 黄色
        'ERROR': '\033[31m',      # 红色
        'CRITICAL': '\033[35m',   # 紫色
        'RESET': '\033[0m'        # 重置
    }

    def __init__(self, fmt: str = None):
        super().__init__()
        self.fmt = fmt or "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"

    def format(self, record: logging.LogRecord) -> str:
        """格式化日志记录"""
        # 获取颜色
        level_color = self.COLORS.get(record.levelname, '')
        reset_color = self.COLORS['RESET']

        # 格式化消息
        formatted = super().format(record)

        # 只对级别名称添加颜色
        if level_color:
            formatted = formatted.replace(
                record.levelname,
                f"{level_color}{record.levelname}{reset_color}"
            )

        return formatted


class AsyncHandler(logging.Handler):
    """异步日志处理器"""

    def __init__(self, target_handler: logging.Handler, buffer_size: int = 1000):
        super().__init__()
        self.target_handler = target_handler
        self.buffer = []
        self.buffer_size = buffer_size
        self.lock = threading.Lock()
        self.flush_thread = None
        self.stop_event = threading.Event()

    def emit(self, record: logging.LogRecord) -> None:
        """发送日志记录"""
        with self.lock:
            self.buffer.append(record)
            if len(self.buffer) >= self.buffer_size:
                self._flush_buffer()

    def _flush_buffer(self) -> None:
        """刷新缓冲区"""
        if not self.buffer:
            return

        records_to_flush = self.buffer.copy()
        self.buffer.clear()

        # 在后台线程中处理
        def flush_records():
            for record in records_to_flush:
                try:
                    self.target_handler.emit(record)
                except Exception:
                    # 避免日志处理中的异常影响主线程
                    pass

        thread = threading.Thread(target=flush_records, daemon=True)
        thread.start()

    def close(self) -> None:
        """关闭处理器"""
        self.stop_event.set()
        with self.lock:
            self._flush_buffer()
        self.target_handler.close()
        super().close()


class ContextFilter(logging.Filter):
    """上下文过滤器，添加请求级别的上下文信息"""

    def __init__(self):
        super().__init__()
        self._context = threading.local()

    def set_context(self, **kwargs) -> None:
        """设置上下文"""
        for key, value in kwargs.items():
            setattr(self._context, key, value)

    def clear_context(self) -> None:
        """清空上下文"""
        self._context = threading.local()

    def filter(self, record: logging.LogRecord) -> bool:
        """过滤并添加上下文"""
        # 添加上下文字段到记录中
        for key in dir(self._context):
            if not key.startswith('_'):
                setattr(record, key, getattr(self._context, key))

        return True


class LoggerManager:
    """日志管理器"""

    def __init__(self):
        self.configs: Dict[str, LogConfig] = {}
        self.loggers: Dict[str, logging.Logger] = {}
        self.context_filter = ContextFilter()
        self._initialized = False

    def initialize(self, config: LogConfig, logger_name: str = "hikyuu_factor") -> None:
        """初始化日志系统"""
        self.configs[logger_name] = config

        # 创建日志目录
        if config.enable_file:
            log_path = Path(config.file_path)
            log_path.parent.mkdir(parents=True, exist_ok=True)

        # 配置根日志器
        root_logger = logging.getLogger()
        root_logger.setLevel(getattr(logging, config.level.upper()))

        # 清除现有处理器
        for handler in root_logger.handlers[:]:
            root_logger.removeHandler(handler)

        # 添加处理器
        handlers = self._create_handlers(config)
        for handler in handlers:
            handler.addFilter(self.context_filter)
            root_logger.addHandler(handler)

        # 创建特定日志器
        logger = logging.getLogger(logger_name)
        self.loggers[logger_name] = logger

        self._initialized = True

    def _create_handlers(self, config: LogConfig) -> List[logging.Handler]:
        """创建日志处理器"""
        handlers = []

        # 控制台处理器
        if config.enable_console:
            console_handler = logging.StreamHandler(sys.stdout)
            if config.format_style == "structured":
                console_handler.setFormatter(StructuredFormatter())
            else:
                console_handler.setFormatter(ColoredFormatter())
            handlers.append(console_handler)

        # 文件处理器
        if config.enable_file:
            if config.enable_rotation:
                file_handler = logging.handlers.RotatingFileHandler(
                    config.file_path,
                    maxBytes=config.max_file_size,
                    backupCount=config.backup_count,
                    encoding='utf-8'
                )
            else:
                file_handler = logging.FileHandler(
                    config.file_path,
                    encoding='utf-8'
                )

            file_handler.setFormatter(StructuredFormatter())

            # 异步处理器包装
            if config.enable_async:
                file_handler = AsyncHandler(file_handler, config.buffer_size)

            handlers.append(file_handler)

        return handlers

    def get_logger(self, name: str = "hikyuu_factor") -> logging.Logger:
        """获取日志器"""
        if not self._initialized:
            # 使用默认配置初始化
            self.initialize(LogConfig())

        return self.loggers.get(name, logging.getLogger(name))

    @contextmanager
    def log_context(self, **kwargs):
        """日志上下文管理器"""
        self.context_filter.set_context(**kwargs)
        try:
            yield
        finally:
            self.context_filter.clear_context()

    def update_config(self, config: LogConfig, logger_name: str = "hikyuu_factor") -> None:
        """更新日志配置"""
        self.initialize(config, logger_name)


# 全局日志管理器实例
log_manager = LoggerManager()


def setup_logging(config: Optional[LogConfig] = None) -> None:
    """设置日志系统"""
    if config is None:
        config = LogConfig()
    log_manager.initialize(config)


def get_logger(name: str = "hikyuu_factor") -> logging.Logger:
    """获取日志器"""
    return log_manager.get_logger(name)


@contextmanager
def log_context(**kwargs):
    """日志上下文管理器"""
    with log_manager.log_context(**kwargs):
        yield


def log_function_call(func):
    """函数调用日志装饰器"""
    def wrapper(*args, **kwargs):
        logger = get_logger()
        func_name = f"{func.__module__}.{func.__name__}"

        # 生成调用ID
        call_id = str(uuid.uuid4())[:8]

        with log_context(function=func_name, call_id=call_id):
            logger.info(f"Function call started", extra={
                "args_count": len(args),
                "kwargs_keys": list(kwargs.keys())
            })

            start_time = datetime.now()
            try:
                result = func(*args, **kwargs)
                end_time = datetime.now()
                duration = (end_time - start_time).total_seconds()

                logger.info(f"Function call completed", extra={
                    "duration_seconds": duration,
                    "success": True
                })
                return result

            except Exception as e:
                end_time = datetime.now()
                duration = (end_time - start_time).total_seconds()

                logger.error(f"Function call failed", extra={
                    "duration_seconds": duration,
                    "error_type": type(e).__name__,
                    "error_message": str(e)
                }, exc_info=True)
                raise

    return wrapper


def log_performance(operation: str, threshold_seconds: float = 1.0):
    """性能日志装饰器"""
    def decorator(func):
        def wrapper(*args, **kwargs):
            logger = get_logger()
            start_time = datetime.now()

            try:
                result = func(*args, **kwargs)
                end_time = datetime.now()
                duration = (end_time - start_time).total_seconds()

                log_level = logging.WARNING if duration > threshold_seconds else logging.INFO
                logger.log(log_level, f"Performance: {operation}", extra={
                    "duration_seconds": duration,
                    "threshold_seconds": threshold_seconds,
                    "exceeded_threshold": duration > threshold_seconds
                })

                return result

            except Exception as e:
                end_time = datetime.now()
                duration = (end_time - start_time).total_seconds()

                logger.error(f"Performance (failed): {operation}", extra={
                    "duration_seconds": duration,
                    "error": str(e)
                }, exc_info=True)
                raise

        return wrapper
    return decorator


class LogCapture:
    """日志捕获器（用于测试）"""

    def __init__(self, logger_name: str = "hikyuu_factor", level: int = logging.DEBUG):
        self.logger = logging.getLogger(logger_name)
        self.handler = logging.handlers.MemoryHandler(capacity=1000)
        self.handler.setLevel(level)
        self.records: List[logging.LogRecord] = []

    def __enter__(self):
        self.handler.setTarget(logging.NullHandler())
        self.logger.addHandler(self.handler)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.logger.removeHandler(self.handler)
        self.records = self.handler.buffer.copy()
        self.handler.close()

    def get_messages(self, level: Optional[str] = None) -> List[str]:
        """获取日志消息"""
        records = self.records
        if level:
            level_num = getattr(logging, level.upper())
            records = [r for r in records if r.levelno >= level_num]

        return [record.getMessage() for record in records]

    def has_message(self, message: str, level: Optional[str] = None) -> bool:
        """检查是否包含特定消息"""
        messages = self.get_messages(level)
        return any(message in msg for msg in messages)


# 预设日志配置
def get_development_config() -> LogConfig:
    """开发环境日志配置"""
    return LogConfig(
        level="DEBUG",
        format_style="simple",
        enable_console=True,
        enable_file=True,
        file_path="logs/dev.log",
        enable_async=False
    )


def get_production_config() -> LogConfig:
    """生产环境日志配置"""
    return LogConfig(
        level="INFO",
        format_style="structured",
        enable_console=False,
        enable_file=True,
        file_path="logs/prod.log",
        max_file_size=500 * 1024 * 1024,  # 500MB
        backup_count=10,
        enable_async=True,
        buffer_size=2000
    )


def get_testing_config() -> LogConfig:
    """测试环境日志配置"""
    return LogConfig(
        level="WARNING",
        format_style="simple",
        enable_console=True,
        enable_file=False,
        enable_async=False
    )