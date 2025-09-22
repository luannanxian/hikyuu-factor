"""
Configuration Management System
配置管理系统，支持多环境配置、动态重载和配置验证
"""

import os
import json
import yaml
from pathlib import Path
from typing import Dict, Any, Optional, Union, Type, List, Callable
from dataclasses import dataclass, field, asdict
from datetime import datetime
import threading
import time
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
import hashlib
import logging

from ..lib.exceptions import ConfigurationException, MissingConfigException, InvalidConfigException

logger = logging.getLogger(__name__)


@dataclass
class ConfigSchema:
    """配置Schema定义"""
    required_keys: List[str] = field(default_factory=list)
    optional_keys: List[str] = field(default_factory=list)
    validators: Dict[str, Callable[[Any], bool]] = field(default_factory=dict)
    type_hints: Dict[str, Type] = field(default_factory=dict)
    default_values: Dict[str, Any] = field(default_factory=dict)

    def validate(self, config: Dict[str, Any]) -> List[str]:
        """验证配置，返回错误列表"""
        errors = []

        # 检查必需字段
        for key in self.required_keys:
            if key not in config:
                errors.append(f"Missing required configuration key: {key}")

        # 检查类型
        for key, expected_type in self.type_hints.items():
            if key in config:
                value = config[key]
                if not isinstance(value, expected_type):
                    errors.append(f"Invalid type for {key}: expected {expected_type.__name__}, got {type(value).__name__}")

        # 检查自定义验证器
        for key, validator in self.validators.items():
            if key in config:
                try:
                    if not validator(config[key]):
                        errors.append(f"Validation failed for key: {key}")
                except Exception as e:
                    errors.append(f"Validation error for key {key}: {e}")

        return errors


class ConfigWatcher(FileSystemEventHandler):
    """配置文件监控器"""

    def __init__(self, config_manager: 'ConfigManager'):
        self.config_manager = config_manager
        self.last_reload = {}

    def on_modified(self, event):
        """文件修改事件处理"""
        if event.is_directory:
            return

        file_path = Path(event.src_path)
        if file_path.suffix in ['.yaml', '.yml', '.json']:
            # 防止重复触发
            now = time.time()
            last_time = self.last_reload.get(file_path, 0)
            if now - last_time < 1.0:  # 1秒内不重复加载
                return

            self.last_reload[file_path] = now
            logger.info(f"Configuration file changed: {file_path}")

            try:
                self.config_manager.reload_config(str(file_path))
            except Exception as e:
                logger.error(f"Failed to reload configuration: {e}")


class ConfigManager:
    """配置管理器"""

    def __init__(self):
        self.configs: Dict[str, Dict[str, Any]] = {}
        self.schemas: Dict[str, ConfigSchema] = {}
        self.file_paths: Dict[str, str] = {}
        self.watchers: Dict[str, Observer] = {}
        self.reload_callbacks: Dict[str, List[Callable]] = {}
        self.lock = threading.RLock()
        self._environment = os.getenv('ENVIRONMENT', 'development')

    @property
    def environment(self) -> str:
        """当前环境"""
        return self._environment

    def set_environment(self, env: str) -> None:
        """设置环境"""
        self._environment = env
        logger.info(f"Environment set to: {env}")

    def register_schema(self, name: str, schema: ConfigSchema) -> None:
        """注册配置Schema"""
        with self.lock:
            self.schemas[name] = schema
            logger.debug(f"Registered schema: {name}")

    def load_config(self, name: str, file_path: Union[str, Path],
                    schema_name: Optional[str] = None,
                    enable_watch: bool = False) -> Dict[str, Any]:
        """加载配置文件"""
        file_path = Path(file_path)

        if not file_path.exists():
            raise MissingConfigException(f"Configuration file not found: {file_path}")

        with self.lock:
            try:
                # 读取配置文件
                config_data = self._read_config_file(file_path)

                # 环境变量替换
                config_data = self._substitute_env_vars(config_data)

                # 应用环境特定配置
                config_data = self._apply_environment_config(config_data)

                # 验证Schema
                if schema_name and schema_name in self.schemas:
                    schema = self.schemas[schema_name]
                    errors = schema.validate(config_data)
                    if errors:
                        raise InvalidConfigException(
                            config_key=name,
                            value=config_data,
                            reason=f"Schema validation failed: {'; '.join(errors)}"
                        )

                    # 应用默认值
                    for key, default_value in schema.default_values.items():
                        if key not in config_data:
                            config_data[key] = default_value

                # 存储配置
                self.configs[name] = config_data
                self.file_paths[name] = str(file_path)

                # 启用文件监控
                if enable_watch:
                    self._enable_file_watcher(name, file_path)

                logger.info(f"Configuration loaded: {name} from {file_path}")
                return config_data

            except Exception as e:
                if isinstance(e, (ConfigurationException, MissingConfigException, InvalidConfigException)):
                    raise
                raise ConfigurationException(f"Failed to load configuration {name}: {e}")

    def _read_config_file(self, file_path: Path) -> Dict[str, Any]:
        """读取配置文件"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                if file_path.suffix.lower() in ['.yaml', '.yml']:
                    return yaml.safe_load(f) or {}
                elif file_path.suffix.lower() == '.json':
                    return json.load(f)
                else:
                    raise InvalidConfigException(
                        config_key=str(file_path),
                        value=file_path.suffix,
                        reason="Unsupported file format. Use .yaml, .yml, or .json"
                    )
        except yaml.YAMLError as e:
            raise InvalidConfigException(
                config_key=str(file_path),
                value="YAML content",
                reason=f"YAML parsing error: {e}"
            )
        except json.JSONDecodeError as e:
            raise InvalidConfigException(
                config_key=str(file_path),
                value="JSON content",
                reason=f"JSON parsing error: {e}"
            )

    def _substitute_env_vars(self, config: Any) -> Any:
        """替换环境变量"""
        if isinstance(config, dict):
            return {key: self._substitute_env_vars(value) for key, value in config.items()}
        elif isinstance(config, list):
            return [self._substitute_env_vars(item) for item in config]
        elif isinstance(config, str):
            # 替换 ${VAR_NAME} 或 ${VAR_NAME:default_value} 格式的环境变量
            import re
            pattern = r'\$\{([^}:]+)(?::([^}]*))?\}'

            def replace_var(match):
                var_name = match.group(1)
                default_value = match.group(2) if match.group(2) is not None else ""
                return os.getenv(var_name, default_value)

            return re.sub(pattern, replace_var, config)
        else:
            return config

    def _apply_environment_config(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """应用环境特定配置"""
        if 'environments' in config and self._environment in config['environments']:
            env_config = config['environments'][self._environment]
            # 深度合并环境配置
            return self._deep_merge(config, env_config)
        return config

    def _deep_merge(self, base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
        """深度合并字典"""
        result = base.copy()
        for key, value in override.items():
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                result[key] = self._deep_merge(result[key], value)
            else:
                result[key] = value
        return result

    def _enable_file_watcher(self, name: str, file_path: Path) -> None:
        """启用文件监控"""
        try:
            observer = Observer()
            event_handler = ConfigWatcher(self)
            observer.schedule(event_handler, str(file_path.parent), recursive=False)
            observer.start()

            self.watchers[name] = observer
            logger.debug(f"File watcher enabled for: {name}")

        except Exception as e:
            logger.warning(f"Failed to enable file watcher for {name}: {e}")

    def get_config(self, name: str) -> Dict[str, Any]:
        """获取配置"""
        with self.lock:
            if name not in self.configs:
                raise MissingConfigException(f"Configuration not loaded: {name}")
            return self.configs[name].copy()

    def get_value(self, name: str, key: str, default: Any = None) -> Any:
        """获取配置值"""
        config = self.get_config(name)
        return self._get_nested_value(config, key, default)

    def _get_nested_value(self, config: Dict[str, Any], key: str, default: Any) -> Any:
        """获取嵌套配置值"""
        keys = key.split('.')
        current = config

        for k in keys:
            if isinstance(current, dict) and k in current:
                current = current[k]
            else:
                return default

        return current

    def set_value(self, name: str, key: str, value: Any) -> None:
        """设置配置值（运行时）"""
        with self.lock:
            if name not in self.configs:
                raise MissingConfigException(f"Configuration not loaded: {name}")

            config = self.configs[name]
            self._set_nested_value(config, key, value)

            # 触发重载回调
            self._trigger_reload_callbacks(name)

    def _set_nested_value(self, config: Dict[str, Any], key: str, value: Any) -> None:
        """设置嵌套配置值"""
        keys = key.split('.')
        current = config

        for k in keys[:-1]:
            if k not in current:
                current[k] = {}
            current = current[k]

        current[keys[-1]] = value

    def reload_config(self, name_or_path: str) -> None:
        """重新加载配置"""
        with self.lock:
            # 如果是文件路径，找到对应的配置名
            config_name = None
            for name, path in self.file_paths.items():
                if path == name_or_path or name == name_or_path:
                    config_name = name
                    break

            if not config_name:
                logger.warning(f"Configuration not found for reload: {name_or_path}")
                return

            file_path = self.file_paths[config_name]

            try:
                old_config = self.configs[config_name].copy()
                new_config = self._read_config_file(Path(file_path))
                new_config = self._substitute_env_vars(new_config)
                new_config = self._apply_environment_config(new_config)

                # 检查配置是否有变化
                if self._config_hash(old_config) != self._config_hash(new_config):
                    self.configs[config_name] = new_config
                    logger.info(f"Configuration reloaded: {config_name}")
                    self._trigger_reload_callbacks(config_name)
                else:
                    logger.debug(f"Configuration unchanged: {config_name}")

            except Exception as e:
                logger.error(f"Failed to reload configuration {config_name}: {e}")

    def _config_hash(self, config: Dict[str, Any]) -> str:
        """计算配置哈希值"""
        config_str = json.dumps(config, sort_keys=True)
        return hashlib.md5(config_str.encode()).hexdigest()

    def add_reload_callback(self, name: str, callback: Callable[[Dict[str, Any]], None]) -> None:
        """添加重载回调"""
        with self.lock:
            if name not in self.reload_callbacks:
                self.reload_callbacks[name] = []
            self.reload_callbacks[name].append(callback)

    def _trigger_reload_callbacks(self, name: str) -> None:
        """触发重载回调"""
        if name in self.reload_callbacks:
            config = self.configs[name]
            for callback in self.reload_callbacks[name]:
                try:
                    callback(config)
                except Exception as e:
                    logger.error(f"Reload callback error for {name}: {e}")

    def export_config(self, name: str, file_path: Union[str, Path],
                      format: str = "yaml", exclude_sensitive: bool = True) -> None:
        """导出配置"""
        config = self.get_config(name)

        if exclude_sensitive:
            config = self._mask_sensitive_data(config)

        file_path = Path(file_path)
        file_path.parent.mkdir(parents=True, exist_ok=True)

        try:
            with open(file_path, 'w', encoding='utf-8') as f:
                if format.lower() in ['yaml', 'yml']:
                    yaml.dump(config, f, default_flow_style=False, allow_unicode=True)
                elif format.lower() == 'json':
                    json.dump(config, f, indent=2, ensure_ascii=False)
                else:
                    raise InvalidConfigException(
                        config_key="export_format",
                        value=format,
                        reason="Unsupported format. Use 'yaml' or 'json'"
                    )

            logger.info(f"Configuration exported: {name} to {file_path}")

        except Exception as e:
            raise ConfigurationException(f"Failed to export configuration {name}: {e}")

    def _mask_sensitive_data(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """屏蔽敏感数据"""
        sensitive_keys = ['password', 'secret', 'key', 'token', 'credential']
        masked_config = config.copy()

        def mask_recursive(obj):
            if isinstance(obj, dict):
                for key, value in obj.items():
                    if any(sensitive in key.lower() for sensitive in sensitive_keys):
                        obj[key] = "***MASKED***"
                    else:
                        mask_recursive(value)
            elif isinstance(obj, list):
                for item in obj:
                    mask_recursive(item)

        mask_recursive(masked_config)
        return masked_config

    def validate_all_configs(self) -> Dict[str, List[str]]:
        """验证所有配置"""
        results = {}
        with self.lock:
            for name, config in self.configs.items():
                if name in self.schemas:
                    schema = self.schemas[name]
                    errors = schema.validate(config)
                    if errors:
                        results[name] = errors

        return results

    def close(self) -> None:
        """关闭配置管理器"""
        with self.lock:
            # 停止文件监控
            for name, observer in self.watchers.items():
                try:
                    observer.stop()
                    observer.join()
                    logger.debug(f"File watcher stopped: {name}")
                except Exception as e:
                    logger.error(f"Error stopping file watcher {name}: {e}")

            self.watchers.clear()
            self.configs.clear()
            self.schemas.clear()
            self.file_paths.clear()
            self.reload_callbacks.clear()


# 全局配置管理器实例
config_manager = ConfigManager()


# 便捷函数
def load_config(name: str, file_path: Union[str, Path], **kwargs) -> Dict[str, Any]:
    """加载配置"""
    return config_manager.load_config(name, file_path, **kwargs)


def get_config(name: str) -> Dict[str, Any]:
    """获取配置"""
    return config_manager.get_config(name)


def get_config_value(name: str, key: str, default: Any = None) -> Any:
    """获取配置值"""
    return config_manager.get_value(name, key, default)


def set_config_value(name: str, key: str, value: Any) -> None:
    """设置配置值"""
    config_manager.set_value(name, key, value)


# 预设配置Schema
def get_database_schema() -> ConfigSchema:
    """数据库配置Schema"""
    return ConfigSchema(
        required_keys=['host', 'port', 'user', 'password', 'database'],
        type_hints={
            'host': str,
            'port': int,
            'user': str,
            'password': str,
            'database': str,
            'pool_size': int,
            'max_overflow': int
        },
        validators={
            'port': lambda x: 1 <= x <= 65535,
            'pool_size': lambda x: x > 0,
            'max_overflow': lambda x: x >= 0
        },
        default_values={
            'charset': 'utf8mb4',
            'pool_size': 10,
            'max_overflow': 20,
            'pool_timeout': 30
        }
    )


def get_logging_schema() -> ConfigSchema:
    """日志配置Schema"""
    return ConfigSchema(
        required_keys=['level'],
        type_hints={
            'level': str,
            'enable_console': bool,
            'enable_file': bool,
            'file_path': str
        },
        validators={
            'level': lambda x: x.upper() in ['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL']
        },
        default_values={
            'enable_console': True,
            'enable_file': True,
            'file_path': 'logs/application.log'
        }
    )


def get_agent_schema() -> ConfigSchema:
    """Agent配置Schema"""
    return ConfigSchema(
        required_keys=['agent_id', 'port'],
        type_hints={
            'agent_id': str,
            'port': int,
            'host': str,
            'debug': bool
        },
        validators={
            'port': lambda x: 1024 <= x <= 65535
        },
        default_values={
            'host': '127.0.0.1',
            'debug': False,
            'enable_cors': True
        }
    )