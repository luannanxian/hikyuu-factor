"""
Hikyuu Framework Wrapper
Hikyuu框架封装和集成，提供统一的Hikyuu初始化和性能优化
"""

import os
import logging
from typing import Dict, List, Optional, Any, Union, Callable
from dataclasses import dataclass, field
from datetime import date, datetime
from pathlib import Path
import threading
import time
from contextlib import contextmanager

# Hikyuu相关导入
try:
    import hikyuu as hku
    from hikyuu import StockManager, KData, Query, Datetime
    HAS_HIKYUU = True
except ImportError:
    HAS_HIKYUU = False
    hku = None
    StockManager = None
    KData = None
    Query = None
    Datetime = None

from ..models.platform_config import PlatformType, OptimizationConfig
from ..lib.exceptions import HikyuuException, ConfigurationException

logger = logging.getLogger(__name__)


@dataclass
class HikyuuConfig:
    """Hikyuu配置"""
    data_dir: str = "data"
    base_info_driver: str = "stock"
    base_info_config: Dict[str, Any] = field(default_factory=dict)
    block_driver: str = "qianlong"
    block_config: Dict[str, Any] = field(default_factory=dict)
    kdata_driver: str = "tdx"
    kdata_config: Dict[str, Any] = field(default_factory=dict)
    preload_day: int = 100
    preload_time: str = ""
    preload_min: str = ""
    max_cache_num: int = 500
    enable_cache: bool = True
    log_level: str = "info"
    cpu_num: int = 0  # 0表示自动检测

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            "data_dir": self.data_dir,
            "base_info_driver": self.base_info_driver,
            "base_info_config": self.base_info_config,
            "block_driver": self.block_driver,
            "block_config": self.block_config,
            "kdata_driver": self.kdata_driver,
            "kdata_config": self.kdata_config,
            "preload_day": self.preload_day,
            "preload_time": self.preload_time,
            "preload_min": self.preload_min,
            "max_cache_num": self.max_cache_num,
            "enable_cache": self.enable_cache,
            "log_level": self.log_level,
            "cpu_num": self.cpu_num
        }


class HikyuuWrapper:
    """Hikyuu框架封装类"""

    def __init__(self):
        self.is_initialized = False
        self.config: Optional[HikyuuConfig] = None
        self.stock_manager: Optional[StockManager] = None
        self.platform_config: Optional[OptimizationConfig] = None
        self._lock = threading.Lock()
        self._performance_stats = {}

    def initialize(self, config: HikyuuConfig,
                   optimization_config: Optional[OptimizationConfig] = None) -> bool:
        """初始化Hikyuu框架"""
        if not HAS_HIKYUU:
            raise HikyuuException("Hikyuu module not available")

        with self._lock:
            if self.is_initialized:
                logger.warning("Hikyuu already initialized")
                return True

            try:
                self.config = config
                self.platform_config = optimization_config or OptimizationConfig.auto_detect()

                # 设置配置
                hikyuu_config = self._prepare_hikyuu_config()

                # 初始化Hikyuu
                logger.info("Initializing Hikyuu framework...")
                hku.hikyuu_init(hikyuu_config)

                # 获取股票管理器
                self.stock_manager = hku.StockManager.instance()

                # 应用平台优化
                self._apply_platform_optimization()

                self.is_initialized = True
                logger.info("Hikyuu framework initialized successfully")

                # 记录初始化统计
                self._record_initialization_stats()

                return True

            except Exception as e:
                logger.error(f"Failed to initialize Hikyuu: {e}")
                raise HikyuuException(f"Initialization failed: {e}")

    def _prepare_hikyuu_config(self) -> Dict[str, Any]:
        """准备Hikyuu配置"""
        config_dict = self.config.to_dict()

        # 应用平台优化配置
        if self.platform_config:
            hikyuu_params = self.platform_config.get_hikyuu_params()
            config_dict.update(hikyuu_params)

        # 确保数据目录存在
        data_dir = Path(config_dict["data_dir"])
        data_dir.mkdir(parents=True, exist_ok=True)

        # 转换为Hikyuu期望的格式
        hikyuu_config = {
            "tmpdir": str(data_dir / "tmp"),
            "datadir": str(data_dir),
            "baseInfoDriver": {
                "name": config_dict["base_info_driver"],
                "config": config_dict["base_info_config"]
            },
            "blockDriver": {
                "name": config_dict["block_driver"],
                "config": config_dict["block_config"]
            },
            "kdataDriver": {
                "name": config_dict["kdata_driver"],
                "config": config_dict["kdata_config"]
            },
            "preloadDay": config_dict["preload_day"],
            "preloadTime": config_dict["preload_time"],
            "preloadMin": config_dict["preload_min"],
            "maxCacheNum": config_dict["max_cache_num"],
            "logLevel": config_dict["log_level"],
            "cpuNum": config_dict["cpu_num"]
        }

        return hikyuu_config

    def _apply_platform_optimization(self) -> None:
        """应用平台优化配置"""
        if not self.platform_config:
            return

        try:
            # 设置线程数
            if self.platform_config.thread_count:
                # Hikyuu内部设置，如果有相关API
                logger.info(f"Platform optimization applied: {self.platform_config.thread_count} threads")

            # 记录优化信息
            logger.info(f"Platform optimization: {self.platform_config.platform_type.value}")
            logger.info(f"SIMD enabled: {self.platform_config.enable_simd}")
            logger.info(f"Parallel enabled: {self.platform_config.enable_parallel}")

        except Exception as e:
            logger.warning(f"Platform optimization application failed: {e}")

    def _record_initialization_stats(self) -> None:
        """记录初始化统计信息"""
        try:
            stock_count = len(self.get_stock_list()) if self.stock_manager else 0
            self._performance_stats = {
                "initialization_time": datetime.now(),
                "stock_count": stock_count,
                "platform_type": self.platform_config.platform_type.value if self.platform_config else "unknown",
                "cache_enabled": self.config.enable_cache if self.config else False
            }
            logger.info(f"Hikyuu initialized with {stock_count} stocks")

        except Exception as e:
            logger.warning(f"Failed to record initialization stats: {e}")

    def is_ready(self) -> bool:
        """检查Hikyuu是否就绪"""
        return self.is_initialized and self.stock_manager is not None

    def get_stock_list(self, market: Optional[str] = None) -> List[Dict[str, Any]]:
        """获取股票列表"""
        if not self.is_ready():
            raise HikyuuException("Hikyuu not initialized")

        try:
            stock_list = []
            market_info = hku.getStockTypeInfo()

            for market_code, market_data in market_info.items():
                if market and market.upper() != market_code.upper():
                    continue

                for stock in market_data:
                    stock_info = {
                        "code": stock.market_code + stock.code,
                        "name": stock.name,
                        "market": stock.market_code,
                        "type": stock.type,
                        "start_date": stock.start_datetime.date() if stock.start_datetime else None,
                        "end_date": stock.end_datetime.date() if stock.end_datetime else None
                    }
                    stock_list.append(stock_info)

            return stock_list

        except Exception as e:
            raise HikyuuException(f"Failed to get stock list: {e}")

    def get_stock(self, code: str) -> Optional[Any]:
        """获取股票对象"""
        if not self.is_ready():
            raise HikyuuException("Hikyuu not initialized")

        try:
            return hku.getStock(code)
        except Exception as e:
            logger.warning(f"Failed to get stock {code}: {e}")
            return None

    def get_kdata(self, code: str, query: Optional[Any] = None) -> Optional[Any]:
        """获取K线数据"""
        if not self.is_ready():
            raise HikyuuException("Hikyuu not initialized")

        try:
            stock = self.get_stock(code)
            if not stock:
                return None

            if query is None:
                query = hku.QueryByIndex(-1000)  # 默认获取最近1000个数据点

            return stock.getKData(query)

        except Exception as e:
            raise HikyuuException(f"Failed to get kdata for {code}: {e}")

    def calculate_indicator(self, indicator_name: str, kdata: Any, **params) -> Optional[Any]:
        """计算技术指标"""
        if not self.is_ready():
            raise HikyuuException("Hikyuu not initialized")

        try:
            # 获取指标函数
            indicator_func = getattr(hku, indicator_name.upper(), None)
            if not indicator_func:
                raise HikyuuException(f"Unknown indicator: {indicator_name}")

            # 调用指标函数
            if params:
                return indicator_func(kdata, **params)
            else:
                return indicator_func(kdata)

        except Exception as e:
            raise HikyuuException(f"Failed to calculate indicator {indicator_name}: {e}")

    @contextmanager
    def performance_monitor(self, operation: str):
        """性能监控上下文管理器"""
        start_time = time.time()
        start_memory = self._get_memory_usage()

        try:
            yield
        finally:
            end_time = time.time()
            end_memory = self._get_memory_usage()

            duration = end_time - start_time
            memory_delta = end_memory - start_memory

            logger.info(f"Performance [{operation}]: {duration:.3f}s, memory: {memory_delta:.1f}MB")

    def _get_memory_usage(self) -> float:
        """获取内存使用量（MB）"""
        try:
            import psutil
            process = psutil.Process()
            return process.memory_info().rss / 1024 / 1024
        except:
            return 0.0

    def get_performance_stats(self) -> Dict[str, Any]:
        """获取性能统计"""
        stats = self._performance_stats.copy()

        if self.platform_config:
            stats["platform_config"] = self.platform_config.to_dict()

        if self.config:
            stats["hikyuu_config"] = self.config.to_dict()

        return stats

    def reload_data(self) -> bool:
        """重新加载数据"""
        if not self.is_ready():
            raise HikyuuException("Hikyuu not initialized")

        try:
            # 重新加载股票管理器
            self.stock_manager.reload()
            logger.info("Hikyuu data reloaded")
            return True

        except Exception as e:
            logger.error(f"Failed to reload Hikyuu data: {e}")
            return False

    def shutdown(self) -> None:
        """关闭Hikyuu"""
        with self._lock:
            if self.is_initialized:
                try:
                    # Hikyuu没有显式的shutdown方法，但可以清理资源
                    self.stock_manager = None
                    self.is_initialized = False
                    logger.info("Hikyuu shutdown completed")

                except Exception as e:
                    logger.error(f"Error during Hikyuu shutdown: {e}")


class HikyuuFactorCalculator:
    """Hikyuu因子计算器"""

    def __init__(self, wrapper: HikyuuWrapper):
        self.wrapper = wrapper

    def calculate_momentum_factor(self, stock_codes: List[str], periods: List[int],
                                  start_date: date, end_date: date) -> Dict[str, Any]:
        """计算动量因子"""
        if not self.wrapper.is_ready():
            raise HikyuuException("Hikyuu not ready")

        results = {}

        with self.wrapper.performance_monitor("momentum_calculation"):
            for code in stock_codes:
                try:
                    kdata = self.wrapper.get_kdata(
                        code,
                        hku.QueryByDate(
                            hku.Datetime(start_date),
                            hku.Datetime(end_date)
                        )
                    )

                    if not kdata or len(kdata) == 0:
                        continue

                    stock_results = {}
                    for period in periods:
                        # 计算动量（价格变化率）
                        momentum = self.wrapper.calculate_indicator("ROCP", kdata, period)
                        if momentum:
                            stock_results[f"momentum_{period}d"] = [
                                float(momentum[i]) for i in range(len(momentum))
                                if not momentum[i].isnan()
                            ]

                    if stock_results:
                        results[code] = stock_results

                except Exception as e:
                    logger.warning(f"Failed to calculate momentum for {code}: {e}")

        return results

    def calculate_technical_factors(self, stock_codes: List[str],
                                    indicators: List[str],
                                    start_date: date, end_date: date) -> Dict[str, Any]:
        """计算技术因子"""
        if not self.wrapper.is_ready():
            raise HikyuuException("Hikyuu not ready")

        results = {}

        with self.wrapper.performance_monitor("technical_calculation"):
            for code in stock_codes:
                try:
                    kdata = self.wrapper.get_kdata(
                        code,
                        hku.QueryByDate(
                            hku.Datetime(start_date),
                            hku.Datetime(end_date)
                        )
                    )

                    if not kdata or len(kdata) == 0:
                        continue

                    stock_results = {}
                    for indicator in indicators:
                        try:
                            if indicator.upper() == "RSI":
                                rsi = self.wrapper.calculate_indicator("RSI", kdata, 14)
                                if rsi:
                                    stock_results["rsi"] = [
                                        float(rsi[i]) for i in range(len(rsi))
                                        if not rsi[i].isnan()
                                    ]

                            elif indicator.upper() == "MACD":
                                macd = self.wrapper.calculate_indicator("MACD", kdata)
                                if macd:
                                    stock_results["macd"] = [
                                        float(macd[i]) for i in range(len(macd))
                                        if not macd[i].isnan()
                                    ]

                            elif indicator.upper() == "BOLL":
                                boll = self.wrapper.calculate_indicator("BOLL", kdata)
                                if boll:
                                    # BOLL返回上轨、中轨、下轨
                                    stock_results["boll_upper"] = [
                                        float(boll.getResult(0)[i]) for i in range(len(boll))
                                        if not boll.getResult(0)[i].isnan()
                                    ]
                                    stock_results["boll_middle"] = [
                                        float(boll.getResult(1)[i]) for i in range(len(boll))
                                        if not boll.getResult(1)[i].isnan()
                                    ]
                                    stock_results["boll_lower"] = [
                                        float(boll.getResult(2)[i]) for i in range(len(boll))
                                        if not boll.getResult(2)[i].isnan()
                                    ]

                        except Exception as e:
                            logger.warning(f"Failed to calculate {indicator} for {code}: {e}")

                    if stock_results:
                        results[code] = stock_results

                except Exception as e:
                    logger.warning(f"Failed to calculate technical factors for {code}: {e}")

        return results


# 全局Hikyuu实例
hikyuu_wrapper = HikyuuWrapper()


def initialize_hikyuu(config: Optional[HikyuuConfig] = None,
                      optimization_config: Optional[OptimizationConfig] = None) -> bool:
    """初始化全局Hikyuu实例"""
    if config is None:
        config = get_default_hikyuu_config()

    return hikyuu_wrapper.initialize(config, optimization_config)


def get_hikyuu_wrapper() -> HikyuuWrapper:
    """获取全局Hikyuu实例"""
    return hikyuu_wrapper


def get_default_hikyuu_config() -> HikyuuConfig:
    """获取默认Hikyuu配置"""
    # 检测平台并应用优化
    platform_config = OptimizationConfig.auto_detect()

    return HikyuuConfig(
        data_dir="data/hikyuu",
        base_info_driver="stock",
        kdata_driver="tdx",
        preload_day=100,
        max_cache_num=500,
        enable_cache=True,
        log_level="info",
        cpu_num=platform_config.thread_count or 0
    )


def check_hikyuu_availability() -> Dict[str, Any]:
    """检查Hikyuu可用性"""
    return {
        "available": HAS_HIKYUU,
        "version": getattr(hku, '__version__', 'unknown') if HAS_HIKYUU else None,
        "initialized": hikyuu_wrapper.is_initialized,
        "ready": hikyuu_wrapper.is_ready()
    }