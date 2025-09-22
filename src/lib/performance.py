"""
Performance Optimization Module
性能优化模块 - 针对Apple Silicon ARM和x86_64平台优化
"""

import platform
import os
import multiprocessing as mp
import psutil
import numpy as np
import pandas as pd
from typing import Dict, Any, Optional, List
import time
import logging
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
from functools import wraps

from lib.environment import env_manager


class PlatformDetector:
    """平台检测器"""

    @staticmethod
    def get_platform_info() -> Dict[str, Any]:
        """获取平台信息"""
        return {
            'system': platform.system(),
            'machine': platform.machine(),
            'processor': platform.processor(),
            'python_version': platform.python_version(),
            'cpu_count': mp.cpu_count(),
            'memory_gb': round(psutil.virtual_memory().total / (1024**3), 2),
            'is_apple_silicon': platform.machine() == 'arm64' and platform.system() == 'Darwin',
            'is_x86_64': platform.machine() in ['x86_64', 'AMD64'],
            'has_avx': 'avx' in os.environ.get('CPU_FLAGS', '').lower(),
            'has_neon': platform.machine() == 'arm64'
        }

    @staticmethod
    def is_apple_silicon() -> bool:
        """检测是否为Apple Silicon"""
        return platform.machine() == 'arm64' and platform.system() == 'Darwin'

    @staticmethod
    def is_x86_64() -> bool:
        """检测是否为x86_64架构"""
        return platform.machine() in ['x86_64', 'AMD64']


class PerformanceOptimizer:
    """性能优化器"""

    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.platform_info = PlatformDetector.get_platform_info()
        self._setup_numpy_optimization()
        self._setup_pandas_optimization()

    def _setup_numpy_optimization(self):
        """设置NumPy优化"""
        try:
            if self.platform_info['is_apple_silicon']:
                # Apple Silicon优化
                os.environ['VECLIB_MAXIMUM_THREADS'] = str(min(8, mp.cpu_count()))
                os.environ['OPENBLAS_NUM_THREADS'] = str(min(8, mp.cpu_count()))
                self.logger.info("已启用Apple Silicon NEON优化")
            elif self.platform_info['is_x86_64']:
                # x86_64优化
                os.environ['MKL_NUM_THREADS'] = str(min(8, mp.cpu_count()))
                os.environ['NUMEXPR_NUM_THREADS'] = str(min(8, mp.cpu_count()))
                self.logger.info("已启用x86_64 AVX优化")
        except Exception as e:
            self.logger.warning(f"NumPy优化设置失败: {e}")

    def _setup_pandas_optimization(self):
        """设置Pandas优化"""
        try:
            # 启用最大性能模式
            pd.set_option('compute.use_bottleneck', True)
            pd.set_option('compute.use_numexpr', True)

            # 内存优化
            pd.set_option('mode.copy_on_write', True)

            self.logger.info("已启用Pandas性能优化")
        except Exception as e:
            self.logger.warning(f"Pandas优化设置失败: {e}")

    def optimize_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """优化DataFrame内存使用"""
        if df.empty:
            return df

        original_memory = df.memory_usage(deep=True).sum()

        # 优化数值类型
        for col in df.select_dtypes(include=['int64']).columns:
            col_min = df[col].min()
            col_max = df[col].max()

            if col_min >= 0:
                if col_max < np.iinfo(np.uint8).max:
                    df[col] = df[col].astype(np.uint8)
                elif col_max < np.iinfo(np.uint16).max:
                    df[col] = df[col].astype(np.uint16)
                elif col_max < np.iinfo(np.uint32).max:
                    df[col] = df[col].astype(np.uint32)
            else:
                if col_min > np.iinfo(np.int8).min and col_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif col_min > np.iinfo(np.int16).min and col_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif col_min > np.iinfo(np.int32).min and col_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)

        # 优化浮点类型
        for col in df.select_dtypes(include=['float64']).columns:
            df[col] = pd.to_numeric(df[col], downcast='float')

        # 优化字符串类型
        for col in df.select_dtypes(include=['object']).columns:
            if df[col].dtype == 'object':
                try:
                    df[col] = df[col].astype('category')
                except:
                    pass

        optimized_memory = df.memory_usage(deep=True).sum()
        reduction = (original_memory - optimized_memory) / original_memory * 100

        self.logger.debug(f"DataFrame内存优化: 减少{reduction:.1f}%")

        return df

    def get_optimal_worker_count(self, data_size: int = 1000) -> int:
        """获取最优工作进程数"""
        cpu_count = mp.cpu_count()
        memory_gb = self.platform_info['memory_gb']

        if self.platform_info['is_apple_silicon']:
            # Apple Silicon M系列优化
            if cpu_count >= 10:  # M1 Pro/Max/Ultra
                base_workers = min(cpu_count - 2, 12)
            else:  # M1/M2 Base
                base_workers = min(cpu_count - 1, 6)
        else:
            # x86_64优化
            base_workers = min(cpu_count - 1, 8)

        # 根据内存调整
        memory_factor = min(memory_gb / 8, 2.0)  # 8GB为基准
        optimal_workers = int(base_workers * memory_factor)

        # 根据数据大小调整
        if data_size > 10000:
            optimal_workers = min(optimal_workers, cpu_count)
        elif data_size < 1000:
            optimal_workers = max(1, optimal_workers // 2)

        return max(1, min(optimal_workers, cpu_count))

    def get_optimal_chunk_size(self, total_size: int, worker_count: int) -> int:
        """获取最优批次大小"""
        if self.platform_info['is_apple_silicon']:
            # Apple Silicon倾向于较大的批次以利用NEON
            base_chunk = max(50, total_size // (worker_count * 2))
        else:
            # x86_64倾向于较小的批次以保持负载均衡
            base_chunk = max(25, total_size // (worker_count * 4))

        # 根据内存限制
        memory_limit_chunk = int(self.platform_info['memory_gb'] * 100)  # 每GB内存处理100只股票

        return min(base_chunk, memory_limit_chunk, 200)  # 最大不超过200


class PerformanceMonitor:
    """性能监控器"""

    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.metrics = {}

    def start_timing(self, operation: str):
        """开始计时"""
        self.metrics[operation] = {
            'start_time': time.perf_counter(),
            'start_memory': psutil.Process().memory_info().rss / 1024 / 1024,  # MB
            'start_cpu': psutil.cpu_percent()
        }

    def end_timing(self, operation: str) -> Dict[str, Any]:
        """结束计时并返回指标"""
        if operation not in self.metrics:
            return {}

        end_time = time.perf_counter()
        end_memory = psutil.Process().memory_info().rss / 1024 / 1024
        end_cpu = psutil.cpu_percent()

        start_metrics = self.metrics[operation]

        result = {
            'duration_seconds': end_time - start_metrics['start_time'],
            'memory_peak_mb': end_memory,
            'memory_delta_mb': end_memory - start_metrics['start_memory'],
            'cpu_usage_percent': end_cpu,
            'operation': operation
        }

        self.logger.info(
            f"性能指标 {operation}: "
            f"耗时={result['duration_seconds']:.2f}s, "
            f"内存峰值={result['memory_peak_mb']:.1f}MB, "
            f"CPU使用率={result['cpu_usage_percent']:.1f}%"
        )

        return result


def performance_optimized(func):
    """性能优化装饰器"""
    @wraps(func)
    async def wrapper(*args, **kwargs):
        monitor = PerformanceMonitor()
        operation_name = f"{func.__module__}.{func.__name__}"

        monitor.start_timing(operation_name)
        try:
            result = await func(*args, **kwargs)
            return result
        finally:
            monitor.end_timing(operation_name)

    return wrapper


class FactorCalculationOptimizer:
    """因子计算性能优化器"""

    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.platform_optimizer = PerformanceOptimizer()
        self.monitor = PerformanceMonitor()

        # 性能目标：30分钟全市场计算
        self.performance_targets = {
            'full_market_calculation_minutes': 30,
            'stocks_per_minute': 200,  # 6000股票/30分钟
            'factors_per_stock': 20,
            'max_memory_usage_gb': 8
        }

    def estimate_calculation_time(self, stock_count: int, factor_count: int) -> Dict[str, Any]:
        """估算计算时间"""
        platform_info = self.platform_optimizer.platform_info

        # 基准性能（每分钟处理股票数）
        if platform_info['is_apple_silicon']:
            if platform_info['cpu_count'] >= 10:  # M1 Pro/Max/Ultra
                base_rate = 300
            else:  # M1/M2 Base
                base_rate = 200
        else:  # x86_64
            base_rate = 150

        # 根据因子数量调整
        factor_adjustment = max(0.8, 1.0 - (factor_count - 10) * 0.02)
        adjusted_rate = base_rate * factor_adjustment

        estimated_minutes = (stock_count * factor_count) / adjusted_rate

        return {
            'estimated_minutes': estimated_minutes,
            'estimated_hours': estimated_minutes / 60,
            'target_minutes': self.performance_targets['full_market_calculation_minutes'],
            'meets_target': estimated_minutes <= self.performance_targets['full_market_calculation_minutes'],
            'performance_ratio': self.performance_targets['full_market_calculation_minutes'] / estimated_minutes,
            'recommended_worker_count': self.platform_optimizer.get_optimal_worker_count(stock_count),
            'recommended_chunk_size': self.platform_optimizer.get_optimal_chunk_size(
                stock_count,
                self.platform_optimizer.get_optimal_worker_count(stock_count)
            )
        }

    def optimize_calculation_strategy(self, stock_count: int, factor_count: int) -> Dict[str, Any]:
        """优化计算策略"""
        estimate = self.estimate_calculation_time(stock_count, factor_count)

        strategy = {
            'use_parallel': stock_count > 100,
            'worker_count': estimate['recommended_worker_count'],
            'chunk_size': estimate['recommended_chunk_size'],
            'memory_optimization': True,
            'cache_intermediate_results': stock_count > 1000,
            'use_vectorized_operations': True
        }

        # 如果不满足性能目标，调整策略
        if not estimate['meets_target']:
            strategy['worker_count'] = min(
                mp.cpu_count(),
                int(strategy['worker_count'] * 1.5)
            )
            strategy['chunk_size'] = min(
                strategy['chunk_size'],
                max(10, stock_count // (strategy['worker_count'] * 3))
            )
            strategy['aggressive_optimization'] = True

            self.logger.warning(
                f"性能目标不满足，调整策略: worker_count={strategy['worker_count']}, "
                f"chunk_size={strategy['chunk_size']}"
            )

        return strategy


# 全局优化器实例
platform_optimizer = PerformanceOptimizer()
calculation_optimizer = FactorCalculationOptimizer()
performance_monitor = PerformanceMonitor()