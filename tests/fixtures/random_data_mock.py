"""
随机数据Mock - Random Data Mock

生成测试场景和边界条件数据，模拟各种异常情况
"""
import random
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Union, Callable
import string
import json


class MockRandomGenerator:
    """
    随机数据生成器

    提供可重现的随机数据生成，支持各种测试场景
    """

    def __init__(self, seed: Optional[int] = None):
        """
        初始化随机数据生成器

        Args:
            seed: 随机种子，确保可重现性
        """
        self.seed = seed
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)

    def random_string(self,
                     length: int = 10,
                     charset: str = None) -> str:
        """
        生成随机字符串

        Args:
            length: 字符串长度
            charset: 字符集，默认为字母和数字

        Returns:
            随机字符串
        """
        if charset is None:
            charset = string.ascii_letters + string.digits

        return ''.join(random.choices(charset, k=length))

    def random_stock_code(self, market: str = "random") -> str:
        """
        生成随机股票代码

        Args:
            market: 市场类型 ("sh", "sz", "random")

        Returns:
            股票代码
        """
        if market == "random":
            market = random.choice(["sh", "sz"])

        if market == "sh":
            # 上证股票代码
            code_num = random.randint(600000, 605000)
            return f"sh{code_num:06d}"
        elif market == "sz":
            # 深证股票代码
            code_num = random.randint(1, 999999)
            return f"sz{code_num:06d}"
        else:
            raise ValueError(f"Unknown market: {market}")

    def random_price_series(self,
                           length: int = 100,
                           initial_price: float = 10.0,
                           volatility: float = 0.02,
                           trend: float = 0.0) -> List[float]:
        """
        生成随机价格序列

        Args:
            length: 序列长度
            initial_price: 初始价格
            volatility: 波动率
            trend: 趋势

        Returns:
            价格序列
        """
        prices = [initial_price]
        current_price = initial_price

        for _ in range(length - 1):
            daily_return = np.random.normal(trend / 252, volatility)
            current_price *= (1 + daily_return)
            prices.append(max(0.01, current_price))  # 避免负价格

        return prices

    def random_factor_values(self,
                           length: int = 100,
                           distribution: str = "normal") -> List[float]:
        """
        生成随机因子值

        Args:
            length: 序列长度
            distribution: 分布类型 ("normal", "uniform", "exponential")

        Returns:
            因子值序列
        """
        if distribution == "normal":
            return np.random.normal(0, 1, length).tolist()
        elif distribution == "uniform":
            return np.random.uniform(-2, 2, length).tolist()
        elif distribution == "exponential":
            return np.random.exponential(1, length).tolist()
        else:
            raise ValueError(f"Unknown distribution: {distribution}")

    def random_signals(self,
                      length: int = 100,
                      signal_probability: float = 0.3) -> List[int]:
        """
        生成随机交易信号

        Args:
            length: 序列长度
            signal_probability: 产生信号的概率

        Returns:
            信号序列 (1: 买入, 0: 持有, -1: 卖出)
        """
        signals = []

        for _ in range(length):
            rand_val = random.random()

            if rand_val < signal_probability / 2:
                signals.append(1)  # 买入
            elif rand_val < signal_probability:
                signals.append(-1)  # 卖出
            else:
                signals.append(0)  # 持有

        return signals

    def random_market_data(self, symbol: str) -> Dict[str, Any]:
        """
        生成随机市场数据

        Args:
            symbol: 股票代码

        Returns:
            市场数据字典
        """
        base_price = random.uniform(5, 100)
        change_pct = random.uniform(-0.1, 0.1)

        return {
            "symbol": symbol,
            "price": round(base_price, 2),
            "change": round(base_price * change_pct, 2),
            "change_percent": round(change_pct * 100, 2),
            "volume": random.randint(100000, 50000000),
            "turnover": random.uniform(1000000, 1000000000),
            "high": round(base_price * random.uniform(1.0, 1.05), 2),
            "low": round(base_price * random.uniform(0.95, 1.0), 2),
            "open": round(base_price * random.uniform(0.98, 1.02), 2),
            "timestamp": datetime.now().isoformat()
        }


def generate_test_scenarios(scenario_types: List[str] = None) -> Dict[str, Dict[str, Any]]:
    """
    生成测试场景数据

    Args:
        scenario_types: 场景类型列表

    Returns:
        测试场景字典
    """
    if scenario_types is None:
        scenario_types = [
            "normal_market",
            "bull_market",
            "bear_market",
            "high_volatility",
            "low_liquidity",
            "market_crash",
            "flash_crash",
            "circuit_breaker"
        ]

    scenarios = {}

    for scenario_type in scenario_types:
        if scenario_type == "normal_market":
            scenarios[scenario_type] = {
                "volatility": 0.015,
                "trend": 0.08,
                "volume_multiplier": 1.0,
                "correlation": 0.3,
                "description": "正常市场条件"
            }

        elif scenario_type == "bull_market":
            scenarios[scenario_type] = {
                "volatility": 0.012,
                "trend": 0.25,
                "volume_multiplier": 1.2,
                "correlation": 0.7,
                "description": "牛市行情"
            }

        elif scenario_type == "bear_market":
            scenarios[scenario_type] = {
                "volatility": 0.025,
                "trend": -0.15,
                "volume_multiplier": 0.8,
                "correlation": 0.8,
                "description": "熊市行情"
            }

        elif scenario_type == "high_volatility":
            scenarios[scenario_type] = {
                "volatility": 0.045,
                "trend": 0.0,
                "volume_multiplier": 1.8,
                "correlation": 0.2,
                "description": "高波动率市场"
            }

        elif scenario_type == "low_liquidity":
            scenarios[scenario_type] = {
                "volatility": 0.030,
                "trend": 0.02,
                "volume_multiplier": 0.3,
                "correlation": 0.1,
                "description": "低流动性市场"
            }

        elif scenario_type == "market_crash":
            scenarios[scenario_type] = {
                "volatility": 0.080,
                "trend": -0.50,
                "volume_multiplier": 3.0,
                "correlation": 0.95,
                "description": "市场崩盘"
            }

        elif scenario_type == "flash_crash":
            scenarios[scenario_type] = {
                "volatility": 0.150,
                "trend": -0.20,
                "volume_multiplier": 5.0,
                "correlation": 0.99,
                "description": "闪电崩盘",
                "duration_minutes": 30
            }

        elif scenario_type == "circuit_breaker":
            scenarios[scenario_type] = {
                "volatility": 0.100,
                "trend": -0.10,
                "volume_multiplier": 10.0,
                "correlation": 0.90,
                "description": "熔断机制触发",
                "halt_probability": 0.5
            }

    return scenarios


def create_edge_case_data(data_type: str = "price") -> Dict[str, Any]:
    """
    创建边界条件数据

    Args:
        data_type: 数据类型 ("price", "volume", "factor", "signal")

    Returns:
        边界条件数据字典
    """
    edge_cases = {}

    if data_type == "price":
        edge_cases = {
            "zero_price": 0.0,
            "negative_price": -1.0,
            "very_small_price": 0.001,
            "very_large_price": 999999.99,
            "nan_price": float('nan'),
            "inf_price": float('inf'),
            "price_with_high_precision": 123.456789123456
        }

    elif data_type == "volume":
        edge_cases = {
            "zero_volume": 0,
            "negative_volume": -1000,
            "very_large_volume": 999999999999,
            "nan_volume": float('nan'),
            "inf_volume": float('inf')
        }

    elif data_type == "factor":
        edge_cases = {
            "zero_factor": 0.0,
            "very_small_factor": 1e-10,
            "very_large_factor": 1e10,
            "negative_factor": -999.99,
            "nan_factor": float('nan'),
            "inf_factor": float('inf'),
            "negative_inf_factor": float('-inf')
        }

    elif data_type == "signal":
        edge_cases = {
            "invalid_signal_value": 999,
            "float_signal": 1.5,
            "negative_signal": -999,
            "string_signal": "invalid",
            "none_signal": None,
            "nan_signal": float('nan')
        }

    return edge_cases


def simulate_error_conditions(error_types: List[str] = None) -> Dict[str, Callable]:
    """
    模拟错误条件

    Args:
        error_types: 错误类型列表

    Returns:
        错误模拟函数字典
    """
    if error_types is None:
        error_types = [
            "connection_error",
            "timeout_error",
            "data_corruption",
            "memory_error",
            "disk_full",
            "permission_denied",
            "rate_limit_exceeded"
        ]

    error_simulators = {}

    for error_type in error_types:
        if error_type == "connection_error":
            def connection_error():
                raise ConnectionError("模拟连接错误")
            error_simulators[error_type] = connection_error

        elif error_type == "timeout_error":
            def timeout_error():
                raise TimeoutError("模拟超时错误")
            error_simulators[error_type] = timeout_error

        elif error_type == "data_corruption":
            def data_corruption(data):
                # 随机破坏数据
                if isinstance(data, str):
                    # 随机替换字符
                    data_list = list(data)
                    if data_list:
                        idx = random.randint(0, len(data_list) - 1)
                        data_list[idx] = random.choice(string.printable)
                    return ''.join(data_list)
                elif isinstance(data, (list, tuple)):
                    # 随机替换元素
                    if data:
                        data_list = list(data)
                        idx = random.randint(0, len(data_list) - 1)
                        data_list[idx] = None
                    return data_list
                return data
            error_simulators[error_type] = data_corruption

        elif error_type == "memory_error":
            def memory_error():
                raise MemoryError("模拟内存不足错误")
            error_simulators[error_type] = memory_error

        elif error_type == "disk_full":
            def disk_full():
                raise OSError("模拟磁盘空间不足")
            error_simulators[error_type] = disk_full

        elif error_type == "permission_denied":
            def permission_denied():
                raise PermissionError("模拟权限拒绝错误")
            error_simulators[error_type] = permission_denied

        elif error_type == "rate_limit_exceeded":
            def rate_limit_exceeded():
                return {
                    "status": "error",
                    "error_code": "RATE_LIMIT_EXCEEDED",
                    "message": "API调用频率超限",
                    "retry_after": 60
                }
            error_simulators[error_type] = rate_limit_exceeded

    return error_simulators


class DataCorruptor:
    """
    数据损坏模拟器

    模拟各种数据损坏情况
    """

    @staticmethod
    def corrupt_json(json_str: str, corruption_rate: float = 0.1) -> str:
        """
        损坏JSON数据

        Args:
            json_str: JSON字符串
            corruption_rate: 损坏率

        Returns:
            损坏的JSON字符串
        """
        if random.random() > corruption_rate:
            return json_str

        corruption_types = [
            "missing_bracket",
            "missing_quote",
            "invalid_character",
            "truncated"
        ]

        corruption_type = random.choice(corruption_types)

        if corruption_type == "missing_bracket":
            # 移除随机括号
            brackets = ['{', '}', '[', ']']
            for bracket in brackets:
                if bracket in json_str:
                    json_str = json_str.replace(bracket, '', 1)
                    break

        elif corruption_type == "missing_quote":
            # 移除随机引号
            if '"' in json_str:
                idx = json_str.find('"')
                json_str = json_str[:idx] + json_str[idx+1:]

        elif corruption_type == "invalid_character":
            # 插入无效字符
            if json_str:
                idx = random.randint(0, len(json_str) - 1)
                json_str = json_str[:idx] + '\x00' + json_str[idx:]

        elif corruption_type == "truncated":
            # 截断数据
            if len(json_str) > 10:
                cutoff = random.randint(10, len(json_str) - 1)
                json_str = json_str[:cutoff]

        return json_str

    @staticmethod
    def corrupt_numerical_data(data: List[float],
                             corruption_rate: float = 0.05) -> List[float]:
        """
        损坏数值数据

        Args:
            data: 数值数据列表
            corruption_rate: 损坏率

        Returns:
            损坏的数值数据
        """
        corrupted_data = data.copy()

        for i in range(len(corrupted_data)):
            if random.random() < corruption_rate:
                corruption_type = random.choice([
                    "nan", "inf", "negative_inf", "zero", "outlier"
                ])

                if corruption_type == "nan":
                    corrupted_data[i] = float('nan')
                elif corruption_type == "inf":
                    corrupted_data[i] = float('inf')
                elif corruption_type == "negative_inf":
                    corrupted_data[i] = float('-inf')
                elif corruption_type == "zero":
                    corrupted_data[i] = 0.0
                elif corruption_type == "outlier":
                    # 生成极端异常值
                    corrupted_data[i] *= random.uniform(100, 1000)

        return corrupted_data


def create_stress_test_data(data_size: str = "medium") -> Dict[str, Any]:
    """
    创建压力测试数据

    Args:
        data_size: 数据规模 ("small", "medium", "large", "xlarge")

    Returns:
        压力测试数据
    """
    size_mapping = {
        "small": 1000,
        "medium": 10000,
        "large": 100000,
        "xlarge": 1000000
    }

    if data_size not in size_mapping:
        raise ValueError(f"Unknown data size: {data_size}")

    data_count = size_mapping[data_size]

    generator = MockRandomGenerator(seed=42)

    return {
        "size": data_size,
        "record_count": data_count,
        "stock_codes": [generator.random_stock_code() for _ in range(min(data_count // 100, 5000))],
        "price_data": generator.random_price_series(data_count),
        "factor_data": generator.random_factor_values(data_count),
        "signals": generator.random_signals(data_count),
        "memory_estimate_mb": data_count * 0.001,  # 粗略估计
        "description": f"压力测试数据集 - {data_size} 规模"
    }