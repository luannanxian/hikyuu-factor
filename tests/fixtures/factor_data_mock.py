"""
因子数据Mock生成器 - Factor Data Mock Generators

专为A股量化因子系统提供因子数据Mock生成
"""
import random
from datetime import datetime, timedelta, date
from typing import Dict, List, Optional, Any, Union, Tuple
import pandas as pd
import numpy as np


class MockFactorDataGenerator:
    """
    因子数据Mock生成器

    生成各类量化因子数据，用于测试
    """

    def __init__(self, seed: Optional[int] = None):
        """
        初始化因子数据生成器

        Args:
            seed: 随机种子，确保可重现性
        """
        self.seed = seed
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)

        # 因子分类定义
        self.factor_categories = {
            "momentum": ["momentum_5d", "momentum_10d", "momentum_20d", "momentum_60d", "rsi", "macd"],
            "value": ["pe_ratio", "pb_ratio", "ps_ratio", "pcf_ratio", "ev_ebitda"],
            "quality": ["roe", "roa", "roic", "debt_ratio", "profit_margin", "asset_turnover"],
            "growth": ["revenue_growth", "profit_growth", "eps_growth", "book_value_growth"],
            "technical": ["bollinger_upper", "bollinger_lower", "sma_20", "ema_20", "volume_ratio"],
            "volatility": ["std_20d", "beta", "downside_deviation", "max_drawdown"],
            "liquidity": ["turnover_rate", "avg_volume", "bid_ask_spread", "market_impact"]
        }

    def generate_factor_definition(self, factor_name: str = None, category: str = None) -> Dict[str, Any]:
        """
        生成因子定义

        Args:
            factor_name: 因子名称
            category: 因子类别

        Returns:
            因子定义字典
        """
        if category is None:
            category = random.choice(list(self.factor_categories.keys()))

        if factor_name is None:
            factor_name = random.choice(self.factor_categories[category])

        # 生成因子ID
        factor_id = f"{category}_{factor_name}_{random.randint(1000, 9999)}"

        return {
            "factor_id": factor_id,
            "factor_name": factor_name,
            "category": category,
            "description": f"{category} 类型的 {factor_name} 因子",
            "data_type": "numeric",
            "frequency": random.choice(["daily", "weekly", "monthly"]),
            "lookback_period": random.randint(5, 60),
            "update_time": datetime.now(),
            "is_active": True,
            "parameters": self._generate_factor_parameters(factor_name),
            "calculation_method": f"calculate_{factor_name}",
            "data_source": random.choice(["hikyuu", "wind", "tushare", "manual"])
        }

    def _generate_factor_parameters(self, factor_name: str) -> Dict[str, Any]:
        """
        生成因子计算参数

        Args:
            factor_name: 因子名称

        Returns:
            参数字典
        """
        parameters = {}

        if "momentum" in factor_name:
            parameters = {
                "period": int(factor_name.split("_")[-1].replace("d", "")) if "_" in factor_name else 20,
                "method": random.choice(["simple", "log", "pct"])
            }
        elif factor_name == "rsi":
            parameters = {
                "period": 14,
                "overbought": 70,
                "oversold": 30
            }
        elif factor_name == "macd":
            parameters = {
                "fast_period": 12,
                "slow_period": 26,
                "signal_period": 9
            }
        elif "ratio" in factor_name:
            parameters = {
                "ttm": True,
                "method": "latest"
            }
        elif "growth" in factor_name:
            parameters = {
                "period": "annual",
                "method": "yoy"
            }
        elif "bollinger" in factor_name:
            parameters = {
                "period": 20,
                "std_multiplier": 2.0
            }
        elif "sma" in factor_name or "ema" in factor_name:
            parameters = {
                "period": int(factor_name.split("_")[-1]) if "_" in factor_name else 20
            }

        return parameters

    def generate_factor_values(self,
                             factor_name: str,
                             stock_codes: List[str],
                             start_date: Union[str, datetime] = None,
                             end_date: Union[str, datetime] = None,
                             days: int = 100) -> pd.DataFrame:
        """
        生成因子值数据

        Args:
            factor_name: 因子名称
            stock_codes: 股票代码列表
            start_date: 开始日期
            end_date: 结束日期
            days: 天数

        Returns:
            因子值DataFrame
        """
        # 处理日期
        if start_date is None and end_date is None:
            end_date = datetime.now().date()
            start_date = end_date - timedelta(days=days)
        elif isinstance(start_date, str):
            start_date = datetime.strptime(start_date, "%Y-%m-%d").date()
        elif isinstance(end_date, str):
            end_date = datetime.strptime(end_date, "%Y-%m-%d").date()

        # 生成交易日期
        dates = []
        current_date = start_date
        while current_date <= end_date:
            if current_date.weekday() < 5:  # 交易日
                dates.append(current_date)
            current_date += timedelta(days=1)

        # 生成因子值
        factor_data = []

        for stock_code in stock_codes:
            # 为每个股票生成因子时间序列
            factor_values = self._generate_factor_time_series(factor_name, len(dates))

            for date, value in zip(dates, factor_values):
                factor_data.append({
                    "datetime": date,
                    "stock_code": stock_code,
                    "factor_name": factor_name,
                    "factor_value": value,
                    "is_valid": not (np.isnan(value) or np.isinf(value)),
                    "update_time": datetime.now()
                })

        return pd.DataFrame(factor_data)

    def _generate_factor_time_series(self, factor_name: str, length: int) -> List[float]:
        """
        生成因子时间序列

        Args:
            factor_name: 因子名称
            length: 序列长度

        Returns:
            因子值序列
        """
        # 根据因子类型生成不同特征的时间序列
        if "momentum" in factor_name:
            # 动量因子：具有趋势性和均值回复
            values = np.random.normal(0, 0.1, length)
            # 添加趋势成分
            trend = np.linspace(-0.1, 0.1, length)
            values += trend
            # 添加均值回复
            for i in range(1, length):
                values[i] += -0.1 * values[i-1]

        elif factor_name in ["pe_ratio", "pb_ratio", "ps_ratio"]:
            # 估值因子：正值，对数正态分布
            values = np.random.lognormal(2.0, 0.5, length)

        elif factor_name in ["roe", "roa", "roic"]:
            # 盈利能力因子：0-1之间
            values = np.random.beta(2, 5, length)

        elif factor_name == "debt_ratio":
            # 负债率：0-1之间，偏向较低值
            values = np.random.beta(2, 3, length)

        elif "growth" in factor_name:
            # 成长因子：可正可负，均值略为正
            values = np.random.normal(0.1, 0.3, length)

        elif factor_name == "rsi":
            # RSI：0-100之间
            values = np.random.beta(2, 2, length) * 100

        elif "std" in factor_name or "volatility" in factor_name:
            # 波动率因子：正值
            values = np.random.gamma(2, 0.01, length)

        elif "volume" in factor_name or "turnover" in factor_name:
            # 成交量相关因子：正值，右偏分布
            values = np.random.lognormal(0, 1, length)

        else:
            # 默认：标准正态分布
            values = np.random.normal(0, 1, length)

        # 随机引入缺失值
        missing_prob = 0.05
        for i in range(length):
            if random.random() < missing_prob:
                values[i] = np.nan

        return values.tolist()

    def generate_factor_exposure_matrix(self,
                                      stock_codes: List[str],
                                      factor_names: List[str],
                                      date: Union[str, datetime] = None) -> pd.DataFrame:
        """
        生成因子暴露度矩阵

        Args:
            stock_codes: 股票代码列表
            factor_names: 因子名称列表
            date: 日期

        Returns:
            因子暴露度矩阵DataFrame
        """
        if date is None:
            date = datetime.now().date()
        elif isinstance(date, str):
            date = datetime.strptime(date, "%Y-%m-%d").date()

        exposure_data = []

        for stock_code in stock_codes:
            row_data = {"datetime": date, "stock_code": stock_code}

            for factor_name in factor_names:
                # 生成单个因子暴露度
                exposure = self._generate_single_factor_exposure(factor_name)
                row_data[factor_name] = exposure

            exposure_data.append(row_data)

        return pd.DataFrame(exposure_data)

    def _generate_single_factor_exposure(self, factor_name: str) -> float:
        """
        生成单个因子暴露度

        Args:
            factor_name: 因子名称

        Returns:
            因子暴露度
        """
        # 标准化的因子暴露度（均值为0，标准差为1）
        return random.normalvariate(0, 1)

    def generate_factor_return_series(self,
                                    factor_names: List[str],
                                    start_date: Union[str, datetime] = None,
                                    end_date: Union[str, datetime] = None,
                                    days: int = 252) -> pd.DataFrame:
        """
        生成因子收益率序列

        Args:
            factor_names: 因子名称列表
            start_date: 开始日期
            end_date: 结束日期
            days: 天数

        Returns:
            因子收益率DataFrame
        """
        # 处理日期
        if start_date is None and end_date is None:
            end_date = datetime.now().date()
            start_date = end_date - timedelta(days=days)
        elif isinstance(start_date, str):
            start_date = datetime.strptime(start_date, "%Y-%m-%d").date()
        elif isinstance(end_date, str):
            end_date = datetime.strptime(end_date, "%Y-%m-%d").date()

        # 生成交易日期
        dates = []
        current_date = start_date
        while current_date <= end_date:
            if current_date.weekday() < 5:
                dates.append(current_date)
            current_date += timedelta(days=1)

        factor_returns = []

        for date in dates:
            row_data = {"datetime": date}

            for factor_name in factor_names:
                # 生成因子收益率（日频）
                factor_return = np.random.normal(0, 0.02)  # 2%年化波动率
                row_data[factor_name] = factor_return

            factor_returns.append(row_data)

        return pd.DataFrame(factor_returns)

    def generate_factor_correlation_matrix(self, factor_names: List[str]) -> pd.DataFrame:
        """
        生成因子相关性矩阵

        Args:
            factor_names: 因子名称列表

        Returns:
            相关性矩阵DataFrame
        """
        n_factors = len(factor_names)

        # 生成随机相关性矩阵
        # 首先生成随机矩阵
        random_matrix = np.random.randn(n_factors, n_factors)

        # 使Gram矩阵正定
        correlation_matrix = np.dot(random_matrix, random_matrix.T)

        # 标准化为相关性矩阵
        d = np.sqrt(np.diag(correlation_matrix))
        correlation_matrix = correlation_matrix / np.outer(d, d)

        # 确保对角线为1
        np.fill_diagonal(correlation_matrix, 1.0)

        # 转换为DataFrame
        df = pd.DataFrame(correlation_matrix, index=factor_names, columns=factor_names)

        return df


def create_factor_library(categories: List[str] = None) -> List[Dict[str, Any]]:
    """
    创建因子库

    Args:
        categories: 因子类别列表

    Returns:
        因子定义列表
    """
    generator = MockFactorDataGenerator(seed=42)

    if categories is None:
        categories = list(generator.factor_categories.keys())

    factor_library = []

    for category in categories:
        for factor_name in generator.factor_categories[category]:
            factor_def = generator.generate_factor_definition(factor_name, category)
            factor_library.append(factor_def)

    return factor_library


def create_test_factor_dataset(stock_codes: List[str],
                             factor_names: List[str],
                             start_date: str = "2024-01-01",
                             end_date: str = "2024-12-31") -> Dict[str, pd.DataFrame]:
    """
    创建测试用因子数据集

    Args:
        stock_codes: 股票代码列表
        factor_names: 因子名称列表
        start_date: 开始日期
        end_date: 结束日期

    Returns:
        因子数据字典
    """
    generator = MockFactorDataGenerator(seed=42)
    dataset = {}

    for factor_name in factor_names:
        dataset[factor_name] = generator.generate_factor_values(
            factor_name=factor_name,
            stock_codes=stock_codes,
            start_date=start_date,
            end_date=end_date
        )

    return dataset


def create_factor_performance_data(factor_name: str, periods: int = 252) -> Dict[str, Any]:
    """
    创建因子表现数据

    Args:
        factor_name: 因子名称
        periods: 周期数

    Returns:
        因子表现数据
    """
    generator = MockFactorDataGenerator(seed=42)

    # 生成因子收益率序列
    returns = np.random.normal(0.001, 0.02, periods)  # 日收益率

    # 计算累计收益率
    cumulative_returns = np.cumprod(1 + returns) - 1

    # 计算统计指标
    annual_return = np.mean(returns) * 252
    volatility = np.std(returns) * np.sqrt(252)
    sharpe_ratio = annual_return / volatility if volatility > 0 else 0
    max_drawdown = np.min(cumulative_returns - np.maximum.accumulate(cumulative_returns))

    # 计算IC统计
    ic_values = np.random.normal(0.05, 0.1, periods)  # 信息系数
    ic_mean = np.mean(ic_values)
    ic_std = np.std(ic_values)
    ic_ir = ic_mean / ic_std if ic_std > 0 else 0

    return {
        "factor_name": factor_name,
        "performance_period": periods,
        "annual_return": annual_return,
        "volatility": volatility,
        "sharpe_ratio": sharpe_ratio,
        "max_drawdown": max_drawdown,
        "ic_mean": ic_mean,
        "ic_std": ic_std,
        "ic_ir": ic_ir,
        "win_rate": np.sum(returns > 0) / len(returns),
        "returns": returns.tolist(),
        "cumulative_returns": cumulative_returns.tolist(),
        "ic_values": ic_values.tolist()
    }