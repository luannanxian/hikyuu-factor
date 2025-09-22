"""
信号数据Mock生成器 - Signal Data Mock Generators

专为A股量化因子系统提供交易信号数据Mock生成
"""
import random
from datetime import datetime, timedelta, date
from typing import Dict, List, Optional, Any, Union, Tuple
import pandas as pd
import numpy as np
from enum import Enum


class SignalType(Enum):
    """信号类型枚举"""
    BUY = 1
    SELL = -1
    HOLD = 0


class SignalStatus(Enum):
    """信号状态枚举"""
    PENDING = "pending"
    CONFIRMED = "confirmed"
    EXECUTED = "executed"
    CANCELLED = "cancelled"
    EXPIRED = "expired"


class MockSignalDataGenerator:
    """
    交易信号Mock生成器

    生成各类交易信号数据，用于测试
    """

    def __init__(self, seed: Optional[int] = None):
        """
        初始化信号数据生成器

        Args:
            seed: 随机种子，确保可重现性
        """
        self.seed = seed
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)

        # 策略类型定义
        self.strategy_types = [
            "momentum_strategy",
            "mean_reversion_strategy",
            "value_strategy",
            "growth_strategy",
            "multi_factor_strategy",
            "pairs_trading_strategy",
            "statistical_arbitrage",
            "trend_following"
        ]

        # 信号强度等级
        self.signal_strengths = ["weak", "medium", "strong", "very_strong"]

    def generate_signal_definition(self, strategy_name: str = None) -> Dict[str, Any]:
        """
        生成信号定义

        Args:
            strategy_name: 策略名称

        Returns:
            信号定义字典
        """
        if strategy_name is None:
            strategy_name = random.choice(self.strategy_types)

        signal_id = f"{strategy_name}_{random.randint(1000, 9999)}"

        return {
            "signal_id": signal_id,
            "strategy_name": strategy_name,
            "description": f"{strategy_name} 策略生成的交易信号",
            "signal_type": random.choice(["directional", "pairs", "spread", "volatility"]),
            "frequency": random.choice(["intraday", "daily", "weekly"]),
            "holding_period": random.randint(1, 30),  # 持有天数
            "max_positions": random.randint(10, 100),
            "risk_level": random.choice(["low", "medium", "high"]),
            "created_time": datetime.now(),
            "is_active": True,
            "parameters": self._generate_strategy_parameters(strategy_name)
        }

    def _generate_strategy_parameters(self, strategy_name: str) -> Dict[str, Any]:
        """
        生成策略参数

        Args:
            strategy_name: 策略名称

        Returns:
            参数字典
        """
        parameters = {}

        if "momentum" in strategy_name:
            parameters = {
                "lookback_period": random.randint(5, 60),
                "threshold": random.uniform(0.01, 0.05),
                "rebalance_frequency": random.choice(["daily", "weekly"])
            }
        elif "mean_reversion" in strategy_name:
            parameters = {
                "mean_period": random.randint(20, 100),
                "std_threshold": random.uniform(1.5, 3.0),
                "exit_threshold": random.uniform(0.5, 1.0)
            }
        elif "value" in strategy_name:
            parameters = {
                "pe_threshold": random.uniform(10, 20),
                "pb_threshold": random.uniform(0.5, 2.0),
                "rebalance_frequency": "monthly"
            }
        elif "growth" in strategy_name:
            parameters = {
                "growth_threshold": random.uniform(0.15, 0.30),
                "quality_filter": True,
                "min_market_cap": random.randint(1000000000, 10000000000)
            }
        elif "multi_factor" in strategy_name:
            parameters = {
                "factor_weights": {
                    "momentum": random.uniform(0.2, 0.4),
                    "value": random.uniform(0.2, 0.4),
                    "quality": random.uniform(0.2, 0.4)
                },
                "rebalance_frequency": "monthly"
            }
        elif "pairs" in strategy_name:
            parameters = {
                "correlation_threshold": random.uniform(0.7, 0.9),
                "spread_threshold": random.uniform(1.5, 2.5),
                "half_life": random.randint(5, 20)
            }

        return parameters

    def generate_trading_signals(self,
                               strategy_name: str,
                               stock_codes: List[str],
                               start_date: Union[str, datetime] = None,
                               end_date: Union[str, datetime] = None,
                               days: int = 100,
                               signal_frequency: float = 0.1) -> pd.DataFrame:
        """
        生成交易信号数据

        Args:
            strategy_name: 策略名称
            stock_codes: 股票代码列表
            start_date: 开始日期
            end_date: 结束日期
            days: 天数
            signal_frequency: 信号频率

        Returns:
            交易信号DataFrame
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

        signals = []

        for date in dates:
            for stock_code in stock_codes:
                # 根据信号频率决定是否产生信号
                if random.random() < signal_frequency:
                    signal = self._generate_single_signal(
                        strategy_name, stock_code, date
                    )
                    signals.append(signal)

        return pd.DataFrame(signals)

    def _generate_single_signal(self,
                              strategy_name: str,
                              stock_code: str,
                              signal_date: date) -> Dict[str, Any]:
        """
        生成单个交易信号

        Args:
            strategy_name: 策略名称
            stock_code: 股票代码
            signal_date: 信号日期

        Returns:
            信号字典
        """
        # 生成信号类型（买入/卖出/持有）
        signal_type = random.choice([SignalType.BUY.value, SignalType.SELL.value])

        # 生成信号强度
        signal_strength = random.choice(self.signal_strengths)

        # 生成信号置信度
        confidence = random.uniform(0.5, 1.0)

        # 生成目标价格
        current_price = random.uniform(10, 100)
        if signal_type == SignalType.BUY.value:
            target_price = current_price * random.uniform(1.05, 1.20)
            stop_loss = current_price * random.uniform(0.90, 0.95)
        else:
            target_price = current_price * random.uniform(0.80, 0.95)
            stop_loss = current_price * random.uniform(1.05, 1.10)

        # 生成信号ID
        signal_id = f"{strategy_name}_{stock_code}_{signal_date.strftime('%Y%m%d')}_{random.randint(100, 999)}"

        return {
            "signal_id": signal_id,
            "strategy_name": strategy_name,
            "stock_code": stock_code,
            "signal_date": signal_date,
            "signal_type": signal_type,
            "signal_strength": signal_strength,
            "confidence": confidence,
            "current_price": round(current_price, 2),
            "target_price": round(target_price, 2),
            "stop_loss": round(stop_loss, 2),
            "expected_return": round((target_price - current_price) / current_price, 4),
            "max_risk": round((stop_loss - current_price) / current_price, 4),
            "position_size": random.uniform(0.01, 0.05),  # 仓位大小
            "holding_period": random.randint(1, 30),  # 预期持有天数
            "status": SignalStatus.PENDING.value,
            "created_time": datetime.combine(signal_date, datetime.now().time()),
            "factor_scores": self._generate_factor_scores(),
            "risk_metrics": self._generate_risk_metrics()
        }

    def _generate_factor_scores(self) -> Dict[str, float]:
        """
        生成因子评分

        Returns:
            因子评分字典
        """
        return {
            "momentum_score": random.uniform(-1, 1),
            "value_score": random.uniform(-1, 1),
            "quality_score": random.uniform(-1, 1),
            "growth_score": random.uniform(-1, 1),
            "technical_score": random.uniform(-1, 1),
            "composite_score": random.uniform(-1, 1)
        }

    def _generate_risk_metrics(self) -> Dict[str, float]:
        """
        生成风险指标

        Returns:
            风险指标字典
        """
        return {
            "volatility": random.uniform(0.10, 0.50),
            "beta": random.uniform(0.5, 2.0),
            "var_95": random.uniform(0.02, 0.10),
            "max_drawdown": random.uniform(0.05, 0.20),
            "correlation_risk": random.uniform(0.1, 0.8),
            "liquidity_risk": random.uniform(0.1, 0.5)
        }

    def generate_signal_performance(self,
                                  signal_ids: List[str],
                                  evaluation_date: Union[str, datetime] = None) -> pd.DataFrame:
        """
        生成信号表现数据

        Args:
            signal_ids: 信号ID列表
            evaluation_date: 评估日期

        Returns:
            信号表现DataFrame
        """
        if evaluation_date is None:
            evaluation_date = datetime.now().date()
        elif isinstance(evaluation_date, str):
            evaluation_date = datetime.strptime(evaluation_date, "%Y-%m-%d").date()

        performance_data = []

        for signal_id in signal_ids:
            performance = {
                "signal_id": signal_id,
                "evaluation_date": evaluation_date,
                "actual_return": random.uniform(-0.20, 0.30),
                "predicted_return": random.uniform(-0.15, 0.25),
                "hit_rate": random.choice([True, False]),
                "days_held": random.randint(1, 30),
                "max_profit": random.uniform(0, 0.25),
                "max_loss": random.uniform(-0.15, 0),
                "profit_factor": random.uniform(0.5, 3.0),
                "sharpe_ratio": random.uniform(-1.0, 2.0),
                "execution_slippage": random.uniform(0, 0.005),
                "transaction_cost": random.uniform(0.001, 0.003)
            }
            performance_data.append(performance)

        return pd.DataFrame(performance_data)

    def generate_portfolio_signals(self,
                                 strategy_name: str,
                                 target_stocks: int = 50,
                                 rebalance_date: Union[str, datetime] = None) -> pd.DataFrame:
        """
        生成组合信号

        Args:
            strategy_name: 策略名称
            target_stocks: 目标股票数量
            rebalance_date: 调仓日期

        Returns:
            组合信号DataFrame
        """
        if rebalance_date is None:
            rebalance_date = datetime.now().date()
        elif isinstance(rebalance_date, str):
            rebalance_date = datetime.strptime(rebalance_date, "%Y-%m-%d").date()

        # 生成股票池
        stock_codes = [f"sh{600000 + i:06d}" for i in range(0, 100)]
        stock_codes.extend([f"sz{i:06d}" for i in range(1, 101)])

        # 随机选择股票
        selected_stocks = random.sample(stock_codes, target_stocks)

        portfolio_signals = []

        total_weight = 0
        for i, stock_code in enumerate(selected_stocks):
            # 生成权重（确保总和为1）
            if i == len(selected_stocks) - 1:
                weight = 1.0 - total_weight
            else:
                weight = random.uniform(0.01, 0.05)
                total_weight += weight

            signal = {
                "strategy_name": strategy_name,
                "rebalance_date": rebalance_date,
                "stock_code": stock_code,
                "signal_type": SignalType.BUY.value,  # 组合信号主要是买入
                "target_weight": round(weight, 4),
                "current_weight": random.uniform(0, 0.03),  # 当前权重
                "weight_change": round(weight - random.uniform(0, 0.03), 4),
                "expected_return": random.uniform(0.05, 0.20),
                "expected_risk": random.uniform(0.10, 0.30),
                "factor_exposure": self._generate_factor_scores(),
                "rank": i + 1,
                "sector": random.choice(["金融", "科技", "消费", "医药", "工业", "能源"]),
                "market_cap": random.uniform(1e9, 1e11)
            }
            portfolio_signals.append(signal)

        return pd.DataFrame(portfolio_signals)

    def generate_risk_control_signals(self,
                                    portfolio_signals: pd.DataFrame,
                                    risk_budget: float = 0.15) -> List[Dict[str, Any]]:
        """
        生成风险控制信号

        Args:
            portfolio_signals: 组合信号
            risk_budget: 风险预算

        Returns:
            风险控制信号列表
        """
        risk_signals = []

        # 检查组合整体风险
        total_risk = np.sqrt(np.sum(portfolio_signals["expected_risk"]**2 * portfolio_signals["target_weight"]**2))

        if total_risk > risk_budget:
            risk_signals.append({
                "signal_type": "risk_alert",
                "alert_level": "high",
                "message": f"组合风险 {total_risk:.3f} 超过预算 {risk_budget:.3f}",
                "recommendation": "减少高风险股票权重",
                "timestamp": datetime.now()
            })

        # 检查单个股票权重
        max_weight = portfolio_signals["target_weight"].max()
        if max_weight > 0.10:  # 单股票权重超过10%
            risk_signals.append({
                "signal_type": "concentration_alert",
                "alert_level": "medium",
                "message": f"单股票权重 {max_weight:.3f} 过于集中",
                "recommendation": "分散投资",
                "timestamp": datetime.now()
            })

        # 检查行业集中度
        sector_weights = portfolio_signals.groupby("sector")["target_weight"].sum()
        max_sector_weight = sector_weights.max()
        if max_sector_weight > 0.30:  # 单行业权重超过30%
            risk_signals.append({
                "signal_type": "sector_concentration_alert",
                "alert_level": "medium",
                "message": f"行业集中度 {max_sector_weight:.3f} 过高",
                "recommendation": "增加行业分散度",
                "timestamp": datetime.now()
            })

        return risk_signals


def create_sample_signals(strategy_names: List[str],
                        stock_codes: List[str],
                        days: int = 30) -> Dict[str, pd.DataFrame]:
    """
    创建样本信号数据

    Args:
        strategy_names: 策略名称列表
        stock_codes: 股票代码列表
        days: 天数

    Returns:
        策略信号字典
    """
    generator = MockSignalDataGenerator(seed=42)
    signals_dict = {}

    for strategy_name in strategy_names:
        signals_dict[strategy_name] = generator.generate_trading_signals(
            strategy_name=strategy_name,
            stock_codes=stock_codes,
            days=days,
            signal_frequency=0.05  # 5%的信号频率
        )

    return signals_dict


def create_backtest_signals(strategy_name: str,
                          stock_codes: List[str],
                          start_date: str = "2024-01-01",
                          end_date: str = "2024-12-31") -> pd.DataFrame:
    """
    创建回测信号数据

    Args:
        strategy_name: 策略名称
        stock_codes: 股票代码列表
        start_date: 开始日期
        end_date: 结束日期

    Returns:
        回测信号DataFrame
    """
    generator = MockSignalDataGenerator(seed=42)

    return generator.generate_trading_signals(
        strategy_name=strategy_name,
        stock_codes=stock_codes,
        start_date=start_date,
        end_date=end_date,
        signal_frequency=0.03  # 3%的信号频率，适合回测
    )


def create_live_trading_scenario(strategy_name: str,
                               universe_size: int = 100) -> Dict[str, Any]:
    """
    创建实盘交易场景

    Args:
        strategy_name: 策略名称
        universe_size: 股票池大小

    Returns:
        实盘交易场景数据
    """
    generator = MockSignalDataGenerator(seed=42)

    # 生成股票池
    stock_codes = [f"sh{600000 + i:06d}" for i in range(0, universe_size // 2)]
    stock_codes.extend([f"sz{i:06d}" for i in range(1, universe_size // 2 + 1)])

    # 生成当日信号
    today_signals = generator.generate_trading_signals(
        strategy_name=strategy_name,
        stock_codes=stock_codes,
        days=1,
        signal_frequency=0.08  # 8%的信号频率
    )

    # 生成组合信号
    portfolio_signals = generator.generate_portfolio_signals(
        strategy_name=strategy_name,
        target_stocks=50,
        rebalance_date=datetime.now().date()
    )

    # 生成风险控制信号
    risk_signals = generator.generate_risk_control_signals(portfolio_signals)

    return {
        "strategy_name": strategy_name,
        "trading_date": datetime.now().date(),
        "today_signals": today_signals,
        "portfolio_signals": portfolio_signals,
        "risk_signals": risk_signals,
        "signal_count": len(today_signals),
        "portfolio_size": len(portfolio_signals),
        "risk_alert_count": len(risk_signals)
    }