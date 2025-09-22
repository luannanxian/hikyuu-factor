"""
Hikyuu Framework Data Models

基于Hikyuu量化框架的核心数据模型，提供：
1. 因子数据模型 - 扩展Hikyuu原生数据结构
2. 交易信号模型 - 基于Hikyuu Stock和KData
3. Point-in-Time数据访问 - 防止前视偏差
4. 投资组合模型 - 基于Hikyuu Portfolio

所有模型都直接使用Hikyuu的C++核心，确保最高性能。
"""

from dataclasses import dataclass, field
from datetime import datetime, date
from typing import Dict, List, Optional, Union, Any, Tuple
from decimal import Decimal
from enum import Enum
import pandas as pd
import numpy as np

try:
    import hikyuu as hk
    from hikyuu import Stock, KData, StockManager, Query, FINANCE
    HIKYUU_AVAILABLE = True
except ImportError:
    # 用于测试环境或Hikyuu未安装的情况
    HIKYUU_AVAILABLE = False
    Stock = Any
    KData = Any
    StockManager = Any
    Query = Any
    FINANCE = Any


class FactorType(Enum):
    """因子类型枚举"""
    MOMENTUM = "momentum"        # 动量因子
    VALUE = "value"             # 价值因子
    QUALITY = "quality"         # 质量因子
    GROWTH = "growth"           # 成长因子
    VOLATILITY = "volatility"   # 波动率因子
    LIQUIDITY = "liquidity"     # 流动性因子
    SENTIMENT = "sentiment"     # 情绪因子
    TECHNICAL = "technical"     # 技术因子
    FUNDAMENTAL = "fundamental" # 基本面因子
    MACRO = "macro"            # 宏观因子
    CUSTOM = "custom"          # 自定义因子


class SignalType(Enum):
    """交易信号类型"""
    BUY = "buy"
    SELL = "sell"
    HOLD = "hold"


class PositionType(Enum):
    """持仓类型"""
    LONG = "long"
    SHORT = "short"
    CASH = "cash"


@dataclass
class FactorData:
    """
    因子数据模型

    基于Hikyuu Stock和KData的因子数据封装，提供Point-in-Time访问。
    """
    factor_name: str
    factor_type: FactorType
    stock_code: str
    calculation_date: date
    factor_value: float
    factor_score: Optional[float] = None  # 标准化后的因子分数 [0,1]

    # Hikyuu原生对象引用
    hikyuu_stock: Optional[Stock] = None
    hikyuu_kdata: Optional[KData] = None

    # 元数据
    calculation_method: Optional[str] = None
    lookback_period: Optional[int] = None
    data_source: Optional[str] = None
    quality_score: Optional[float] = None

    # Point-in-Time保证
    as_of_date: Optional[date] = None  # 数据可用时间点

    def __post_init__(self):
        """初始化后验证"""
        if self.as_of_date is None:
            self.as_of_date = self.calculation_date

        # 验证因子值
        if pd.isna(self.factor_value) or np.isinf(self.factor_value):
            raise ValueError(f"Invalid factor value: {self.factor_value}")

        # 验证因子分数范围
        if self.factor_score is not None:
            if not (0 <= self.factor_score <= 1):
                raise ValueError(f"Factor score must be in [0,1], got: {self.factor_score}")

    @classmethod
    def from_hikyuu_stock(
        cls,
        stock: Stock,
        factor_name: str,
        factor_type: FactorType,
        calculation_date: date,
        factor_value: float,
        **kwargs
    ) -> 'FactorData':
        """从Hikyuu Stock对象创建因子数据"""
        if not HIKYUU_AVAILABLE:
            raise RuntimeError("Hikyuu framework not available")

        return cls(
            factor_name=factor_name,
            factor_type=factor_type,
            stock_code=stock.market_code,
            calculation_date=calculation_date,
            factor_value=factor_value,
            hikyuu_stock=stock,
            **kwargs
        )

    def get_stock_info(self) -> Dict[str, Any]:
        """获取股票基本信息"""
        if not self.hikyuu_stock:
            return {"stock_code": self.stock_code}

        return {
            "stock_code": self.hikyuu_stock.market_code,
            "name": self.hikyuu_stock.name,
            "market": self.hikyuu_stock.market,
            "type": self.hikyuu_stock.type,
            "valid": self.hikyuu_stock.valid,
            "start_date": self.hikyuu_stock.start_datetime,
            "last_date": self.hikyuu_stock.last_datetime
        }

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典格式"""
        return {
            "factor_name": self.factor_name,
            "factor_type": self.factor_type.value,
            "stock_code": self.stock_code,
            "calculation_date": self.calculation_date.isoformat(),
            "factor_value": self.factor_value,
            "factor_score": self.factor_score,
            "calculation_method": self.calculation_method,
            "lookback_period": self.lookback_period,
            "data_source": self.data_source,
            "quality_score": self.quality_score,
            "as_of_date": self.as_of_date.isoformat() if self.as_of_date else None
        }


@dataclass
class FactorCalculationRequest:
    """
    因子计算请求模型

    定义因子计算任务的完整参数，支持批量计算和增量更新。
    """
    request_id: str
    factor_name: str
    factor_type: FactorType

    # 计算范围
    stock_codes: List[str]
    start_date: date
    end_date: date

    # 计算参数
    calculation_params: Dict[str, Any] = field(default_factory=dict)
    lookback_period: Optional[int] = None

    # 执行控制
    priority: int = 1  # 1=低, 2=中, 3=高
    max_parallel_workers: int = 4
    chunk_size: int = 100  # 批处理大小

    # 数据源配置
    data_source: str = "hikyuu_default"
    use_cache: bool = True
    force_recalculate: bool = False

    # 输出控制
    output_format: str = "dataframe"  # dataframe, json, parquet
    normalize_scores: bool = True

    created_at: datetime = field(default_factory=datetime.now)
    created_by: Optional[str] = None

    def __post_init__(self):
        """初始化后验证"""
        if self.start_date > self.end_date:
            raise ValueError("start_date must be <= end_date")

        if not self.stock_codes:
            raise ValueError("stock_codes cannot be empty")

        if self.priority not in [1, 2, 3]:
            raise ValueError("priority must be 1, 2, or 3")

    def get_hikyuu_stocks(self) -> List[Stock]:
        """获取Hikyuu Stock对象列表"""
        if not HIKYUU_AVAILABLE:
            raise RuntimeError("Hikyuu framework not available")

        sm = StockManager.instance()
        stocks = []

        for code in self.stock_codes:
            stock = sm.get_stock(code)
            if stock.valid:
                stocks.append(stock)

        return stocks

    def create_hikyuu_query(self) -> Query:
        """创建Hikyuu查询对象"""
        if not HIKYUU_AVAILABLE:
            raise RuntimeError("Hikyuu framework not available")

        start_dt = hk.Datetime(self.start_date)
        end_dt = hk.Datetime(self.end_date)

        return Query(start_dt, end_dt)

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典格式"""
        return {
            "request_id": self.request_id,
            "factor_name": self.factor_name,
            "factor_type": self.factor_type.value,
            "stock_codes": self.stock_codes,
            "start_date": self.start_date.isoformat(),
            "end_date": self.end_date.isoformat(),
            "calculation_params": self.calculation_params,
            "lookback_period": self.lookback_period,
            "priority": self.priority,
            "max_parallel_workers": self.max_parallel_workers,
            "chunk_size": self.chunk_size,
            "data_source": self.data_source,
            "use_cache": self.use_cache,
            "force_recalculate": self.force_recalculate,
            "output_format": self.output_format,
            "normalize_scores": self.normalize_scores,
            "created_at": self.created_at.isoformat(),
            "created_by": self.created_by
        }


@dataclass
class FactorCalculationResult:
    """
    因子计算结果模型

    包含计算结果、性能指标和质量评估。
    """
    request_id: str
    factor_name: str
    factor_type: FactorType

    # 计算结果
    factor_data: List[FactorData]
    calculation_date: datetime

    # 执行统计
    total_stocks: int
    successful_calculations: int
    failed_calculations: int
    execution_time_seconds: float

    # 质量指标
    data_quality_score: Optional[float] = None
    coverage_ratio: Optional[float] = None  # 有效数据覆盖率
    outlier_ratio: Optional[float] = None   # 异常值比例

    # 统计信息
    factor_statistics: Optional[Dict[str, float]] = None

    # 错误信息
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)

    def __post_init__(self):
        """计算派生指标"""
        if self.coverage_ratio is None:
            self.coverage_ratio = self.successful_calculations / self.total_stocks

        # 计算基本统计信息
        if self.factor_data and not self.factor_statistics:
            values = [fd.factor_value for fd in self.factor_data if not pd.isna(fd.factor_value)]
            if values:
                self.factor_statistics = {
                    "mean": float(np.mean(values)),
                    "std": float(np.std(values)),
                    "min": float(np.min(values)),
                    "max": float(np.max(values)),
                    "median": float(np.median(values)),
                    "count": len(values)
                }

    def to_dataframe(self) -> pd.DataFrame:
        """转换为DataFrame格式"""
        if not self.factor_data:
            return pd.DataFrame()

        data = []
        for fd in self.factor_data:
            row = fd.to_dict()
            data.append(row)

        return pd.DataFrame(data)

    def get_successful_data(self) -> List[FactorData]:
        """获取成功计算的因子数据"""
        return [fd for fd in self.factor_data if not pd.isna(fd.factor_value)]

    def get_failed_stocks(self) -> List[str]:
        """获取计算失败的股票代码"""
        successful_stocks = {fd.stock_code for fd in self.get_successful_data()}
        all_stocks = {fd.stock_code for fd in self.factor_data}
        return list(all_stocks - successful_stocks)

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典格式"""
        return {
            "request_id": self.request_id,
            "factor_name": self.factor_name,
            "factor_type": self.factor_type.value,
            "calculation_date": self.calculation_date.isoformat(),
            "total_stocks": self.total_stocks,
            "successful_calculations": self.successful_calculations,
            "failed_calculations": self.failed_calculations,
            "execution_time_seconds": self.execution_time_seconds,
            "data_quality_score": self.data_quality_score,
            "coverage_ratio": self.coverage_ratio,
            "outlier_ratio": self.outlier_ratio,
            "factor_statistics": self.factor_statistics,
            "errors": self.errors,
            "warnings": self.warnings,
            "factor_data_count": len(self.factor_data)
        }


@dataclass
class TradingSignal:
    """
    交易信号模型

    基于因子计算结果生成的交易信号，包含完整的执行信息。
    """
    signal_id: str
    stock_code: str
    signal_type: SignalType
    signal_strength: float  # [0, 1]

    # 信号生成信息
    generation_date: datetime
    effective_date: date
    expiry_date: Optional[date] = None

    # 基于的因子
    source_factors: List[str] = field(default_factory=list)
    factor_scores: Dict[str, float] = field(default_factory=dict)

    # 交易参数
    target_position: Optional[float] = None  # 目标仓位 [0, 1]
    position_type: PositionType = PositionType.LONG

    # 风险参数
    max_position_size: Optional[float] = None
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None

    # Hikyuu集成
    hikyuu_stock: Optional[Stock] = None
    current_price: Optional[float] = None

    # 执行状态
    status: str = "pending"  # pending, confirmed, executed, cancelled
    confirmation_required: bool = True

    # 元数据
    strategy_name: Optional[str] = None
    created_by: Optional[str] = None
    risk_score: Optional[float] = None

    def __post_init__(self):
        """初始化后验证"""
        if not (0 <= self.signal_strength <= 1):
            raise ValueError(f"Signal strength must be in [0,1], got: {self.signal_strength}")

        if self.target_position is not None:
            if not (0 <= self.target_position <= 1):
                raise ValueError(f"Target position must be in [0,1], got: {self.target_position}")

    @classmethod
    def from_factor_data(
        cls,
        signal_id: str,
        factor_data: List[FactorData],
        signal_config: Dict[str, Any],
        **kwargs
    ) -> 'TradingSignal':
        """从因子数据创建交易信号"""
        if not factor_data:
            raise ValueError("factor_data cannot be empty")

        # 使用第一个因子的股票代码
        stock_code = factor_data[0].stock_code

        # 确保所有因子都是同一只股票
        if not all(fd.stock_code == stock_code for fd in factor_data):
            raise ValueError("All factor data must be for the same stock")

        # 计算综合信号强度
        factor_scores = {fd.factor_name: fd.factor_score or 0.5 for fd in factor_data}

        # 简单平均作为信号强度（实际实现中应该使用加权）
        signal_strength = np.mean(list(factor_scores.values()))

        # 根据信号强度确定信号类型
        buy_threshold = signal_config.get('buy_threshold', 0.7)
        sell_threshold = signal_config.get('sell_threshold', 0.3)

        if signal_strength >= buy_threshold:
            signal_type = SignalType.BUY
        elif signal_strength <= sell_threshold:
            signal_type = SignalType.SELL
        else:
            signal_type = SignalType.HOLD

        return cls(
            signal_id=signal_id,
            stock_code=stock_code,
            signal_type=signal_type,
            signal_strength=signal_strength,
            generation_date=datetime.now(),
            effective_date=date.today(),
            source_factors=[fd.factor_name for fd in factor_data],
            factor_scores=factor_scores,
            hikyuu_stock=factor_data[0].hikyuu_stock,
            **kwargs
        )

    def get_current_price(self) -> Optional[float]:
        """获取当前价格"""
        if not self.hikyuu_stock or not HIKYUU_AVAILABLE:
            return self.current_price

        try:
            # 获取最新收盘价
            kdata = self.hikyuu_stock.get_kdata()
            if len(kdata) > 0:
                return float(kdata[-1].close)
        except Exception:
            pass

        return self.current_price

    def update_price(self):
        """更新当前价格"""
        self.current_price = self.get_current_price()

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典格式"""
        return {
            "signal_id": self.signal_id,
            "stock_code": self.stock_code,
            "signal_type": self.signal_type.value,
            "signal_strength": self.signal_strength,
            "generation_date": self.generation_date.isoformat(),
            "effective_date": self.effective_date.isoformat(),
            "expiry_date": self.expiry_date.isoformat() if self.expiry_date else None,
            "source_factors": self.source_factors,
            "factor_scores": self.factor_scores,
            "target_position": self.target_position,
            "position_type": self.position_type.value,
            "max_position_size": self.max_position_size,
            "stop_loss": self.stop_loss,
            "take_profit": self.take_profit,
            "current_price": self.current_price,
            "status": self.status,
            "confirmation_required": self.confirmation_required,
            "strategy_name": self.strategy_name,
            "created_by": self.created_by,
            "risk_score": self.risk_score
        }


@dataclass
class PortfolioPosition:
    """
    投资组合持仓模型

    表示投资组合中的单个持仓，支持与Hikyuu Portfolio集成。
    """
    position_id: str
    stock_code: str
    position_type: PositionType

    # 持仓数量和价值
    quantity: int  # 股数
    cost_price: float  # 成本价
    current_price: float  # 当前价
    market_value: float  # 市值

    # 持仓比例
    weight: float  # 在投资组合中的权重 [0, 1]

    # 日期信息
    open_date: date
    last_update: datetime

    # 收益信息
    unrealized_pnl: float = 0.0  # 浮动盈亏
    realized_pnl: float = 0.0    # 已实现盈亏

    # Hikyuu集成
    hikyuu_stock: Optional[Stock] = None

    # 风险控制
    stop_loss_price: Optional[float] = None
    take_profit_price: Optional[float] = None
    max_loss_ratio: Optional[float] = None

    # 来源信号
    source_signal_id: Optional[str] = None

    def __post_init__(self):
        """计算派生字段"""
        self.market_value = self.quantity * self.current_price
        self.unrealized_pnl = (self.current_price - self.cost_price) * self.quantity

        if not (0 <= self.weight <= 1):
            raise ValueError(f"Weight must be in [0,1], got: {self.weight}")

    def update_price(self, new_price: float):
        """更新当前价格和相关计算"""
        self.current_price = new_price
        self.market_value = self.quantity * self.current_price
        self.unrealized_pnl = (self.current_price - self.cost_price) * self.quantity
        self.last_update = datetime.now()

    def get_return_ratio(self) -> float:
        """获取收益率"""
        if self.cost_price == 0:
            return 0.0
        return (self.current_price - self.cost_price) / self.cost_price

    def should_stop_loss(self) -> bool:
        """检查是否触发止损"""
        if self.stop_loss_price is None:
            return False

        if self.position_type == PositionType.LONG:
            return self.current_price <= self.stop_loss_price
        else:  # SHORT
            return self.current_price >= self.stop_loss_price

    def should_take_profit(self) -> bool:
        """检查是否触发止盈"""
        if self.take_profit_price is None:
            return False

        if self.position_type == PositionType.LONG:
            return self.current_price >= self.take_profit_price
        else:  # SHORT
            return self.current_price <= self.take_profit_price

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典格式"""
        return {
            "position_id": self.position_id,
            "stock_code": self.stock_code,
            "position_type": self.position_type.value,
            "quantity": self.quantity,
            "cost_price": self.cost_price,
            "current_price": self.current_price,
            "market_value": self.market_value,
            "weight": self.weight,
            "open_date": self.open_date.isoformat(),
            "last_update": self.last_update.isoformat(),
            "unrealized_pnl": self.unrealized_pnl,
            "realized_pnl": self.realized_pnl,
            "return_ratio": self.get_return_ratio(),
            "stop_loss_price": self.stop_loss_price,
            "take_profit_price": self.take_profit_price,
            "max_loss_ratio": self.max_loss_ratio,
            "source_signal_id": self.source_signal_id
        }


# 工具函数
def create_factor_data_from_hikyuu(
    stock: Stock,
    factor_name: str,
    factor_type: FactorType,
    calculation_date: date,
    calculation_method: str,
    **params
) -> FactorData:
    """
    从Hikyuu Stock对象创建因子数据的便利函数

    这是一个工厂函数，简化因子数据的创建过程。
    """
    if not HIKYUU_AVAILABLE:
        raise RuntimeError("Hikyuu framework not available")

    # 这里应该实现具体的因子计算逻辑
    # 暂时返回一个示例值
    factor_value = 0.0  # 实际实现中应该根据calculation_method计算

    return FactorData.from_hikyuu_stock(
        stock=stock,
        factor_name=factor_name,
        factor_type=factor_type,
        calculation_date=calculation_date,
        factor_value=factor_value,
        calculation_method=calculation_method,
        **params
    )


def batch_create_factor_requests(
    factor_configs: List[Dict[str, Any]],
    stock_universe: List[str],
    date_range: Tuple[date, date],
    **common_params
) -> List[FactorCalculationRequest]:
    """
    批量创建因子计算请求

    Args:
        factor_configs: 因子配置列表
        stock_universe: 股票池
        date_range: 日期范围 (start_date, end_date)
        **common_params: 通用参数

    Returns:
        因子计算请求列表
    """
    requests = []
    start_date, end_date = date_range

    for i, config in enumerate(factor_configs):
        request = FactorCalculationRequest(
            request_id=f"batch_{i}_{config['factor_name']}",
            factor_name=config['factor_name'],
            factor_type=FactorType(config['factor_type']),
            stock_codes=stock_universe,
            start_date=start_date,
            end_date=end_date,
            calculation_params=config.get('params', {}),
            **common_params
        )
        requests.append(request)

    return requests