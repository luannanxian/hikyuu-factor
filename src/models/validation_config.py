"""
Validation Configuration Models
验证配置相关的数据模型，支持可配置的训练/验证/测试期间划分
"""

import enum
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple, Union
import json


class ValidationPeriodType(enum.Enum):
    """验证期间类型"""
    TRAINING = "training"
    VALIDATION = "validation"
    TEST = "test"
    OUT_OF_SAMPLE = "out_of_sample"


class ValidationMetricType(enum.Enum):
    """验证指标类型"""
    IC = "ic"  # Information Coefficient
    RANK_IC = "rank_ic"  # Rank Information Coefficient
    LAYERED_RETURNS = "layered_returns"  # 分层收益
    TURNOVER_RATE = "turnover_rate"  # 换手率
    SHARPE_RATIO = "sharpe_ratio"  # 夏普比率
    MAX_DRAWDOWN = "max_drawdown"  # 最大回撤
    VOLATILITY = "volatility"  # 波动率
    CUMULATIVE_RETURN = "cumulative_return"  # 累计收益


@dataclass
class ValidationPeriod:
    """验证期间定义"""
    period_type: ValidationPeriodType
    start_date: date
    end_date: date
    description: Optional[str] = None

    def __post_init__(self):
        """初始化后验证"""
        if self.start_date >= self.end_date:
            raise ValueError("start_date must be before end_date")

    @property
    def duration_days(self) -> int:
        """期间长度（天数）"""
        return (self.end_date - self.start_date).days

    @property
    def duration_years(self) -> float:
        """期间长度（年数）"""
        return self.duration_days / 365.25

    def contains_date(self, check_date: date) -> bool:
        """检查日期是否在期间内"""
        return self.start_date <= check_date <= self.end_date

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            "period_type": self.period_type.value,
            "start_date": self.start_date.isoformat(),
            "end_date": self.end_date.isoformat(),
            "description": self.description
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ValidationPeriod':
        """从字典创建"""
        return cls(
            period_type=ValidationPeriodType(data["period_type"]),
            start_date=date.fromisoformat(data["start_date"]),
            end_date=date.fromisoformat(data["end_date"]),
            description=data.get("description")
        )


@dataclass
class ValidationConfig:
    """验证配置类"""
    config_name: str
    periods: List[ValidationPeriod]
    enabled_metrics: List[ValidationMetricType] = field(default_factory=lambda: [
        ValidationMetricType.IC,
        ValidationMetricType.RANK_IC,
        ValidationMetricType.LAYERED_RETURNS
    ])
    layered_groups: int = 10  # 分层数量
    rebalance_frequency: str = "monthly"  # 调仓频率: daily, weekly, monthly
    benchmark: Optional[str] = "000300.SH"  # 基准指数
    min_stock_count: int = 100  # 最少股票数量
    max_stock_count: Optional[int] = None  # 最多股票数量
    exclude_st: bool = True  # 排除ST股票
    exclude_suspended: bool = True  # 排除停牌股票
    custom_params: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        """初始化后验证"""
        self.validate()

    def validate(self) -> None:
        """验证配置"""
        if not self.periods:
            raise ValueError("At least one validation period is required")

        if self.layered_groups < 2:
            raise ValueError("layered_groups must be >= 2")

        if self.min_stock_count < 10:
            raise ValueError("min_stock_count must be >= 10")

        if self.max_stock_count and self.max_stock_count < self.min_stock_count:
            raise ValueError("max_stock_count must be >= min_stock_count")

        # 检查期间重叠
        self._check_period_overlaps()

        # 检查期间顺序
        self._check_period_order()

    def _check_period_overlaps(self) -> None:
        """检查期间是否重叠"""
        for i, period1 in enumerate(self.periods):
            for j, period2 in enumerate(self.periods[i+1:], i+1):
                if (period1.start_date <= period2.end_date and
                    period2.start_date <= period1.end_date):
                    raise ValueError(f"Periods {i} and {j} overlap")

    def _check_period_order(self) -> None:
        """检查期间顺序"""
        sorted_periods = sorted(self.periods, key=lambda p: p.start_date)
        if sorted_periods != self.periods:
            raise ValueError("Periods must be ordered by start_date")

    def get_period_by_type(self, period_type: ValidationPeriodType) -> Optional[ValidationPeriod]:
        """根据类型获取期间"""
        for period in self.periods:
            if period.period_type == period_type:
                return period
        return None

    def get_periods_by_type(self, period_type: ValidationPeriodType) -> List[ValidationPeriod]:
        """根据类型获取所有期间"""
        return [p for p in self.periods if p.period_type == period_type]

    def get_date_period_type(self, check_date: date) -> Optional[ValidationPeriodType]:
        """获取指定日期所属的期间类型"""
        for period in self.periods:
            if period.contains_date(check_date):
                return period.period_type
        return None

    @property
    def total_duration_years(self) -> float:
        """总时长（年）"""
        if not self.periods:
            return 0.0

        start_date = min(p.start_date for p in self.periods)
        end_date = max(p.end_date for p in self.periods)
        return (end_date - start_date).days / 365.25

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            "config_name": self.config_name,
            "periods": [p.to_dict() for p in self.periods],
            "enabled_metrics": [m.value for m in self.enabled_metrics],
            "layered_groups": self.layered_groups,
            "rebalance_frequency": self.rebalance_frequency,
            "benchmark": self.benchmark,
            "min_stock_count": self.min_stock_count,
            "max_stock_count": self.max_stock_count,
            "exclude_st": self.exclude_st,
            "exclude_suspended": self.exclude_suspended,
            "custom_params": self.custom_params
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ValidationConfig':
        """从字典创建"""
        return cls(
            config_name=data["config_name"],
            periods=[ValidationPeriod.from_dict(p) for p in data["periods"]],
            enabled_metrics=[ValidationMetricType(m) for m in data.get("enabled_metrics", [])],
            layered_groups=data.get("layered_groups", 10),
            rebalance_frequency=data.get("rebalance_frequency", "monthly"),
            benchmark=data.get("benchmark"),
            min_stock_count=data.get("min_stock_count", 100),
            max_stock_count=data.get("max_stock_count"),
            exclude_st=data.get("exclude_st", True),
            exclude_suspended=data.get("exclude_suspended", True),
            custom_params=data.get("custom_params", {})
        )

    def to_json(self) -> str:
        """转换为JSON字符串"""
        return json.dumps(self.to_dict(), indent=2, ensure_ascii=False)

    @classmethod
    def from_json(cls, json_str: str) -> 'ValidationConfig':
        """从JSON字符串创建"""
        data = json.loads(json_str)
        return cls.from_dict(data)


class ValidationConfigPresets:
    """验证配置预设模板"""

    @staticmethod
    def default_config() -> ValidationConfig:
        """默认配置 (2010-2016训练, 2017-2020验证, 2021-至今测试)"""
        periods = [
            ValidationPeriod(
                period_type=ValidationPeriodType.TRAINING,
                start_date=date(2010, 1, 1),
                end_date=date(2016, 12, 31),
                description="训练期：2010-2016"
            ),
            ValidationPeriod(
                period_type=ValidationPeriodType.VALIDATION,
                start_date=date(2017, 1, 1),
                end_date=date(2020, 12, 31),
                description="验证期：2017-2020"
            ),
            ValidationPeriod(
                period_type=ValidationPeriodType.TEST,
                start_date=date(2021, 1, 1),
                end_date=date.today(),
                description="测试期：2021-至今"
            )
        ]

        return ValidationConfig(
            config_name="default",
            periods=periods,
            enabled_metrics=[
                ValidationMetricType.IC,
                ValidationMetricType.RANK_IC,
                ValidationMetricType.LAYERED_RETURNS,
                ValidationMetricType.SHARPE_RATIO
            ]
        )

    @staticmethod
    def short_term_config() -> ValidationConfig:
        """短期验证配置 (最近3年数据)"""
        end_date = date.today()
        start_date = end_date - timedelta(days=3*365)
        mid_date = start_date + timedelta(days=2*365)

        periods = [
            ValidationPeriod(
                period_type=ValidationPeriodType.TRAINING,
                start_date=start_date,
                end_date=mid_date,
                description="训练期：近3年前2年"
            ),
            ValidationPeriod(
                period_type=ValidationPeriodType.TEST,
                start_date=mid_date + timedelta(days=1),
                end_date=end_date,
                description="测试期：近1年"
            )
        ]

        return ValidationConfig(
            config_name="short_term",
            periods=periods,
            rebalance_frequency="weekly"
        )

    @staticmethod
    def long_term_config() -> ValidationConfig:
        """长期验证配置 (2005-至今)"""
        periods = [
            ValidationPeriod(
                period_type=ValidationPeriodType.TRAINING,
                start_date=date(2005, 1, 1),
                end_date=date(2012, 12, 31),
                description="训练期：2005-2012"
            ),
            ValidationPeriod(
                period_type=ValidationPeriodType.VALIDATION,
                start_date=date(2013, 1, 1),
                end_date=date(2018, 12, 31),
                description="验证期：2013-2018"
            ),
            ValidationPeriod(
                period_type=ValidationPeriodType.TEST,
                start_date=date(2019, 1, 1),
                end_date=date.today(),
                description="测试期：2019-至今"
            )
        ]

        return ValidationConfig(
            config_name="long_term",
            periods=periods,
            enabled_metrics=[m for m in ValidationMetricType]  # 所有指标
        )

    @staticmethod
    def crisis_aware_config() -> ValidationConfig:
        """危机感知配置 (包含特殊市场环境)"""
        periods = [
            ValidationPeriod(
                period_type=ValidationPeriodType.TRAINING,
                start_date=date(2010, 1, 1),
                end_date=date(2014, 12, 31),
                description="训练期：2010-2014 (牛市前)"
            ),
            ValidationPeriod(
                period_type=ValidationPeriodType.VALIDATION,
                start_date=date(2015, 1, 1),
                end_date=date(2016, 12, 31),
                description="验证期：2015-2016 (股灾期)"
            ),
            ValidationPeriod(
                period_type=ValidationPeriodType.TEST,
                start_date=date(2017, 1, 1),
                end_date=date(2019, 12, 31),
                description="测试期：2017-2019 (震荡期)"
            ),
            ValidationPeriod(
                period_type=ValidationPeriodType.OUT_OF_SAMPLE,
                start_date=date(2020, 1, 1),
                end_date=date.today(),
                description="样本外：2020-至今 (疫情+复苏)"
            )
        ]

        return ValidationConfig(
            config_name="crisis_aware",
            periods=periods,
            enabled_metrics=[
                ValidationMetricType.IC,
                ValidationMetricType.RANK_IC,
                ValidationMetricType.LAYERED_RETURNS,
                ValidationMetricType.MAX_DRAWDOWN,
                ValidationMetricType.VOLATILITY
            ],
            layered_groups=5,  # 更少分层，关注极端表现
            rebalance_frequency="monthly"
        )

    @staticmethod
    def rolling_window_config(window_years: int = 3) -> ValidationConfig:
        """滚动窗口配置"""
        end_date = date.today()
        start_date = end_date - timedelta(days=window_years*365)

        periods = [
            ValidationPeriod(
                period_type=ValidationPeriodType.TRAINING,
                start_date=start_date,
                end_date=end_date,
                description=f"滚动窗口：近{window_years}年"
            )
        ]

        return ValidationConfig(
            config_name=f"rolling_{window_years}y",
            periods=periods,
            rebalance_frequency="daily",
            layered_groups=20  # 更细的分层
        )

    @classmethod
    def get_all_presets(cls) -> Dict[str, ValidationConfig]:
        """获取所有预设配置"""
        return {
            "default": cls.default_config(),
            "short_term": cls.short_term_config(),
            "long_term": cls.long_term_config(),
            "crisis_aware": cls.crisis_aware_config(),
            "rolling_3y": cls.rolling_window_config(3),
            "rolling_5y": cls.rolling_window_config(5)
        }

    @classmethod
    def get_preset_names(cls) -> List[str]:
        """获取所有预设配置名称"""
        return list(cls.get_all_presets().keys())


class ValidationConfigValidator:
    """验证配置验证器"""

    @staticmethod
    def validate_config(config: ValidationConfig) -> List[str]:
        """验证配置并返回警告信息"""
        warnings = []

        # 检查期间长度
        for period in config.periods:
            if period.duration_years < 1:
                warnings.append(f"Period {period.period_type.value} is less than 1 year")
            elif period.duration_years > 10:
                warnings.append(f"Period {period.period_type.value} is longer than 10 years")

        # 检查训练/验证/测试比例
        training_periods = config.get_periods_by_type(ValidationPeriodType.TRAINING)
        validation_periods = config.get_periods_by_type(ValidationPeriodType.VALIDATION)
        test_periods = config.get_periods_by_type(ValidationPeriodType.TEST)

        total_training_years = sum(p.duration_years for p in training_periods)
        total_validation_years = sum(p.duration_years for p in validation_periods)
        total_test_years = sum(p.duration_years for p in test_periods)

        if total_training_years < 2:
            warnings.append("Training period should be at least 2 years")

        if validation_periods and total_validation_years < 1:
            warnings.append("Validation period should be at least 1 year")

        if test_periods and total_test_years < 1:
            warnings.append("Test period should be at least 1 year")

        # 检查股票数量要求
        if config.min_stock_count < 50:
            warnings.append("min_stock_count is quite low, may affect statistical significance")

        return warnings

    @staticmethod
    def suggest_improvements(config: ValidationConfig) -> List[str]:
        """建议配置改进"""
        suggestions = []

        # 建议启用的指标
        recommended_metrics = [
            ValidationMetricType.IC,
            ValidationMetricType.RANK_IC,
            ValidationMetricType.LAYERED_RETURNS,
            ValidationMetricType.TURNOVER_RATE
        ]

        missing_metrics = set(recommended_metrics) - set(config.enabled_metrics)
        if missing_metrics:
            suggestions.append(f"Consider enabling metrics: {[m.value for m in missing_metrics]}")

        # 建议分层数量
        if config.layered_groups < 5:
            suggestions.append("Consider using at least 5 layered groups for better analysis")
        elif config.layered_groups > 20:
            suggestions.append("Too many layered groups may reduce statistical significance")

        # 建议调仓频率
        if config.rebalance_frequency == "daily":
            suggestions.append("Daily rebalancing may incur high transaction costs")

        return suggestions