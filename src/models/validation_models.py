"""
Validation and Risk Assessment Data Models

验证和风险评估相关的数据模型，提供：
1. 因子验证规则和结果模型
2. 风险评估和控制模型
3. 业务逻辑验证模型
4. 统计和相关性验证模型

支持多层级验证和风险控制策略。
"""

from dataclasses import dataclass, field
from datetime import datetime, date
from typing import Dict, List, Optional, Any, Union, Tuple
from enum import Enum
import uuid
import json


class ValidationRuleType(Enum):
    """验证规则类型"""
    STATISTICAL = "statistical"
    CORRELATION = "correlation"
    STABILITY = "stability"
    BUSINESS_LOGIC = "business_logic"
    PERFORMANCE = "performance"
    DATA_QUALITY = "data_quality"


class ValidationSeverity(Enum):
    """验证严重性级别"""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


class RiskLevel(Enum):
    """风险级别"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class RiskCategory(Enum):
    """风险类别"""
    MARKET_RISK = "market_risk"
    POSITION_RISK = "position_risk"
    LIQUIDITY_RISK = "liquidity_risk"
    COMPLIANCE_RISK = "compliance_risk"
    OPERATIONAL_RISK = "operational_risk"
    MODEL_RISK = "model_risk"


@dataclass
class ValidationRule:
    """
    验证规则模型

    定义具体的验证规则，包括规则类型、参数和阈值。
    """
    rule_id: str
    rule_name: str
    rule_type: ValidationRuleType

    # 规则参数
    parameters: Dict[str, Any] = field(default_factory=dict)
    thresholds: Dict[str, float] = field(default_factory=dict)

    # 规则配置
    enabled: bool = True
    severity: ValidationSeverity = ValidationSeverity.WARNING

    # 适用范围
    applicable_factor_types: List[str] = field(default_factory=list)
    excluded_stocks: List[str] = field(default_factory=list)

    # 描述信息
    description: Optional[str] = None
    remediation_suggestion: Optional[str] = None

    # 元数据
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)
    created_by: Optional[str] = None

    def __post_init__(self):
        """初始化后处理"""
        if not self.rule_id:
            self.rule_id = str(uuid.uuid4())

    def is_applicable_to_factor(self, factor_type: str) -> bool:
        """检查规则是否适用于指定因子类型"""
        if not self.applicable_factor_types:
            return True  # 空列表表示适用于所有因子
        return factor_type in self.applicable_factor_types

    def is_stock_excluded(self, stock_code: str) -> bool:
        """检查股票是否被排除"""
        return stock_code in self.excluded_stocks

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典格式"""
        return {
            "rule_id": self.rule_id,
            "rule_name": self.rule_name,
            "rule_type": self.rule_type.value,
            "parameters": self.parameters,
            "thresholds": self.thresholds,
            "enabled": self.enabled,
            "severity": self.severity.value,
            "applicable_factor_types": self.applicable_factor_types,
            "excluded_stocks": self.excluded_stocks,
            "description": self.description,
            "remediation_suggestion": self.remediation_suggestion,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
            "created_by": self.created_by
        }


@dataclass
class ValidationIssue:
    """
    验证问题模型

    表示验证过程中发现的具体问题。
    """
    issue_id: str
    rule_id: str
    rule_name: str

    # 问题描述
    category: str
    severity: ValidationSeverity
    description: str

    # 影响范围
    affected_factor: Optional[str] = None
    affected_stocks: List[str] = field(default_factory=list)
    affected_date_range: Optional[Tuple[date, date]] = None

    # 问题值
    actual_value: Optional[float] = None
    expected_value: Optional[float] = None
    threshold_value: Optional[float] = None

    # 建议措施
    recommendation: Optional[str] = None
    auto_fixable: bool = False

    # 时间信息
    detected_at: datetime = field(default_factory=datetime.now)

    # 元数据
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        """初始化后处理"""
        if not self.issue_id:
            self.issue_id = str(uuid.uuid4())

    def get_severity_score(self) -> int:
        """获取严重性分数（用于排序）"""
        severity_scores = {
            ValidationSeverity.INFO: 1,
            ValidationSeverity.WARNING: 2,
            ValidationSeverity.ERROR: 3,
            ValidationSeverity.CRITICAL: 4
        }
        return severity_scores.get(self.severity, 0)

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典格式"""
        return {
            "issue_id": self.issue_id,
            "rule_id": self.rule_id,
            "rule_name": self.rule_name,
            "category": self.category,
            "severity": self.severity.value,
            "description": self.description,
            "affected_factor": self.affected_factor,
            "affected_stocks": self.affected_stocks,
            "affected_date_range": [
                self.affected_date_range[0].isoformat(),
                self.affected_date_range[1].isoformat()
            ] if self.affected_date_range else None,
            "actual_value": self.actual_value,
            "expected_value": self.expected_value,
            "threshold_value": self.threshold_value,
            "recommendation": self.recommendation,
            "auto_fixable": self.auto_fixable,
            "detected_at": self.detected_at.isoformat(),
            "metadata": self.metadata
        }


@dataclass
class ValidationResult:
    """
    验证结果模型

    包含完整的验证结果信息，包括通过/失败状态和详细问题列表。
    """
    validation_id: str
    factor_name: str
    validation_date: datetime

    # 验证状态
    passed: bool
    validation_score: float  # [0, 1] 综合验证分数

    # 规则结果
    rule_results: Dict[str, Dict[str, Any]] = field(default_factory=dict)

    # 问题列表
    issues: List[ValidationIssue] = field(default_factory=list)

    # 统计信息
    total_rules: int = 0
    passed_rules: int = 0
    failed_rules: int = 0

    # 数据统计
    data_points_validated: int = 0
    invalid_data_points: int = 0

    # 元数据
    validation_config: Optional[Dict[str, Any]] = None
    execution_time_seconds: Optional[float] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        """计算派生指标"""
        if not self.validation_id:
            self.validation_id = str(uuid.uuid4())

        # 计算规则统计
        if self.rule_results:
            self.total_rules = len(self.rule_results)
            self.passed_rules = sum(1 for result in self.rule_results.values() if result.get('passed', False))
            self.failed_rules = self.total_rules - self.passed_rules

        # 计算综合分数（如果未提供）
        if self.validation_score == 0 and self.total_rules > 0:
            self.validation_score = self.passed_rules / self.total_rules

    def add_issue(self, issue: ValidationIssue) -> None:
        """添加验证问题"""
        self.issues.append(issue)

    def get_issues_by_severity(self, severity: ValidationSeverity) -> List[ValidationIssue]:
        """按严重性获取问题"""
        return [issue for issue in self.issues if issue.severity == severity]

    def get_critical_issues(self) -> List[ValidationIssue]:
        """获取关键问题"""
        return self.get_issues_by_severity(ValidationSeverity.CRITICAL)

    def get_error_issues(self) -> List[ValidationIssue]:
        """获取错误问题"""
        return self.get_issues_by_severity(ValidationSeverity.ERROR)

    def has_blocking_issues(self) -> bool:
        """检查是否有阻塞性问题"""
        return len(self.get_critical_issues()) > 0 or len(self.get_error_issues()) > 0

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典格式"""
        return {
            "validation_id": self.validation_id,
            "factor_name": self.factor_name,
            "validation_date": self.validation_date.isoformat(),
            "passed": self.passed,
            "validation_score": self.validation_score,
            "rule_results": self.rule_results,
            "issues": [issue.to_dict() for issue in self.issues],
            "total_rules": self.total_rules,
            "passed_rules": self.passed_rules,
            "failed_rules": self.failed_rules,
            "data_points_validated": self.data_points_validated,
            "invalid_data_points": self.invalid_data_points,
            "validation_config": self.validation_config,
            "execution_time_seconds": self.execution_time_seconds,
            "metadata": self.metadata
        }


@dataclass
class RiskFactor:
    """
    风险因子模型

    定义具体的风险因子及其度量方法。
    """
    factor_id: str
    factor_name: str
    category: RiskCategory

    # 风险度量
    current_value: float
    threshold_value: float
    max_acceptable_value: float

    # 风险级别
    risk_level: RiskLevel

    # 描述信息
    description: Optional[str] = None
    measurement_method: Optional[str] = None

    # 时间信息
    measured_at: datetime = field(default_factory=datetime.now)

    # 元数据
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        """初始化后处理"""
        if not self.factor_id:
            self.factor_id = str(uuid.uuid4())

    def is_exceeded(self) -> bool:
        """检查是否超过阈值"""
        return self.current_value > self.threshold_value

    def is_critical(self) -> bool:
        """检查是否达到关键水平"""
        return self.current_value > self.max_acceptable_value

    def get_excess_ratio(self) -> float:
        """获取超额比例"""
        if self.threshold_value == 0:
            return 0.0
        return max(0.0, (self.current_value - self.threshold_value) / self.threshold_value)

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典格式"""
        return {
            "factor_id": self.factor_id,
            "factor_name": self.factor_name,
            "category": self.category.value,
            "current_value": self.current_value,
            "threshold_value": self.threshold_value,
            "max_acceptable_value": self.max_acceptable_value,
            "risk_level": self.risk_level.value,
            "description": self.description,
            "measurement_method": self.measurement_method,
            "measured_at": self.measured_at.isoformat(),
            "is_exceeded": self.is_exceeded(),
            "is_critical": self.is_critical(),
            "excess_ratio": self.get_excess_ratio(),
            "metadata": self.metadata
        }


@dataclass
class RiskAssessment:
    """
    风险评估模型

    包含完整的风险评估结果，支持多维度风险分析。
    """
    assessment_id: str
    assessment_date: datetime

    # 总体风险
    overall_risk_score: float  # [0, 1]
    overall_risk_level: RiskLevel

    # 分类风险
    risk_factors: List[RiskFactor] = field(default_factory=list)
    risk_breakdown: Dict[str, float] = field(default_factory=dict)

    # 风险警告
    risk_warnings: List[str] = field(default_factory=list)
    critical_alerts: List[str] = field(default_factory=list)

    # 推荐措施
    recommendations: List[str] = field(default_factory=list)

    # 相关数据
    portfolio_value: Optional[float] = None
    max_loss_estimate: Optional[float] = None  # VaR
    confidence_level: float = 0.95

    # 过滤结果（用于信号风险评估）
    filtered_signals: Optional[List[Any]] = None
    rejected_signals: Optional[List[Any]] = None

    # 元数据
    assessment_config: Optional[Dict[str, Any]] = None
    execution_time_seconds: Optional[float] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        """初始化后处理"""
        if not self.assessment_id:
            self.assessment_id = str(uuid.uuid4())

        # 计算风险分解
        if self.risk_factors and not self.risk_breakdown:
            self.risk_breakdown = self._calculate_risk_breakdown()

        # 确定总体风险级别
        if not hasattr(self, '_overall_risk_level_set'):
            self.overall_risk_level = self._determine_overall_risk_level()

    def _calculate_risk_breakdown(self) -> Dict[str, float]:
        """计算风险分解"""
        breakdown = {}

        # 按类别分组
        category_factors = {}
        for factor in self.risk_factors:
            category = factor.category.value
            if category not in category_factors:
                category_factors[category] = []
            category_factors[category].append(factor)

        # 计算每个类别的风险
        for category, factors in category_factors.items():
            category_risk = max(factor.current_value for factor in factors)
            breakdown[category] = category_risk

        return breakdown

    def _determine_overall_risk_level(self) -> RiskLevel:
        """确定总体风险级别"""
        if self.overall_risk_score >= 0.8:
            return RiskLevel.CRITICAL
        elif self.overall_risk_score >= 0.6:
            return RiskLevel.HIGH
        elif self.overall_risk_score >= 0.3:
            return RiskLevel.MEDIUM
        else:
            return RiskLevel.LOW

    def add_risk_factor(self, risk_factor: RiskFactor) -> None:
        """添加风险因子"""
        self.risk_factors.append(risk_factor)

        # 重新计算风险分解
        self.risk_breakdown = self._calculate_risk_breakdown()

    def get_critical_risk_factors(self) -> List[RiskFactor]:
        """获取关键风险因子"""
        return [factor for factor in self.risk_factors if factor.is_critical()]

    def get_exceeded_risk_factors(self) -> List[RiskFactor]:
        """获取超阈值风险因子"""
        return [factor for factor in self.risk_factors if factor.is_exceeded()]

    def get_risk_factors_by_category(self, category: RiskCategory) -> List[RiskFactor]:
        """按类别获取风险因子"""
        return [factor for factor in self.risk_factors if factor.category == category]

    def has_blocking_risks(self) -> bool:
        """检查是否有阻塞性风险"""
        return (self.overall_risk_level in [RiskLevel.CRITICAL, RiskLevel.HIGH] or
                len(self.get_critical_risk_factors()) > 0)

    def calculate_var(self, confidence_level: float = 0.95) -> Optional[float]:
        """计算风险价值（VaR）"""
        if not self.portfolio_value:
            return None

        # 简化的VaR计算（实际实现应该使用更复杂的模型）
        var_ratio = self.overall_risk_score * (1 - confidence_level) * 2
        return self.portfolio_value * var_ratio

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典格式"""
        return {
            "assessment_id": self.assessment_id,
            "assessment_date": self.assessment_date.isoformat(),
            "overall_risk_score": self.overall_risk_score,
            "overall_risk_level": self.overall_risk_level.value,
            "risk_factors": [factor.to_dict() for factor in self.risk_factors],
            "risk_breakdown": self.risk_breakdown,
            "risk_warnings": self.risk_warnings,
            "critical_alerts": self.critical_alerts,
            "recommendations": self.recommendations,
            "portfolio_value": self.portfolio_value,
            "max_loss_estimate": self.max_loss_estimate,
            "confidence_level": self.confidence_level,
            "var_95": self.calculate_var(0.95),
            "var_99": self.calculate_var(0.99),
            "has_blocking_risks": self.has_blocking_risks(),
            "assessment_config": self.assessment_config,
            "execution_time_seconds": self.execution_time_seconds,
            "metadata": self.metadata
        }


# 预定义验证规则
def create_statistical_validation_rules() -> List[ValidationRule]:
    """创建统计验证规则"""
    rules = []

    # 正态性检验
    rules.append(ValidationRule(
        rule_id="stat_normality_test",
        rule_name="因子分布正态性检验",
        rule_type=ValidationRuleType.STATISTICAL,
        parameters={"test_method": "shapiro_wilk"},
        thresholds={"p_value_min": 0.05},
        severity=ValidationSeverity.WARNING,
        description="检验因子值是否符合正态分布"
    ))

    # 极值检测
    rules.append(ValidationRule(
        rule_id="stat_outlier_detection",
        rule_name="因子极值检测",
        rule_type=ValidationRuleType.STATISTICAL,
        parameters={"method": "iqr", "multiplier": 3.0},
        thresholds={"max_outlier_ratio": 0.05},
        severity=ValidationSeverity.ERROR,
        description="检测因子值中的异常极值"
    ))

    # 缺失值检查
    rules.append(ValidationRule(
        rule_id="stat_missing_data",
        rule_name="因子缺失值检查",
        rule_type=ValidationRuleType.STATISTICAL,
        parameters={},
        thresholds={"max_missing_ratio": 0.1},
        severity=ValidationSeverity.WARNING,
        description="检查因子数据的缺失比例"
    ))

    return rules


def create_correlation_validation_rules() -> List[ValidationRule]:
    """创建相关性验证规则"""
    rules = []

    # 因子间相关性
    rules.append(ValidationRule(
        rule_id="corr_factor_correlation",
        rule_name="因子间相关性检查",
        rule_type=ValidationRuleType.CORRELATION,
        parameters={"correlation_method": "pearson"},
        thresholds={"max_correlation": 0.8},
        severity=ValidationSeverity.WARNING,
        description="检查因子间的过高相关性"
    ))

    return rules


def create_risk_assessment_config() -> Dict[str, Any]:
    """创建风险评估配置"""
    return {
        "position_limits": {
            "max_single_position": 0.1,
            "max_sector_exposure": 0.3,
            "max_total_leverage": 1.0
        },
        "market_risk": {
            "max_beta": 1.5,
            "max_tracking_error": 0.15,
            "var_confidence": 0.95
        },
        "liquidity_risk": {
            "min_avg_volume": 1000000,
            "max_impact_cost": 0.02
        },
        "compliance": {
            "forbidden_stocks": [],
            "sector_limits": {
                "financial": 0.4,
                "technology": 0.5
            }
        }
    }