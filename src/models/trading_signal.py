"""
Trading Signal Models
交易信号相关的数据模型，支持人工确认工作流程和审计追踪
"""

import enum
import hashlib
import json
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Union
from decimal import Decimal
import uuid


class SignalType(enum.Enum):
    """信号类型"""
    BUY = "buy"
    SELL = "sell"
    HOLD = "hold"
    REDUCE = "reduce"
    INCREASE = "increase"


class SignalStatus(enum.Enum):
    """信号状态"""
    PENDING = "pending"  # 待确认
    CONFIRMED = "confirmed"  # 已确认
    REJECTED = "rejected"  # 已拒绝
    EXECUTED = "executed"  # 已执行
    EXPIRED = "expired"  # 已过期
    CANCELLED = "cancelled"  # 已取消


class SignalPriority(enum.Enum):
    """信号优先级"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    URGENT = "urgent"


class RiskLevel(enum.Enum):
    """风险级别"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    EXTREME = "extreme"


@dataclass
class SignalExplanation:
    """信号解释"""
    factor_contributions: Dict[str, float]  # 因子贡献度
    reasoning: str  # 推理说明
    confidence_score: float  # 置信度分数 (0-1)
    supporting_evidence: List[str] = field(default_factory=list)  # 支撑证据
    risk_warnings: List[str] = field(default_factory=list)  # 风险警告
    model_version: Optional[str] = None  # 模型版本

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            "factor_contributions": self.factor_contributions,
            "reasoning": self.reasoning,
            "confidence_score": self.confidence_score,
            "supporting_evidence": self.supporting_evidence,
            "risk_warnings": self.risk_warnings,
            "model_version": self.model_version
        }


@dataclass
class RiskCheck:
    """风险检查结果"""
    risk_level: RiskLevel
    checks_passed: List[str]  # 通过的检查项
    checks_failed: List[str]  # 失败的检查项
    risk_metrics: Dict[str, float]  # 风险指标
    recommendations: List[str] = field(default_factory=list)  # 建议
    max_position_size: Optional[float] = None  # 最大仓位
    stop_loss_level: Optional[float] = None  # 止损位

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            "risk_level": self.risk_level.value,
            "checks_passed": self.checks_passed,
            "checks_failed": self.checks_failed,
            "risk_metrics": self.risk_metrics,
            "recommendations": self.recommendations,
            "max_position_size": self.max_position_size,
            "stop_loss_level": self.stop_loss_level
        }


@dataclass
class ConfirmationAction:
    """确认动作记录"""
    action: str  # "confirm", "reject", "modify"
    user_id: str
    timestamp: datetime
    reason: str
    modified_params: Optional[Dict[str, Any]] = None  # 修改的参数
    ip_address: Optional[str] = None
    user_agent: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            "action": self.action,
            "user_id": self.user_id,
            "timestamp": self.timestamp.isoformat(),
            "reason": self.reason,
            "modified_params": self.modified_params,
            "ip_address": self.ip_address,
            "user_agent": self.user_agent
        }


@dataclass
class TradingSignal:
    """交易信号主类"""
    signal_id: str
    stock_code: str
    signal_type: SignalType
    target_price: Decimal
    target_quantity: int
    created_at: datetime
    valid_until: datetime
    status: SignalStatus = SignalStatus.PENDING
    priority: SignalPriority = SignalPriority.MEDIUM

    # 信号生成相关
    strategy_id: str = ""
    factor_signals: Dict[str, float] = field(default_factory=dict)  # 各因子信号强度
    explanation: Optional[SignalExplanation] = None

    # 风险管理
    risk_check: Optional[RiskCheck] = None
    max_position_size: Optional[float] = None
    stop_loss_price: Optional[Decimal] = None
    take_profit_price: Optional[Decimal] = None

    # 确认工作流
    confirmation_required: bool = True
    confirmation_timeout_minutes: int = 30
    confirmation_actions: List[ConfirmationAction] = field(default_factory=list)
    confirmed_by: Optional[str] = None
    confirmed_at: Optional[datetime] = None

    # 执行相关
    executed_price: Optional[Decimal] = None
    executed_quantity: Optional[int] = None
    executed_at: Optional[datetime] = None
    execution_fees: Optional[Decimal] = None

    # 元数据
    tags: List[str] = field(default_factory=list)
    custom_fields: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        """初始化后处理"""
        if not self.signal_id:
            self.signal_id = self._generate_signal_id()

        # 检查过期时间
        if self.valid_until <= self.created_at:
            raise ValueError("valid_until must be after created_at")

    def _generate_signal_id(self) -> str:
        """生成信号ID"""
        content = f"{self.stock_code}_{self.signal_type.value}_{self.created_at.isoformat()}"
        return f"SIG_{hashlib.md5(content.encode()).hexdigest()[:12].upper()}"

    @property
    def is_expired(self) -> bool:
        """检查是否已过期"""
        return datetime.now() > self.valid_until

    @property
    def time_to_expiry(self) -> timedelta:
        """距离过期的时间"""
        return self.valid_until - datetime.now()

    @property
    def confirmation_deadline(self) -> datetime:
        """确认截止时间"""
        return self.created_at + timedelta(minutes=self.confirmation_timeout_minutes)

    @property
    def is_confirmation_expired(self) -> bool:
        """确认是否已超时"""
        if not self.confirmation_required:
            return False
        return datetime.now() > self.confirmation_deadline

    @property
    def needs_confirmation(self) -> bool:
        """是否需要确认"""
        return (self.confirmation_required and
                self.status == SignalStatus.PENDING and
                not self.is_confirmation_expired)

    def add_confirmation_action(self, action: ConfirmationAction) -> None:
        """添加确认动作"""
        self.confirmation_actions.append(action)

        if action.action == "confirm":
            self.status = SignalStatus.CONFIRMED
            self.confirmed_by = action.user_id
            self.confirmed_at = action.timestamp

            # 应用修改的参数
            if action.modified_params:
                self._apply_modifications(action.modified_params)

        elif action.action == "reject":
            self.status = SignalStatus.REJECTED

    def _apply_modifications(self, modifications: Dict[str, Any]) -> None:
        """应用确认时的修改"""
        if "target_price" in modifications:
            self.target_price = Decimal(str(modifications["target_price"]))

        if "target_quantity" in modifications:
            self.target_quantity = int(modifications["target_quantity"])

        if "stop_loss_price" in modifications:
            self.stop_loss_price = Decimal(str(modifications["stop_loss_price"]))

        if "take_profit_price" in modifications:
            self.take_profit_price = Decimal(str(modifications["take_profit_price"]))

    def confirm(self, user_id: str, reason: str = "",
                modifications: Optional[Dict[str, Any]] = None,
                ip_address: Optional[str] = None) -> bool:
        """确认信号"""
        if not self.needs_confirmation:
            return False

        action = ConfirmationAction(
            action="confirm",
            user_id=user_id,
            timestamp=datetime.now(),
            reason=reason,
            modified_params=modifications,
            ip_address=ip_address
        )

        self.add_confirmation_action(action)
        return True

    def reject(self, user_id: str, reason: str,
               ip_address: Optional[str] = None) -> bool:
        """拒绝信号"""
        if self.status != SignalStatus.PENDING:
            return False

        action = ConfirmationAction(
            action="reject",
            user_id=user_id,
            timestamp=datetime.now(),
            reason=reason,
            ip_address=ip_address
        )

        self.add_confirmation_action(action)
        return True

    def cancel(self, user_id: str, reason: str) -> bool:
        """取消信号"""
        if self.status in [SignalStatus.EXECUTED, SignalStatus.EXPIRED]:
            return False

        self.status = SignalStatus.CANCELLED

        action = ConfirmationAction(
            action="cancel",
            user_id=user_id,
            timestamp=datetime.now(),
            reason=reason
        )

        self.confirmation_actions.append(action)
        return True

    def execute(self, executed_price: Decimal, executed_quantity: int,
                execution_fees: Optional[Decimal] = None) -> bool:
        """执行信号"""
        if self.status != SignalStatus.CONFIRMED:
            return False

        self.executed_price = executed_price
        self.executed_quantity = executed_quantity
        self.executed_at = datetime.now()
        self.execution_fees = execution_fees or Decimal('0')
        self.status = SignalStatus.EXECUTED

        return True

    def update_expiry_status(self) -> None:
        """更新过期状态"""
        if self.is_expired and self.status == SignalStatus.PENDING:
            self.status = SignalStatus.EXPIRED
        elif self.is_confirmation_expired and self.status == SignalStatus.PENDING:
            self.status = SignalStatus.EXPIRED

    def get_execution_summary(self) -> Dict[str, Any]:
        """获取执行摘要"""
        if self.status != SignalStatus.EXECUTED:
            return {}

        pnl = None
        if self.executed_price and self.target_price:
            if self.signal_type == SignalType.BUY:
                pnl = float(self.target_price - self.executed_price) * self.executed_quantity
            elif self.signal_type == SignalType.SELL:
                pnl = float(self.executed_price - self.target_price) * self.executed_quantity

        return {
            "signal_id": self.signal_id,
            "stock_code": self.stock_code,
            "signal_type": self.signal_type.value,
            "target_price": float(self.target_price),
            "executed_price": float(self.executed_price) if self.executed_price else None,
            "target_quantity": self.target_quantity,
            "executed_quantity": self.executed_quantity,
            "execution_fees": float(self.execution_fees) if self.execution_fees else None,
            "estimated_pnl": pnl,
            "executed_at": self.executed_at.isoformat() if self.executed_at else None
        }

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            "signal_id": self.signal_id,
            "stock_code": self.stock_code,
            "signal_type": self.signal_type.value,
            "target_price": str(self.target_price),
            "target_quantity": self.target_quantity,
            "created_at": self.created_at.isoformat(),
            "valid_until": self.valid_until.isoformat(),
            "status": self.status.value,
            "priority": self.priority.value,
            "strategy_id": self.strategy_id,
            "factor_signals": self.factor_signals,
            "explanation": self.explanation.to_dict() if self.explanation else None,
            "risk_check": self.risk_check.to_dict() if self.risk_check else None,
            "max_position_size": self.max_position_size,
            "stop_loss_price": str(self.stop_loss_price) if self.stop_loss_price else None,
            "take_profit_price": str(self.take_profit_price) if self.take_profit_price else None,
            "confirmation_required": self.confirmation_required,
            "confirmation_timeout_minutes": self.confirmation_timeout_minutes,
            "confirmation_actions": [a.to_dict() for a in self.confirmation_actions],
            "confirmed_by": self.confirmed_by,
            "confirmed_at": self.confirmed_at.isoformat() if self.confirmed_at else None,
            "executed_price": str(self.executed_price) if self.executed_price else None,
            "executed_quantity": self.executed_quantity,
            "executed_at": self.executed_at.isoformat() if self.executed_at else None,
            "execution_fees": str(self.execution_fees) if self.execution_fees else None,
            "tags": self.tags,
            "custom_fields": self.custom_fields
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'TradingSignal':
        """从字典创建"""
        # 解析确认动作
        confirmation_actions = []
        for action_data in data.get("confirmation_actions", []):
            action_data_copy = action_data.copy()
            action_data_copy["timestamp"] = datetime.fromisoformat(action_data_copy["timestamp"])
            confirmation_actions.append(ConfirmationAction(**action_data_copy))

        # 解析信号解释
        explanation = None
        if data.get("explanation"):
            explanation = SignalExplanation(**data["explanation"])

        # 解析风险检查
        risk_check = None
        if data.get("risk_check"):
            risk_data = data["risk_check"].copy()
            risk_data["risk_level"] = RiskLevel(risk_data["risk_level"])
            risk_check = RiskCheck(**risk_data)

        return cls(
            signal_id=data["signal_id"],
            stock_code=data["stock_code"],
            signal_type=SignalType(data["signal_type"]),
            target_price=Decimal(data["target_price"]),
            target_quantity=data["target_quantity"],
            created_at=datetime.fromisoformat(data["created_at"]),
            valid_until=datetime.fromisoformat(data["valid_until"]),
            status=SignalStatus(data.get("status", "pending")),
            priority=SignalPriority(data.get("priority", "medium")),
            strategy_id=data.get("strategy_id", ""),
            factor_signals=data.get("factor_signals", {}),
            explanation=explanation,
            risk_check=risk_check,
            max_position_size=data.get("max_position_size"),
            stop_loss_price=Decimal(data["stop_loss_price"]) if data.get("stop_loss_price") else None,
            take_profit_price=Decimal(data["take_profit_price"]) if data.get("take_profit_price") else None,
            confirmation_required=data.get("confirmation_required", True),
            confirmation_timeout_minutes=data.get("confirmation_timeout_minutes", 30),
            confirmation_actions=confirmation_actions,
            confirmed_by=data.get("confirmed_by"),
            confirmed_at=datetime.fromisoformat(data["confirmed_at"]) if data.get("confirmed_at") else None,
            executed_price=Decimal(data["executed_price"]) if data.get("executed_price") else None,
            executed_quantity=data.get("executed_quantity"),
            executed_at=datetime.fromisoformat(data["executed_at"]) if data.get("executed_at") else None,
            execution_fees=Decimal(data["execution_fees"]) if data.get("execution_fees") else None,
            tags=data.get("tags", []),
            custom_fields=data.get("custom_fields", {})
        )

    def to_json(self) -> str:
        """转换为JSON字符串"""
        return json.dumps(self.to_dict(), indent=2, ensure_ascii=False)

    @classmethod
    def from_json(cls, json_str: str) -> 'TradingSignal':
        """从JSON字符串创建"""
        data = json.loads(json_str)
        return cls.from_dict(data)


class SignalBatch:
    """信号批次管理"""

    def __init__(self, batch_id: Optional[str] = None):
        self.batch_id = batch_id or str(uuid.uuid4())
        self.signals: List[TradingSignal] = []
        self.created_at = datetime.now()
        self.batch_status = "active"

    def add_signal(self, signal: TradingSignal) -> None:
        """添加信号到批次"""
        self.signals.append(signal)

    def get_signals_by_status(self, status: SignalStatus) -> List[TradingSignal]:
        """根据状态获取信号"""
        return [s for s in self.signals if s.status == status]

    def get_pending_confirmations(self) -> List[TradingSignal]:
        """获取待确认的信号"""
        return [s for s in self.signals if s.needs_confirmation]

    def get_expired_signals(self) -> List[TradingSignal]:
        """获取已过期的信号"""
        return [s for s in self.signals if s.is_expired or s.is_confirmation_expired]

    def update_all_expiry_status(self) -> None:
        """更新所有信号的过期状态"""
        for signal in self.signals:
            signal.update_expiry_status()

    def get_batch_summary(self) -> Dict[str, Any]:
        """获取批次摘要"""
        status_counts = {}
        for status in SignalStatus:
            status_counts[status.value] = len(self.get_signals_by_status(status))

        return {
            "batch_id": self.batch_id,
            "total_signals": len(self.signals),
            "status_breakdown": status_counts,
            "pending_confirmations": len(self.get_pending_confirmations()),
            "expired_signals": len(self.get_expired_signals()),
            "created_at": self.created_at.isoformat()
        }