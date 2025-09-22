"""
Audit and Compliance Data Models

审计和合规相关的数据模型，提供：
1. 操作审计和日志记录
2. 人工确认和决策跟踪
3. 工作流步骤记录
4. 合规性检查和报告

确保完整的审计跟踪和监管合规。
"""

from dataclasses import dataclass, field
from datetime import datetime, date
from typing import Dict, List, Optional, Any, Union
from enum import Enum
import uuid
import json


class AuditEventType(Enum):
    """审计事件类型"""
    USER_ACTION = "user_action"
    SYSTEM_ACTION = "system_action"
    DATA_ACCESS = "data_access"
    CALCULATION = "calculation"
    VALIDATION = "validation"
    RISK_CHECK = "risk_check"
    SIGNAL_GENERATION = "signal_generation"
    CONFIRMATION = "confirmation"
    ERROR = "error"
    SECURITY = "security"


class ConfirmationStatus(Enum):
    """确认状态"""
    PENDING = "pending"
    APPROVED = "approved"
    REJECTED = "rejected"
    PARTIAL = "partial"
    MODIFIED = "modified"
    EXPIRED = "expired"
    CANCELLED = "cancelled"


class WorkflowStepType(Enum):
    """工作流步骤类型"""
    DATA_PREPARATION = "data_preparation"
    FACTOR_CALCULATION = "factor_calculation"
    VALIDATION = "validation"
    RISK_ASSESSMENT = "risk_assessment"
    SIGNAL_GENERATION = "signal_generation"
    HUMAN_REVIEW = "human_review"
    EXECUTION = "execution"
    REPORTING = "reporting"


class ComplianceStatus(Enum):
    """合规状态"""
    COMPLIANT = "compliant"
    NON_COMPLIANT = "non_compliant"
    UNDER_REVIEW = "under_review"
    EXEMPT = "exempt"


@dataclass
class AuditEntry:
    """
    审计条目模型

    记录系统中的所有重要操作和事件。
    """
    audit_id: str
    event_type: AuditEventType
    event_name: str

    # 时间信息
    timestamp: datetime = field(default_factory=datetime.now)
    event_date: date = field(default_factory=date.today)

    # 用户信息
    user_id: Optional[str] = None
    user_role: Optional[str] = None
    session_id: Optional[str] = None

    # 系统信息
    component: Optional[str] = None  # 触发事件的系统组件
    version: Optional[str] = None
    environment: Optional[str] = None  # dev, test, prod

    # 事件详情
    description: str = ""
    event_data: Dict[str, Any] = field(default_factory=dict)

    # 影响信息
    affected_entities: List[str] = field(default_factory=list)
    affected_data: Optional[Dict[str, Any]] = None

    # 结果信息
    success: bool = True
    error_message: Optional[str] = None
    error_code: Optional[str] = None

    # 安全相关
    ip_address: Optional[str] = None
    user_agent: Optional[str] = None

    # 关联信息
    parent_audit_id: Optional[str] = None  # 父级审计记录
    workflow_id: Optional[str] = None
    correlation_id: Optional[str] = None

    # 合规标记
    compliance_relevant: bool = False
    retention_period_days: int = 2555  # 7年默认保留期

    # 元数据
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        """初始化后处理"""
        if not self.audit_id:
            self.audit_id = str(uuid.uuid4())

        # 设置事件日期
        if not hasattr(self, '_event_date_set'):
            self.event_date = self.timestamp.date()

    @classmethod
    def create_user_action(
        cls,
        user_id: str,
        action_name: str,
        description: str,
        **kwargs
    ) -> 'AuditEntry':
        """创建用户操作审计记录"""
        return cls(
            audit_id=str(uuid.uuid4()),
            event_type=AuditEventType.USER_ACTION,
            event_name=action_name,
            user_id=user_id,
            description=description,
            **kwargs
        )

    @classmethod
    def create_system_action(
        cls,
        component: str,
        action_name: str,
        description: str,
        **kwargs
    ) -> 'AuditEntry':
        """创建系统操作审计记录"""
        return cls(
            audit_id=str(uuid.uuid4()),
            event_type=AuditEventType.SYSTEM_ACTION,
            event_name=action_name,
            component=component,
            description=description,
            **kwargs
        )

    @classmethod
    def create_error_event(
        cls,
        component: str,
        error_message: str,
        error_code: Optional[str] = None,
        **kwargs
    ) -> 'AuditEntry':
        """创建错误事件审计记录"""
        return cls(
            audit_id=str(uuid.uuid4()),
            event_type=AuditEventType.ERROR,
            event_name="error_occurred",
            component=component,
            description=f"Error occurred: {error_message}",
            success=False,
            error_message=error_message,
            error_code=error_code,
            **kwargs
        )

    def add_affected_entity(self, entity_id: str, entity_type: str) -> None:
        """添加受影响的实体"""
        entity_info = f"{entity_type}:{entity_id}"
        if entity_info not in self.affected_entities:
            self.affected_entities.append(entity_info)

    def mark_compliance_relevant(self, retention_days: Optional[int] = None) -> None:
        """标记为合规相关"""
        self.compliance_relevant = True
        if retention_days:
            self.retention_period_days = retention_days

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典格式"""
        return {
            "audit_id": self.audit_id,
            "event_type": self.event_type.value,
            "event_name": self.event_name,
            "timestamp": self.timestamp.isoformat(),
            "event_date": self.event_date.isoformat(),
            "user_id": self.user_id,
            "user_role": self.user_role,
            "session_id": self.session_id,
            "component": self.component,
            "version": self.version,
            "environment": self.environment,
            "description": self.description,
            "event_data": self.event_data,
            "affected_entities": self.affected_entities,
            "affected_data": self.affected_data,
            "success": self.success,
            "error_message": self.error_message,
            "error_code": self.error_code,
            "ip_address": self.ip_address,
            "user_agent": self.user_agent,
            "parent_audit_id": self.parent_audit_id,
            "workflow_id": self.workflow_id,
            "correlation_id": self.correlation_id,
            "compliance_relevant": self.compliance_relevant,
            "retention_period_days": self.retention_period_days,
            "metadata": self.metadata
        }

    def to_json(self) -> str:
        """转换为JSON字符串"""
        return json.dumps(self.to_dict(), ensure_ascii=False, indent=2)


@dataclass
class ConfirmationRecord:
    """
    确认记录模型

    记录人工确认的完整过程和结果。
    """
    confirmation_id: str
    confirmation_type: str  # signal_confirmation, risk_override, etc.

    # 确认对象
    target_object_type: str  # trading_signal, risk_assessment, etc.
    target_object_id: str

    # 确认状态
    status: ConfirmationStatus
    decision: str  # approved, rejected, modified, etc.

    # 确认人员
    confirmer_id: str
    confirmer_role: str
    confirmer_name: Optional[str] = None

    # 时间信息
    created_at: datetime = field(default_factory=datetime.now)
    confirmed_at: Optional[datetime] = None
    expires_at: Optional[datetime] = None

    # 确认内容
    confirmation_message: Optional[str] = None
    user_comments: Optional[str] = None
    rejection_reason: Optional[str] = None

    # 修改信息（如果适用）
    original_values: Optional[Dict[str, Any]] = None
    modified_values: Optional[Dict[str, Any]] = None

    # 上下文信息
    context_data: Dict[str, Any] = field(default_factory=dict)
    risk_assessment: Optional[Dict[str, Any]] = None

    # 流程信息
    workflow_step_id: Optional[str] = None
    approval_chain: List[str] = field(default_factory=list)

    # 合规信息
    compliance_checked: bool = False
    compliance_notes: Optional[str] = None

    # 元数据
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        """初始化后处理"""
        if not self.confirmation_id:
            self.confirmation_id = str(uuid.uuid4())

    @classmethod
    def create_pending_confirmation(
        cls,
        target_object_type: str,
        target_object_id: str,
        confirmation_type: str,
        confirmer_id: str,
        expires_in_hours: int = 24,
        **kwargs
    ) -> 'ConfirmationRecord':
        """创建待确认记录"""
        from datetime import timedelta

        expires_at = datetime.now() + timedelta(hours=expires_in_hours)

        return cls(
            confirmation_id=str(uuid.uuid4()),
            confirmation_type=confirmation_type,
            target_object_type=target_object_type,
            target_object_id=target_object_id,
            status=ConfirmationStatus.PENDING,
            decision="pending",
            confirmer_id=confirmer_id,
            expires_at=expires_at,
            **kwargs
        )

    def approve(
        self,
        comments: Optional[str] = None,
        modified_values: Optional[Dict[str, Any]] = None
    ) -> None:
        """批准确认"""
        self.status = ConfirmationStatus.MODIFIED if modified_values else ConfirmationStatus.APPROVED
        self.decision = "approved"
        self.confirmed_at = datetime.now()
        self.user_comments = comments
        if modified_values:
            self.modified_values = modified_values

    def reject(self, reason: str, comments: Optional[str] = None) -> None:
        """拒绝确认"""
        self.status = ConfirmationStatus.REJECTED
        self.decision = "rejected"
        self.confirmed_at = datetime.now()
        self.rejection_reason = reason
        self.user_comments = comments

    def is_expired(self) -> bool:
        """检查是否过期"""
        if not self.expires_at:
            return False
        return datetime.now() > self.expires_at

    def mark_expired(self) -> None:
        """标记为过期"""
        if self.status == ConfirmationStatus.PENDING:
            self.status = ConfirmationStatus.EXPIRED
            self.decision = "expired"

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典格式"""
        return {
            "confirmation_id": self.confirmation_id,
            "confirmation_type": self.confirmation_type,
            "target_object_type": self.target_object_type,
            "target_object_id": self.target_object_id,
            "status": self.status.value,
            "decision": self.decision,
            "confirmer_id": self.confirmer_id,
            "confirmer_role": self.confirmer_role,
            "confirmer_name": self.confirmer_name,
            "created_at": self.created_at.isoformat(),
            "confirmed_at": self.confirmed_at.isoformat() if self.confirmed_at else None,
            "expires_at": self.expires_at.isoformat() if self.expires_at else None,
            "confirmation_message": self.confirmation_message,
            "user_comments": self.user_comments,
            "rejection_reason": self.rejection_reason,
            "original_values": self.original_values,
            "modified_values": self.modified_values,
            "context_data": self.context_data,
            "risk_assessment": self.risk_assessment,
            "workflow_step_id": self.workflow_step_id,
            "approval_chain": self.approval_chain,
            "compliance_checked": self.compliance_checked,
            "compliance_notes": self.compliance_notes,
            "is_expired": self.is_expired(),
            "metadata": self.metadata
        }


@dataclass
class WorkflowStep:
    """
    工作流步骤模型

    记录工作流中的每个步骤执行情况。
    """
    step_id: str
    workflow_id: str
    step_type: WorkflowStepType
    step_name: str

    # 执行顺序
    sequence_number: int
    parent_step_id: Optional[str] = None

    # 状态信息
    status: str = "pending"  # pending, running, completed, failed, skipped
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None

    # 执行结果
    success: bool = False
    result_data: Optional[Dict[str, Any]] = None
    error_message: Optional[str] = None

    # 执行者信息
    executor_type: str = "system"  # system, user, agent
    executor_id: Optional[str] = None

    # 输入输出
    input_data: Optional[Dict[str, Any]] = None
    output_data: Optional[Dict[str, Any]] = None

    # 质量指标
    execution_time_seconds: Optional[float] = None
    resource_usage: Optional[Dict[str, float]] = None

    # 审计信息
    audit_entries: List[str] = field(default_factory=list)  # 相关的审计记录ID

    # 元数据
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        """初始化后处理"""
        if not self.step_id:
            self.step_id = str(uuid.uuid4())

    def start_execution(self, executor_id: Optional[str] = None) -> None:
        """开始执行步骤"""
        self.status = "running"
        self.started_at = datetime.now()
        if executor_id:
            self.executor_id = executor_id

    def complete_execution(
        self,
        success: bool = True,
        result_data: Optional[Dict[str, Any]] = None,
        error_message: Optional[str] = None
    ) -> None:
        """完成步骤执行"""
        self.status = "completed" if success else "failed"
        self.completed_at = datetime.now()
        self.success = success
        self.result_data = result_data
        self.error_message = error_message

        # 计算执行时间
        if self.started_at:
            delta = self.completed_at - self.started_at
            self.execution_time_seconds = delta.total_seconds()

    def skip_execution(self, reason: str) -> None:
        """跳过步骤执行"""
        self.status = "skipped"
        self.completed_at = datetime.now()
        self.success = True
        self.metadata["skip_reason"] = reason

    def add_audit_entry(self, audit_id: str) -> None:
        """添加审计记录关联"""
        if audit_id not in self.audit_entries:
            self.audit_entries.append(audit_id)

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典格式"""
        return {
            "step_id": self.step_id,
            "workflow_id": self.workflow_id,
            "step_type": self.step_type.value,
            "step_name": self.step_name,
            "sequence_number": self.sequence_number,
            "parent_step_id": self.parent_step_id,
            "status": self.status,
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
            "success": self.success,
            "result_data": self.result_data,
            "error_message": self.error_message,
            "executor_type": self.executor_type,
            "executor_id": self.executor_id,
            "input_data": self.input_data,
            "output_data": self.output_data,
            "execution_time_seconds": self.execution_time_seconds,
            "resource_usage": self.resource_usage,
            "audit_entries": self.audit_entries,
            "metadata": self.metadata
        }


@dataclass
class ComplianceCheck:
    """
    合规检查模型

    记录合规性检查的结果和状态。
    """
    check_id: str
    check_type: str  # regulatory, internal_policy, risk_limit, etc.
    check_name: str

    # 检查对象
    target_entity_type: str
    target_entity_id: str

    # 检查结果
    status: ComplianceStatus
    passed: bool

    # 规则信息
    regulation_reference: Optional[str] = None
    policy_reference: Optional[str] = None
    rule_version: Optional[str] = None

    # 检查详情
    check_criteria: Dict[str, Any] = field(default_factory=dict)
    actual_values: Dict[str, Any] = field(default_factory=dict)
    violations: List[str] = field(default_factory=list)

    # 时间信息
    checked_at: datetime = field(default_factory=datetime.now)
    valid_until: Optional[datetime] = None

    # 检查人员
    checker_id: Optional[str] = None
    reviewer_id: Optional[str] = None

    # 豁免信息
    exemption_granted: bool = False
    exemption_reason: Optional[str] = None
    exemption_valid_until: Optional[datetime] = None

    # 元数据
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        """初始化后处理"""
        if not self.check_id:
            self.check_id = str(uuid.uuid4())

    def add_violation(self, violation_description: str) -> None:
        """添加违规描述"""
        if violation_description not in self.violations:
            self.violations.append(violation_description)
            self.passed = False
            if self.status == ComplianceStatus.COMPLIANT:
                self.status = ComplianceStatus.NON_COMPLIANT

    def grant_exemption(
        self,
        reason: str,
        valid_until: Optional[datetime] = None,
        grantor_id: Optional[str] = None
    ) -> None:
        """授予豁免"""
        self.exemption_granted = True
        self.exemption_reason = reason
        self.exemption_valid_until = valid_until
        self.status = ComplianceStatus.EXEMPT
        if grantor_id:
            self.metadata["exemption_grantor"] = grantor_id

    def is_exemption_valid(self) -> bool:
        """检查豁免是否有效"""
        if not self.exemption_granted:
            return False
        if not self.exemption_valid_until:
            return True
        return datetime.now() <= self.exemption_valid_until

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典格式"""
        return {
            "check_id": self.check_id,
            "check_type": self.check_type,
            "check_name": self.check_name,
            "target_entity_type": self.target_entity_type,
            "target_entity_id": self.target_entity_id,
            "status": self.status.value,
            "passed": self.passed,
            "regulation_reference": self.regulation_reference,
            "policy_reference": self.policy_reference,
            "rule_version": self.rule_version,
            "check_criteria": self.check_criteria,
            "actual_values": self.actual_values,
            "violations": self.violations,
            "checked_at": self.checked_at.isoformat(),
            "valid_until": self.valid_until.isoformat() if self.valid_until else None,
            "checker_id": self.checker_id,
            "reviewer_id": self.reviewer_id,
            "exemption_granted": self.exemption_granted,
            "exemption_reason": self.exemption_reason,
            "exemption_valid_until": self.exemption_valid_until.isoformat() if self.exemption_valid_until else None,
            "is_exemption_valid": self.is_exemption_valid(),
            "metadata": self.metadata
        }


# 工具函数
def create_workflow_audit_chain(
    workflow_id: str,
    steps: List[WorkflowStep],
    user_id: Optional[str] = None
) -> List[AuditEntry]:
    """为工作流创建审计链"""
    audit_entries = []

    # 工作流开始审计
    start_audit = AuditEntry.create_system_action(
        component="workflow_engine",
        action_name="workflow_started",
        description=f"Workflow {workflow_id} started",
        workflow_id=workflow_id,
        user_id=user_id,
        compliance_relevant=True
    )
    audit_entries.append(start_audit)

    # 为每个步骤创建审计记录
    for step in steps:
        step_audit = AuditEntry.create_system_action(
            component="workflow_engine",
            action_name=f"step_{step.step_type.value}",
            description=f"Workflow step {step.step_name} executed",
            workflow_id=workflow_id,
            event_data={
                "step_id": step.step_id,
                "step_type": step.step_type.value,
                "status": step.status,
                "success": step.success
            },
            success=step.success,
            compliance_relevant=True
        )
        audit_entries.append(step_audit)

        # 关联审计记录到步骤
        step.add_audit_entry(step_audit.audit_id)

    return audit_entries


def create_signal_confirmation_request(
    signal_id: str,
    signal_data: Dict[str, Any],
    risk_assessment: Dict[str, Any],
    confirmer_id: str
) -> ConfirmationRecord:
    """创建信号确认请求"""
    return ConfirmationRecord.create_pending_confirmation(
        target_object_type="trading_signal",
        target_object_id=signal_id,
        confirmation_type="signal_confirmation",
        confirmer_id=confirmer_id,
        confirmer_role="trader",
        context_data=signal_data,
        risk_assessment=risk_assessment,
        expires_in_hours=4  # 4小时内确认
    )


def audit_user_login(
    user_id: str,
    ip_address: str,
    user_agent: str,
    success: bool = True
) -> AuditEntry:
    """创建用户登录审计记录"""
    return AuditEntry.create_user_action(
        user_id=user_id,
        action_name="user_login",
        description=f"User {user_id} login {'successful' if success else 'failed'}",
        success=success,
        ip_address=ip_address,
        user_agent=user_agent,
        compliance_relevant=True,
        metadata={"security_event": True}
    )


def audit_data_access(
    user_id: str,
    data_type: str,
    data_id: str,
    access_type: str = "read"
) -> AuditEntry:
    """创建数据访问审计记录"""
    return AuditEntry(
        audit_id=str(uuid.uuid4()),
        event_type=AuditEventType.DATA_ACCESS,
        event_name=f"data_{access_type}",
        user_id=user_id,
        description=f"User {user_id} {access_type} {data_type} {data_id}",
        event_data={
            "data_type": data_type,
            "data_id": data_id,
            "access_type": access_type
        },
        compliance_relevant=True
    )