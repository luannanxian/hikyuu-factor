"""
Audit Log Models
审计日志相关的数据模型，支持哈希链验证和不可变性
"""

import enum
import hashlib
import json
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional, Any, Union
import uuid


class AuditEventType(enum.Enum):
    """审计事件类型"""
    USER_LOGIN = "user_login"
    USER_LOGOUT = "user_logout"
    SIGNAL_CREATED = "signal_created"
    SIGNAL_CONFIRMED = "signal_confirmed"
    SIGNAL_REJECTED = "signal_rejected"
    SIGNAL_EXECUTED = "signal_executed"
    FACTOR_CALCULATED = "factor_calculated"
    FACTOR_REGISTERED = "factor_registered"
    DATA_UPDATED = "data_updated"
    CONFIG_CHANGED = "config_changed"
    SYSTEM_ERROR = "system_error"
    SECURITY_ALERT = "security_alert"
    API_ACCESS = "api_access"
    DATABASE_OPERATION = "database_operation"
    FILE_ACCESS = "file_access"
    CUSTOM = "custom"


class AuditSeverity(enum.Enum):
    """审计严重性级别"""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


@dataclass
class AuditContext:
    """审计上下文信息"""
    user_id: Optional[str] = None
    session_id: Optional[str] = None
    ip_address: Optional[str] = None
    user_agent: Optional[str] = None
    request_id: Optional[str] = None
    correlation_id: Optional[str] = None
    source_system: Optional[str] = None
    additional_data: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            "user_id": self.user_id,
            "session_id": self.session_id,
            "ip_address": self.ip_address,
            "user_agent": self.user_agent,
            "request_id": self.request_id,
            "correlation_id": self.correlation_id,
            "source_system": self.source_system,
            "additional_data": self.additional_data
        }


@dataclass
class AuditLog:
    """审计日志记录"""
    log_id: str
    event_type: AuditEventType
    timestamp: datetime
    severity: AuditSeverity
    message: str
    details: Dict[str, Any] = field(default_factory=dict)
    context: Optional[AuditContext] = None

    # 哈希链相关
    previous_hash: Optional[str] = None
    current_hash: Optional[str] = None
    sequence_number: int = 0

    # 完整性验证
    is_verified: bool = False
    verification_timestamp: Optional[datetime] = None

    def __post_init__(self):
        """初始化后处理"""
        if not self.log_id:
            self.log_id = self._generate_log_id()

        if not self.current_hash:
            self.current_hash = self._calculate_hash()

    def _generate_log_id(self) -> str:
        """生成日志ID"""
        content = f"{self.event_type.value}_{self.timestamp.isoformat()}_{uuid.uuid4().hex[:8]}"
        return f"AUDIT_{hashlib.md5(content.encode()).hexdigest()[:16].upper()}"

    def _calculate_hash(self) -> str:
        """计算当前记录的哈希值"""
        # 构建用于哈希的内容
        hash_content = {
            "log_id": self.log_id,
            "event_type": self.event_type.value,
            "timestamp": self.timestamp.isoformat(),
            "severity": self.severity.value,
            "message": self.message,
            "details": self.details,
            "context": self.context.to_dict() if self.context else None,
            "previous_hash": self.previous_hash,
            "sequence_number": self.sequence_number
        }

        # 转换为确定性的JSON字符串
        content_str = json.dumps(hash_content, sort_keys=True, ensure_ascii=False)

        # 计算SHA-256哈希
        return hashlib.sha256(content_str.encode('utf-8')).hexdigest()

    def update_hash_chain(self, previous_hash: Optional[str], sequence_number: int) -> None:
        """更新哈希链信息"""
        self.previous_hash = previous_hash
        self.sequence_number = sequence_number
        self.current_hash = self._calculate_hash()

    def verify_integrity(self, expected_previous_hash: Optional[str] = None) -> bool:
        """验证记录完整性"""
        # 重新计算哈希值
        calculated_hash = self._calculate_hash()

        # 检查哈希值是否匹配
        hash_valid = calculated_hash == self.current_hash

        # 检查前一个哈希值（如果提供）
        chain_valid = True
        if expected_previous_hash is not None:
            chain_valid = self.previous_hash == expected_previous_hash

        # 更新验证状态
        self.is_verified = hash_valid and chain_valid
        self.verification_timestamp = datetime.now()

        return self.is_verified

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            "log_id": self.log_id,
            "event_type": self.event_type.value,
            "timestamp": self.timestamp.isoformat(),
            "severity": self.severity.value,
            "message": self.message,
            "details": self.details,
            "context": self.context.to_dict() if self.context else None,
            "previous_hash": self.previous_hash,
            "current_hash": self.current_hash,
            "sequence_number": self.sequence_number,
            "is_verified": self.is_verified,
            "verification_timestamp": self.verification_timestamp.isoformat() if self.verification_timestamp else None
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'AuditLog':
        """从字典创建"""
        context = None
        if data.get("context"):
            context = AuditContext(**data["context"])

        return cls(
            log_id=data["log_id"],
            event_type=AuditEventType(data["event_type"]),
            timestamp=datetime.fromisoformat(data["timestamp"]),
            severity=AuditSeverity(data["severity"]),
            message=data["message"],
            details=data.get("details", {}),
            context=context,
            previous_hash=data.get("previous_hash"),
            current_hash=data.get("current_hash"),
            sequence_number=data.get("sequence_number", 0),
            is_verified=data.get("is_verified", False),
            verification_timestamp=datetime.fromisoformat(data["verification_timestamp"]) if data.get("verification_timestamp") else None
        )

    def to_json(self) -> str:
        """转换为JSON字符串"""
        return json.dumps(self.to_dict(), indent=2, ensure_ascii=False)

    @classmethod
    def from_json(cls, json_str: str) -> 'AuditLog':
        """从JSON字符串创建"""
        data = json.loads(json_str)
        return cls.from_dict(data)


class AuditLogChain:
    """审计日志链管理器"""

    def __init__(self):
        self.logs: List[AuditLog] = []
        self.current_sequence = 0

    def add_log(self, log: AuditLog) -> None:
        """添加日志到链中"""
        # 获取前一个哈希值
        previous_hash = None
        if self.logs:
            previous_hash = self.logs[-1].current_hash

        # 更新哈希链信息
        self.current_sequence += 1
        log.update_hash_chain(previous_hash, self.current_sequence)

        # 添加到链中
        self.logs.append(log)

    def create_log(self, event_type: AuditEventType, message: str,
                   severity: AuditSeverity = AuditSeverity.INFO,
                   details: Optional[Dict[str, Any]] = None,
                   context: Optional[AuditContext] = None) -> AuditLog:
        """创建并添加新的审计日志"""
        log = AuditLog(
            log_id="",  # 将在__post_init__中生成
            event_type=event_type,
            timestamp=datetime.now(),
            severity=severity,
            message=message,
            details=details or {},
            context=context
        )

        self.add_log(log)
        return log

    def verify_chain(self) -> bool:
        """验证整个链的完整性"""
        if not self.logs:
            return True

        for i, log in enumerate(self.logs):
            expected_previous_hash = None
            if i > 0:
                expected_previous_hash = self.logs[i-1].current_hash

            if not log.verify_integrity(expected_previous_hash):
                return False

            # 检查序列号
            if log.sequence_number != i + 1:
                return False

        return True

    def find_tampered_logs(self) -> List[int]:
        """查找被篡改的日志记录索引"""
        tampered_indices = []

        for i, log in enumerate(self.logs):
            expected_previous_hash = None
            if i > 0:
                expected_previous_hash = self.logs[i-1].current_hash

            if not log.verify_integrity(expected_previous_hash):
                tampered_indices.append(i)

        return tampered_indices

    def get_logs_by_type(self, event_type: AuditEventType) -> List[AuditLog]:
        """根据事件类型获取日志"""
        return [log for log in self.logs if log.event_type == event_type]

    def get_logs_by_severity(self, severity: AuditSeverity) -> List[AuditLog]:
        """根据严重性级别获取日志"""
        return [log for log in self.logs if log.severity == severity]

    def get_logs_by_user(self, user_id: str) -> List[AuditLog]:
        """根据用户ID获取日志"""
        return [log for log in self.logs
                if log.context and log.context.user_id == user_id]

    def get_logs_in_timerange(self, start_time: datetime, end_time: datetime) -> List[AuditLog]:
        """获取时间范围内的日志"""
        return [log for log in self.logs
                if start_time <= log.timestamp <= end_time]

    def search_logs(self, query: str) -> List[AuditLog]:
        """搜索日志内容"""
        query_lower = query.lower()
        results = []

        for log in self.logs:
            if (query_lower in log.message.lower() or
                query_lower in str(log.details).lower() or
                query_lower in log.log_id.lower()):
                results.append(log)

        return results

    def export_chain(self, file_path: str, include_verification: bool = True) -> None:
        """导出审计链到文件"""
        # 如果需要，先验证链
        if include_verification:
            chain_valid = self.verify_chain()
        else:
            chain_valid = None

        export_data = {
            "chain_metadata": {
                "total_logs": len(self.logs),
                "current_sequence": self.current_sequence,
                "export_timestamp": datetime.now().isoformat(),
                "chain_valid": chain_valid
            },
            "logs": [log.to_dict() for log in self.logs]
        }

        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(export_data, f, indent=2, ensure_ascii=False)

    def import_chain(self, file_path: str) -> bool:
        """从文件导入审计链"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)

            # 清空当前链
            self.logs = []
            self.current_sequence = 0

            # 导入日志
            for log_data in data.get("logs", []):
                log = AuditLog.from_dict(log_data)
                self.logs.append(log)

            # 更新序列号
            if self.logs:
                self.current_sequence = max(log.sequence_number for log in self.logs)

            return True

        except Exception:
            return False

    def get_chain_summary(self) -> Dict[str, Any]:
        """获取链摘要信息"""
        if not self.logs:
            return {
                "total_logs": 0,
                "chain_valid": True,
                "first_log_time": None,
                "last_log_time": None,
                "event_type_distribution": {},
                "severity_distribution": {}
            }

        # 统计事件类型分布
        event_type_counts = {}
        for event_type in AuditEventType:
            event_type_counts[event_type.value] = len(self.get_logs_by_type(event_type))

        # 统计严重性分布
        severity_counts = {}
        for severity in AuditSeverity:
            severity_counts[severity.value] = len(self.get_logs_by_severity(severity))

        return {
            "total_logs": len(self.logs),
            "current_sequence": self.current_sequence,
            "chain_valid": self.verify_chain(),
            "first_log_time": self.logs[0].timestamp.isoformat(),
            "last_log_time": self.logs[-1].timestamp.isoformat(),
            "event_type_distribution": event_type_counts,
            "severity_distribution": severity_counts
        }


class AuditLogger:
    """审计日志记录器"""

    def __init__(self, chain: Optional[AuditLogChain] = None):
        self.chain = chain or AuditLogChain()

    def log_user_action(self, user_id: str, action: str, details: Optional[Dict[str, Any]] = None,
                        context: Optional[AuditContext] = None) -> AuditLog:
        """记录用户操作"""
        if not context:
            context = AuditContext(user_id=user_id)
        elif not context.user_id:
            context.user_id = user_id

        return self.chain.create_log(
            event_type=AuditEventType.API_ACCESS,
            message=f"User {user_id} performed action: {action}",
            details=details or {},
            context=context
        )

    def log_signal_event(self, signal_id: str, event_type: str, user_id: Optional[str] = None,
                         details: Optional[Dict[str, Any]] = None) -> AuditLog:
        """记录信号相关事件"""
        event_mapping = {
            "created": AuditEventType.SIGNAL_CREATED,
            "confirmed": AuditEventType.SIGNAL_CONFIRMED,
            "rejected": AuditEventType.SIGNAL_REJECTED,
            "executed": AuditEventType.SIGNAL_EXECUTED
        }

        audit_event_type = event_mapping.get(event_type, AuditEventType.CUSTOM)

        context = None
        if user_id:
            context = AuditContext(user_id=user_id)

        event_details = {"signal_id": signal_id}
        if details:
            event_details.update(details)

        return self.chain.create_log(
            event_type=audit_event_type,
            message=f"Signal {signal_id} {event_type}",
            details=event_details,
            context=context
        )

    def log_system_event(self, event: str, severity: AuditSeverity = AuditSeverity.INFO,
                         details: Optional[Dict[str, Any]] = None) -> AuditLog:
        """记录系统事件"""
        return self.chain.create_log(
            event_type=AuditEventType.SYSTEM_ERROR if severity in [AuditSeverity.ERROR, AuditSeverity.CRITICAL] else AuditEventType.API_ACCESS,
            message=f"System event: {event}",
            severity=severity,
            details=details or {}
        )

    def log_security_event(self, event: str, user_id: Optional[str] = None,
                           ip_address: Optional[str] = None,
                           details: Optional[Dict[str, Any]] = None) -> AuditLog:
        """记录安全事件"""
        context = AuditContext(user_id=user_id, ip_address=ip_address)

        return self.chain.create_log(
            event_type=AuditEventType.SECURITY_ALERT,
            message=f"Security event: {event}",
            severity=AuditSeverity.WARNING,
            details=details or {},
            context=context
        )