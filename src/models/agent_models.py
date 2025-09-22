"""
Agent Communication Data Models

Agent间通信的数据模型，提供：
1. 标准化的消息格式
2. 任务请求和响应模型
3. 状态同步和错误处理
4. RESTful API集成支持

所有模型都遵循OpenAPI 3.0.3规范，确保跨Agent兼容性。
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional, Any, Union
from enum import Enum
import uuid
import json


class MessageType(Enum):
    """消息类型枚举"""
    REQUEST = "request"
    RESPONSE = "response"
    NOTIFICATION = "notification"
    ERROR = "error"
    HEARTBEAT = "heartbeat"


class TaskStatus(Enum):
    """任务状态枚举"""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    PAUSED = "paused"


class Priority(Enum):
    """优先级枚举"""
    LOW = 1
    MEDIUM = 2
    HIGH = 3
    URGENT = 4


class AgentType(Enum):
    """Agent类型枚举"""
    DATA_MANAGER = "data_manager"
    FACTOR_CALCULATION = "factor_calculation"
    VALIDATION = "validation"
    SIGNAL_GENERATION = "signal_generation"


@dataclass
class AgentMessage:
    """
    Agent间通信的基础消息模型

    提供标准化的消息格式，支持请求-响应和通知模式。
    """
    message_id: str
    message_type: MessageType
    sender_agent: AgentType
    receiver_agent: Optional[AgentType] = None  # None表示广播

    # 消息内容
    payload: Dict[str, Any] = field(default_factory=dict)

    # 时间信息
    timestamp: datetime = field(default_factory=datetime.now)
    expires_at: Optional[datetime] = None

    # 相关性
    correlation_id: Optional[str] = None  # 用于关联请求和响应
    reply_to: Optional[str] = None       # 回复目标

    # 传输控制
    priority: Priority = Priority.MEDIUM
    retry_count: int = 0
    max_retries: int = 3

    # 元数据
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        """初始化后处理"""
        if not self.message_id:
            self.message_id = str(uuid.uuid4())

        # 设置默认的correlation_id
        if not self.correlation_id and self.message_type == MessageType.REQUEST:
            self.correlation_id = self.message_id

    @classmethod
    def create_request(
        cls,
        sender: AgentType,
        receiver: AgentType,
        payload: Dict[str, Any],
        **kwargs
    ) -> 'AgentMessage':
        """创建请求消息"""
        return cls(
            message_id=str(uuid.uuid4()),
            message_type=MessageType.REQUEST,
            sender_agent=sender,
            receiver_agent=receiver,
            payload=payload,
            **kwargs
        )

    @classmethod
    def create_response(
        cls,
        original_request: 'AgentMessage',
        sender: AgentType,
        payload: Dict[str, Any],
        **kwargs
    ) -> 'AgentMessage':
        """创建响应消息"""
        return cls(
            message_id=str(uuid.uuid4()),
            message_type=MessageType.RESPONSE,
            sender_agent=sender,
            receiver_agent=original_request.sender_agent,
            payload=payload,
            correlation_id=original_request.correlation_id,
            reply_to=original_request.message_id,
            **kwargs
        )

    @classmethod
    def create_notification(
        cls,
        sender: AgentType,
        payload: Dict[str, Any],
        receiver: Optional[AgentType] = None,
        **kwargs
    ) -> 'AgentMessage':
        """创建通知消息（广播或单播）"""
        return cls(
            message_id=str(uuid.uuid4()),
            message_type=MessageType.NOTIFICATION,
            sender_agent=sender,
            receiver_agent=receiver,
            payload=payload,
            **kwargs
        )

    @classmethod
    def create_error(
        cls,
        sender: AgentType,
        error_code: str,
        error_message: str,
        original_message: Optional['AgentMessage'] = None,
        **kwargs
    ) -> 'AgentMessage':
        """创建错误消息"""
        payload = {
            "error_code": error_code,
            "error_message": error_message,
            "original_message_id": original_message.message_id if original_message else None
        }

        return cls(
            message_id=str(uuid.uuid4()),
            message_type=MessageType.ERROR,
            sender_agent=sender,
            receiver_agent=original_message.sender_agent if original_message else None,
            payload=payload,
            correlation_id=original_message.correlation_id if original_message else None,
            **kwargs
        )

    def is_expired(self) -> bool:
        """检查消息是否过期"""
        if self.expires_at is None:
            return False
        return datetime.now() > self.expires_at

    def can_retry(self) -> bool:
        """检查是否可以重试"""
        return self.retry_count < self.max_retries

    def increment_retry(self) -> None:
        """增加重试计数"""
        self.retry_count += 1

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典格式"""
        return {
            "message_id": self.message_id,
            "message_type": self.message_type.value,
            "sender_agent": self.sender_agent.value,
            "receiver_agent": self.receiver_agent.value if self.receiver_agent else None,
            "payload": self.payload,
            "timestamp": self.timestamp.isoformat(),
            "expires_at": self.expires_at.isoformat() if self.expires_at else None,
            "correlation_id": self.correlation_id,
            "reply_to": self.reply_to,
            "priority": self.priority.value,
            "retry_count": self.retry_count,
            "max_retries": self.max_retries,
            "metadata": self.metadata
        }

    def to_json(self) -> str:
        """转换为JSON字符串"""
        return json.dumps(self.to_dict(), ensure_ascii=False, indent=2)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'AgentMessage':
        """从字典创建消息对象"""
        # 处理枚举字段
        data = data.copy()
        data['message_type'] = MessageType(data['message_type'])
        data['sender_agent'] = AgentType(data['sender_agent'])
        if data['receiver_agent']:
            data['receiver_agent'] = AgentType(data['receiver_agent'])
        data['priority'] = Priority(data['priority'])

        # 处理日期时间字段
        data['timestamp'] = datetime.fromisoformat(data['timestamp'])
        if data['expires_at']:
            data['expires_at'] = datetime.fromisoformat(data['expires_at'])

        return cls(**data)


@dataclass
class AgentResponse:
    """
    Agent响应模型

    标准化的Agent响应格式，包含状态、数据和错误信息。
    """
    success: bool
    message: str
    data: Optional[Any] = None
    error_code: Optional[str] = None
    error_details: Optional[Dict[str, Any]] = None

    # 执行信息
    execution_time_ms: Optional[float] = None
    timestamp: datetime = field(default_factory=datetime.now)

    # 分页信息（用于大数据集）
    total_count: Optional[int] = None
    page: Optional[int] = None
    page_size: Optional[int] = None

    # 元数据
    metadata: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def success_response(
        cls,
        message: str = "Success",
        data: Optional[Any] = None,
        **kwargs
    ) -> 'AgentResponse':
        """创建成功响应"""
        return cls(
            success=True,
            message=message,
            data=data,
            **kwargs
        )

    @classmethod
    def error_response(
        cls,
        message: str,
        error_code: Optional[str] = None,
        error_details: Optional[Dict[str, Any]] = None,
        **kwargs
    ) -> 'AgentResponse':
        """创建错误响应"""
        return cls(
            success=False,
            message=message,
            error_code=error_code,
            error_details=error_details,
            **kwargs
        )

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典格式"""
        result = {
            "success": self.success,
            "message": self.message,
            "timestamp": self.timestamp.isoformat()
        }

        if self.data is not None:
            result["data"] = self.data

        if not self.success:
            result["error_code"] = self.error_code
            result["error_details"] = self.error_details

        if self.execution_time_ms is not None:
            result["execution_time_ms"] = self.execution_time_ms

        # 分页信息
        if self.total_count is not None:
            result["pagination"] = {
                "total_count": self.total_count,
                "page": self.page,
                "page_size": self.page_size
            }

        if self.metadata:
            result["metadata"] = self.metadata

        return result

    def to_json(self) -> str:
        """转换为JSON字符串"""
        return json.dumps(self.to_dict(), ensure_ascii=False, indent=2)


@dataclass
class TaskRequest:
    """
    任务请求模型

    定义Agent间的任务请求格式，支持异步任务管理。
    """
    task_id: str
    task_type: str
    task_name: str

    # 任务参数
    parameters: Dict[str, Any] = field(default_factory=dict)
    input_data: Optional[Any] = None

    # 执行控制
    priority: Priority = Priority.MEDIUM
    timeout_seconds: Optional[int] = None
    max_retries: int = 3

    # 回调配置
    callback_url: Optional[str] = None
    callback_events: List[str] = field(default_factory=list)  # started, progress, completed, failed

    # 依赖关系
    depends_on: List[str] = field(default_factory=list)  # 依赖的任务ID

    # 调度信息
    scheduled_at: Optional[datetime] = None
    deadline: Optional[datetime] = None

    # 创建信息
    created_at: datetime = field(default_factory=datetime.now)
    created_by: Optional[str] = None

    # 元数据
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        """初始化后处理"""
        if not self.task_id:
            self.task_id = str(uuid.uuid4())

        # 设置默认回调事件
        if not self.callback_events:
            self.callback_events = ["completed", "failed"]

    def to_agent_message(
        self,
        sender: AgentType,
        receiver: AgentType
    ) -> AgentMessage:
        """转换为Agent消息"""
        return AgentMessage.create_request(
            sender=sender,
            receiver=receiver,
            payload={
                "task_request": self.to_dict()
            }
        )

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典格式"""
        return {
            "task_id": self.task_id,
            "task_type": self.task_type,
            "task_name": self.task_name,
            "parameters": self.parameters,
            "input_data": self.input_data,
            "priority": self.priority.value,
            "timeout_seconds": self.timeout_seconds,
            "max_retries": self.max_retries,
            "callback_url": self.callback_url,
            "callback_events": self.callback_events,
            "depends_on": self.depends_on,
            "scheduled_at": self.scheduled_at.isoformat() if self.scheduled_at else None,
            "deadline": self.deadline.isoformat() if self.deadline else None,
            "created_at": self.created_at.isoformat(),
            "created_by": self.created_by,
            "metadata": self.metadata
        }


@dataclass
class TaskResult:
    """
    任务结果模型

    包含任务执行的完整结果信息。
    """
    task_id: str
    status: TaskStatus
    message: str

    # 结果数据
    result_data: Optional[Any] = None
    output_files: List[str] = field(default_factory=list)

    # 执行信息
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    execution_time_seconds: Optional[float] = None

    # 进度信息
    progress_percentage: float = 0.0
    current_step: Optional[str] = None
    total_steps: Optional[int] = None

    # 错误信息
    error_code: Optional[str] = None
    error_message: Optional[str] = None
    error_details: Optional[Dict[str, Any]] = None
    stack_trace: Optional[str] = None

    # 资源使用
    cpu_usage_percent: Optional[float] = None
    memory_usage_mb: Optional[float] = None
    disk_usage_mb: Optional[float] = None

    # 输出统计
    processed_items: Optional[int] = None
    successful_items: Optional[int] = None
    failed_items: Optional[int] = None

    # 元数据
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        """计算派生字段"""
        if self.started_at and self.completed_at and not self.execution_time_seconds:
            delta = self.completed_at - self.started_at
            self.execution_time_seconds = delta.total_seconds()

    @classmethod
    def create_pending(cls, task_id: str) -> 'TaskResult':
        """创建待处理状态的任务结果"""
        return cls(
            task_id=task_id,
            status=TaskStatus.PENDING,
            message="Task is pending"
        )

    @classmethod
    def create_running(cls, task_id: str, current_step: Optional[str] = None) -> 'TaskResult':
        """创建运行中状态的任务结果"""
        return cls(
            task_id=task_id,
            status=TaskStatus.RUNNING,
            message="Task is running",
            started_at=datetime.now(),
            current_step=current_step
        )

    @classmethod
    def create_completed(
        cls,
        task_id: str,
        result_data: Optional[Any] = None,
        message: str = "Task completed successfully"
    ) -> 'TaskResult':
        """创建完成状态的任务结果"""
        return cls(
            task_id=task_id,
            status=TaskStatus.COMPLETED,
            message=message,
            result_data=result_data,
            completed_at=datetime.now(),
            progress_percentage=100.0
        )

    @classmethod
    def create_failed(
        cls,
        task_id: str,
        error_message: str,
        error_code: Optional[str] = None,
        error_details: Optional[Dict[str, Any]] = None
    ) -> 'TaskResult':
        """创建失败状态的任务结果"""
        return cls(
            task_id=task_id,
            status=TaskStatus.FAILED,
            message=f"Task failed: {error_message}",
            error_code=error_code,
            error_message=error_message,
            error_details=error_details,
            completed_at=datetime.now()
        )

    def update_progress(
        self,
        percentage: float,
        current_step: Optional[str] = None,
        message: Optional[str] = None
    ) -> None:
        """更新进度信息"""
        self.progress_percentage = max(0.0, min(100.0, percentage))
        if current_step:
            self.current_step = current_step
        if message:
            self.message = message

    def mark_completed(
        self,
        result_data: Optional[Any] = None,
        message: str = "Task completed successfully"
    ) -> None:
        """标记任务完成"""
        self.status = TaskStatus.COMPLETED
        self.completed_at = datetime.now()
        self.progress_percentage = 100.0
        self.message = message
        if result_data is not None:
            self.result_data = result_data

    def mark_failed(
        self,
        error_message: str,
        error_code: Optional[str] = None,
        error_details: Optional[Dict[str, Any]] = None
    ) -> None:
        """标记任务失败"""
        self.status = TaskStatus.FAILED
        self.completed_at = datetime.now()
        self.error_message = error_message
        self.error_code = error_code
        self.error_details = error_details
        self.message = f"Task failed: {error_message}"

    def to_agent_response(self) -> AgentResponse:
        """转换为Agent响应"""
        if self.status == TaskStatus.COMPLETED:
            return AgentResponse.success_response(
                message=self.message,
                data=self.to_dict()
            )
        elif self.status == TaskStatus.FAILED:
            return AgentResponse.error_response(
                message=self.message,
                error_code=self.error_code,
                error_details=self.error_details
            )
        else:
            return AgentResponse.success_response(
                message=self.message,
                data=self.to_dict()
            )

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典格式"""
        return {
            "task_id": self.task_id,
            "status": self.status.value,
            "message": self.message,
            "result_data": self.result_data,
            "output_files": self.output_files,
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
            "execution_time_seconds": self.execution_time_seconds,
            "progress_percentage": self.progress_percentage,
            "current_step": self.current_step,
            "total_steps": self.total_steps,
            "error_code": self.error_code,
            "error_message": self.error_message,
            "error_details": self.error_details,
            "stack_trace": self.stack_trace,
            "cpu_usage_percent": self.cpu_usage_percent,
            "memory_usage_mb": self.memory_usage_mb,
            "disk_usage_mb": self.disk_usage_mb,
            "processed_items": self.processed_items,
            "successful_items": self.successful_items,
            "failed_items": self.failed_items,
            "metadata": self.metadata
        }


# 工具函数
def create_heartbeat_message(agent: AgentType) -> AgentMessage:
    """创建心跳消息"""
    return AgentMessage(
        message_id=str(uuid.uuid4()),
        message_type=MessageType.HEARTBEAT,
        sender_agent=agent,
        payload={
            "status": "alive",
            "timestamp": datetime.now().isoformat()
        }
    )


def create_batch_task_request(
    task_requests: List[TaskRequest],
    batch_name: str,
    **kwargs
) -> TaskRequest:
    """创建批量任务请求"""
    batch_id = str(uuid.uuid4())

    return TaskRequest(
        task_id=batch_id,
        task_type="batch",
        task_name=batch_name,
        parameters={
            "batch_tasks": [req.to_dict() for req in task_requests],
            "parallel_execution": kwargs.get("parallel_execution", True),
            "fail_fast": kwargs.get("fail_fast", False)
        },
        **kwargs
    )


def parse_agent_message(message_json: str) -> AgentMessage:
    """解析JSON格式的Agent消息"""
    try:
        data = json.loads(message_json)
        return AgentMessage.from_dict(data)
    except (json.JSONDecodeError, KeyError, ValueError) as e:
        raise ValueError(f"Invalid agent message format: {e}")


def validate_message_flow(
    request: AgentMessage,
    response: AgentMessage
) -> bool:
    """验证请求-响应消息流的一致性"""
    if request.message_type != MessageType.REQUEST:
        return False

    if response.message_type != MessageType.RESPONSE:
        return False

    if response.correlation_id != request.correlation_id:
        return False

    if response.reply_to != request.message_id:
        return False

    return True