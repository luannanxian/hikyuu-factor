"""
Agent Models Unit Tests

基于真实Hikyuu框架的Agent模型单元测试
不使用mock数据，测试AgentMessage, AgentResponse, TaskRequest, TaskResult等核心模型
"""

import pytest
import json
from datetime import datetime, timedelta
from typing import Dict, List, Any

from src.models.agent_models import (
    MessageType, TaskStatus, Priority, AgentType,
    AgentMessage, AgentResponse, TaskRequest, TaskResult,
    create_heartbeat_message, create_batch_task_request,
    parse_agent_message, validate_message_flow
)


class TestEnums:
    """枚举类型单元测试"""

    def test_message_type_enum(self):
        """测试MessageType枚举"""
        assert MessageType.REQUEST.value == "request"
        assert MessageType.RESPONSE.value == "response"
        assert MessageType.NOTIFICATION.value == "notification"
        assert MessageType.ERROR.value == "error"
        assert MessageType.HEARTBEAT.value == "heartbeat"

        # 验证所有消息类型
        message_types = [mt.value for mt in MessageType]
        expected_types = ["request", "response", "notification", "error", "heartbeat"]
        assert set(message_types) == set(expected_types)

    def test_task_status_enum(self):
        """测试TaskStatus枚举"""
        assert TaskStatus.PENDING.value == "pending"
        assert TaskStatus.RUNNING.value == "running"
        assert TaskStatus.COMPLETED.value == "completed"
        assert TaskStatus.FAILED.value == "failed"
        assert TaskStatus.CANCELLED.value == "cancelled"
        assert TaskStatus.PAUSED.value == "paused"

    def test_priority_enum(self):
        """测试Priority枚举"""
        assert Priority.LOW.value == 1
        assert Priority.MEDIUM.value == 2
        assert Priority.HIGH.value == 3
        assert Priority.URGENT.value == 4

        # 验证优先级排序
        priorities = [p.value for p in Priority]
        assert priorities == sorted(priorities)

    def test_agent_type_enum(self):
        """测试AgentType枚举"""
        assert AgentType.DATA_MANAGER.value == "data_manager"
        assert AgentType.FACTOR_CALCULATION.value == "factor_calculation"
        assert AgentType.VALIDATION.value == "validation"
        assert AgentType.SIGNAL_GENERATION.value == "signal_generation"


class TestAgentMessage:
    """AgentMessage模型单元测试"""

    @pytest.fixture
    def basic_message(self):
        """创建基本的Agent消息"""
        return AgentMessage(
            message_id="test_msg_001",
            message_type=MessageType.REQUEST,
            sender_agent=AgentType.DATA_MANAGER,
            receiver_agent=AgentType.FACTOR_CALCULATION,
            payload={"action": "calculate_factor", "factor_name": "momentum_20d"}
        )

    def test_agent_message_initialization(self, basic_message):
        """测试AgentMessage基本初始化"""
        message = basic_message

        assert message.message_id == "test_msg_001"
        assert message.message_type == MessageType.REQUEST
        assert message.sender_agent == AgentType.DATA_MANAGER
        assert message.receiver_agent == AgentType.FACTOR_CALCULATION
        assert message.payload["action"] == "calculate_factor"
        assert message.priority == Priority.MEDIUM  # 默认值
        assert message.retry_count == 0
        assert message.max_retries == 3

    def test_agent_message_post_init(self):
        """测试AgentMessage初始化后处理"""
        # 测试自动生成message_id
        message = AgentMessage(
            message_id="",  # 空ID
            message_type=MessageType.REQUEST,
            sender_agent=AgentType.DATA_MANAGER
        )
        assert message.message_id != ""
        assert len(message.message_id) > 0

        # 测试REQUEST类型自动设置correlation_id
        request_msg = AgentMessage(
            message_id="req_001",
            message_type=MessageType.REQUEST,
            sender_agent=AgentType.DATA_MANAGER
        )
        assert request_msg.correlation_id == request_msg.message_id

    def test_create_request_message(self):
        """测试创建请求消息"""
        payload = {"action": "get_stock_data", "stock_code": "sh600000"}

        message = AgentMessage.create_request(
            sender=AgentType.FACTOR_CALCULATION,
            receiver=AgentType.DATA_MANAGER,
            payload=payload
        )

        assert message.message_type == MessageType.REQUEST
        assert message.sender_agent == AgentType.FACTOR_CALCULATION
        assert message.receiver_agent == AgentType.DATA_MANAGER
        assert message.payload == payload
        assert message.correlation_id == message.message_id

    def test_create_response_message(self):
        """测试创建响应消息"""
        # 先创建请求
        request = AgentMessage.create_request(
            sender=AgentType.FACTOR_CALCULATION,
            receiver=AgentType.DATA_MANAGER,
            payload={"action": "get_data"}
        )

        # 创建响应
        response_payload = {"result": "success", "data": [1, 2, 3]}
        response = AgentMessage.create_response(
            original_request=request,
            sender=AgentType.DATA_MANAGER,
            payload=response_payload
        )

        assert response.message_type == MessageType.RESPONSE
        assert response.sender_agent == AgentType.DATA_MANAGER
        assert response.receiver_agent == AgentType.FACTOR_CALCULATION
        assert response.payload == response_payload
        assert response.correlation_id == request.correlation_id
        assert response.reply_to == request.message_id

    def test_create_notification_message(self):
        """测试创建通知消息"""
        # 单播通知
        unicast_payload = {"event": "data_updated", "timestamp": datetime.now().isoformat()}
        unicast_msg = AgentMessage.create_notification(
            sender=AgentType.DATA_MANAGER,
            receiver=AgentType.FACTOR_CALCULATION,
            payload=unicast_payload
        )

        assert unicast_msg.message_type == MessageType.NOTIFICATION
        assert unicast_msg.sender_agent == AgentType.DATA_MANAGER
        assert unicast_msg.receiver_agent == AgentType.FACTOR_CALCULATION

        # 广播通知
        broadcast_payload = {"event": "system_maintenance", "scheduled_time": "2024-02-01T02:00:00"}
        broadcast_msg = AgentMessage.create_notification(
            sender=AgentType.DATA_MANAGER,
            payload=broadcast_payload
        )

        assert broadcast_msg.message_type == MessageType.NOTIFICATION
        assert broadcast_msg.receiver_agent is None  # 广播

    def test_create_error_message(self):
        """测试创建错误消息"""
        # 创建原始请求
        original_request = AgentMessage.create_request(
            sender=AgentType.FACTOR_CALCULATION,
            receiver=AgentType.DATA_MANAGER,
            payload={"action": "invalid_action"}
        )

        # 创建错误响应
        error_msg = AgentMessage.create_error(
            sender=AgentType.DATA_MANAGER,
            error_code="INVALID_ACTION",
            error_message="The requested action is not supported",
            original_message=original_request
        )

        assert error_msg.message_type == MessageType.ERROR
        assert error_msg.sender_agent == AgentType.DATA_MANAGER
        assert error_msg.receiver_agent == AgentType.FACTOR_CALCULATION
        assert error_msg.payload["error_code"] == "INVALID_ACTION"
        assert error_msg.payload["error_message"] == "The requested action is not supported"
        assert error_msg.payload["original_message_id"] == original_request.message_id
        assert error_msg.correlation_id == original_request.correlation_id

    def test_message_expiration(self):
        """测试消息过期功能"""
        # 未设置过期时间的消息
        msg_no_expiry = AgentMessage(
            message_id="no_expiry",
            message_type=MessageType.REQUEST,
            sender_agent=AgentType.DATA_MANAGER
        )
        assert msg_no_expiry.is_expired() is False

        # 未过期的消息
        future_time = datetime.now() + timedelta(hours=1)
        msg_not_expired = AgentMessage(
            message_id="not_expired",
            message_type=MessageType.REQUEST,
            sender_agent=AgentType.DATA_MANAGER,
            expires_at=future_time
        )
        assert msg_not_expired.is_expired() is False

        # 已过期的消息
        past_time = datetime.now() - timedelta(hours=1)
        msg_expired = AgentMessage(
            message_id="expired",
            message_type=MessageType.REQUEST,
            sender_agent=AgentType.DATA_MANAGER,
            expires_at=past_time
        )
        assert msg_expired.is_expired() is True

    def test_message_retry_functionality(self):
        """测试消息重试功能"""
        message = AgentMessage(
            message_id="retry_test",
            message_type=MessageType.REQUEST,
            sender_agent=AgentType.DATA_MANAGER,
            max_retries=2
        )

        # 初始状态
        assert message.retry_count == 0
        assert message.can_retry() is True

        # 第一次重试
        message.increment_retry()
        assert message.retry_count == 1
        assert message.can_retry() is True

        # 第二次重试
        message.increment_retry()
        assert message.retry_count == 2
        assert message.can_retry() is False

        # 超过重试次数
        message.increment_retry()
        assert message.retry_count == 3
        assert message.can_retry() is False

    def test_message_serialization(self, basic_message):
        """测试消息序列化"""
        # 转换为字典
        msg_dict = basic_message.to_dict()

        assert isinstance(msg_dict, dict)
        assert msg_dict["message_id"] == basic_message.message_id
        assert msg_dict["message_type"] == basic_message.message_type.value
        assert msg_dict["sender_agent"] == basic_message.sender_agent.value
        assert msg_dict["receiver_agent"] == basic_message.receiver_agent.value
        assert msg_dict["payload"] == basic_message.payload

        # 转换为JSON
        msg_json = basic_message.to_json()
        assert isinstance(msg_json, str)

        # 验证JSON可以被解析
        parsed_dict = json.loads(msg_json)
        assert parsed_dict["message_id"] == basic_message.message_id

    def test_message_deserialization(self, basic_message):
        """测试消息反序列化"""
        # 序列化后反序列化
        msg_dict = basic_message.to_dict()
        restored_message = AgentMessage.from_dict(msg_dict)

        assert restored_message.message_id == basic_message.message_id
        assert restored_message.message_type == basic_message.message_type
        assert restored_message.sender_agent == basic_message.sender_agent
        assert restored_message.receiver_agent == basic_message.receiver_agent
        assert restored_message.payload == basic_message.payload

    def test_message_with_metadata(self):
        """测试包含元数据的消息"""
        metadata = {
            "trace_id": "trace_001",
            "span_id": "span_001",
            "user_id": "user_123"
        }

        message = AgentMessage(
            message_id="metadata_test",
            message_type=MessageType.REQUEST,
            sender_agent=AgentType.DATA_MANAGER,
            metadata=metadata
        )

        assert message.metadata == metadata

        # 验证序列化包含元数据
        msg_dict = message.to_dict()
        assert msg_dict["metadata"] == metadata


class TestAgentResponse:
    """AgentResponse模型单元测试"""

    def test_success_response_creation(self):
        """测试成功响应创建"""
        data = {"result": "success", "count": 100}
        response = AgentResponse.success_response(
            message="Operation completed successfully",
            data=data,
            execution_time_ms=250.5
        )

        assert response.success is True
        assert response.message == "Operation completed successfully"
        assert response.data == data
        assert response.execution_time_ms == 250.5
        assert response.error_code is None

    def test_error_response_creation(self):
        """测试错误响应创建"""
        error_details = {"field": "factor_name", "issue": "required"}
        response = AgentResponse.error_response(
            message="Validation failed",
            error_code="VALIDATION_ERROR",
            error_details=error_details
        )

        assert response.success is False
        assert response.message == "Validation failed"
        assert response.error_code == "VALIDATION_ERROR"
        assert response.error_details == error_details
        assert response.data is None

    def test_response_with_pagination(self):
        """测试包含分页信息的响应"""
        response = AgentResponse.success_response(
            message="Data retrieved",
            data=[1, 2, 3, 4, 5],
            total_count=1000,
            page=1,
            page_size=5
        )

        assert response.total_count == 1000
        assert response.page == 1
        assert response.page_size == 5

    def test_response_serialization(self):
        """测试响应序列化"""
        response = AgentResponse.success_response(
            message="Test response",
            data={"key": "value"},
            execution_time_ms=100.0
        )

        # 转换为字典
        resp_dict = response.to_dict()
        assert resp_dict["success"] is True
        assert resp_dict["message"] == "Test response"
        assert resp_dict["data"]["key"] == "value"
        assert resp_dict["execution_time_ms"] == 100.0

        # 转换为JSON
        resp_json = response.to_json()
        assert isinstance(resp_json, str)
        parsed_dict = json.loads(resp_json)
        assert parsed_dict["success"] is True

    def test_error_response_serialization(self):
        """测试错误响应序列化"""
        response = AgentResponse.error_response(
            message="Error occurred",
            error_code="TEST_ERROR",
            error_details={"detail": "test"}
        )

        resp_dict = response.to_dict()
        assert resp_dict["success"] is False
        assert resp_dict["error_code"] == "TEST_ERROR"
        assert resp_dict["error_details"]["detail"] == "test"

    def test_response_with_metadata(self):
        """测试包含元数据的响应"""
        metadata = {"processing_node": "node_001", "cache_hit": True}
        response = AgentResponse.success_response(
            message="Success",
            metadata=metadata
        )

        assert response.metadata == metadata

        resp_dict = response.to_dict()
        assert resp_dict["metadata"] == metadata


class TestTaskRequest:
    """TaskRequest模型单元测试"""

    @pytest.fixture
    def basic_task_request(self):
        """创建基本的任务请求"""
        return TaskRequest(
            task_id="task_001",
            task_type="factor_calculation",
            task_name="计算动量因子",
            parameters={
                "factor_name": "momentum_20d",
                "stock_codes": ["sh600000", "sz000001"],
                "start_date": "2024-01-01",
                "end_date": "2024-01-31"
            }
        )

    def test_task_request_initialization(self, basic_task_request):
        """测试TaskRequest基本初始化"""
        task = basic_task_request

        assert task.task_id == "task_001"
        assert task.task_type == "factor_calculation"
        assert task.task_name == "计算动量因子"
        assert task.parameters["factor_name"] == "momentum_20d"
        assert task.priority == Priority.MEDIUM
        assert task.max_retries == 3
        assert task.created_at is not None

    def test_task_request_post_init(self):
        """测试TaskRequest初始化后处理"""
        # 测试自动生成task_id
        task = TaskRequest(
            task_id="",
            task_type="test",
            task_name="Test Task"
        )
        assert task.task_id != ""
        assert len(task.task_id) > 0

        # 测试默认回调事件
        assert "completed" in task.callback_events
        assert "failed" in task.callback_events

    def test_task_request_with_dependencies(self):
        """测试包含依赖关系的任务请求"""
        task = TaskRequest(
            task_id="dependent_task",
            task_type="validation",
            task_name="验证因子",
            depends_on=["task_001", "task_002"]
        )

        assert task.depends_on == ["task_001", "task_002"]

    def test_task_request_with_scheduling(self):
        """测试包含调度信息的任务请求"""
        scheduled_time = datetime.now() + timedelta(hours=1)
        deadline = datetime.now() + timedelta(hours=6)

        task = TaskRequest(
            task_id="scheduled_task",
            task_type="data_update",
            task_name="定时数据更新",
            scheduled_at=scheduled_time,
            deadline=deadline
        )

        assert task.scheduled_at == scheduled_time
        assert task.deadline == deadline

    def test_task_request_to_agent_message(self, basic_task_request):
        """测试任务请求转换为Agent消息"""
        message = basic_task_request.to_agent_message(
            sender=AgentType.FACTOR_CALCULATION,
            receiver=AgentType.DATA_MANAGER
        )

        assert isinstance(message, AgentMessage)
        assert message.message_type == MessageType.REQUEST
        assert message.sender_agent == AgentType.FACTOR_CALCULATION
        assert message.receiver_agent == AgentType.DATA_MANAGER
        assert "task_request" in message.payload

    def test_task_request_serialization(self, basic_task_request):
        """测试任务请求序列化"""
        task_dict = basic_task_request.to_dict()

        assert isinstance(task_dict, dict)
        assert task_dict["task_id"] == basic_task_request.task_id
        assert task_dict["task_type"] == basic_task_request.task_type
        assert task_dict["priority"] == basic_task_request.priority.value
        assert task_dict["parameters"] == basic_task_request.parameters

    def test_task_request_with_callback(self):
        """测试包含回调配置的任务请求"""
        task = TaskRequest(
            task_id="callback_task",
            task_type="signal_generation",
            task_name="生成交易信号",
            callback_url="http://callback.example.com/webhook",
            callback_events=["started", "progress", "completed", "failed"]
        )

        assert task.callback_url == "http://callback.example.com/webhook"
        assert task.callback_events == ["started", "progress", "completed", "failed"]

    def test_task_request_priority_types(self):
        """测试不同优先级的任务请求"""
        priorities = [Priority.LOW, Priority.MEDIUM, Priority.HIGH, Priority.URGENT]

        for priority in priorities:
            task = TaskRequest(
                task_id=f"task_{priority.name.lower()}",
                task_type="test",
                task_name=f"Test {priority.name}",
                priority=priority
            )
            assert task.priority == priority


class TestTaskResult:
    """TaskResult模型单元测试"""

    @pytest.fixture
    def basic_task_result(self):
        """创建基本的任务结果"""
        return TaskResult(
            task_id="result_001",
            status=TaskStatus.COMPLETED,
            message="Task completed successfully",
            result_data={"factor_values": [0.1, 0.2, 0.3]},
            execution_time_seconds=5.5
        )

    def test_task_result_initialization(self, basic_task_result):
        """测试TaskResult基本初始化"""
        result = basic_task_result

        assert result.task_id == "result_001"
        assert result.status == TaskStatus.COMPLETED
        assert result.message == "Task completed successfully"
        assert result.result_data["factor_values"] == [0.1, 0.2, 0.3]
        assert result.execution_time_seconds == 5.5

    def test_task_result_post_init_time_calculation(self):
        """测试TaskResult初始化后时间计算"""
        start_time = datetime.now()
        end_time = start_time + timedelta(seconds=10)

        result = TaskResult(
            task_id="time_test",
            status=TaskStatus.COMPLETED,
            message="Test",
            started_at=start_time,
            completed_at=end_time
        )

        # 应该自动计算执行时间
        assert result.execution_time_seconds == 10.0

    def test_create_pending_result(self):
        """测试创建待处理状态结果"""
        result = TaskResult.create_pending("pending_task")

        assert result.task_id == "pending_task"
        assert result.status == TaskStatus.PENDING
        assert result.message == "Task is pending"

    def test_create_running_result(self):
        """测试创建运行中状态结果"""
        result = TaskResult.create_running("running_task", current_step="数据加载")

        assert result.task_id == "running_task"
        assert result.status == TaskStatus.RUNNING
        assert result.message == "Task is running"
        assert result.current_step == "数据加载"
        assert result.started_at is not None

    def test_create_completed_result(self):
        """测试创建完成状态结果"""
        result_data = {"output": "success"}
        result = TaskResult.create_completed(
            "completed_task",
            result_data=result_data,
            message="Custom completion message"
        )

        assert result.task_id == "completed_task"
        assert result.status == TaskStatus.COMPLETED
        assert result.message == "Custom completion message"
        assert result.result_data == result_data
        assert result.progress_percentage == 100.0
        assert result.completed_at is not None

    def test_create_failed_result(self):
        """测试创建失败状态结果"""
        error_details = {"line": 42, "file": "calculation.py"}
        result = TaskResult.create_failed(
            "failed_task",
            error_message="Division by zero",
            error_code="MATH_ERROR",
            error_details=error_details
        )

        assert result.task_id == "failed_task"
        assert result.status == TaskStatus.FAILED
        assert result.error_message == "Division by zero"
        assert result.error_code == "MATH_ERROR"
        assert result.error_details == error_details
        assert result.completed_at is not None

    def test_update_progress(self):
        """测试更新进度"""
        result = TaskResult.create_running("progress_task")

        # 更新进度
        result.update_progress(25.0, current_step="数据验证", message="验证中...")

        assert result.progress_percentage == 25.0
        assert result.current_step == "数据验证"
        assert result.message == "验证中..."

        # 测试边界值
        result.update_progress(-10.0)  # 应该被限制为0
        assert result.progress_percentage == 0.0

        result.update_progress(150.0)  # 应该被限制为100
        assert result.progress_percentage == 100.0

    def test_mark_completed(self):
        """测试标记任务完成"""
        result = TaskResult.create_running("completion_test")
        completion_data = {"final_result": "success"}

        result.mark_completed(
            result_data=completion_data,
            message="Successfully processed all data"
        )

        assert result.status == TaskStatus.COMPLETED
        assert result.progress_percentage == 100.0
        assert result.message == "Successfully processed all data"
        assert result.result_data == completion_data
        assert result.completed_at is not None

    def test_mark_failed(self):
        """测试标记任务失败"""
        result = TaskResult.create_running("failure_test")
        error_details = {"exception": "ValueError", "context": "data processing"}

        result.mark_failed(
            error_message="Invalid input data format",
            error_code="DATA_FORMAT_ERROR",
            error_details=error_details
        )

        assert result.status == TaskStatus.FAILED
        assert result.error_message == "Invalid input data format"
        assert result.error_code == "DATA_FORMAT_ERROR"
        assert result.error_details == error_details
        assert result.message == "Task failed: Invalid input data format"
        assert result.completed_at is not None

    def test_task_result_to_agent_response(self):
        """测试任务结果转换为Agent响应"""
        # 成功的任务结果
        completed_result = TaskResult.create_completed("success_task", result_data={"value": 42})
        success_response = completed_result.to_agent_response()

        assert isinstance(success_response, AgentResponse)
        assert success_response.success is True

        # 失败的任务结果
        failed_result = TaskResult.create_failed("failed_task", "Error occurred", "TEST_ERROR")
        error_response = failed_result.to_agent_response()

        assert isinstance(error_response, AgentResponse)
        assert error_response.success is False
        assert error_response.error_code == "TEST_ERROR"

        # 运行中的任务结果
        running_result = TaskResult.create_running("running_task")
        running_response = running_result.to_agent_response()

        assert isinstance(running_response, AgentResponse)
        assert running_response.success is True

    def test_task_result_with_resource_usage(self):
        """测试包含资源使用信息的任务结果"""
        result = TaskResult(
            task_id="resource_test",
            status=TaskStatus.COMPLETED,
            message="Completed with resource monitoring",
            cpu_usage_percent=75.5,
            memory_usage_mb=512.0,
            disk_usage_mb=1024.0
        )

        assert result.cpu_usage_percent == 75.5
        assert result.memory_usage_mb == 512.0
        assert result.disk_usage_mb == 1024.0

    def test_task_result_with_statistics(self):
        """测试包含统计信息的任务结果"""
        result = TaskResult(
            task_id="stats_test",
            status=TaskStatus.COMPLETED,
            message="Completed with statistics",
            processed_items=1000,
            successful_items=950,
            failed_items=50
        )

        assert result.processed_items == 1000
        assert result.successful_items == 950
        assert result.failed_items == 50

    def test_task_result_serialization(self, basic_task_result):
        """测试任务结果序列化"""
        result_dict = basic_task_result.to_dict()

        assert isinstance(result_dict, dict)
        assert result_dict["task_id"] == basic_task_result.task_id
        assert result_dict["status"] == basic_task_result.status.value
        assert result_dict["message"] == basic_task_result.message
        assert result_dict["result_data"] == basic_task_result.result_data


class TestUtilityFunctions:
    """工具函数单元测试"""

    def test_create_heartbeat_message(self):
        """测试创建心跳消息"""
        heartbeat = create_heartbeat_message(AgentType.DATA_MANAGER)

        assert isinstance(heartbeat, AgentMessage)
        assert heartbeat.message_type == MessageType.HEARTBEAT
        assert heartbeat.sender_agent == AgentType.DATA_MANAGER
        assert "status" in heartbeat.payload
        assert heartbeat.payload["status"] == "alive"

    def test_create_batch_task_request(self):
        """测试创建批量任务请求"""
        individual_tasks = [
            TaskRequest(
                task_id="task_1",
                task_type="calculation",
                task_name="计算任务1"
            ),
            TaskRequest(
                task_id="task_2",
                task_type="validation",
                task_name="验证任务2"
            )
        ]

        batch_request = create_batch_task_request(
            task_requests=individual_tasks,
            batch_name="批量因子计算"
        )

        assert batch_request.task_type == "batch"
        assert batch_request.task_name == "批量因子计算"
        assert "batch_tasks" in batch_request.parameters
        assert len(batch_request.parameters["batch_tasks"]) == 2

    def test_parse_agent_message(self):
        """测试解析Agent消息"""
        # 创建原始消息
        original_message = AgentMessage.create_request(
            sender=AgentType.DATA_MANAGER,
            receiver=AgentType.FACTOR_CALCULATION,
            payload={"action": "test"}
        )

        # 序列化为JSON
        message_json = original_message.to_json()

        # 解析回消息对象
        parsed_message = parse_agent_message(message_json)

        assert isinstance(parsed_message, AgentMessage)
        assert parsed_message.message_id == original_message.message_id
        assert parsed_message.message_type == original_message.message_type
        assert parsed_message.sender_agent == original_message.sender_agent
        assert parsed_message.receiver_agent == original_message.receiver_agent

    def test_parse_agent_message_invalid_json(self):
        """测试解析无效JSON消息"""
        invalid_json = "{ invalid json }"

        with pytest.raises(ValueError, match="Invalid agent message format"):
            parse_agent_message(invalid_json)

    def test_validate_message_flow(self):
        """测试验证消息流"""
        # 创建正确的请求-响应流
        request = AgentMessage.create_request(
            sender=AgentType.FACTOR_CALCULATION,
            receiver=AgentType.DATA_MANAGER,
            payload={"action": "get_data"}
        )

        response = AgentMessage.create_response(
            original_request=request,
            sender=AgentType.DATA_MANAGER,
            payload={"result": "success"}
        )

        # 应该验证通过
        assert validate_message_flow(request, response) is True

        # 创建不匹配的响应
        wrong_response = AgentMessage.create_response(
            original_request=request,
            sender=AgentType.VALIDATION,  # 错误的发送者
            payload={"result": "success"}
        )
        wrong_response.correlation_id = "wrong_correlation"  # 错误的关联ID

        # 应该验证失败
        assert validate_message_flow(request, wrong_response) is False

    def test_validate_message_flow_wrong_types(self):
        """测试验证错误类型的消息流"""
        # 非请求消息
        notification = AgentMessage.create_notification(
            sender=AgentType.DATA_MANAGER,
            payload={"event": "update"}
        )

        response = AgentMessage.create_request(
            sender=AgentType.FACTOR_CALCULATION,
            receiver=AgentType.DATA_MANAGER,
            payload={"action": "test"}
        )

        assert validate_message_flow(notification, response) is False


class TestMessageFlowIntegration:
    """消息流集成测试"""

    def test_complete_request_response_flow(self):
        """测试完整的请求-响应流程"""
        # Step 1: 创建请求
        request_payload = {
            "action": "calculate_factor",
            "factor_name": "momentum_20d",
            "stock_codes": ["sh600000", "sz000001"],
            "date_range": ["2024-01-01", "2024-01-31"]
        }

        request = AgentMessage.create_request(
            sender=AgentType.FACTOR_CALCULATION,
            receiver=AgentType.DATA_MANAGER,
            payload=request_payload
        )

        # Step 2: 创建响应
        response_payload = {
            "status": "success",
            "factor_data": [
                {"stock_code": "sh600000", "factor_value": 0.15},
                {"stock_code": "sz000001", "factor_value": 0.08}
            ]
        }

        response = AgentMessage.create_response(
            original_request=request,
            sender=AgentType.DATA_MANAGER,
            payload=response_payload
        )

        # Step 3: 验证消息流
        assert validate_message_flow(request, response) is True

        # Step 4: 验证消息内容
        assert request.payload["action"] == "calculate_factor"
        assert response.payload["status"] == "success"
        assert len(response.payload["factor_data"]) == 2

    def test_error_handling_flow(self):
        """测试错误处理流程"""
        # Step 1: 创建可能出错的请求
        request = AgentMessage.create_request(
            sender=AgentType.FACTOR_CALCULATION,
            receiver=AgentType.DATA_MANAGER,
            payload={"action": "invalid_action"}
        )

        # Step 2: 创建错误响应
        error_message = AgentMessage.create_error(
            sender=AgentType.DATA_MANAGER,
            error_code="INVALID_ACTION",
            error_message="The requested action 'invalid_action' is not supported",
            original_message=request
        )

        # Step 3: 验证错误消息结构
        assert error_message.message_type == MessageType.ERROR
        assert error_message.correlation_id == request.correlation_id
        assert error_message.payload["error_code"] == "INVALID_ACTION"

    def test_notification_broadcast_flow(self):
        """测试通知广播流程"""
        # 系统广播通知
        system_notification = AgentMessage.create_notification(
            sender=AgentType.DATA_MANAGER,
            payload={
                "event": "market_data_updated",
                "update_time": datetime.now().isoformat(),
                "affected_stocks": ["sh600000", "sz000001", "sh600036"],
                "data_type": "realtime_quotes"
            }
        )

        # 验证广播消息
        assert system_notification.receiver_agent is None  # 广播
        assert system_notification.payload["event"] == "market_data_updated"
        assert len(system_notification.payload["affected_stocks"]) == 3

    def test_task_execution_flow(self):
        """测试任务执行流程"""
        # Step 1: 创建任务请求
        task_request = TaskRequest(
            task_id="integration_test_task",
            task_type="factor_calculation",
            task_name="集成测试因子计算",
            parameters={
                "factor_name": "rsi_14d",
                "stock_codes": ["sh600000"],
                "lookback_period": 14
            }
        )

        # Step 2: 转换为Agent消息
        task_message = task_request.to_agent_message(
            sender=AgentType.FACTOR_CALCULATION,
            receiver=AgentType.DATA_MANAGER
        )

        # Step 3: 创建任务结果
        task_result = TaskResult.create_completed(
            task_request.task_id,
            result_data={
                "factor_values": [65.5, 72.3, 58.9],
                "calculation_time": "2024-01-15T10:30:00",
                "data_quality": "high"
            }
        )

        # Step 4: 转换为Agent响应
        result_response = task_result.to_agent_response()

        # Step 5: 验证整个流程
        assert task_message.payload["task_request"]["task_id"] == task_request.task_id
        assert result_response.success is True
        assert result_response.data["result_data"]["calculation_time"] == "2024-01-15T10:30:00"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])