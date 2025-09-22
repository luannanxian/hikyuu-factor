"""
BaseAgent Unit Tests

基于真实Hikyuu框架的BaseAgent单元测试
不使用mock数据，测试Agent的核心功能
"""

import pytest
import asyncio
import time
from datetime import datetime, timedelta
from typing import Dict, Any
from unittest.mock import patch

from src.agents.base_agent import BaseAgent
from src.models.agent_models import (
    AgentType, MessageType, TaskStatus, Priority,
    AgentMessage, AgentResponse, TaskRequest, TaskResult
)


class TestAgent(BaseAgent):
    """测试用Agent实现"""

    def __init__(self, config=None):
        super().__init__(
            agent_type=AgentType.TEST,
            config=config or {}
        )
        self.init_called = False
        self.cleanup_called = False
        self.processed_tasks = []

    async def initialize(self) -> bool:
        """测试初始化方法"""
        self.init_called = True
        # 模拟初始化延迟
        await asyncio.sleep(0.1)
        return True

    async def process_task(self, task_request: TaskRequest) -> TaskResult:
        """测试任务处理方法"""
        self.processed_tasks.append(task_request)

        # 模拟任务处理
        if task_request.task_type == "test_success":
            result = TaskResult.create_pending(task_request.task_id)
            result.mark_completed(
                result_data={"result": "success", "input": task_request.task_data},
                message="Task completed successfully"
            )
            return result
        elif task_request.task_type == "test_error":
            result = TaskResult.create_pending(task_request.task_id)
            result.mark_failed(
                error_message="Simulated task error",
                error_code="TEST_ERROR"
            )
            return result
        else:
            result = TaskResult.create_pending(task_request.task_id)
            result.mark_completed(
                result_data={"processed": True},
                message="Default task processing"
            )
            return result

    async def cleanup(self) -> None:
        """测试清理方法"""
        self.cleanup_called = True
        await asyncio.sleep(0.05)


class TestBaseAgentCore:
    """BaseAgent核心功能测试"""

    @pytest.fixture
    def test_agent(self):
        """创建测试Agent实例"""
        config = {
            'host': '127.0.0.1',
            'port': 8999,
            'debug': True
        }
        return TestAgent(config)

    def test_agent_initialization(self, test_agent):
        """测试Agent基础初始化"""
        assert test_agent.agent_type == AgentType.TEST
        assert test_agent.agent_id is not None
        assert test_agent.agent_id.startswith("test_")
        assert test_agent.status == "initializing"
        assert test_agent.start_time is None
        assert test_agent.message_count == 0
        assert test_agent.task_count == 0
        assert test_agent.host == '127.0.0.1'
        assert test_agent.port == 8999
        assert test_agent.debug is True

    def test_agent_id_generation(self):
        """测试Agent ID生成"""
        agent1 = TestAgent()
        agent2 = TestAgent()

        # Agent ID应该是唯一的
        assert agent1.agent_id != agent2.agent_id
        assert agent1.agent_id.startswith("test_")
        assert agent2.agent_id.startswith("test_")

    def test_custom_agent_id(self):
        """测试自定义Agent ID"""
        custom_id = "custom_test_agent_001"
        agent = TestAgent({'agent_id': custom_id})
        # 注意：当前实现不支持通过config设置agent_id
        # 这个测试展示了当前的行为
        assert agent.agent_id != custom_id  # 当前实现会生成新ID

        # 直接设置agent_id的方式
        agent_with_custom_id = BaseAgent.__new__(TestAgent)
        agent_with_custom_id.__init__(config={})
        agent_with_custom_id.agent_id = custom_id
        assert agent_with_custom_id.agent_id == custom_id

    @pytest.mark.asyncio
    async def test_agent_startup_lifecycle(self, test_agent):
        """测试Agent启动生命周期"""
        # 初始状态
        assert test_agent.status == "initializing"
        assert not test_agent.init_called

        # 启动Agent
        await test_agent.start()

        # 验证启动后状态
        assert test_agent.status == "running"
        assert test_agent.init_called
        assert test_agent.start_time is not None
        assert isinstance(test_agent.start_time, datetime)

    @pytest.mark.asyncio
    async def test_agent_shutdown_lifecycle(self, test_agent):
        """测试Agent关闭生命周期"""
        # 先启动
        await test_agent.start()
        assert test_agent.status == "running"

        # 关闭Agent
        await test_agent.stop()

        # 验证关闭后状态
        assert test_agent.status == "stopped"
        assert test_agent.cleanup_called

    @pytest.mark.asyncio
    async def test_agent_initialization_failure(self):
        """测试Agent初始化失败情况"""
        class FailingAgent(BaseAgent):
            async def initialize(self) -> bool:
                return False  # 模拟初始化失败

            async def process_task(self, task_request: TaskRequest) -> TaskResult:
                pass

            async def cleanup(self) -> None:
                pass

        agent = FailingAgent(AgentType.TEST)

        with pytest.raises(RuntimeError, match="Agent initialization failed"):
            await agent.start()

        assert agent.status == "failed"

    def test_capabilities_default(self, test_agent):
        """测试默认能力列表"""
        capabilities = asyncio.run(test_agent.get_capabilities())

        expected_capabilities = [
            "message_processing",
            "task_execution",
            "health_monitoring",
            "api_interface"
        ]

        assert capabilities == expected_capabilities

    def test_status_information(self, test_agent):
        """测试状态信息获取"""
        status = test_agent.get_status()

        assert status["agent_id"] == test_agent.agent_id
        assert status["agent_type"] == "test"
        assert status["status"] == "initializing"
        assert status["start_time"] is None
        assert status["uptime_seconds"] == 0
        assert status["message_count"] == 0
        assert status["task_count"] == 0
        assert status["running_tasks"] == 0
        assert "config" in status


class TestBaseAgentMessageHandling:
    """BaseAgent消息处理测试"""

    @pytest.fixture
    def test_agent(self):
        return TestAgent()

    @pytest.mark.asyncio
    async def test_heartbeat_message_handling(self, test_agent):
        """测试心跳消息处理"""
        message = AgentMessage.create_heartbeat(
            sender=AgentType.DATA_MANAGER,
            receiver=AgentType.TEST
        )

        response = await test_agent.handle_agent_message(message)

        assert response.success is True
        assert response.message == "Heartbeat acknowledged"
        assert response.data["agent_id"] == test_agent.agent_id
        assert response.data["status"] == test_agent.status
        assert "timestamp" in response.data
        assert test_agent.message_count == 1

    @pytest.mark.asyncio
    async def test_notification_message_handling(self, test_agent):
        """测试通知消息处理"""
        message = AgentMessage.create_notification(
            sender=AgentType.DATA_MANAGER,
            receiver=AgentType.TEST,
            payload={"event": "data_updated", "timestamp": datetime.now().isoformat()}
        )

        response = await test_agent.handle_agent_message(message)

        assert response.success is True
        assert response.message == "Notification received"
        assert test_agent.message_count == 1

    @pytest.mark.asyncio
    async def test_expired_message_handling(self, test_agent):
        """测试过期消息处理"""
        # 创建已过期的消息
        message = AgentMessage(
            message_id="test_expired",
            message_type=MessageType.REQUEST,
            sender_agent=AgentType.DATA_MANAGER,
            receiver_agent=AgentType.TEST,
            payload={"action": "test"},
            timestamp=datetime.now() - timedelta(hours=1),  # 1小时前
            expires_at=datetime.now() - timedelta(minutes=30)  # 30分钟前过期
        )

        response = await test_agent.handle_agent_message(message)

        assert response.success is False
        assert response.error_code == "MESSAGE_EXPIRED"
        assert response.message == "Message expired"

    @pytest.mark.asyncio
    async def test_custom_action_handler(self, test_agent):
        """测试自定义动作处理器"""
        # 注册自定义处理器
        async def custom_handler(payload: Dict[str, Any]) -> Dict[str, Any]:
            return {
                "processed_data": payload.get("data", ""),
                "handler": "custom"
            }

        test_agent.register_message_handler("custom_action", custom_handler)

        # 发送请求消息
        message = AgentMessage.create_request(
            sender=AgentType.DATA_MANAGER,
            receiver=AgentType.TEST,
            payload={"action": "custom_action", "data": "test_data"}
        )

        response = await test_agent.handle_agent_message(message)

        assert response.success is True
        assert response.message == "Action custom_action completed"
        assert response.data["processed_data"] == "test_data"
        assert response.data["handler"] == "custom"

    @pytest.mark.asyncio
    async def test_unknown_action_handling(self, test_agent):
        """测试未知动作处理"""
        message = AgentMessage.create_request(
            sender=AgentType.DATA_MANAGER,
            receiver=AgentType.TEST,
            payload={"action": "unknown_action"}
        )

        response = await test_agent.handle_agent_message(message)

        assert response.success is False
        assert response.error_code == "UNKNOWN_ACTION"
        assert "Unknown action: unknown_action" in response.message

    @pytest.mark.asyncio
    async def test_message_handler_exception(self, test_agent):
        """测试消息处理器异常情况"""
        # 注册会抛出异常的处理器
        async def failing_handler(payload: Dict[str, Any]) -> Dict[str, Any]:
            raise ValueError("Handler error for testing")

        test_agent.register_message_handler("failing_action", failing_handler)

        message = AgentMessage.create_request(
            sender=AgentType.DATA_MANAGER,
            receiver=AgentType.TEST,
            payload={"action": "failing_action"}
        )

        response = await test_agent.handle_agent_message(message)

        assert response.success is False
        assert response.error_code == "ACTION_EXECUTION_ERROR"
        assert "Handler error for testing" in response.message


class TestBaseAgentTaskExecution:
    """BaseAgent任务执行测试"""

    @pytest.fixture
    def test_agent(self):
        return TestAgent()

    @pytest.mark.asyncio
    async def test_successful_task_execution(self, test_agent):
        """测试成功的任务执行"""
        task_request = TaskRequest(
            task_id="test_task_001",
            task_type="test_success",
            priority=Priority.MEDIUM,
            task_data={"input": "test_data"}
        )

        # 直接测试任务处理
        result = await test_agent.process_task(task_request)

        assert result.task_id == "test_task_001"
        assert result.status == TaskStatus.COMPLETED
        assert result.result_data["result"] == "success"
        assert result.result_data["input"]["input"] == "test_data"
        assert len(test_agent.processed_tasks) == 1

    @pytest.mark.asyncio
    async def test_failed_task_execution(self, test_agent):
        """测试失败的任务执行"""
        task_request = TaskRequest(
            task_id="test_task_002",
            task_type="test_error",
            priority=Priority.HIGH,
            task_data={"input": "error_data"}
        )

        result = await test_agent.process_task(task_request)

        assert result.task_id == "test_task_002"
        assert result.status == TaskStatus.FAILED
        assert result.error_message == "Simulated task error"
        assert result.error_code == "TEST_ERROR"

    @pytest.mark.asyncio
    async def test_task_execution_through_interface(self, test_agent):
        """测试通过_execute_task接口的任务执行"""
        await test_agent.start()

        task_request = TaskRequest(
            task_id="test_task_003",
            task_type="test_success",
            priority=Priority.LOW,
            task_data={"test": "data"}
        )

        # 将任务添加到运行队列
        test_agent.running_tasks[task_request.task_id] = TaskResult.create_pending(task_request.task_id)

        # 执行任务
        await test_agent._execute_task(task_request)

        # 验证任务被移动到历史记录
        assert task_request.task_id not in test_agent.running_tasks
        assert len(test_agent.task_history) == 1
        assert test_agent.task_history[0].task_id == task_request.task_id
        assert test_agent.task_history[0].status == TaskStatus.COMPLETED
        assert test_agent.task_count == 1

    @pytest.mark.asyncio
    async def test_task_history_management(self, test_agent):
        """测试任务历史记录管理"""
        await test_agent.start()

        # 添加大量任务到历史记录（超过限制）
        for i in range(1050):  # 超过1000的限制
            task_result = TaskResult.create_pending(f"task_{i}")
            task_result.mark_completed(result_data={"index": i}, message="Test task")
            test_agent.task_history.append(task_result)

        # 执行一个新任务来触发历史记录清理
        task_request = TaskRequest(
            task_id="cleanup_trigger_task",
            task_type="test_success",
            priority=Priority.MEDIUM,
            task_data={}
        )

        test_agent.running_tasks[task_request.task_id] = TaskResult.create_pending(task_request.task_id)
        await test_agent._execute_task(task_request)

        # 验证历史记录被清理到500条
        assert len(test_agent.task_history) == 501  # 500 + 新任务

    def test_task_result_creation_and_status_transitions(self):
        """测试TaskResult的创建和状态转换"""
        task_id = "status_test_task"

        # 创建待处理任务
        result = TaskResult.create_pending(task_id)
        assert result.task_id == task_id
        assert result.status == TaskStatus.PENDING
        assert result.created_at is not None
        assert result.started_at is None
        assert result.completed_at is None

        # 开始执行
        agent_id = "test_agent_001"
        result.start_execution(agent_id)
        assert result.status == TaskStatus.RUNNING
        assert result.executing_agent == agent_id
        assert result.started_at is not None

        # 标记完成
        result_data = {"output": "success"}
        message = "Task completed"
        result.mark_completed(result_data, message)
        assert result.status == TaskStatus.COMPLETED
        assert result.result_data == result_data
        assert result.message == message
        assert result.completed_at is not None

        # 计算执行时间
        execution_time = result.get_execution_time_seconds()
        assert execution_time > 0
        assert isinstance(execution_time, float)

    def test_task_result_failure_handling(self):
        """测试TaskResult失败处理"""
        result = TaskResult.create_pending("failure_test_task")
        result.start_execution("test_agent")

        # 标记失败
        error_msg = "Task processing failed"
        error_code = "PROCESSING_ERROR"
        result.mark_failed(error_msg, error_code)

        assert result.status == TaskStatus.FAILED
        assert result.error_message == error_msg
        assert result.error_code == error_code
        assert result.completed_at is not None


class TestBaseAgentAPIIntegration:
    """BaseAgent API集成测试"""

    @pytest.fixture
    def test_agent(self):
        return TestAgent({'port': 8998})  # 使用不同端口避免冲突

    def test_api_app_creation(self, test_agent):
        """测试FastAPI应用创建"""
        assert test_agent.app is not None
        assert test_agent.app.title == "Test Agent"
        assert "Agent for test" in test_agent.app.description
        assert test_agent.app.version == "1.0.0"

    def test_api_routes_setup(self, test_agent):
        """测试API路由设置"""
        # 获取所有路由路径
        routes = [route.path for route in test_agent.app.routes if hasattr(route, 'path')]

        expected_routes = [
            "/health",
            "/info",
            "/message",
            "/task",
            "/task/{task_id}",
            "/tasks"
        ]

        for expected_route in expected_routes:
            assert expected_route in routes

    @pytest.mark.asyncio
    async def test_health_check_endpoint(self, test_agent):
        """测试健康检查端点"""
        from fastapi.testclient import TestClient

        client = TestClient(test_agent.app)
        response = client.get("/health")

        assert response.status_code == 200
        data = response.json()

        assert data["agent_id"] == test_agent.agent_id
        assert data["agent_type"] == "test"
        assert data["status"] == "initializing"
        assert data["uptime_seconds"] == 0
        assert data["message_count"] == 0
        assert data["task_count"] == 0
        assert data["running_tasks"] == 0

    @pytest.mark.asyncio
    async def test_agent_info_endpoint(self, test_agent):
        """测试Agent信息端点"""
        from fastapi.testclient import TestClient

        client = TestClient(test_agent.app)
        response = client.get("/info")

        assert response.status_code == 200
        data = response.json()

        assert data["agent_id"] == test_agent.agent_id
        assert data["agent_type"] == "test"
        assert "config" in data
        assert "capabilities" in data
        assert "api_endpoints" in data

        # 验证能力列表
        expected_capabilities = [
            "message_processing",
            "task_execution",
            "health_monitoring",
            "api_interface"
        ]
        assert data["capabilities"] == expected_capabilities

    @pytest.mark.asyncio
    async def test_message_endpoint(self, test_agent):
        """测试消息处理端点"""
        from fastapi.testclient import TestClient

        client = TestClient(test_agent.app)

        # 测试心跳消息
        message_data = {
            "message_id": "test_msg_001",
            "message_type": "heartbeat",
            "sender_agent": "data_manager",
            "receiver_agent": "test",
            "payload": {},
            "timestamp": datetime.now().isoformat(),
            "expires_at": (datetime.now() + timedelta(minutes=5)).isoformat()
        }

        response = client.post("/message", json=message_data)

        assert response.status_code == 200
        data = response.json()

        assert data["success"] is True
        assert data["message"] == "Heartbeat acknowledged"
        assert data["data"]["agent_id"] == test_agent.agent_id

    @pytest.mark.asyncio
    async def test_task_submission_endpoint(self, test_agent):
        """测试任务提交端点"""
        from fastapi.testclient import TestClient

        client = TestClient(test_agent.app)

        task_data = {
            "task_id": "api_test_task",
            "task_type": "test_success",
            "priority": "medium",
            "task_data": {"api_test": True}
        }

        response = client.post("/task", json=task_data)

        assert response.status_code == 200
        data = response.json()

        assert data["task_id"] == "api_test_task"
        assert data["status"] == "accepted"
        assert "Task submitted for processing" in data["message"]

    @pytest.mark.asyncio
    async def test_task_status_endpoint(self, test_agent):
        """测试任务状态查询端点"""
        from fastapi.testclient import TestClient

        client = TestClient(test_agent.app)

        # 先创建一个任务
        task_result = TaskResult.create_pending("status_test_task")
        task_result.mark_completed(
            result_data={"test": "completed"},
            message="Test completed"
        )
        test_agent.task_history.append(task_result)

        # 查询任务状态
        response = client.get("/task/status_test_task")

        assert response.status_code == 200
        data = response.json()

        assert data["task_id"] == "status_test_task"
        assert data["status"] == "completed"
        assert data["result_data"]["test"] == "completed"

    @pytest.mark.asyncio
    async def test_task_status_not_found(self, test_agent):
        """测试任务状态查询不存在的任务"""
        from fastapi.testclient import TestClient

        client = TestClient(test_agent.app)
        response = client.get("/task/nonexistent_task")

        assert response.status_code == 404
        assert "Task not found" in response.json()["detail"]

    @pytest.mark.asyncio
    async def test_tasks_list_endpoint(self, test_agent):
        """测试任务列表端点"""
        from fastapi.testclient import TestClient

        client = TestClient(test_agent.app)

        # 添加一些测试数据
        running_task = TaskResult.create_pending("running_task")
        running_task.start_execution("test_agent")
        test_agent.running_tasks["running_task"] = running_task

        completed_task = TaskResult.create_pending("completed_task")
        completed_task.mark_completed({"result": "done"}, "Completed")
        test_agent.task_history.append(completed_task)

        response = client.get("/tasks")

        assert response.status_code == 200
        data = response.json()

        assert "running_tasks" in data
        assert "completed_tasks_count" in data
        assert "recent_tasks" in data

        assert "running_task" in data["running_tasks"]
        assert data["completed_tasks_count"] == 1
        assert len(data["recent_tasks"]) == 1
        assert data["recent_tasks"][0]["task_id"] == "completed_task"


class TestBaseAgentCommunication:
    """BaseAgent通信功能测试"""

    @pytest.fixture
    def test_agent(self):
        return TestAgent()

    @pytest.mark.asyncio
    async def test_send_message_to_agent(self, test_agent):
        """测试向其他Agent发送消息"""
        response = await test_agent.send_message_to_agent(
            target_agent=AgentType.DATA_MANAGER,
            action="test_action",
            payload={"data": "test_payload"}
        )

        assert response.success is True
        assert "Message sent to data_manager" in response.message
        assert "message_id" in response.data

    def test_message_handler_registration(self, test_agent):
        """测试消息处理器注册"""
        async def test_handler(payload: Dict[str, Any]) -> Dict[str, Any]:
            return {"handled": True, "payload": payload}

        # 注册处理器
        test_agent.register_message_handler("test_action", test_handler)

        # 验证处理器被注册
        assert "test_action" in test_agent.message_handlers
        assert test_agent.message_handlers["test_action"] == test_handler

    @pytest.mark.asyncio
    async def test_multiple_message_handlers(self, test_agent):
        """测试多个消息处理器"""
        handlers_called = []

        async def handler1(payload: Dict[str, Any]) -> Dict[str, Any]:
            handlers_called.append("handler1")
            return {"handler": "1", "data": payload}

        async def handler2(payload: Dict[str, Any]) -> Dict[str, Any]:
            handlers_called.append("handler2")
            return {"handler": "2", "data": payload}

        # 注册多个处理器
        test_agent.register_message_handler("action1", handler1)
        test_agent.register_message_handler("action2", handler2)

        # 发送不同动作的消息
        message1 = AgentMessage.create_request(
            sender=AgentType.DATA_MANAGER,
            receiver=AgentType.TEST,
            payload={"action": "action1", "test": "data1"}
        )

        message2 = AgentMessage.create_request(
            sender=AgentType.DATA_MANAGER,
            receiver=AgentType.TEST,
            payload={"action": "action2", "test": "data2"}
        )

        response1 = await test_agent.handle_agent_message(message1)
        response2 = await test_agent.handle_agent_message(message2)

        assert response1.success is True
        assert response2.success is True
        assert response1.data["handler"] == "1"
        assert response2.data["handler"] == "2"
        assert len(handlers_called) == 2
        assert "handler1" in handlers_called
        assert "handler2" in handlers_called


class TestBaseAgentPerformance:
    """BaseAgent性能测试"""

    @pytest.fixture
    def test_agent(self):
        return TestAgent()

    @pytest.mark.asyncio
    async def test_concurrent_message_processing(self, test_agent):
        """测试并发消息处理"""
        # 创建多个消息
        messages = []
        for i in range(10):
            message = AgentMessage.create_heartbeat(
                sender=AgentType.DATA_MANAGER,
                receiver=AgentType.TEST
            )
            messages.append(message)

        # 并发处理消息
        start_time = time.time()
        responses = await asyncio.gather(
            *[test_agent.handle_agent_message(msg) for msg in messages]
        )
        end_time = time.time()

        # 验证所有消息都被处理
        assert len(responses) == 10
        for response in responses:
            assert response.success is True
            assert response.message == "Heartbeat acknowledged"

        # 验证消息计数
        assert test_agent.message_count == 10

        # 验证处理时间合理（应该很快）
        processing_time = end_time - start_time
        assert processing_time < 1.0  # 应该在1秒内完成

    @pytest.mark.asyncio
    async def test_concurrent_task_processing(self, test_agent):
        """测试并发任务处理"""
        await test_agent.start()

        # 创建多个任务
        tasks = []
        for i in range(5):
            task_request = TaskRequest(
                task_id=f"concurrent_task_{i}",
                task_type="test_success",
                priority=Priority.MEDIUM,
                task_data={"index": i}
            )
            tasks.append(task_request)

        # 并发处理任务
        start_time = time.time()
        results = await asyncio.gather(
            *[test_agent.process_task(task) for task in tasks]
        )
        end_time = time.time()

        # 验证所有任务都被处理
        assert len(results) == 5
        for i, result in enumerate(results):
            assert result.status == TaskStatus.COMPLETED
            assert result.result_data["input"]["index"] == i

        # 验证处理时间
        processing_time = end_time - start_time
        assert processing_time < 2.0  # 应该在合理时间内完成

    @pytest.mark.asyncio
    async def test_memory_usage_with_large_task_history(self, test_agent):
        """测试大量任务历史的内存使用"""
        import sys

        # 记录初始内存使用
        initial_size = sys.getsizeof(test_agent.task_history)

        # 添加大量任务到历史记录
        for i in range(100):
            task_result = TaskResult.create_pending(f"memory_test_task_{i}")
            task_result.mark_completed(
                result_data={"index": i, "data": "x" * 100},  # 添加一些数据
                message=f"Task {i} completed"
            )
            test_agent.task_history.append(task_result)

        # 验证历史记录大小
        assert len(test_agent.task_history) == 100

        # 检查内存使用增长是否合理
        final_size = sys.getsizeof(test_agent.task_history)
        size_increase = final_size - initial_size

        # 内存增长应该与任务数量成正比，但不应该过大
        assert size_increase > 0
        assert size_increase < 1000000  # 不应该超过1MB

    def test_agent_state_consistency_under_load(self, test_agent):
        """测试高负载下Agent状态一致性"""
        initial_status = test_agent.get_status()

        # 模拟高负载操作
        for i in range(100):
            # 增加消息计数
            test_agent.message_count += 1

            # 添加任务历史
            task_result = TaskResult.create_pending(f"load_test_task_{i}")
            task_result.mark_completed({"result": i}, f"Task {i}")
            test_agent.task_history.append(task_result)

        final_status = test_agent.get_status()

        # 验证状态一致性
        assert final_status["agent_id"] == initial_status["agent_id"]
        assert final_status["agent_type"] == initial_status["agent_type"]
        assert final_status["message_count"] == 100
        assert len(test_agent.task_history) == 100


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])