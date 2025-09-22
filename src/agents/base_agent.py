"""
Base Agent Class

Agent架构的基础类，提供：
1. Agent生命周期管理
2. 消息处理和通信
3. RESTful API框架
4. 健康检查和监控
5. 配置管理和日志
"""

import asyncio
import logging
import uuid
from datetime import datetime
from typing import Dict, List, Optional, Any, Callable
from abc import ABC, abstractmethod
from fastapi import FastAPI, HTTPException, BackgroundTasks
import uvicorn

from models.agent_models import (
    AgentMessage, AgentResponse, TaskRequest, TaskResult,
    AgentType, MessageType, TaskStatus, Priority
)
from models.audit_models import AuditEntry, AuditEventType


class BaseAgent(ABC):
    """
    Agent基类

    所有Agent的基础实现，提供通用功能和标准接口。
    """

    def __init__(
        self,
        agent_type: AgentType,
        agent_id: Optional[str] = None,
        config: Optional[Dict[str, Any]] = None
    ):
        self.agent_type = agent_type
        self.agent_id = agent_id or f"{agent_type.value}_{uuid.uuid4().hex[:8]}"
        self.config = config or {}

        # 基础配置
        self.host = self.config.get('host', '0.0.0.0')
        self.port = self.config.get('port', 8000)
        self.debug = self.config.get('debug', False)

        # 状态管理
        self.status = "initializing"
        self.start_time = None
        self.message_count = 0
        self.task_count = 0

        # 日志配置
        self.logger = logging.getLogger(f"{self.agent_type.value}.{self.agent_id}")

        # FastAPI应用
        self.app = FastAPI(
            title=f"{self.agent_type.value.title()} Agent",
            description=f"Agent for {self.agent_type.value}",
            version="1.0.0"
        )

        # 任务管理
        self.running_tasks: Dict[str, TaskResult] = {}
        self.task_history: List[TaskResult] = []

        # 消息处理器注册
        self.message_handlers: Dict[str, Callable] = {}

        # 初始化API路由
        self._setup_api_routes()

        self.logger.info(f"Agent {self.agent_id} initialized")

    @abstractmethod
    async def initialize(self) -> bool:
        """
        Agent初始化方法

        子类必须实现此方法进行特定的初始化操作
        """
        pass

    @abstractmethod
    async def process_task(self, task_request: TaskRequest) -> TaskResult:
        """
        处理任务请求

        子类必须实现此方法处理特定的任务类型
        """
        pass

    @abstractmethod
    async def cleanup(self) -> None:
        """
        Agent清理方法

        子类实现资源清理逻辑
        """
        pass

    def _setup_api_routes(self) -> None:
        """设置API路由"""

        @self.app.get("/health")
        async def health_check() -> Dict[str, Any]:
            """健康检查接口"""
            return {
                "agent_id": self.agent_id,
                "agent_type": self.agent_type.value,
                "status": self.status,
                "uptime_seconds": (datetime.now() - self.start_time).total_seconds() if self.start_time else 0,
                "message_count": self.message_count,
                "task_count": self.task_count,
                "running_tasks": len(self.running_tasks)
            }

        @self.app.get("/info")
        async def agent_info() -> Dict[str, Any]:
            """Agent信息接口"""
            return {
                "agent_id": self.agent_id,
                "agent_type": self.agent_type.value,
                "config": {k: v for k, v in self.config.items() if not k.startswith('_')},
                "capabilities": await self.get_capabilities(),
                "api_endpoints": [
                    {"path": route.path, "methods": list(route.methods)}
                    for route in self.app.routes
                    if hasattr(route, 'methods')
                ]
            }

        @self.app.post("/message")
        async def handle_message(message_data: dict) -> Dict[str, Any]:
            """消息处理接口"""
            try:
                message = AgentMessage.from_dict(message_data)
                response = await self.handle_agent_message(message)
                return response.to_dict()
            except Exception as e:
                self.logger.error(f"Message handling failed: {e}")
                return AgentResponse.error_response(
                    message=str(e),
                    error_code="MESSAGE_PROCESSING_ERROR"
                ).to_dict()

        @self.app.post("/task")
        async def submit_task(task_data: dict, background_tasks: BackgroundTasks) -> Dict[str, Any]:
            """任务提交接口"""
            try:
                task_request = TaskRequest(**task_data)

                # 创建任务结果
                task_result = TaskResult.create_pending(task_request.task_id)
                self.running_tasks[task_request.task_id] = task_result

                # 异步执行任务
                background_tasks.add_task(self._execute_task, task_request)

                return {
                    "task_id": task_request.task_id,
                    "status": "accepted",
                    "message": "Task submitted for processing"
                }
            except Exception as e:
                self.logger.error(f"Task submission failed: {e}")
                raise HTTPException(status_code=400, detail=str(e))

        @self.app.get("/task/{task_id}")
        async def get_task_status(task_id: str) -> Dict[str, Any]:
            """任务状态查询接口"""
            if task_id in self.running_tasks:
                return self.running_tasks[task_id].to_dict()

            # 在历史记录中查找
            for task in self.task_history:
                if task.task_id == task_id:
                    return task.to_dict()

            raise HTTPException(status_code=404, detail="Task not found")

        @self.app.get("/tasks")
        async def list_tasks() -> Dict[str, Any]:
            """任务列表接口"""
            return {
                "running_tasks": {
                    task_id: result.to_dict()
                    for task_id, result in self.running_tasks.items()
                },
                "completed_tasks_count": len(self.task_history),
                "recent_tasks": [
                    task.to_dict() for task in self.task_history[-10:]
                ]
            }

    async def get_capabilities(self) -> List[str]:
        """获取Agent能力列表"""
        return [
            "message_processing",
            "task_execution",
            "health_monitoring",
            "api_interface"
        ]

    async def start(self) -> None:
        """启动Agent"""
        try:
            self.logger.info(f"Starting agent {self.agent_id}...")

            # 执行子类初始化
            if await self.initialize():
                self.status = "running"
                self.start_time = datetime.now()
                self.logger.info(f"Agent {self.agent_id} started successfully")
            else:
                self.status = "failed"
                self.logger.error(f"Agent {self.agent_id} initialization failed")
                raise RuntimeError("Agent initialization failed")

        except Exception as e:
            self.status = "failed"
            self.logger.error(f"Agent startup failed: {e}")
            raise

    async def stop(self) -> None:
        """停止Agent"""
        try:
            self.logger.info(f"Stopping agent {self.agent_id}...")
            self.status = "stopping"

            # 等待运行中的任务完成
            if self.running_tasks:
                self.logger.info(f"Waiting for {len(self.running_tasks)} running tasks to complete...")
                await asyncio.sleep(1)  # 给任务一些时间完成

            # 执行清理
            await self.cleanup()

            self.status = "stopped"
            self.logger.info(f"Agent {self.agent_id} stopped")

        except Exception as e:
            self.logger.error(f"Agent shutdown failed: {e}")
            self.status = "error"

    def run(self) -> None:
        """运行Agent服务器"""
        uvicorn.run(
            self.app,
            host=self.host,
            port=self.port,
            log_level="info" if not self.debug else "debug"
        )

    async def handle_agent_message(self, message: AgentMessage) -> AgentResponse:
        """处理Agent消息"""
        self.message_count += 1

        try:
            # 检查消息是否过期
            if message.is_expired():
                return AgentResponse.error_response(
                    message="Message expired",
                    error_code="MESSAGE_EXPIRED"
                )

            # 根据消息类型处理
            if message.message_type == MessageType.REQUEST:
                return await self._handle_request_message(message)
            elif message.message_type == MessageType.NOTIFICATION:
                return await self._handle_notification_message(message)
            elif message.message_type == MessageType.HEARTBEAT:
                return await self._handle_heartbeat_message(message)
            else:
                return AgentResponse.error_response(
                    message=f"Unsupported message type: {message.message_type}",
                    error_code="UNSUPPORTED_MESSAGE_TYPE"
                )

        except Exception as e:
            self.logger.error(f"Message handling error: {e}")
            return AgentResponse.error_response(
                message=str(e),
                error_code="MESSAGE_HANDLING_ERROR"
            )

    async def _handle_request_message(self, message: AgentMessage) -> AgentResponse:
        """处理请求消息"""
        action = message.payload.get('action')

        if action in self.message_handlers:
            handler = self.message_handlers[action]
            try:
                result = await handler(message.payload)
                return AgentResponse.success_response(
                    message=f"Action {action} completed",
                    data=result
                )
            except Exception as e:
                return AgentResponse.error_response(
                    message=f"Action {action} failed: {str(e)}",
                    error_code="ACTION_EXECUTION_ERROR"
                )
        else:
            return AgentResponse.error_response(
                message=f"Unknown action: {action}",
                error_code="UNKNOWN_ACTION"
            )

    async def _handle_notification_message(self, message: AgentMessage) -> AgentResponse:
        """处理通知消息"""
        # 记录通知
        self.logger.info(f"Received notification from {message.sender_agent.value}: {message.payload}")

        return AgentResponse.success_response(
            message="Notification received"
        )

    async def _handle_heartbeat_message(self, message: AgentMessage) -> AgentResponse:
        """处理心跳消息"""
        return AgentResponse.success_response(
            message="Heartbeat acknowledged",
            data={
                "agent_id": self.agent_id,
                "status": self.status,
                "timestamp": datetime.now().isoformat()
            }
        )

    async def _execute_task(self, task_request: TaskRequest) -> None:
        """执行任务"""
        task_result = self.running_tasks.get(task_request.task_id)
        if not task_result:
            return

        try:
            # 标记任务开始
            task_result.start_execution(self.agent_id)
            self.task_count += 1

            # 执行实际任务
            result = await self.process_task(task_request)

            # 更新结果
            self.running_tasks[task_request.task_id] = result
            result.mark_completed(
                result_data=result.result_data,
                message="Task completed successfully"
            )

        except Exception as e:
            self.logger.error(f"Task execution failed: {e}")
            task_result.mark_failed(
                error_message=str(e),
                error_code="TASK_EXECUTION_ERROR"
            )

        finally:
            # 移动到历史记录
            final_result = self.running_tasks.pop(task_request.task_id, task_result)
            self.task_history.append(final_result)

            # 限制历史记录大小
            if len(self.task_history) > 1000:
                self.task_history = self.task_history[-500:]

    def register_message_handler(self, action: str, handler: Callable) -> None:
        """注册消息处理器"""
        self.message_handlers[action] = handler
        self.logger.info(f"Registered message handler for action: {action}")

    async def send_message_to_agent(
        self,
        target_agent: AgentType,
        action: str,
        payload: Dict[str, Any],
        target_url: Optional[str] = None
    ) -> AgentResponse:
        """向其他Agent发送消息"""
        message = AgentMessage.create_request(
            sender=self.agent_type,
            receiver=target_agent,
            payload={"action": action, **payload}
        )

        # 这里应该实现实际的HTTP调用到目标Agent
        # 为简化实现，返回模拟响应
        self.logger.info(f"Sending message to {target_agent.value}: {action}")

        return AgentResponse.success_response(
            message=f"Message sent to {target_agent.value}",
            data={"message_id": message.message_id}
        )

    def get_status(self) -> Dict[str, Any]:
        """获取Agent状态"""
        return {
            "agent_id": self.agent_id,
            "agent_type": self.agent_type.value,
            "status": self.status,
            "start_time": self.start_time.isoformat() if self.start_time else None,
            "uptime_seconds": (datetime.now() - self.start_time).total_seconds() if self.start_time else 0,
            "message_count": self.message_count,
            "task_count": self.task_count,
            "running_tasks": len(self.running_tasks),
            "config": {k: v for k, v in self.config.items() if not k.startswith('_')}
        }
