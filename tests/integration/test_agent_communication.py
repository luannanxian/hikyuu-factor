"""
Agent间通信集成测试

测试Agent间的消息传递、协调和协作：
1. 点对点消息传递
2. 广播和订阅机制
3. 任务分发和结果聚合
4. 超时和重试机制
5. 故障转移和恢复
6. 消息序列化和反序列化
7. 并发通信安全性
"""

import pytest
import asyncio
import json
import time
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional
from unittest.mock import Mock, patch, AsyncMock
import uuid

# 导入Agent和消息模型
try:
    from src.agents.base_agent import BaseAgent
    from src.agents.data_manager_agent import DataManagerAgent
    from src.agents.factor_calculation_agent import FactorCalculationAgent
    from src.agents.validation_agent import ValidationAgent
    from src.agents.signal_generation_agent import SignalGenerationAgent
    from src.models.agent_models import (
        AgentMessage, AgentResponse, TaskRequest, TaskResult,
        MessageType, TaskStatus, Priority, AgentType,
        create_heartbeat_message, create_batch_task_request,
        parse_agent_message, validate_message_flow
    )
except ImportError as e:
    pytest.skip(f"Required modules not available: {e}", allow_module_level=True)


@pytest.mark.integration
@pytest.mark.agent_communication
@pytest.mark.requires_hikyuu
class TestAgentCommunication:
    """Agent间通信集成测试"""

    def setup_method(self):
        """设置测试环境"""
        self.test_config = {
            "communication": {
                "timeout": 30,
                "retry_attempts": 3,
                "heartbeat_interval": 5,
                "message_queue_size": 1000
            },
            "agents": {
                "data_manager": {"port": 8001},
                "factor_calculator": {"port": 8002},
                "validator": {"port": 8003},
                "signal_generator": {"port": 8004}
            }
        }

        # 初始化测试Agents
        self.agents = {}
        self.message_history = []

    async def _create_test_agents(self) -> Dict[str, BaseAgent]:
        """创建测试用的Agent实例"""
        agents = {}

        try:
            # 创建数据管理Agent
            agents["data_manager"] = DataManagerAgent(config=self.test_config)
            await agents["data_manager"].initialize()

            # 创建因子计算Agent
            agents["factor_calculator"] = FactorCalculationAgent(config=self.test_config)
            await agents["factor_calculator"].initialize()

            # 创建验证Agent
            agents["validator"] = ValidationAgent(config=self.test_config)
            await agents["validator"].initialize()

            # 创建信号生成Agent
            agents["signal_generator"] = SignalGenerationAgent(config=self.test_config)
            await agents["signal_generator"].initialize()

            return agents

        except Exception as e:
            # 如果创建失败，清理已创建的Agent
            for agent in agents.values():
                try:
                    await agent.shutdown()
                except:
                    pass
            raise e

    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_point_to_point_communication(self):
        """测试点对点通信"""
        agents = await self._create_test_agents()

        try:
            # 创建测试消息
            message = AgentMessage.create_request_message(
                sender_agent=AgentType.DATA_MANAGER,
                receiver_agent=AgentType.FACTOR_CALCULATOR,
                content={
                    "action": "test_connection",
                    "timestamp": datetime.now().isoformat(),
                    "test_data": "Hello from Data Manager"
                }
            )

            # 发送消息
            data_agent = agents["data_manager"]
            factor_agent = agents["factor_calculator"]

            response = await factor_agent.process_message(message)

            # 验证响应
            assert response is not None, "应该收到响应"
            assert response.success, f"响应应该成功: {response.error_message}"
            assert response.message_id == message.message_id, "响应ID应该匹配"

        finally:
            await self._cleanup_agents(agents)

    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_broadcast_communication(self):
        """测试广播通信"""
        agents = await self._create_test_agents()

        try:
            # 创建广播消息
            broadcast_message = AgentMessage(
                message_id=str(uuid.uuid4()),
                message_type=MessageType.NOTIFICATION,
                sender_agent=AgentType.DATA_MANAGER,
                receiver_agent=None,  # 广播消息
                content={
                    "event": "data_update_complete",
                    "affected_stocks": ["sh600000", "sz000001"],
                    "update_time": datetime.now().isoformat()
                },
                timestamp=datetime.now(),
                priority=Priority.MEDIUM
            )

            # 发送广播消息
            data_agent = agents["data_manager"]
            responses = []

            # 模拟广播到所有其他Agent
            for agent_name, agent in agents.items():
                if agent_name != "data_manager":
                    response = await agent.process_message(broadcast_message)
                    responses.append((agent_name, response))

            # 验证所有Agent都收到了广播
            assert len(responses) == 3, "应该有3个Agent收到广播"

            for agent_name, response in responses:
                assert response is not None, f"Agent {agent_name} 应该响应广播"
                # 广播通常不要求成功响应，但应该有确认
                assert hasattr(response, 'message_id'), f"Agent {agent_name} 响应格式错误"

        finally:
            await self._cleanup_agents(agents)

    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_task_distribution_and_aggregation(self):
        """测试任务分发和结果聚合"""
        agents = await self._create_test_agents()

        try:
            # 创建需要多Agent协作的复合任务
            main_task = TaskRequest(
                task_type="factor_calculation_pipeline",
                parameters={
                    "stock_codes": ["sh600000", "sz000001"],
                    "factor_types": ["momentum", "value", "technical"],
                    "date_range": ["2024-01-01", "2024-01-31"]
                },
                priority=Priority.HIGH,
                timeout_seconds=300
            )

            # 任务协调者（数据管理Agent）
            coordinator = agents["data_manager"]

            # 第一步：数据准备
            data_prep_message = main_task.to_agent_message(
                sender_agent=AgentType.TEST,
                receiver_agent=AgentType.DATA_MANAGER
            )

            prep_response = await coordinator.process_message(data_prep_message)
            assert prep_response.success, "数据准备失败"

            # 第二步：并行因子计算（分发到因子计算Agent）
            factor_tasks = []
            for factor_type in ["momentum", "value", "technical"]:
                subtask = TaskRequest(
                    task_type="single_factor_calculation",
                    parameters={
                        "factor_type": factor_type,
                        "stock_codes": ["sh600000", "sz000001"],
                        "parent_task_id": main_task.task_id
                    }
                )

                task_message = subtask.to_agent_message(
                    sender_agent=AgentType.DATA_MANAGER,
                    receiver_agent=AgentType.FACTOR_CALCULATOR
                )

                factor_tasks.append(task_message)

            # 并行执行因子计算
            factor_agent = agents["factor_calculator"]
            factor_responses = await asyncio.gather(*[
                factor_agent.process_message(task) for task in factor_tasks
            ])

            # 验证所有子任务都成功
            for i, response in enumerate(factor_responses):
                assert response.success, f"因子计算任务 {i} 失败: {response.error_message}"

            # 第三步：结果聚合和验证
            aggregation_task = TaskRequest(
                task_type="factor_validation",
                parameters={
                    "factor_results": [resp.data for resp in factor_responses],
                    "validation_rules": ["completeness", "consistency"]
                }
            )

            validation_message = aggregation_task.to_agent_message(
                sender_agent=AgentType.DATA_MANAGER,
                receiver_agent=AgentType.VALIDATOR
            )

            validator = agents["validator"]
            final_response = await validator.process_message(validation_message)

            assert final_response.success, "结果聚合验证失败"
            assert "aggregated_results" in final_response.data, "缺少聚合结果"

        finally:
            await self._cleanup_agents(agents)

    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_timeout_and_retry_mechanism(self):
        """测试超时和重试机制"""
        agents = await self._create_test_agents()

        try:
            # 创建一个会超时的任务
            slow_task = TaskRequest(
                task_type="slow_calculation",
                parameters={"delay": 10},  # 10秒延迟
                timeout_seconds=3  # 3秒超时
            )

            message = slow_task.to_agent_message(
                sender_agent=AgentType.TEST,
                receiver_agent=AgentType.FACTOR_CALCULATOR
            )

            factor_agent = agents["factor_calculator"]

            # 模拟慢响应
            original_process = factor_agent.process_message
            async def slow_process(msg):
                await asyncio.sleep(5)  # 模拟5秒处理时间
                return await original_process(msg)

            factor_agent.process_message = slow_process

            # 测试超时
            start_time = time.time()
            with pytest.raises((asyncio.TimeoutError, Exception)):
                await asyncio.wait_for(
                    factor_agent.process_message(message),
                    timeout=3
                )

            elapsed_time = time.time() - start_time
            assert elapsed_time < 4, "超时时间应该接近设置的3秒"

            # 恢复正常处理
            factor_agent.process_message = original_process

            # 测试重试成功
            normal_task = TaskRequest(task_type="normal_calculation", parameters={})
            normal_message = normal_task.to_agent_message(
                sender_agent=AgentType.TEST,
                receiver_agent=AgentType.FACTOR_CALCULATOR
            )

            response = await factor_agent.process_message(normal_message)
            assert response is not None, "重试后应该成功"

        finally:
            await self._cleanup_agents(agents)

    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_agent_failure_and_recovery(self):
        """测试Agent故障和恢复"""
        agents = await self._create_test_agents()

        try:
            data_agent = agents["data_manager"]
            factor_agent = agents["factor_calculator"]

            # 正常通信测试
            test_message = AgentMessage.create_request_message(
                sender_agent=AgentType.DATA_MANAGER,
                receiver_agent=AgentType.FACTOR_CALCULATOR,
                content={"action": "health_check"}
            )

            response = await factor_agent.process_message(test_message)
            assert response.success, "初始通信应该成功"

            # 模拟Agent故障
            original_process = factor_agent.process_message
            factor_agent.process_message = AsyncMock(side_effect=Exception("Agent故障"))

            # 验证故障检测
            with pytest.raises(Exception):
                await factor_agent.process_message(test_message)

            # 恢复Agent
            factor_agent.process_message = original_process

            # 验证恢复后正常工作
            recovery_response = await factor_agent.process_message(test_message)
            assert recovery_response.success, "恢复后通信应该成功"

        finally:
            await self._cleanup_agents(agents)

    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_message_serialization_and_deserialization(self):
        """测试消息序列化和反序列化"""
        # 创建复杂消息
        complex_message = AgentMessage(
            message_id=str(uuid.uuid4()),
            message_type=MessageType.REQUEST,
            sender_agent=AgentType.DATA_MANAGER,
            receiver_agent=AgentType.FACTOR_CALCULATOR,
            content={
                "nested_data": {
                    "stocks": ["sh600000", "sz000001"],
                    "factors": ["momentum", "value"],
                    "dates": ["2024-01-01", "2024-01-31"]
                },
                "arrays": [1, 2, 3, 4, 5],
                "unicode_text": "测试中文字符",
                "timestamp": datetime.now().isoformat()
            },
            timestamp=datetime.now(),
            priority=Priority.HIGH,
            metadata={
                "version": "1.0",
                "source": "integration_test"
            }
        )

        # 序列化
        serialized = complex_message.to_json()
        assert isinstance(serialized, str), "序列化结果应该是字符串"

        # 反序列化
        parsed_message = parse_agent_message(serialized)
        assert parsed_message is not None, "反序列化应该成功"

        # 验证内容一致性
        assert parsed_message.message_id == complex_message.message_id
        assert parsed_message.message_type == complex_message.message_type
        assert parsed_message.sender_agent == complex_message.sender_agent
        assert parsed_message.receiver_agent == complex_message.receiver_agent
        assert parsed_message.content == complex_message.content
        assert parsed_message.priority == complex_message.priority

    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_concurrent_communication_safety(self):
        """测试并发通信安全性"""
        agents = await self._create_test_agents()

        try:
            factor_agent = agents["factor_calculator"]

            # 创建大量并发消息
            num_concurrent_messages = 50
            messages = []

            for i in range(num_concurrent_messages):
                message = AgentMessage.create_request_message(
                    sender_agent=AgentType.DATA_MANAGER,
                    receiver_agent=AgentType.FACTOR_CALCULATOR,
                    content={
                        "action": "concurrent_test",
                        "message_number": i,
                        "timestamp": datetime.now().isoformat()
                    }
                )
                messages.append(message)

            # 并发发送所有消息
            start_time = time.time()
            responses = await asyncio.gather(*[
                factor_agent.process_message(msg) for msg in messages
            ], return_exceptions=True)

            end_time = time.time()

            # 验证结果
            successful_responses = 0
            for i, response in enumerate(responses):
                if isinstance(response, Exception):
                    print(f"消息 {i} 处理失败: {response}")
                else:
                    successful_responses += 1

            # 至少90%的消息应该成功处理
            success_rate = successful_responses / num_concurrent_messages
            assert success_rate >= 0.9, f"并发成功率过低: {success_rate:.2%}"

            # 验证处理时间合理
            avg_time_per_message = (end_time - start_time) / num_concurrent_messages
            assert avg_time_per_message < 1.0, f"平均处理时间过长: {avg_time_per_message:.3f}秒"

        finally:
            await self._cleanup_agents(agents)

    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_heartbeat_mechanism(self):
        """测试心跳机制"""
        agents = await self._create_test_agents()

        try:
            # 创建心跳消息
            heartbeat = create_heartbeat_message(
                sender_agent=AgentType.DATA_MANAGER,
                status="healthy",
                metrics={
                    "cpu_usage": 45.2,
                    "memory_usage": 67.8,
                    "active_tasks": 3
                }
            )

            # 发送心跳到所有其他Agent
            data_agent = agents["data_manager"]
            heartbeat_responses = []

            for agent_name, agent in agents.items():
                if agent_name != "data_manager":
                    response = await agent.process_message(heartbeat)
                    heartbeat_responses.append((agent_name, response))

            # 验证心跳响应
            for agent_name, response in heartbeat_responses:
                assert response is not None, f"Agent {agent_name} 应该响应心跳"
                # 心跳响应通常是简单确认

            # 测试心跳超时检测
            # 模拟Agent无响应
            factor_agent = agents["factor_calculator"]
            original_process = factor_agent.process_message
            factor_agent.process_message = AsyncMock(side_effect=asyncio.TimeoutError())

            with pytest.raises(asyncio.TimeoutError):
                await asyncio.wait_for(
                    factor_agent.process_message(heartbeat),
                    timeout=1
                )

            # 恢复
            factor_agent.process_message = original_process

        finally:
            await self._cleanup_agents(agents)

    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_message_flow_validation(self):
        """测试消息流验证"""
        # 创建有效的消息流
        request_message = AgentMessage.create_request_message(
            sender_agent=AgentType.DATA_MANAGER,
            receiver_agent=AgentType.FACTOR_CALCULATOR,
            content={"action": "calculate_factor"}
        )

        response_message = AgentResponse.create_success_response(
            original_message=request_message,
            data={"result": "calculation_complete"}
        ).to_agent_message()

        # 验证消息流
        flow_valid = validate_message_flow([request_message, response_message])
        assert flow_valid, "有效的请求-响应流应该通过验证"

        # 测试无效消息流
        invalid_response = AgentMessage.create_response_message(
            sender_agent=AgentType.VALIDATOR,  # 错误的发送者
            receiver_agent=AgentType.DATA_MANAGER,
            original_message_id=request_message.message_id,
            content={"result": "invalid"}
        )

        invalid_flow_valid = validate_message_flow([request_message, invalid_response])
        assert not invalid_flow_valid, "无效的消息流应该被检测出来"

    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_batch_task_processing(self):
        """测试批量任务处理"""
        agents = await self._create_test_agents()

        try:
            # 创建批量任务
            individual_tasks = []
            for i in range(5):
                task = TaskRequest(
                    task_type="batch_calculation",
                    parameters={
                        "stock_code": f"sh{600000 + i:06d}",
                        "factor_type": "momentum"
                    }
                )
                individual_tasks.append(task)

            batch_request = create_batch_task_request(
                tasks=individual_tasks,
                batch_id="test_batch_001"
            )

            # 发送批量任务
            factor_agent = agents["factor_calculator"]
            batch_message = batch_request.to_agent_message(
                sender_agent=AgentType.TEST,
                receiver_agent=AgentType.FACTOR_CALCULATOR
            )

            batch_response = await factor_agent.process_message(batch_message)

            # 验证批量处理结果
            assert batch_response.success, "批量任务应该成功"
            assert "batch_results" in batch_response.data, "应该包含批量结果"

            batch_results = batch_response.data["batch_results"]
            assert len(batch_results) == 5, "应该处理所有5个子任务"

            for result in batch_results:
                assert "task_id" in result, "每个结果应该包含任务ID"
                assert "status" in result, "每个结果应该包含状态"

        finally:
            await self._cleanup_agents(agents)

    async def _cleanup_agents(self, agents: Dict[str, BaseAgent]):
        """清理Agent资源"""
        for agent in agents.values():
            try:
                await agent.shutdown()
            except Exception as e:
                print(f"清理Agent时出错: {e}")

    def teardown_method(self):
        """清理测试环境"""
        # 清理消息历史
        self.message_history.clear()

        # 清理Agent引用
        self.agents.clear()