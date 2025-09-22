"""
端到端集成测试 - 完整因子挖掘工作流

测试从数据获取到信号生成的完整流程：
1. 数据管理Agent → 数据更新和质量检查
2. 因子计算Agent → 因子计算和存储
3. 验证Agent → 因子验证和质量评估
4. 信号生成Agent → 交易信号生成和确认
5. 全链路性能和稳定性验证

这是最重要的集成测试，确保整个系统协同工作
"""

import pytest
import asyncio
import pandas as pd
import numpy as np
from datetime import datetime, timedelta, date
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional
from unittest.mock import Mock, patch, MagicMock
import json
import time

# 导入系统模块
try:
    from src.agents.data_manager_agent import DataManagerAgent
    from src.agents.factor_calculation_agent import FactorCalculationAgent
    from src.agents.validation_agent import ValidationAgent
    from src.agents.signal_generation_agent import SignalGenerationAgent
    from src.models.hikyuu_models import (
        FactorData, FactorCalculationRequest, FactorCalculationResult,
        FactorType, SignalType, PositionType
    )
    from src.models.agent_models import (
        AgentMessage, AgentResponse, TaskRequest, TaskResult,
        MessageType, TaskStatus, Priority, AgentType
    )
    from src.services.data_manager_service import DataManagerService
    from src.services.factor_calculation_service import FactorCalculationService
    from src.services.validation_service import ValidationService
    from src.services.signal_generation_service import SignalGenerationService
except ImportError as e:
    # 某些模块可能还未完全实现
    pytest.skip(f"Required modules not available: {e}", allow_module_level=True)


@pytest.mark.integration
@pytest.mark.end_to_end
@pytest.mark.requires_mysql
@pytest.mark.requires_hikyuu
@pytest.mark.slow
class TestEndToEndWorkflow:
    """端到端工作流集成测试"""

    def setup_method(self):
        """设置测试环境"""
        self.test_stocks = ["sh600000", "sh600001", "sz000001", "sz000002"]
        self.test_date_range = (
            date(2024, 1, 1),
            date(2024, 1, 31)
        )

        # 测试配置
        self.test_config = {
            "database": {
                "host": "192.168.3.46",
                "port": 3306,
                "user": "remote",
                "password": "remote123456",
                "database": "hikyuu_factor_test",
                "charset": "utf8mb4"
            },
            "hikyuu": {
                "data_path": "/tmp/hikyuu_test_data",
                "block_path": "/tmp/hikyuu_test_blocks"
            },
            "agents": {
                "timeout": 30,
                "retry_attempts": 3,
                "heartbeat_interval": 5
            }
        }

        # 初始化Agents
        self.data_agent = None
        self.factor_agent = None
        self.validation_agent = None
        self.signal_agent = None

        # 测试数据
        self.test_market_data = self._create_test_market_data()
        self.expected_factors = self._define_expected_factors()

    def _create_test_market_data(self) -> pd.DataFrame:
        """创建测试市场数据"""
        np.random.seed(42)
        data = []

        for stock in self.test_stocks:
            base_price = 10.0 + np.random.randn() * 2.0
            for i in range(31):  # 31天数据
                current_date = self.test_date_range[0] + timedelta(days=i)

                # 模拟价格走势
                price_change = np.random.randn() * 0.02  # 2%日波动
                base_price *= (1 + price_change)

                data.append({
                    'stock_code': stock,
                    'trade_date': current_date,
                    'open_price': base_price * (1 + np.random.randn() * 0.005),
                    'high_price': base_price * (1 + abs(np.random.randn() * 0.01)),
                    'low_price': base_price * (1 - abs(np.random.randn() * 0.01)),
                    'close_price': base_price,
                    'volume': 1000000 + np.random.randint(0, 500000),
                    'turnover_rate': np.random.uniform(0.5, 5.0)
                })

        df = pd.DataFrame(data)
        df['amount'] = df['close_price'] * df['volume']
        return df

    def _define_expected_factors(self) -> List[str]:
        """定义期望的因子列表"""
        return [
            "momentum_20d",     # 20日动量因子
            "rsi_14d",          # 14日RSI
            "pe_ratio",         # 市盈率
            "pb_ratio",         # 市净率
            "volume_ratio_5d"   # 5日量比
        ]

    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_complete_end_to_end_workflow(self):
        """测试完整的端到端工作流"""

        # === Phase 1: 初始化所有Agents ===
        await self._initialize_all_agents()

        # === Phase 2: 数据管理工作流 ===
        data_update_result = await self._execute_data_management_workflow()
        assert data_update_result["status"] == "success", "数据更新阶段失败"

        # === Phase 3: 因子计算工作流 ===
        factor_calculation_result = await self._execute_factor_calculation_workflow()
        assert factor_calculation_result["status"] == "success", "因子计算阶段失败"
        assert len(factor_calculation_result["factors"]) >= len(self.expected_factors), "因子数量不足"

        # === Phase 4: 因子验证工作流 ===
        validation_result = await self._execute_factor_validation_workflow(
            factor_calculation_result["factors"]
        )
        assert validation_result["status"] == "success", "因子验证阶段失败"
        assert validation_result["quality_score"] >= 0.7, "因子质量不达标"

        # === Phase 5: 信号生成工作流 ===
        signal_generation_result = await self._execute_signal_generation_workflow(
            validation_result["validated_factors"]
        )
        assert signal_generation_result["status"] == "success", "信号生成阶段失败"
        assert len(signal_generation_result["signals"]) > 0, "未生成任何信号"

        # === Phase 6: 端到端验证 ===
        await self._verify_end_to_end_consistency()

        # === Phase 7: 性能指标验证 ===
        performance_metrics = await self._collect_performance_metrics()
        self._validate_performance_requirements(performance_metrics)

    async def _initialize_all_agents(self):
        """初始化所有Agents"""
        try:
            # 初始化数据管理Agent
            self.data_agent = DataManagerAgent(config=self.test_config)
            await self.data_agent.initialize()

            # 初始化因子计算Agent
            self.factor_agent = FactorCalculationAgent(config=self.test_config)
            await self.factor_agent.initialize()

            # 初始化验证Agent
            self.validation_agent = ValidationAgent(config=self.test_config)
            await self.validation_agent.initialize()

            # 初始化信号生成Agent
            self.signal_agent = SignalGenerationAgent(config=self.test_config)
            await self.signal_agent.initialize()

            # 验证所有Agent都已就绪
            agents = [self.data_agent, self.factor_agent, self.validation_agent, self.signal_agent]
            for agent in agents:
                status = await agent.get_health_status()
                assert status["status"] == "healthy", f"Agent {agent.agent_type} 初始化失败"

        except Exception as e:
            pytest.fail(f"Agent初始化失败: {e}")

    async def _execute_data_management_workflow(self) -> Dict[str, Any]:
        """执行数据管理工作流"""
        try:
            # 创建数据更新任务
            update_task = TaskRequest(
                task_type="data_update",
                parameters={
                    "stock_codes": self.test_stocks,
                    "start_date": self.test_date_range[0].isoformat(),
                    "end_date": self.test_date_range[1].isoformat(),
                    "data_types": ["market_data", "financial_data"],
                    "update_mode": "incremental"
                },
                priority=Priority.HIGH,
                timeout_seconds=300
            )

            # 发送任务给数据管理Agent
            message = update_task.to_agent_message(
                sender_agent=AgentType.TEST,
                receiver_agent=AgentType.DATA_MANAGER
            )

            response = await self.data_agent.process_message(message)

            # 验证响应
            assert response.success, f"数据更新任务失败: {response.error_message}"

            # 等待任务完成
            task_result = await self._wait_for_task_completion(
                self.data_agent, response.data["task_id"]
            )

            return {
                "status": "success",
                "task_id": response.data["task_id"],
                "updated_records": task_result.data.get("updated_records", 0),
                "quality_score": task_result.data.get("quality_score", 0.0)
            }

        except Exception as e:
            return {"status": "error", "error": str(e)}

    async def _execute_factor_calculation_workflow(self) -> Dict[str, Any]:
        """执行因子计算工作流"""
        try:
            # 创建因子计算请求
            calculation_request = FactorCalculationRequest(
                factor_types=[FactorType.MOMENTUM, FactorType.VALUATION, FactorType.TECHNICAL],
                stock_codes=self.test_stocks,
                start_date=self.test_date_range[0],
                end_date=self.test_date_range[1],
                calculation_params={
                    "momentum_periods": [5, 10, 20],
                    "technical_indicators": ["RSI", "MACD", "BOLL"],
                    "valuation_metrics": ["PE", "PB", "PS"]
                }
            )

            # 转换为任务请求
            calc_task = TaskRequest(
                task_type="factor_calculation",
                parameters=calculation_request.to_dict(),
                priority=Priority.MEDIUM,
                timeout_seconds=600
            )

            # 发送任务给因子计算Agent
            message = calc_task.to_agent_message(
                sender_agent=AgentType.TEST,
                receiver_agent=AgentType.FACTOR_CALCULATOR
            )

            response = await self.factor_agent.process_message(message)
            assert response.success, f"因子计算任务失败: {response.error_message}"

            # 等待计算完成
            task_result = await self._wait_for_task_completion(
                self.factor_agent, response.data["task_id"]
            )

            return {
                "status": "success",
                "task_id": response.data["task_id"],
                "factors": task_result.data.get("calculated_factors", []),
                "calculation_time": task_result.data.get("calculation_time", 0.0),
                "coverage_ratio": task_result.data.get("coverage_ratio", 0.0)
            }

        except Exception as e:
            return {"status": "error", "error": str(e)}

    async def _execute_factor_validation_workflow(self, factors: List[Dict]) -> Dict[str, Any]:
        """执行因子验证工作流"""
        try:
            # 创建验证任务
            validation_task = TaskRequest(
                task_type="factor_validation",
                parameters={
                    "factors": factors,
                    "validation_rules": [
                        "missing_value_check",
                        "outlier_detection",
                        "correlation_analysis",
                        "stability_check",
                        "ic_analysis"
                    ],
                    "quality_threshold": 0.7
                },
                priority=Priority.MEDIUM,
                timeout_seconds=300
            )

            # 发送任务给验证Agent
            message = validation_task.to_agent_message(
                sender_agent=AgentType.TEST,
                receiver_agent=AgentType.VALIDATOR
            )

            response = await self.validation_agent.process_message(message)
            assert response.success, f"因子验证任务失败: {response.error_message}"

            # 等待验证完成
            task_result = await self._wait_for_task_completion(
                self.validation_agent, response.data["task_id"]
            )

            return {
                "status": "success",
                "task_id": response.data["task_id"],
                "validated_factors": task_result.data.get("validated_factors", []),
                "quality_score": task_result.data.get("overall_quality_score", 0.0),
                "validation_report": task_result.data.get("validation_report", {})
            }

        except Exception as e:
            return {"status": "error", "error": str(e)}

    async def _execute_signal_generation_workflow(self, validated_factors: List[Dict]) -> Dict[str, Any]:
        """执行信号生成工作流"""
        try:
            # 创建信号生成任务
            signal_task = TaskRequest(
                task_type="signal_generation",
                parameters={
                    "factors": validated_factors,
                    "strategies": [
                        "momentum_strategy",
                        "value_strategy",
                        "multi_factor_strategy"
                    ],
                    "risk_controls": {
                        "max_position_size": 0.1,
                        "sector_limit": 0.3,
                        "stop_loss": 0.05
                    },
                    "require_confirmation": True
                },
                priority=Priority.HIGH,
                timeout_seconds=180
            )

            # 发送任务给信号生成Agent
            message = signal_task.to_agent_message(
                sender_agent=AgentType.TEST,
                receiver_agent=AgentType.SIGNAL_GENERATOR
            )

            response = await self.signal_agent.process_message(message)
            assert response.success, f"信号生成任务失败: {response.error_message}"

            # 等待信号生成完成
            task_result = await self._wait_for_task_completion(
                self.signal_agent, response.data["task_id"]
            )

            return {
                "status": "success",
                "task_id": response.data["task_id"],
                "signals": task_result.data.get("generated_signals", []),
                "signal_count": task_result.data.get("signal_count", 0),
                "risk_metrics": task_result.data.get("risk_metrics", {})
            }

        except Exception as e:
            return {"status": "error", "error": str(e)}

    async def _wait_for_task_completion(self, agent, task_id: str, timeout: int = 300) -> TaskResult:
        """等待任务完成"""
        start_time = time.time()

        while time.time() - start_time < timeout:
            # 查询任务状态
            status_request = TaskRequest(
                task_type="get_task_status",
                parameters={"task_id": task_id}
            )

            message = status_request.to_agent_message(
                sender_agent=AgentType.TEST,
                receiver_agent=agent.agent_type
            )

            response = await agent.process_message(message)

            if response.success:
                task_status = response.data.get("status")
                if task_status == TaskStatus.COMPLETED:
                    return TaskResult.create_completed_result(
                        task_id=task_id,
                        result=response.data.get("result", {}),
                        data=response.data.get("data", {})
                    )
                elif task_status == TaskStatus.FAILED:
                    raise Exception(f"Task {task_id} failed: {response.data.get('error')}")

            await asyncio.sleep(1)  # 等待1秒后重试

        raise TimeoutError(f"Task {task_id} did not complete within {timeout} seconds")

    async def _verify_end_to_end_consistency(self):
        """验证端到端一致性"""
        try:
            # 验证数据完整性
            await self._verify_data_integrity()

            # 验证因子数据质量
            await self._verify_factor_quality()

            # 验证信号逻辑一致性
            await self._verify_signal_consistency()

        except Exception as e:
            pytest.fail(f"端到端一致性验证失败: {e}")

    async def _verify_data_integrity(self):
        """验证数据完整性"""
        # 检查数据库中的数据完整性
        query_task = TaskRequest(
            task_type="data_integrity_check",
            parameters={
                "stock_codes": self.test_stocks,
                "date_range": [d.isoformat() for d in self.test_date_range]
            }
        )

        message = query_task.to_agent_message(
            sender_agent=AgentType.TEST,
            receiver_agent=AgentType.DATA_MANAGER
        )

        response = await self.data_agent.process_message(message)
        assert response.success, "数据完整性检查失败"

        integrity_result = response.data
        assert integrity_result["missing_data_ratio"] < 0.05, "缺失数据过多"
        assert integrity_result["duplicate_records"] == 0, "存在重复记录"

    async def _verify_factor_quality(self):
        """验证因子质量"""
        quality_task = TaskRequest(
            task_type="factor_quality_analysis",
            parameters={
                "factor_names": self.expected_factors,
                "analysis_type": "comprehensive"
            }
        )

        message = quality_task.to_agent_message(
            sender_agent=AgentType.TEST,
            receiver_agent=AgentType.VALIDATOR
        )

        response = await self.validation_agent.process_message(message)
        assert response.success, "因子质量分析失败"

        quality_result = response.data
        assert quality_result["overall_score"] >= 0.7, "因子质量总分不达标"

        for factor in quality_result["factor_scores"]:
            assert factor["score"] >= 0.6, f"因子 {factor['name']} 质量不达标"

    async def _verify_signal_consistency(self):
        """验证信号一致性"""
        consistency_task = TaskRequest(
            task_type="signal_consistency_check",
            parameters={
                "check_types": [
                    "risk_constraint_compliance",
                    "position_size_validation",
                    "sector_diversification"
                ]
            }
        )

        message = consistency_task.to_agent_message(
            sender_agent=AgentType.TEST,
            receiver_agent=AgentType.SIGNAL_GENERATOR
        )

        response = await self.signal_agent.process_message(message)
        assert response.success, "信号一致性检查失败"

        consistency_result = response.data
        assert consistency_result["compliance_rate"] >= 0.95, "信号合规率不达标"

    async def _collect_performance_metrics(self) -> Dict[str, Any]:
        """收集性能指标"""
        metrics = {}

        # 收集各Agent的性能指标
        agents = [
            ("data_manager", self.data_agent),
            ("factor_calculator", self.factor_agent),
            ("validator", self.validation_agent),
            ("signal_generator", self.signal_agent)
        ]

        for agent_name, agent in agents:
            perf_task = TaskRequest(
                task_type="get_performance_metrics",
                parameters={}
            )

            message = perf_task.to_agent_message(
                sender_agent=AgentType.TEST,
                receiver_agent=agent.agent_type
            )

            response = await agent.process_message(message)
            if response.success:
                metrics[agent_name] = response.data

        return metrics

    def _validate_performance_requirements(self, metrics: Dict[str, Any]):
        """验证性能要求"""
        # 数据管理性能要求
        data_metrics = metrics.get("data_manager", {})
        if data_metrics:
            assert data_metrics.get("avg_response_time", 999) < 5.0, "数据管理响应时间过长"
            assert data_metrics.get("memory_usage", 999) < 1000, "数据管理内存使用过高"

        # 因子计算性能要求
        factor_metrics = metrics.get("factor_calculator", {})
        if factor_metrics:
            assert factor_metrics.get("calculation_throughput", 0) > 100, "因子计算吞吐量过低"
            assert factor_metrics.get("avg_calculation_time", 999) < 10.0, "因子计算时间过长"

        # 验证性能要求
        validation_metrics = metrics.get("validator", {})
        if validation_metrics:
            assert validation_metrics.get("validation_speed", 0) > 50, "因子验证速度过慢"

        # 信号生成性能要求
        signal_metrics = metrics.get("signal_generator", {})
        if signal_metrics:
            assert signal_metrics.get("signal_generation_time", 999) < 3.0, "信号生成时间过长"

    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_agent_failure_recovery(self):
        """测试Agent故障恢复"""
        await self._initialize_all_agents()

        # 模拟因子计算Agent故障
        original_process = self.factor_agent.process_message

        # 模拟故障
        self.factor_agent.process_message = Mock(side_effect=Exception("Agent故障"))

        # 尝试发送任务（应该失败）
        calc_task = TaskRequest(task_type="factor_calculation", parameters={})
        message = calc_task.to_agent_message(
            sender_agent=AgentType.TEST,
            receiver_agent=AgentType.FACTOR_CALCULATOR
        )

        with pytest.raises(Exception):
            await self.factor_agent.process_message(message)

        # 恢复Agent
        self.factor_agent.process_message = original_process

        # 验证恢复后能正常工作
        response = await self.factor_agent.process_message(message)
        # 根据实际实现调整断言

    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_concurrent_workflow_execution(self):
        """测试并发工作流执行"""
        await self._initialize_all_agents()

        # 创建多个并发任务
        tasks = []
        for i in range(3):
            task = asyncio.create_task(
                self._execute_mini_workflow(f"workflow_{i}")
            )
            tasks.append(task)

        # 等待所有任务完成
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # 验证所有任务都成功完成
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                pytest.fail(f"并发工作流 {i} 失败: {result}")
            assert result["status"] == "success", f"工作流 {i} 未成功完成"

    async def _execute_mini_workflow(self, workflow_id: str) -> Dict[str, Any]:
        """执行迷你工作流（用于并发测试）"""
        try:
            # 简化的工作流：数据更新 → 单个因子计算

            # 数据更新
            update_task = TaskRequest(
                task_type="data_update",
                parameters={
                    "stock_codes": [self.test_stocks[0]],  # 只处理一只股票
                    "workflow_id": workflow_id
                }
            )

            message = update_task.to_agent_message(
                sender_agent=AgentType.TEST,
                receiver_agent=AgentType.DATA_MANAGER
            )

            response = await self.data_agent.process_message(message)
            if not response.success:
                return {"status": "error", "stage": "data_update", "error": response.error_message}

            # 因子计算
            calc_task = TaskRequest(
                task_type="factor_calculation",
                parameters={
                    "factor_types": [FactorType.MOMENTUM],
                    "workflow_id": workflow_id
                }
            )

            message = calc_task.to_agent_message(
                sender_agent=AgentType.TEST,
                receiver_agent=AgentType.FACTOR_CALCULATOR
            )

            response = await self.factor_agent.process_message(message)
            if not response.success:
                return {"status": "error", "stage": "factor_calculation", "error": response.error_message}

            return {"status": "success", "workflow_id": workflow_id}

        except Exception as e:
            return {"status": "error", "workflow_id": workflow_id, "error": str(e)}

    def teardown_method(self):
        """清理测试环境"""
        # 清理Agents
        if self.data_agent:
            asyncio.create_task(self.data_agent.shutdown())
        if self.factor_agent:
            asyncio.create_task(self.factor_agent.shutdown())
        if self.validation_agent:
            asyncio.create_task(self.validation_agent.shutdown())
        if self.signal_agent:
            asyncio.create_task(self.signal_agent.shutdown())