"""
Signal Generation Agent

信号生成Agent实现，提供：
1. 交易信号生成和策略执行
2. 强制人工确认机制
3. 信号质量评估和风险控制
4. 实时监控和预警

基于Phase 6 SignalGenerationService实现的微服务Agent。
"""

import asyncio
import logging
from datetime import datetime, date, timedelta
from typing import Dict, List, Optional, Any
from fastapi import HTTPException
from pydantic import BaseModel

from .base_agent import BaseAgent
from ..models.agent_models import AgentType, TaskRequest, TaskResult, AgentResponse
from ..services.signal_generation_service import SignalGenerationService
from ..models.audit_models import AuditEntry, AuditEventType


class SignalGenerationRequest(BaseModel):
    """信号生成请求模型"""
    strategy_id: str
    factor_inputs: Dict[str, Any]
    universe: Optional[List[str]] = None
    signal_date: Optional[str] = None
    signal_params: Optional[Dict[str, Any]] = {}
    require_confirmation: bool = True


class SignalConfirmationRequest(BaseModel):
    """信号确认请求模型"""
    signal_id: str
    confirmation_status: str  # "approved", "rejected", "modified"
    operator_id: str
    confirmation_notes: Optional[str] = None
    modifications: Optional[Dict[str, Any]] = None


class StrategyExecutionRequest(BaseModel):
    """策略执行请求模型"""
    strategy_config: Dict[str, Any]
    execution_mode: str = "simulation"  # "simulation", "paper_trading", "live"
    risk_limits: Optional[Dict[str, Any]] = None
    notification_settings: Optional[Dict[str, Any]] = None


class SignalMonitoringRequest(BaseModel):
    """信号监控请求模型"""
    signal_ids: Optional[List[str]] = None
    monitoring_period: str = "real_time"  # "real_time", "daily", "weekly"
    alert_conditions: Optional[Dict[str, Any]] = None


class SignalGenerationAgent(BaseAgent):
    """
    信号生成Agent

    基于SignalGenerationService实现的微服务Agent，
    提供RESTful API和Agent间通信接口。
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        # 默认配置
        default_config = {
            'host': '0.0.0.0',
            'port': 8004,  # 信号生成Agent默认端口
            'debug': False,
            'enable_audit': True,
            'signal_generation': {
                'strategy_engine': {
                    'supported_strategies': ['momentum', 'mean_reversion', 'factor_model'],
                    'max_concurrent_strategies': 10,
                    'strategy_timeout_minutes': 15
                },
                'signal_generator': {
                    'default_universe_size': 300,
                    'signal_strength_threshold': 0.6,
                    'max_signals_per_day': 50,
                    'position_sizing_method': 'equal_weight'
                },
                'confirmation_system': {
                    'require_human_confirmation': True,
                    'confirmation_timeout_hours': 2,
                    'auto_reject_after_timeout': True,
                    'notification_channels': ['email', 'webhook']
                },
                'risk_controller': {
                    'max_position_size': 0.05,
                    'max_sector_exposure': 0.3,
                    'max_daily_turnover': 0.2,
                    'stop_loss_threshold': -0.05
                },
                'signal_monitor': {
                    'real_time_monitoring': True,
                    'performance_tracking': True,
                    'anomaly_detection': True,
                    'alert_thresholds': {
                        'signal_decay': 0.3,
                        'unexpected_correlation': 0.8
                    }
                }
            }
        }

        # 合并配置
        if config:
            default_config.update(config)

        super().__init__(AgentType.SIGNAL_GENERATION, config=default_config)

        # 信号生成服务
        self.signal_service = None

        # 注册API路由
        self._setup_signal_api_routes()

        # 注册消息处理器
        self._setup_message_handlers()

    async def initialize(self) -> bool:
        """初始化信号生成Agent"""
        try:
            self.logger.info("Initializing Signal Generation Agent...")

            # 初始化信号生成服务
            self.signal_service = SignalGenerationService(self.config.get('signal_generation', {}))

            # 创建审计记录
            if self.config.get('enable_audit', True):
                audit_entry = AuditEntry.create_system_action(
                    component="SignalGenerationAgent",
                    action_name="agent_initialization",
                    description=f"Signal Generation Agent {self.agent_id} initialized"
                )
                self.logger.info(f"Audit entry created: {audit_entry.audit_id}")

            self.logger.info("Signal Generation Agent initialized successfully")
            return True

        except Exception as e:
            self.logger.error(f"Signal Generation Agent initialization failed: {e}")
            return False

    async def cleanup(self) -> None:
        """清理资源"""
        self.logger.info("Cleaning up Signal Generation Agent...")
        # 这里可以添加资源清理逻辑

    def _setup_signal_api_routes(self):
        """设置信号生成相关的API路由"""

        @self.app.post("/signals/generate")
        async def generate_signals(request: SignalGenerationRequest):
            """信号生成接口"""
            try:
                # 解析日期
                signal_date = datetime.fromisoformat(request.signal_date).date() if request.signal_date else date.today()

                # 生成交易信号
                result = await self.signal_service.signal_generator.generate_signals(
                    strategy_id=request.strategy_id,
                    factor_inputs=request.factor_inputs,
                    universe=request.universe,
                    signal_date=signal_date,
                    **request.signal_params
                )

                # 如果需要人工确认，创建确认记录
                if request.require_confirmation and result['success']:
                    confirmation_result = await self.signal_service.confirmation_system.create_confirmation_request(
                        signal_data=result['data'],
                        request_metadata={
                            'strategy_id': request.strategy_id,
                            'signal_date': signal_date.isoformat(),
                            'agent_id': self.agent_id
                        }
                    )
                    result['data']['confirmation_required'] = True
                    result['data']['confirmation_id'] = confirmation_result.get('confirmation_id')

                return {
                    "success": True,
                    "message": "Signal generation completed",
                    "data": result
                }

            except Exception as e:
                self.logger.error(f"Signal generation failed: {e}")
                raise HTTPException(status_code=500, detail=str(e))

        @self.app.post("/signals/confirm")
        async def confirm_signals(request: SignalConfirmationRequest):
            """信号确认接口"""
            try:
                # 处理信号确认
                result = await self.signal_service.confirmation_system.process_confirmation(
                    signal_id=request.signal_id,
                    confirmation_status=request.confirmation_status,
                    operator_id=request.operator_id,
                    confirmation_notes=request.confirmation_notes,
                    modifications=request.modifications
                )

                # 创建审计记录
                if self.config.get('enable_audit', True):
                    audit_entry = AuditEntry.create_user_action(
                        user_id=request.operator_id,
                        component="SignalGenerationAgent",
                        action_name="signal_confirmation",
                        description=f"Signal {request.signal_id} {request.confirmation_status} by {request.operator_id}",
                        action_details={
                            'signal_id': request.signal_id,
                            'status': request.confirmation_status,
                            'notes': request.confirmation_notes
                        }
                    )
                    self.logger.info(f"Confirmation audit entry created: {audit_entry.audit_id}")

                return {
                    "success": True,
                    "message": "Signal confirmation processed",
                    "data": result
                }

            except Exception as e:
                self.logger.error(f"Signal confirmation failed: {e}")
                raise HTTPException(status_code=500, detail=str(e))

        @self.app.post("/strategies/execute")
        async def execute_strategy(request: StrategyExecutionRequest):
            """策略执行接口"""
            try:
                # 执行交易策略
                result = await self.signal_service.strategy_engine.execute_strategy(
                    strategy_config=request.strategy_config,
                    execution_mode=request.execution_mode,
                    risk_limits=request.risk_limits
                )

                return {
                    "success": True,
                    "message": "Strategy execution completed",
                    "data": result
                }

            except Exception as e:
                self.logger.error(f"Strategy execution failed: {e}")
                raise HTTPException(status_code=500, detail=str(e))

        @self.app.post("/signals/monitor")
        async def monitor_signals(request: SignalMonitoringRequest):
            """信号监控接口"""
            try:
                # 启动信号监控
                result = await self.signal_service.signal_monitor.start_monitoring(
                    signal_ids=request.signal_ids,
                    monitoring_period=request.monitoring_period,
                    alert_conditions=request.alert_conditions
                )

                return {
                    "success": True,
                    "message": "Signal monitoring started",
                    "data": result
                }

            except Exception as e:
                self.logger.error(f"Signal monitoring failed: {e}")
                raise HTTPException(status_code=500, detail=str(e))

        @self.app.get("/signals/pending-confirmations")
        async def get_pending_confirmations():
            """获取待确认信号接口"""
            try:
                result = await self.signal_service.confirmation_system.get_pending_confirmations()

                return {
                    "success": True,
                    "message": "Pending confirmations retrieved",
                    "data": result
                }

            except Exception as e:
                self.logger.error(f"Get pending confirmations failed: {e}")
                raise HTTPException(status_code=500, detail=str(e))

        @self.app.get("/signals/status")
        async def get_signal_status():
            """获取信号状态接口"""
            try:
                # 获取信号生成统计
                stats = await self.signal_service.get_signal_statistics()

                return {
                    "agent_status": self.get_status(),
                    "signal_statistics": stats,
                    "active_strategies": ["momentum", "mean_reversion"],  # 简化实现
                    "pending_confirmations": 0,  # 应从确认系统获取实际数量
                    "risk_status": "normal"
                }

            except Exception as e:
                self.logger.error(f"Get signal status failed: {e}")
                raise HTTPException(status_code=500, detail=str(e))

        @self.app.get("/signals/{signal_id}")
        async def get_signal_details(signal_id: str):
            """获取信号详情接口"""
            try:
                result = await self.signal_service.get_signal_details(signal_id)

                if result:
                    return {
                        "success": True,
                        "message": "Signal details retrieved",
                        "data": result
                    }
                else:
                    raise HTTPException(status_code=404, detail=f"Signal {signal_id} not found")

            except HTTPException:
                raise
            except Exception as e:
                self.logger.error(f"Get signal details failed: {e}")
                raise HTTPException(status_code=500, detail=str(e))

        @self.app.post("/risk/check")
        async def check_risk_limits(risk_data: dict):
            """风险检查接口"""
            try:
                result = await self.signal_service.risk_controller.check_risk_limits(risk_data)

                return {
                    "success": True,
                    "message": "Risk check completed",
                    "data": result
                }

            except Exception as e:
                self.logger.error(f"Risk check failed: {e}")
                raise HTTPException(status_code=500, detail=str(e))

    def _setup_message_handlers(self):
        """设置消息处理器"""

        async def handle_signal_generation(payload: Dict[str, Any]) -> Dict[str, Any]:
            """处理信号生成消息"""
            strategy_id = payload.get('strategy_id')
            factor_inputs = payload.get('factor_inputs', {})
            universe = payload.get('universe')
            signal_date = payload.get('signal_date')
            signal_params = payload.get('signal_params', {})
            require_confirmation = payload.get('require_confirmation', True)

            # 解析日期
            signal_dt = datetime.fromisoformat(signal_date).date() if signal_date else date.today()

            result = await self.signal_service.signal_generator.generate_signals(
                strategy_id=strategy_id,
                factor_inputs=factor_inputs,
                universe=universe,
                signal_date=signal_dt,
                **signal_params
            )

            # 如果需要人工确认，创建确认记录
            if require_confirmation and result['success']:
                confirmation_result = await self.signal_service.confirmation_system.create_confirmation_request(
                    signal_data=result['data'],
                    request_metadata={
                        'strategy_id': strategy_id,
                        'signal_date': signal_dt.isoformat(),
                        'agent_id': self.agent_id
                    }
                )
                result['data']['confirmation_required'] = True
                result['data']['confirmation_id'] = confirmation_result.get('confirmation_id')

            return {
                "action": "signal_generation",
                "result": result
            }

        async def handle_signal_confirmation(payload: Dict[str, Any]) -> Dict[str, Any]:
            """处理信号确认消息"""
            signal_id = payload.get('signal_id')
            confirmation_status = payload.get('confirmation_status')
            operator_id = payload.get('operator_id')
            confirmation_notes = payload.get('confirmation_notes')
            modifications = payload.get('modifications')

            result = await self.signal_service.confirmation_system.process_confirmation(
                signal_id=signal_id,
                confirmation_status=confirmation_status,
                operator_id=operator_id,
                confirmation_notes=confirmation_notes,
                modifications=modifications
            )

            return {
                "action": "signal_confirmation",
                "result": result
            }

        async def handle_strategy_execution(payload: Dict[str, Any]) -> Dict[str, Any]:
            """处理策略执行消息"""
            strategy_config = payload.get('strategy_config', {})
            execution_mode = payload.get('execution_mode', 'simulation')
            risk_limits = payload.get('risk_limits')

            result = await self.signal_service.strategy_engine.execute_strategy(
                strategy_config=strategy_config,
                execution_mode=execution_mode,
                risk_limits=risk_limits
            )

            return {
                "action": "strategy_execution",
                "result": result
            }

        async def handle_signal_monitoring(payload: Dict[str, Any]) -> Dict[str, Any]:
            """处理信号监控消息"""
            signal_ids = payload.get('signal_ids')
            monitoring_period = payload.get('monitoring_period', 'real_time')
            alert_conditions = payload.get('alert_conditions')

            result = await self.signal_service.signal_monitor.start_monitoring(
                signal_ids=signal_ids,
                monitoring_period=monitoring_period,
                alert_conditions=alert_conditions
            )

            return {
                "action": "signal_monitoring",
                "result": result
            }

        # 注册消息处理器
        self.register_message_handler("signal_generation", handle_signal_generation)
        self.register_message_handler("signal_confirmation", handle_signal_confirmation)
        self.register_message_handler("strategy_execution", handle_strategy_execution)
        self.register_message_handler("signal_monitoring", handle_signal_monitoring)

    async def process_task(self, task_request: TaskRequest) -> TaskResult:
        """处理任务请求"""
        task_result = TaskResult.create_running(task_request.task_id, "Processing signal generation task")

        try:
            task_type = task_request.task_type
            parameters = task_request.parameters

            if task_type == "signal_generation":
                # 信号生成任务
                strategy_id = parameters.get('strategy_id')
                factor_inputs = parameters.get('factor_inputs', {})
                universe = parameters.get('universe')
                signal_date = datetime.fromisoformat(parameters['signal_date']).date() if parameters.get('signal_date') else date.today()
                signal_params = parameters.get('signal_params', {})
                require_confirmation = parameters.get('require_confirmation', True)

                result = await self.signal_service.signal_generator.generate_signals(
                    strategy_id=strategy_id,
                    factor_inputs=factor_inputs,
                    universe=universe,
                    signal_date=signal_date,
                    **signal_params
                )

                # 如果需要人工确认，创建确认记录
                if require_confirmation and result['success']:
                    confirmation_result = await self.signal_service.confirmation_system.create_confirmation_request(
                        signal_data=result['data'],
                        request_metadata={
                            'strategy_id': strategy_id,
                            'signal_date': signal_date.isoformat(),
                            'agent_id': self.agent_id
                        }
                    )
                    result['data']['confirmation_required'] = True
                    result['data']['confirmation_id'] = confirmation_result.get('confirmation_id')

                task_result.mark_completed(
                    result_data=result,
                    message="Signal generation task completed"
                )

            elif task_type == "strategy_execution":
                # 策略执行任务
                strategy_config = parameters.get('strategy_config', {})
                execution_mode = parameters.get('execution_mode', 'simulation')
                risk_limits = parameters.get('risk_limits')

                result = await self.signal_service.strategy_engine.execute_strategy(
                    strategy_config=strategy_config,
                    execution_mode=execution_mode,
                    risk_limits=risk_limits
                )

                task_result.mark_completed(
                    result_data=result,
                    message="Strategy execution task completed"
                )

            elif task_type == "signal_confirmation":
                # 信号确认任务
                signal_id = parameters.get('signal_id')
                confirmation_status = parameters.get('confirmation_status')
                operator_id = parameters.get('operator_id')
                confirmation_notes = parameters.get('confirmation_notes')
                modifications = parameters.get('modifications')

                result = await self.signal_service.confirmation_system.process_confirmation(
                    signal_id=signal_id,
                    confirmation_status=confirmation_status,
                    operator_id=operator_id,
                    confirmation_notes=confirmation_notes,
                    modifications=modifications
                )

                task_result.mark_completed(
                    result_data=result,
                    message="Signal confirmation task completed"
                )

            else:
                task_result.mark_failed(
                    error_message=f"Unknown task type: {task_type}",
                    error_code="UNKNOWN_TASK_TYPE"
                )

        except Exception as e:
            self.logger.error(f"Task processing failed: {e}")
            task_result.mark_failed(
                error_message=str(e),
                error_code="TASK_PROCESSING_ERROR"
            )

        return task_result

    async def get_capabilities(self) -> List[str]:
        """获取信号生成Agent的能力列表"""
        return [
            "signal_generation",
            "strategy_execution",
            "human_confirmation",
            "risk_control",
            "signal_monitoring",
            "real_time_processing",
            "compliance_checking",
            "performance_tracking"
        ]


# 便利函数：创建和启动信号生成Agent
async def create_signal_generation_agent(config: Optional[Dict[str, Any]] = None) -> SignalGenerationAgent:
    """创建信号生成Agent实例"""
    agent = SignalGenerationAgent(config)
    await agent.start()
    return agent


def run_signal_generation_agent(config: Optional[Dict[str, Any]] = None):
    """运行信号生成Agent服务器"""
    agent = SignalGenerationAgent(config)

    async def startup():
        await agent.start()

    # 添加启动事件处理器
    @agent.app.on_event("startup")
    async def startup_event():
        await startup()

    @agent.app.on_event("shutdown")
    async def shutdown_event():
        await agent.stop()

    # 运行服务器
    agent.run()