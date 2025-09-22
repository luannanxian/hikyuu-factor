"""
Factor Calculation Agent

因子计算Agent实现，提供：
1. 因子注册和管理
2. 高性能因子计算
3. 因子存储和查询
4. 平台优化和监控

基于Phase 6 FactorCalculationService实现的微服务Agent。
"""

import asyncio
import logging
from datetime import datetime, date, timedelta
from typing import Dict, List, Optional, Any
from fastapi import HTTPException
from pydantic import BaseModel

from agents.base_agent import BaseAgent
from models.agent_models import AgentType, TaskRequest, TaskResult, AgentResponse
from services.factor_calculation_service import FactorCalculationService
from models.audit_models import AuditEntry, AuditEventType


class FactorRegistrationRequest(BaseModel):
    """因子注册请求模型"""
    factor_id: str
    factor_name: str
    factor_type: str
    calculation_method: str
    dependencies: Optional[List[str]] = []
    parameters: Optional[Dict[str, Any]] = {}
    description: Optional[str] = None


class FactorCalculationRequest(BaseModel):
    """因子计算请求模型"""
    factor_ids: List[str]
    stock_codes: Optional[List[str]] = None
    start_date: Optional[str] = None
    end_date: Optional[str] = None
    save_results: bool = True
    calculation_params: Optional[Dict[str, Any]] = {}


class FactorQueryRequest(BaseModel):
    """因子查询请求模型"""
    factor_ids: Optional[List[str]] = None
    stock_codes: Optional[List[str]] = None
    start_date: Optional[str] = None
    end_date: Optional[str] = None
    limit: Optional[int] = 1000
    offset: Optional[int] = 0


class FactorCalculationAgent(BaseAgent):
    """
    因子计算Agent

    基于FactorCalculationService实现的微服务Agent，
    提供RESTful API和Agent间通信接口。
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        # 默认配置
        default_config = {
            'host': '0.0.0.0',
            'port': 8002,  # 因子计算Agent默认端口
            'debug': False,
            'enable_audit': True,
            'factor_calculation': {
                'platform_optimizer': {
                    'enable_optimization': True,
                    'cache_optimization': True,
                    'parallel_workers': None  # 自动检测
                },
                'factor_registry': {
                    'auto_discover': True,
                    'registry_file': 'config/factor_registry.yaml'
                },
                'factor_calculator': {
                    'batch_size': 100,
                    'max_workers': 10,
                    'timeout_minutes': 30,
                    'memory_limit_gb': 8
                },
                'factor_storage': {
                    'storage_backend': 'mysql',
                    'compression': True,
                    'partition_by_date': True
                }
            }
        }

        # 合并配置
        if config:
            default_config.update(config)

        super().__init__(AgentType.FACTOR_CALCULATION, config=default_config)

        # 因子计算服务
        self.factor_service = None

        # 注册API路由
        self._setup_factor_api_routes()

        # 注册消息处理器
        self._setup_message_handlers()

    async def initialize(self) -> bool:
        """初始化因子计算Agent"""
        try:
            self.logger.info("Initializing Factor Calculation Agent...")

            # 初始化因子计算服务
            self.factor_service = FactorCalculationService(self.config.get('factor_calculation', {}))

            # 创建审计记录
            if self.config.get('enable_audit', True):
                audit_entry = AuditEntry.create_system_action(
                    component="FactorCalculationAgent",
                    action_name="agent_initialization",
                    description=f"Factor Calculation Agent {self.agent_id} initialized"
                )
                self.logger.info(f"Audit entry created: {audit_entry.audit_id}")

            self.logger.info("Factor Calculation Agent initialized successfully")
            return True

        except Exception as e:
            self.logger.error(f"Factor Calculation Agent initialization failed: {e}")
            return False

    async def cleanup(self) -> None:
        """清理资源"""
        self.logger.info("Cleaning up Factor Calculation Agent...")
        # 这里可以添加资源清理逻辑

    def _setup_factor_api_routes(self) -> None:
        """设置因子计算相关的API路由"""

        @self.app.post("/factors/register")
        async def register_factor(request: FactorRegistrationRequest) -> None:
            """因子注册接口"""
            try:
                # 注册因子
                result = await self.factor_service.factor_registry.register_factor(
                    factor_id=request.factor_id,
                    factor_name=request.factor_name,
                    factor_type=request.factor_type,
                    calculation_method=request.calculation_method,
                    dependencies=request.dependencies,
                    parameters=request.parameters,
                    description=request.description
                )

                return {
                    "success": True,
                    "message": "Factor registered successfully",
                    "data": result
                }

            except Exception as e:
                self.logger.error(f"Factor registration failed: {e}")
                raise HTTPException(status_code=500, detail=str(e))

        @self.app.post("/factors/calculate")
        async def calculate_factors(request: FactorCalculationRequest) -> None:
            """因子计算接口"""
            try:
                # 解析日期
                start_date = datetime.fromisoformat(request.start_date).date() if request.start_date else None
                end_date = datetime.fromisoformat(request.end_date).date() if request.end_date else None

                # 执行因子计算
                result = await self.factor_service.factor_calculator.calculate_factors(
                    factor_ids=request.factor_ids,
                    stock_codes=request.stock_codes,
                    start_date=start_date,
                    end_date=end_date,
                    save_results=request.save_results,
                    **request.calculation_params
                )

                return {
                    "success": True,
                    "message": "Factor calculation completed",
                    "data": result
                }

            except Exception as e:
                self.logger.error(f"Factor calculation failed: {e}")
                raise HTTPException(status_code=500, detail=str(e))

        @self.app.post("/factors/query")
        async def query_factors(request: FactorQueryRequest) -> None:
            """因子查询接口"""
            try:
                # 解析日期
                start_date = datetime.fromisoformat(request.start_date).date() if request.start_date else None
                end_date = datetime.fromisoformat(request.end_date).date() if request.end_date else None

                # 查询因子数据
                result = await self.factor_service.factor_query_engine.query_factors(
                    factor_ids=request.factor_ids,
                    stock_codes=request.stock_codes,
                    start_date=start_date,
                    end_date=end_date,
                    limit=request.limit,
                    offset=request.offset
                )

                # 转换DataFrame为JSON可序列化格式
                if result['success'] and hasattr(result['data'], 'to_dict'):
                    result['data'] = result['data'].to_dict('records')

                return result

            except Exception as e:
                self.logger.error(f"Factor query failed: {e}")
                raise HTTPException(status_code=500, detail=str(e))

        @self.app.get("/factors/registry")
        async def get_factor_registry() -> None:
            """获取因子注册表接口"""
            try:
                result = await self.factor_service.factor_registry.get_all_factors()

                return {
                    "success": True,
                    "message": "Factor registry retrieved",
                    "data": result
                }

            except Exception as e:
                self.logger.error(f"Get factor registry failed: {e}")
                raise HTTPException(status_code=500, detail=str(e))

        @self.app.get("/factors/{factor_id}")
        async def get_factor_info(factor_id: str) -> None:
            """获取单个因子信息接口"""
            try:
                result = await self.factor_service.factor_registry.get_factor_info(factor_id)

                if result:
                    return {
                        "success": True,
                        "message": "Factor info retrieved",
                        "data": result
                    }
                else:
                    raise HTTPException(status_code=404, detail=f"Factor {factor_id} not found")

            except HTTPException:
                raise
            except Exception as e:
                self.logger.error(f"Get factor info failed: {e}")
                raise HTTPException(status_code=500, detail=str(e))

        @self.app.get("/platform/status")
        async def get_platform_status() -> None:
            """获取平台状态接口"""
            try:
                platform_info = await self.factor_service.platform_optimizer.get_platform_info()
                optimization_config = await self.factor_service.platform_optimizer.get_optimization_config()

                return {
                    "agent_status": self.get_status(),
                    "platform_info": platform_info,
                    "optimization_config": optimization_config,
                    "service_status": "active"
                }

            except Exception as e:
                self.logger.error(f"Get platform status failed: {e}")
                raise HTTPException(status_code=500, detail=str(e))

        @self.app.post("/platform/optimize")
        async def optimize_platform() -> None:
            """平台优化接口"""
            try:
                result = await self.factor_service.platform_optimizer.optimize_performance()

                return {
                    "success": True,
                    "message": "Platform optimization completed",
                    "data": result
                }

            except Exception as e:
                self.logger.error(f"Platform optimization failed: {e}")
                raise HTTPException(status_code=500, detail=str(e))

    def _setup_message_handlers(self) -> None:
        """设置消息处理器"""

        async def handle_factor_calculation(payload: Dict[str, Any]) -> Dict[str, Any]:
            """处理因子计算消息"""
            factor_ids = payload.get('factor_ids', [])
            stock_codes = payload.get('stock_codes')
            start_date = payload.get('start_date')
            end_date = payload.get('end_date')
            save_results = payload.get('save_results', True)
            calculation_params = payload.get('calculation_params', {})

            # 解析日期
            start_dt = datetime.fromisoformat(start_date).date() if start_date else None
            end_dt = datetime.fromisoformat(end_date).date() if end_date else None

            result = await self.factor_service.factor_calculator.calculate_factors(
                factor_ids=factor_ids,
                stock_codes=stock_codes,
                start_date=start_dt,
                end_date=end_dt,
                save_results=save_results,
                **calculation_params
            )

            return {
                "action": "factor_calculation",
                "result": result
            }

        async def handle_factor_registration(payload: Dict[str, Any]) -> Dict[str, Any]:
            """处理因子注册消息"""
            result = await self.factor_service.factor_registry.register_factor(
                factor_id=payload.get('factor_id'),
                factor_name=payload.get('factor_name'),
                factor_type=payload.get('factor_type'),
                calculation_method=payload.get('calculation_method'),
                dependencies=payload.get('dependencies', []),
                parameters=payload.get('parameters', {}),
                description=payload.get('description')
            )

            return {
                "action": "factor_registration",
                "result": result
            }

        async def handle_factor_query(payload: Dict[str, Any]) -> Dict[str, Any]:
            """处理因子查询消息"""
            factor_ids = payload.get('factor_ids')
            stock_codes = payload.get('stock_codes')
            start_date = payload.get('start_date')
            end_date = payload.get('end_date')
            limit = payload.get('limit', 1000)
            offset = payload.get('offset', 0)

            # 解析日期
            start_dt = datetime.fromisoformat(start_date).date() if start_date else None
            end_dt = datetime.fromisoformat(end_date).date() if end_date else None

            result = await self.factor_service.factor_query_engine.query_factors(
                factor_ids=factor_ids,
                stock_codes=stock_codes,
                start_date=start_dt,
                end_date=end_dt,
                limit=limit,
                offset=offset
            )

            # 转换DataFrame为可序列化格式
            if result['success'] and hasattr(result['data'], 'to_dict'):
                result['data'] = result['data'].to_dict('records')

            return {
                "action": "factor_query",
                "result": result
            }

        async def handle_platform_optimization(payload: Dict[str, Any]) -> Dict[str, Any]:
            """处理平台优化消息"""
            result = await self.factor_service.platform_optimizer.optimize_performance()

            return {
                "action": "platform_optimization",
                "result": result
            }

        # 注册消息处理器
        self.register_message_handler("factor_calculation", handle_factor_calculation)
        self.register_message_handler("factor_registration", handle_factor_registration)
        self.register_message_handler("factor_query", handle_factor_query)
        self.register_message_handler("platform_optimization", handle_platform_optimization)

    async def process_task(self, task_request: TaskRequest) -> TaskResult:
        """处理任务请求"""
        task_result = TaskResult.create_running(task_request.task_id, "Processing factor calculation task")

        try:
            task_type = task_request.task_type
            parameters = task_request.parameters

            if task_type == "factor_calculation":
                # 因子计算任务
                factor_ids = parameters.get('factor_ids', [])
                stock_codes = parameters.get('stock_codes')
                start_date = datetime.fromisoformat(parameters['start_date']).date() if parameters.get('start_date') else None
                end_date = datetime.fromisoformat(parameters['end_date']).date() if parameters.get('end_date') else None
                save_results = parameters.get('save_results', True)
                calculation_params = parameters.get('calculation_params', {})

                result = await self.factor_service.factor_calculator.calculate_factors(
                    factor_ids=factor_ids,
                    stock_codes=stock_codes,
                    start_date=start_date,
                    end_date=end_date,
                    save_results=save_results,
                    **calculation_params
                )

                task_result.mark_completed(
                    result_data=result,
                    message="Factor calculation task completed"
                )

            elif task_type == "factor_registration":
                # 因子注册任务
                result = await self.factor_service.factor_registry.register_factor(
                    factor_id=parameters.get('factor_id'),
                    factor_name=parameters.get('factor_name'),
                    factor_type=parameters.get('factor_type'),
                    calculation_method=parameters.get('calculation_method'),
                    dependencies=parameters.get('dependencies', []),
                    parameters=parameters.get('parameters', {}),
                    description=parameters.get('description')
                )

                task_result.mark_completed(
                    result_data=result,
                    message="Factor registration task completed"
                )

            elif task_type == "platform_optimization":
                # 平台优化任务
                result = await self.factor_service.platform_optimizer.optimize_performance()

                task_result.mark_completed(
                    result_data=result,
                    message="Platform optimization task completed"
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
        """获取因子计算Agent的能力列表"""
        return [
            "factor_registration",
            "factor_calculation",
            "factor_storage",
            "factor_query",
            "platform_optimization",
            "batch_processing",
            "real_time_calculation",
            "performance_monitoring"
        ]


# 便利函数：创建和启动因子计算Agent
async def create_factor_calculation_agent(config: Optional[Dict[str, Any]] = None) -> FactorCalculationAgent:
    """创建因子计算Agent实例"""
    agent = FactorCalculationAgent(config)
    await agent.start()
    return agent


def run_factor_calculation_agent(config: Optional[Dict[str, Any]] = None) -> None:
    """运行因子计算Agent服务器"""
    agent = FactorCalculationAgent(config)

    async def startup() -> None:
        await agent.start()

    # 添加启动事件处理器
    @agent.app.on_event("startup")
    async def startup_event() -> None:
        await startup()

    @agent.app.on_event("shutdown")
    async def shutdown_event() -> None:
        await agent.stop()

    # 运行服务器
    agent.run()
