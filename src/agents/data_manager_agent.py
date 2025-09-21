"""
Data Manager Agent

数据管理Agent实现，提供：
1. 市场数据更新和同步
2. 数据质量检查和监控
3. 数据异常处理和恢复
4. 数据服务API接口

基于Phase 6 DataManagerService实现的微服务Agent。
"""

import asyncio
import logging
from datetime import datetime, date, timedelta
from typing import Dict, List, Optional, Any
from fastapi import HTTPException
from pydantic import BaseModel

from .base_agent import BaseAgent
from ..models.agent_models import AgentType, TaskRequest, TaskResult, AgentResponse
from ..services.data_manager_service import DataManagerService
from ..models.audit_models import AuditEntry, AuditEventType


class DataUpdateRequest(BaseModel):
    """数据更新请求模型"""
    stock_codes: Optional[List[str]] = None
    start_date: Optional[str] = None
    end_date: Optional[str] = None
    data_types: List[str] = ["kdata"]
    force_update: bool = False


class DataQualityCheckRequest(BaseModel):
    """数据质量检查请求模型"""
    data_type: str = "market_data"
    check_rules: Optional[Dict[str, Any]] = None


class DataManagerAgent(BaseAgent):
    """
    数据管理Agent

    基于DataManagerService实现的微服务Agent，
    提供RESTful API和Agent间通信接口。
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        # 默认配置
        default_config = {
            'host': '0.0.0.0',
            'port': 8001,  # 数据管理Agent默认端口
            'debug': False,
            'enable_audit': True,
            'data_manager': {
                'updater': {
                    'data_sources': ['hikyuu'],
                    'update_frequency': 'daily',
                    'retry_count': 3,
                    'max_workers': 10,
                    'batch_size': 100
                },
                'quality_checker': {
                    'rules': None  # 使用默认规则
                },
                'exception_handler': {
                    'strategies': {
                        'missing_data': 'interpolate',
                        'outlier_data': 'winsorize',
                        'duplicate_data': 'deduplicate'
                    }
                }
            }
        }

        # 合并配置
        if config:
            default_config.update(config)

        super().__init__(AgentType.DATA_MANAGER, config=default_config)

        # 数据管理服务
        self.data_service = None

        # 注册API路由
        self._setup_data_api_routes()

        # 注册消息处理器
        self._setup_message_handlers()

    async def initialize(self) -> bool:
        """初始化数据管理Agent"""
        try:
            self.logger.info("Initializing Data Manager Agent...")

            # 初始化数据管理服务
            self.data_service = DataManagerService(self.config.get('data_manager', {}))

            # 创建审计记录
            if self.config.get('enable_audit', True):
                audit_entry = AuditEntry.create_system_action(
                    component="DataManagerAgent",
                    action_name="agent_initialization",
                    description=f"Data Manager Agent {self.agent_id} initialized"
                )
                self.logger.info(f"Audit entry created: {audit_entry.audit_id}")

            self.logger.info("Data Manager Agent initialized successfully")
            return True

        except Exception as e:
            self.logger.error(f"Data Manager Agent initialization failed: {e}")
            return False

    async def cleanup(self) -> None:
        """清理资源"""
        self.logger.info("Cleaning up Data Manager Agent...")
        # 这里可以添加资源清理逻辑

    def _setup_data_api_routes(self):
        """设置数据管理相关的API路由"""

        @self.app.post("/data/update")
        async def update_market_data(request: DataUpdateRequest):
            """更新市场数据接口"""
            try:
                # 解析日期
                start_date = datetime.fromisoformat(request.start_date).date() if request.start_date else None
                end_date = datetime.fromisoformat(request.end_date).date() if request.end_date else None

                # 调用数据更新服务
                result = await self.data_service.data_updater.update_market_data(
                    stock_codes=request.stock_codes,
                    start_date=start_date,
                    end_date=end_date,
                    data_types=request.data_types
                )

                return {
                    "success": True,
                    "message": "Market data update completed",
                    "data": result
                }

            except Exception as e:
                self.logger.error(f"Market data update failed: {e}")
                raise HTTPException(status_code=500, detail=str(e))

        @self.app.post("/data/quality-check")
        async def check_data_quality(request: DataQualityCheckRequest):
            """数据质量检查接口"""
            try:
                # 获取样本数据进行质量检查
                sample_data = self.data_service._get_sample_data_for_quality_check()

                # 执行质量检查
                result = await self.data_service.quality_checker.check_data_quality(
                    sample_data,
                    data_type=request.data_type
                )

                return {
                    "success": True,
                    "message": "Data quality check completed",
                    "data": result.to_dict()
                }

            except Exception as e:
                self.logger.error(f"Data quality check failed: {e}")
                raise HTTPException(status_code=500, detail=str(e))

        @self.app.post("/data/workflow")
        async def execute_data_workflow(workflow_config: dict):
            """执行数据工作流接口"""
            try:
                result = await self.data_service.execute_data_workflow(workflow_config)

                return {
                    "success": result['success'],
                    "message": "Data workflow completed",
                    "data": result
                }

            except Exception as e:
                self.logger.error(f"Data workflow execution failed: {e}")
                raise HTTPException(status_code=500, detail=str(e))

        @self.app.get("/data/market/{stock_code}")
        async def get_stock_data(
            stock_code: str,
            start_date: Optional[str] = None,
            end_date: Optional[str] = None,
            data_type: str = "kdata"
        ):
            """获取单个股票数据接口"""
            try:
                # 解析日期
                start_dt = datetime.fromisoformat(start_date).date() if start_date else date.today() - timedelta(days=30)
                end_dt = datetime.fromisoformat(end_date).date() if end_date else date.today()

                # 获取股票数据
                result = await self.data_service.get_market_data(
                    stock_codes=[stock_code],
                    start_date=start_dt,
                    end_date=end_dt,
                    data_type=data_type
                )

                if result['success']:
                    # 转换DataFrame为JSON可序列化格式
                    data_dict = result['data'].to_dict('records') if hasattr(result['data'], 'to_dict') else result['data']
                    result['data'] = data_dict

                return result

            except Exception as e:
                self.logger.error(f"Get stock data failed: {e}")
                raise HTTPException(status_code=500, detail=str(e))

        @self.app.get("/data/status")
        async def get_data_status():
            """获取数据状态接口"""
            try:
                # 这里可以添加数据状态统计逻辑
                return {
                    "agent_status": self.get_status(),
                    "data_sources": ["hikyuu"],
                    "last_update": datetime.now().isoformat(),
                    "data_quality": "good"  # 简化实现
                }

            except Exception as e:
                self.logger.error(f"Get data status failed: {e}")
                raise HTTPException(status_code=500, detail=str(e))

    def _setup_message_handlers(self):
        """设置消息处理器"""

        async def handle_data_update(payload: Dict[str, Any]) -> Dict[str, Any]:
            """处理数据更新消息"""
            stock_codes = payload.get('stock_codes')
            start_date = payload.get('start_date')
            end_date = payload.get('end_date')
            data_types = payload.get('data_types', ['kdata'])

            # 解析日期
            start_dt = datetime.fromisoformat(start_date).date() if start_date else None
            end_dt = datetime.fromisoformat(end_date).date() if end_date else None

            result = await self.data_service.data_updater.update_market_data(
                stock_codes=stock_codes,
                start_date=start_dt,
                end_date=end_dt,
                data_types=data_types
            )

            return {
                "action": "data_update",
                "result": result
            }

        async def handle_quality_check(payload: Dict[str, Any]) -> Dict[str, Any]:
            """处理数据质量检查消息"""
            data_type = payload.get('data_type', 'market_data')

            # 获取样本数据
            sample_data = self.data_service._get_sample_data_for_quality_check()

            # 执行质量检查
            result = await self.data_service.quality_checker.check_data_quality(
                sample_data,
                data_type=data_type
            )

            return {
                "action": "quality_check",
                "result": result.to_dict()
            }

        async def handle_get_market_data(payload: Dict[str, Any]) -> Dict[str, Any]:
            """处理获取市场数据消息"""
            stock_codes = payload.get('stock_codes', [])
            start_date = payload.get('start_date')
            end_date = payload.get('end_date')
            data_type = payload.get('data_type', 'kdata')

            # 解析日期
            start_dt = datetime.fromisoformat(start_date).date() if start_date else date.today() - timedelta(days=30)
            end_dt = datetime.fromisoformat(end_date).date() if end_date else date.today()

            result = await self.data_service.get_market_data(
                stock_codes=stock_codes,
                start_date=start_dt,
                end_date=end_dt,
                data_type=data_type
            )

            # 转换DataFrame为可序列化格式
            if result['success'] and hasattr(result['data'], 'to_dict'):
                result['data'] = result['data'].to_dict('records')

            return {
                "action": "get_market_data",
                "result": result
            }

        # 注册消息处理器
        self.register_message_handler("data_update", handle_data_update)
        self.register_message_handler("quality_check", handle_quality_check)
        self.register_message_handler("get_market_data", handle_get_market_data)

    async def process_task(self, task_request: TaskRequest) -> TaskResult:
        """处理任务请求"""
        task_result = TaskResult.create_running(task_request.task_id, "Processing data management task")

        try:
            task_type = task_request.task_type
            parameters = task_request.parameters

            if task_type == "data_update":
                # 数据更新任务
                result = await self.data_service.data_updater.update_market_data(
                    stock_codes=parameters.get('stock_codes'),
                    start_date=datetime.fromisoformat(parameters['start_date']).date() if parameters.get('start_date') else None,
                    end_date=datetime.fromisoformat(parameters['end_date']).date() if parameters.get('end_date') else None,
                    data_types=parameters.get('data_types', ['kdata'])
                )

                task_result.mark_completed(
                    result_data=result,
                    message="Data update task completed"
                )

            elif task_type == "data_workflow":
                # 数据工作流任务
                workflow_config = parameters.get('workflow_config', {})
                result = await self.data_service.execute_data_workflow(workflow_config)

                task_result.mark_completed(
                    result_data=result,
                    message="Data workflow task completed"
                )

            elif task_type == "quality_check":
                # 数据质量检查任务
                sample_data = self.data_service._get_sample_data_for_quality_check()
                result = await self.data_service.quality_checker.check_data_quality(
                    sample_data,
                    data_type=parameters.get('data_type', 'market_data')
                )

                task_result.mark_completed(
                    result_data=result.to_dict(),
                    message="Data quality check task completed"
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
        """获取数据管理Agent的能力列表"""
        return [
            "market_data_update",
            "data_quality_check",
            "data_exception_handling",
            "data_workflow_execution",
            "hikyuu_integration",
            "real_time_monitoring"
        ]


# 便利函数：创建和启动数据管理Agent
async def create_data_manager_agent(config: Optional[Dict[str, Any]] = None) -> DataManagerAgent:
    """创建数据管理Agent实例"""
    agent = DataManagerAgent(config)
    await agent.start()
    return agent


def run_data_manager_agent(config: Optional[Dict[str, Any]] = None):
    """运行数据管理Agent服务器"""
    agent = DataManagerAgent(config)

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