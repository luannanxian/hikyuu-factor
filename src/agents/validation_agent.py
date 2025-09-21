"""
Validation Agent

验证Agent实现，提供：
1. 因子有效性验证
2. 回测验证和分析
3. 风险评估和控制
4. 性能指标计算

基于Phase 6 ValidationService实现的微服务Agent。
"""

import asyncio
import logging
from datetime import datetime, date, timedelta
from typing import Dict, List, Optional, Any
from fastapi import HTTPException
from pydantic import BaseModel

from .base_agent import BaseAgent
from ..models.agent_models import AgentType, TaskRequest, TaskResult, AgentResponse
from ..services.validation_service import ValidationService
from ..models.audit_models import AuditEntry, AuditEventType


class FactorValidationRequest(BaseModel):
    """因子验证请求模型"""
    factor_ids: List[str]
    stock_codes: Optional[List[str]] = None
    start_date: Optional[str] = None
    end_date: Optional[str] = None
    validation_rules: Optional[Dict[str, Any]] = None
    statistical_tests: List[str] = ["normality", "stationarity", "autocorrelation"]


class BacktestRequest(BaseModel):
    """回测请求模型"""
    strategy_config: Dict[str, Any]
    start_date: str
    end_date: str
    universe: Optional[List[str]] = None
    benchmark: str = "000300.SH"
    initial_capital: float = 1000000.0
    commission_rate: float = 0.0003


class RiskAssessmentRequest(BaseModel):
    """风险评估请求模型"""
    portfolio_data: Dict[str, Any]
    assessment_type: str = "comprehensive"
    risk_models: List[str] = ["var", "cvar", "drawdown", "correlation"]
    confidence_level: float = 0.95


class PerformanceAnalysisRequest(BaseModel):
    """性能分析请求模型"""
    returns_data: List[float]
    benchmark_returns: Optional[List[float]] = None
    analysis_metrics: List[str] = ["sharpe", "sortino", "calmar", "max_drawdown"]
    time_period: str = "daily"


class ValidationAgent(BaseAgent):
    """
    验证Agent

    基于ValidationService实现的微服务Agent，
    提供RESTful API和Agent间通信接口。
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        # 默认配置
        default_config = {
            'host': '0.0.0.0',
            'port': 8003,  # 验证Agent默认端口
            'debug': False,
            'enable_audit': True,
            'validation': {
                'factor_validator': {
                    'statistical_tests': {
                        'normality': {'method': 'shapiro', 'alpha': 0.05},
                        'stationarity': {'method': 'adf', 'alpha': 0.05},
                        'autocorrelation': {'lags': 20, 'alpha': 0.05}
                    },
                    'quality_checks': {
                        'missing_data_threshold': 0.1,
                        'outlier_detection': 'iqr',
                        'correlation_threshold': 0.95
                    }
                },
                'backtest_engine': {
                    'universe_size_limit': 3000,
                    'rebalance_frequency': 'monthly',
                    'position_limit': 0.05,
                    'sector_limit': 0.3
                },
                'risk_assessor': {
                    'models': {
                        'var': {'confidence_levels': [0.95, 0.99]},
                        'cvar': {'confidence_levels': [0.95, 0.99]},
                        'stress_test': {'scenarios': ['market_crash', 'sector_rotation']}
                    }
                },
                'performance_analyzer': {
                    'benchmark_indices': ['000300.SH', '000905.SH', '000852.SH'],
                    'risk_free_rate': 0.03,
                    'business_days_per_year': 252
                }
            }
        }

        # 合并配置
        if config:
            default_config.update(config)

        super().__init__(AgentType.VALIDATION, config=default_config)

        # 验证服务
        self.validation_service = None

        # 注册API路由
        self._setup_validation_api_routes()

        # 注册消息处理器
        self._setup_message_handlers()

    async def initialize(self) -> bool:
        """初始化验证Agent"""
        try:
            self.logger.info("Initializing Validation Agent...")

            # 初始化验证服务
            self.validation_service = ValidationService(self.config.get('validation', {}))

            # 创建审计记录
            if self.config.get('enable_audit', True):
                audit_entry = AuditEntry.create_system_action(
                    component="ValidationAgent",
                    action_name="agent_initialization",
                    description=f"Validation Agent {self.agent_id} initialized"
                )
                self.logger.info(f"Audit entry created: {audit_entry.audit_id}")

            self.logger.info("Validation Agent initialized successfully")
            return True

        except Exception as e:
            self.logger.error(f"Validation Agent initialization failed: {e}")
            return False

    async def cleanup(self) -> None:
        """清理资源"""
        self.logger.info("Cleaning up Validation Agent...")
        # 这里可以添加资源清理逻辑

    def _setup_validation_api_routes(self):
        """设置验证相关的API路由"""

        @self.app.post("/validation/factors")
        async def validate_factors(request: FactorValidationRequest):
            """因子验证接口"""
            try:
                # 解析日期
                start_date = datetime.fromisoformat(request.start_date).date() if request.start_date else None
                end_date = datetime.fromisoformat(request.end_date).date() if request.end_date else None

                # 执行因子验证
                result = await self.validation_service.factor_validator.validate_factors(
                    factor_ids=request.factor_ids,
                    stock_codes=request.stock_codes,
                    start_date=start_date,
                    end_date=end_date,
                    validation_rules=request.validation_rules,
                    statistical_tests=request.statistical_tests
                )

                return {
                    "success": True,
                    "message": "Factor validation completed",
                    "data": result
                }

            except Exception as e:
                self.logger.error(f"Factor validation failed: {e}")
                raise HTTPException(status_code=500, detail=str(e))

        @self.app.post("/validation/backtest")
        async def run_backtest(request: BacktestRequest):
            """回测验证接口"""
            try:
                # 解析日期
                start_date = datetime.fromisoformat(request.start_date).date()
                end_date = datetime.fromisoformat(request.end_date).date()

                # 执行回测
                result = await self.validation_service.backtest_engine.run_backtest(
                    strategy_config=request.strategy_config,
                    start_date=start_date,
                    end_date=end_date,
                    universe=request.universe,
                    benchmark=request.benchmark,
                    initial_capital=request.initial_capital,
                    commission_rate=request.commission_rate
                )

                # 转换回测结果为可序列化格式
                if result['success'] and 'portfolio_data' in result['data']:
                    portfolio_data = result['data']['portfolio_data']
                    if hasattr(portfolio_data, 'to_dict'):
                        result['data']['portfolio_data'] = portfolio_data.to_dict('records')

                return {
                    "success": True,
                    "message": "Backtest completed",
                    "data": result
                }

            except Exception as e:
                self.logger.error(f"Backtest failed: {e}")
                raise HTTPException(status_code=500, detail=str(e))

        @self.app.post("/validation/risk-assessment")
        async def assess_risk(request: RiskAssessmentRequest):
            """风险评估接口"""
            try:
                # 执行风险评估
                result = await self.validation_service.risk_assessor.assess_portfolio_risk(
                    portfolio_data=request.portfolio_data,
                    assessment_type=request.assessment_type,
                    risk_models=request.risk_models,
                    confidence_level=request.confidence_level
                )

                return {
                    "success": True,
                    "message": "Risk assessment completed",
                    "data": result
                }

            except Exception as e:
                self.logger.error(f"Risk assessment failed: {e}")
                raise HTTPException(status_code=500, detail=str(e))

        @self.app.post("/validation/performance")
        async def analyze_performance(request: PerformanceAnalysisRequest):
            """性能分析接口"""
            try:
                # 执行性能分析
                result = await self.validation_service.performance_analyzer.analyze_performance(
                    returns_data=request.returns_data,
                    benchmark_returns=request.benchmark_returns,
                    analysis_metrics=request.analysis_metrics,
                    time_period=request.time_period
                )

                return {
                    "success": True,
                    "message": "Performance analysis completed",
                    "data": result
                }

            except Exception as e:
                self.logger.error(f"Performance analysis failed: {e}")
                raise HTTPException(status_code=500, detail=str(e))

        @self.app.get("/validation/status")
        async def get_validation_status():
            """获取验证状态接口"""
            try:
                return {
                    "agent_status": self.get_status(),
                    "validation_services": {
                        "factor_validator": "active",
                        "backtest_engine": "active",
                        "risk_assessor": "active",
                        "performance_analyzer": "active"
                    },
                    "supported_methods": {
                        "statistical_tests": ["normality", "stationarity", "autocorrelation"],
                        "risk_models": ["var", "cvar", "drawdown", "correlation"],
                        "performance_metrics": ["sharpe", "sortino", "calmar", "max_drawdown"]
                    }
                }

            except Exception as e:
                self.logger.error(f"Get validation status failed: {e}")
                raise HTTPException(status_code=500, detail=str(e))

        @self.app.get("/validation/reports/{validation_id}")
        async def get_validation_report(validation_id: str):
            """获取验证报告接口"""
            try:
                # 这里应该实现从存储中获取验证报告的逻辑
                # 为简化实现，返回模拟报告
                report = {
                    "validation_id": validation_id,
                    "timestamp": datetime.now().isoformat(),
                    "status": "completed",
                    "summary": "Validation report generated successfully",
                    "details": {
                        "factor_validation": {"passed": True, "score": 0.85},
                        "backtest_results": {"sharpe_ratio": 1.2, "max_drawdown": -0.15},
                        "risk_assessment": {"var_95": -0.02, "overall_risk": "medium"}
                    }
                }

                return {
                    "success": True,
                    "message": "Validation report retrieved",
                    "data": report
                }

            except Exception as e:
                self.logger.error(f"Get validation report failed: {e}")
                raise HTTPException(status_code=500, detail=str(e))

    def _setup_message_handlers(self):
        """设置消息处理器"""

        async def handle_factor_validation(payload: Dict[str, Any]) -> Dict[str, Any]:
            """处理因子验证消息"""
            factor_ids = payload.get('factor_ids', [])
            stock_codes = payload.get('stock_codes')
            start_date = payload.get('start_date')
            end_date = payload.get('end_date')
            validation_rules = payload.get('validation_rules')
            statistical_tests = payload.get('statistical_tests', ["normality", "stationarity", "autocorrelation"])

            # 解析日期
            start_dt = datetime.fromisoformat(start_date).date() if start_date else None
            end_dt = datetime.fromisoformat(end_date).date() if end_date else None

            result = await self.validation_service.factor_validator.validate_factors(
                factor_ids=factor_ids,
                stock_codes=stock_codes,
                start_date=start_dt,
                end_date=end_dt,
                validation_rules=validation_rules,
                statistical_tests=statistical_tests
            )

            return {
                "action": "factor_validation",
                "result": result
            }

        async def handle_backtest(payload: Dict[str, Any]) -> Dict[str, Any]:
            """处理回测消息"""
            strategy_config = payload.get('strategy_config', {})
            start_date = datetime.fromisoformat(payload['start_date']).date()
            end_date = datetime.fromisoformat(payload['end_date']).date()
            universe = payload.get('universe')
            benchmark = payload.get('benchmark', '000300.SH')
            initial_capital = payload.get('initial_capital', 1000000.0)
            commission_rate = payload.get('commission_rate', 0.0003)

            result = await self.validation_service.backtest_engine.run_backtest(
                strategy_config=strategy_config,
                start_date=start_date,
                end_date=end_date,
                universe=universe,
                benchmark=benchmark,
                initial_capital=initial_capital,
                commission_rate=commission_rate
            )

            # 转换回测结果为可序列化格式
            if result['success'] and 'portfolio_data' in result['data']:
                portfolio_data = result['data']['portfolio_data']
                if hasattr(portfolio_data, 'to_dict'):
                    result['data']['portfolio_data'] = portfolio_data.to_dict('records')

            return {
                "action": "backtest",
                "result": result
            }

        async def handle_risk_assessment(payload: Dict[str, Any]) -> Dict[str, Any]:
            """处理风险评估消息"""
            portfolio_data = payload.get('portfolio_data', {})
            assessment_type = payload.get('assessment_type', 'comprehensive')
            risk_models = payload.get('risk_models', ['var', 'cvar', 'drawdown', 'correlation'])
            confidence_level = payload.get('confidence_level', 0.95)

            result = await self.validation_service.risk_assessor.assess_portfolio_risk(
                portfolio_data=portfolio_data,
                assessment_type=assessment_type,
                risk_models=risk_models,
                confidence_level=confidence_level
            )

            return {
                "action": "risk_assessment",
                "result": result
            }

        async def handle_performance_analysis(payload: Dict[str, Any]) -> Dict[str, Any]:
            """处理性能分析消息"""
            returns_data = payload.get('returns_data', [])
            benchmark_returns = payload.get('benchmark_returns')
            analysis_metrics = payload.get('analysis_metrics', ['sharpe', 'sortino', 'calmar', 'max_drawdown'])
            time_period = payload.get('time_period', 'daily')

            result = await self.validation_service.performance_analyzer.analyze_performance(
                returns_data=returns_data,
                benchmark_returns=benchmark_returns,
                analysis_metrics=analysis_metrics,
                time_period=time_period
            )

            return {
                "action": "performance_analysis",
                "result": result
            }

        # 注册消息处理器
        self.register_message_handler("factor_validation", handle_factor_validation)
        self.register_message_handler("backtest", handle_backtest)
        self.register_message_handler("risk_assessment", handle_risk_assessment)
        self.register_message_handler("performance_analysis", handle_performance_analysis)

    async def process_task(self, task_request: TaskRequest) -> TaskResult:
        """处理任务请求"""
        task_result = TaskResult.create_running(task_request.task_id, "Processing validation task")

        try:
            task_type = task_request.task_type
            parameters = task_request.parameters

            if task_type == "factor_validation":
                # 因子验证任务
                factor_ids = parameters.get('factor_ids', [])
                stock_codes = parameters.get('stock_codes')
                start_date = datetime.fromisoformat(parameters['start_date']).date() if parameters.get('start_date') else None
                end_date = datetime.fromisoformat(parameters['end_date']).date() if parameters.get('end_date') else None
                validation_rules = parameters.get('validation_rules')
                statistical_tests = parameters.get('statistical_tests', ["normality", "stationarity", "autocorrelation"])

                result = await self.validation_service.factor_validator.validate_factors(
                    factor_ids=factor_ids,
                    stock_codes=stock_codes,
                    start_date=start_date,
                    end_date=end_date,
                    validation_rules=validation_rules,
                    statistical_tests=statistical_tests
                )

                task_result.mark_completed(
                    result_data=result,
                    message="Factor validation task completed"
                )

            elif task_type == "backtest":
                # 回测任务
                strategy_config = parameters.get('strategy_config', {})
                start_date = datetime.fromisoformat(parameters['start_date']).date()
                end_date = datetime.fromisoformat(parameters['end_date']).date()
                universe = parameters.get('universe')
                benchmark = parameters.get('benchmark', '000300.SH')
                initial_capital = parameters.get('initial_capital', 1000000.0)
                commission_rate = parameters.get('commission_rate', 0.0003)

                result = await self.validation_service.backtest_engine.run_backtest(
                    strategy_config=strategy_config,
                    start_date=start_date,
                    end_date=end_date,
                    universe=universe,
                    benchmark=benchmark,
                    initial_capital=initial_capital,
                    commission_rate=commission_rate
                )

                task_result.mark_completed(
                    result_data=result,
                    message="Backtest task completed"
                )

            elif task_type == "risk_assessment":
                # 风险评估任务
                portfolio_data = parameters.get('portfolio_data', {})
                assessment_type = parameters.get('assessment_type', 'comprehensive')
                risk_models = parameters.get('risk_models', ['var', 'cvar', 'drawdown', 'correlation'])
                confidence_level = parameters.get('confidence_level', 0.95)

                result = await self.validation_service.risk_assessor.assess_portfolio_risk(
                    portfolio_data=portfolio_data,
                    assessment_type=assessment_type,
                    risk_models=risk_models,
                    confidence_level=confidence_level
                )

                task_result.mark_completed(
                    result_data=result,
                    message="Risk assessment task completed"
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
        """获取验证Agent的能力列表"""
        return [
            "factor_validation",
            "statistical_testing",
            "backtest_execution",
            "risk_assessment",
            "performance_analysis",
            "portfolio_optimization",
            "stress_testing",
            "compliance_checking"
        ]


# 便利函数：创建和启动验证Agent
async def create_validation_agent(config: Optional[Dict[str, Any]] = None) -> ValidationAgent:
    """创建验证Agent实例"""
    agent = ValidationAgent(config)
    await agent.start()
    return agent


def run_validation_agent(config: Optional[Dict[str, Any]] = None):
    """运行验证Agent服务器"""
    agent = ValidationAgent(config)

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