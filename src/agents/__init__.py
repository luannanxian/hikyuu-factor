"""
Agents Package - 微服务Agent架构

基于Phase 6 Services实现的4个核心微服务Agent：
1. DataManagerAgent - 数据管理和质量检查
2. FactorCalculationAgent - 因子计算和存储
3. ValidationAgent - 因子验证和风险评估
4. SignalGenerationAgent - 交易信号生成和人工确认

所有Agent支持：
- RESTful API接口
- Agent间异步通信
- 独立部署和扩展
- 完整的审计跟踪
"""

from agents.base_agent import BaseAgent
from agents.data_manager_agent import DataManagerAgent, create_data_manager_agent, run_data_manager_agent
from agents.factor_calculation_agent import FactorCalculationAgent, create_factor_calculation_agent, run_factor_calculation_agent
from agents.validation_agent import ValidationAgent, create_validation_agent, run_validation_agent
from agents.signal_generation_agent import SignalGenerationAgent, create_signal_generation_agent, run_signal_generation_agent

__all__ = [
    'BaseAgent',
    'DataManagerAgent',
    'FactorCalculationAgent',
    'ValidationAgent',
    'SignalGenerationAgent',
    'create_data_manager_agent',
    'run_data_manager_agent',
    'create_factor_calculation_agent',
    'run_factor_calculation_agent',
    'create_validation_agent',
    'run_validation_agent',
    'create_signal_generation_agent',
    'run_signal_generation_agent'
]
