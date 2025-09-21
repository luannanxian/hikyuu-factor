"""
Core Services Package

基于Hikyuu量化框架的核心业务服务，提供：
1. 数据管理服务 - 基于Hikyuu的数据访问和管理
2. 因子计算服务 - 高性能因子计算引擎
3. 验证服务 - 数据质量和因子验证
4. 信号生成服务 - 交易信号生成和风险控制

所有服务都支持Agent架构和RESTful API。
"""

from .data_manager_service import DataManagerService
from .factor_calculation_service import FactorCalculationService
from .validation_service import ValidationService
from .signal_generation_service import SignalGenerationService

__all__ = [
    'DataManagerService',
    'FactorCalculationService',
    'ValidationService',
    'SignalGenerationService'
]