"""
Factor Calculation Service Unit Tests

基于真实Hikyuu框架的因子计算服务单元测试
不使用mock数据，测试PlatformOptimizer, FactorRegistry, FactorCalculator功能
"""

import pytest
import asyncio
import pandas as pd
import numpy as np
import platform
import multiprocessing as mp
from datetime import datetime, date, timedelta
from typing import Dict, List, Any

from src.services.factor_calculation_service import (
    PlatformOptimizer, FactorRegistry, FactorCalculator
)
from src.models.hikyuu_models import (
    FactorType, FactorCalculationRequest, FactorCalculationResult
)


class TestPlatformOptimizer:
    """PlatformOptimizer组件单元测试"""

    @pytest.fixture
    def platform_optimizer(self):
        """创建PlatformOptimizer实例"""
        return PlatformOptimizer()

    def test_platform_optimizer_initialization(self, platform_optimizer):
        """测试PlatformOptimizer初始化"""
        assert platform_optimizer.platform_info is not None
        assert platform_optimizer.optimization_config is not None

        # 验证平台信息结构
        required_keys = ["system", "machine", "cpu_count", "architecture"]
        for key in required_keys:
            assert key in platform_optimizer.platform_info

        # 验证优化配置结构
        config_keys = ["cpu_optimization", "memory_optimization", "threading_config", "computation_config"]
        for key in config_keys:
            assert key in platform_optimizer.optimization_config

    def test_platform_detection(self, platform_optimizer):
        """测试平台检测功能"""
        platform_info = platform_optimizer.platform_info

        # 验证系统信息
        assert platform_info["system"] in ["Darwin", "Linux", "Windows"]
        assert platform_info["cpu_count"] > 0
        assert platform_info["architecture"] in [
            "apple_silicon", "x86_64", "x86_64_macos", "arm64_linux", "unknown"
        ]

        # 验证特定平台的配置
        if platform_info["architecture"] == "apple_silicon":
            assert "performance_cores" in platform_info
            assert "efficiency_cores" in platform_info
            assert platform_info["cpu_features"] == ["neon", "fp16"]
        elif platform_info["architecture"] in ["x86_64", "x86_64_macos"]:
            assert "avx" in platform_info.get("cpu_features", [])

    def test_optimization_config_apple_silicon(self):
        """测试Apple Silicon优化配置"""
        # 模拟Apple Silicon环境
        original_system = platform.system
        original_machine = platform.machine

        try:
            platform.system = lambda: "Darwin"
            platform.machine = lambda: "arm64"

            optimizer = PlatformOptimizer()
            config = optimizer.optimization_config

            if optimizer.platform_info["architecture"] == "apple_silicon":
                # 验证Apple Silicon特有配置
                assert config["cpu_optimization"]["neon_enabled"] is True
                assert config["cpu_optimization"]["fp16_enabled"] is True
                assert config["threading_config"]["use_performance_cores_only"] is True
                assert config["computation_config"]["use_accelerate"] is True

        finally:
            platform.system = original_system
            platform.machine = original_machine

    def test_optimization_config_x86_64(self):
        """测试x86_64优化配置"""
        # 模拟x86_64环境
        original_system = platform.system
        original_machine = platform.machine

        try:
            platform.system = lambda: "Linux"
            platform.machine = lambda: "x86_64"

            optimizer = PlatformOptimizer()
            config = optimizer.optimization_config

            if optimizer.platform_info["architecture"] == "x86_64":
                # 验证x86_64特有配置
                assert config["cpu_optimization"]["avx_enabled"] is True
                assert config["cpu_optimization"]["avx2_enabled"] is True
                assert config["computation_config"]["use_mkl"] is True

        finally:
            platform.system = original_system
            platform.machine = original_machine

    def test_optimal_worker_count_calculation(self, platform_optimizer):
        """测试最优工作进程数计算"""
        # 测试不同任务大小
        small_task = platform_optimizer.get_optimal_worker_count(500)
        medium_task = platform_optimizer.get_optimal_worker_count(2000)
        large_task = platform_optimizer.get_optimal_worker_count(10000)

        assert isinstance(small_task, int)
        assert isinstance(medium_task, int)
        assert isinstance(large_task, int)

        assert small_task >= 1
        assert medium_task >= small_task
        assert large_task >= 1

        # 验证不超过系统核心数
        assert small_task <= mp.cpu_count()
        assert medium_task <= mp.cpu_count()
        assert large_task <= mp.cpu_count()

    def test_optimize_computation(self, platform_optimizer):
        """测试计算优化功能"""
        # 创建测试数据
        test_data = pd.DataFrame({
            'close': np.random.randn(1000) + 100,
            'volume': np.random.randint(1000, 10000, 1000),
            'date': pd.date_range('2024-01-01', periods=1000)
        })

        # 应用优化
        optimized_data = platform_optimizer.optimize_computation(test_data)

        assert isinstance(optimized_data, pd.DataFrame)
        assert len(optimized_data) == len(test_data)
        assert list(optimized_data.columns) == list(test_data.columns)

    def test_platform_specific_optimizations(self, platform_optimizer):
        """测试平台特定优化方法"""
        test_data = pd.DataFrame({
            'values': np.random.randn(500)
        })

        if platform_optimizer.platform_info["architecture"] == "apple_silicon":
            result = platform_optimizer._apple_silicon_optimization(test_data)
        else:
            result = platform_optimizer._x86_64_optimization(test_data)

        assert isinstance(result, pd.DataFrame)
        assert len(result) == len(test_data)


class TestFactorRegistry:
    """FactorRegistry组件单元测试"""

    @pytest.fixture
    def factor_registry(self):
        """创建FactorRegistry实例"""
        return FactorRegistry()

    def test_factor_registry_initialization(self, factor_registry):
        """测试FactorRegistry初始化"""
        assert factor_registry._factors is not None
        assert factor_registry._factor_categories is not None

        # 验证内置因子已注册
        assert len(factor_registry._factors) > 0

        # 验证基本内置因子存在
        expected_factors = ["momentum_20d", "rsi_14d", "pe_ratio", "pb_ratio", "roe", "volatility_20d"]
        for factor_name in expected_factors:
            assert factor_name in factor_registry._factors

    def test_builtin_factors_registration(self, factor_registry):
        """测试内置因子注册"""
        # 验证各类型因子都有注册
        momentum_factors = factor_registry.list_factors(FactorType.MOMENTUM)
        value_factors = factor_registry.list_factors(FactorType.VALUE)
        quality_factors = factor_registry.list_factors(FactorType.QUALITY)
        volatility_factors = factor_registry.list_factors(FactorType.VOLATILITY)

        assert len(momentum_factors) >= 2  # momentum_20d, rsi_14d
        assert len(value_factors) >= 2     # pe_ratio, pb_ratio
        assert len(quality_factors) >= 1   # roe
        assert len(volatility_factors) >= 1 # volatility_20d

    def test_factor_definition_structure(self, factor_registry):
        """测试因子定义结构"""
        factor_def = factor_registry.get_factor_definition("momentum_20d")

        assert factor_def is not None

        # 验证必需字段
        required_fields = [
            "factor_name", "factor_type", "calculation_method",
            "description", "parameters", "data_requirements",
            "lookback_period", "registered_at", "version"
        ]

        for field in required_fields:
            assert field in factor_def

        assert factor_def["factor_name"] == "momentum_20d"
        assert factor_def["factor_type"] == FactorType.MOMENTUM
        assert callable(factor_def["calculation_method"])
        assert isinstance(factor_def["parameters"], dict)
        assert isinstance(factor_def["data_requirements"], list)
        assert isinstance(factor_def["lookback_period"], int)

    def test_register_custom_factor(self, factor_registry):
        """测试注册自定义因子"""
        def custom_factor_calculation(data: pd.DataFrame, params: Dict[str, Any]) -> pd.Series:
            """自定义因子计算方法"""
            return data['close'].rolling(window=5).mean()

        success = factor_registry.register_factor(
            factor_name="custom_ma5",
            factor_type=FactorType.MOMENTUM,
            calculation_method=custom_factor_calculation,
            description="5日移动平均",
            parameters={"window": 5},
            data_requirements=["close"],
            lookback_period=5
        )

        assert success is True
        assert "custom_ma5" in factor_registry._factors

        # 验证因子可以被检索
        factor_def = factor_registry.get_factor_definition("custom_ma5")
        assert factor_def is not None
        assert factor_def["factor_name"] == "custom_ma5"

    def test_factor_override(self, factor_registry):
        """测试因子覆盖功能"""
        def new_momentum_calculation(data: pd.DataFrame, params: Dict[str, Any]) -> pd.Series:
            """新的动量计算方法"""
            return data['close'].pct_change(10)  # 10日而不是20日

        # 覆盖现有因子
        success = factor_registry.register_factor(
            factor_name="momentum_20d",
            factor_type=FactorType.MOMENTUM,
            calculation_method=new_momentum_calculation,
            description="修改版20日动量",
            parameters={"window": 10},
            data_requirements=["close"],
            lookback_period=10
        )

        assert success is True

        # 验证覆盖生效
        factor_def = factor_registry.get_factor_definition("momentum_20d")
        assert factor_def["parameters"]["window"] == 10
        assert factor_def["lookback_period"] == 10

    def test_list_factors_by_type(self, factor_registry):
        """测试按类型列出因子"""
        all_factors = factor_registry.list_factors()
        momentum_factors = factor_registry.list_factors(FactorType.MOMENTUM)
        value_factors = factor_registry.list_factors(FactorType.VALUE)

        assert isinstance(all_factors, list)
        assert isinstance(momentum_factors, list)
        assert isinstance(value_factors, list)

        assert len(all_factors) >= len(momentum_factors)
        assert len(all_factors) >= len(value_factors)

        # 验证分类正确
        for factor_name in momentum_factors:
            factor_def = factor_registry.get_factor_definition(factor_name)
            assert factor_def["factor_type"] == FactorType.MOMENTUM

    def test_get_factor_info(self, factor_registry):
        """测试获取因子信息"""
        factor_info = factor_registry.get_factor_info("momentum_20d")

        assert isinstance(factor_info, dict)
        assert "factor_name" in factor_info
        assert "factor_type" in factor_info
        assert "description" in factor_info

        # 计算方法不应该在info中
        assert "calculation_method" not in factor_info

        # 因子类型应该是字符串格式
        assert isinstance(factor_info["factor_type"], str)

    def test_nonexistent_factor(self, factor_registry):
        """测试获取不存在的因子"""
        factor_def = factor_registry.get_factor_definition("nonexistent_factor")
        assert factor_def is None

        factor_info = factor_registry.get_factor_info("nonexistent_factor")
        assert factor_info == {}

    def test_builtin_factor_calculations(self, factor_registry):
        """测试内置因子计算方法"""
        # 创建测试数据
        test_data = pd.DataFrame({
            'close': [10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24] * 2,
            'eps': [1.0] * 30,
            'bvps': [5.0] * 30,
            'net_income': [100] * 30,
            'shareholders_equity': [1000] * 30
        })

        # 测试动量因子
        momentum_result = factor_registry._calculate_momentum_20d(test_data, {"window": 20, "method": "return_ratio"})
        assert isinstance(momentum_result, pd.Series)
        assert not momentum_result.dropna().empty

        # 测试RSI因子
        rsi_result = factor_registry._calculate_rsi_14d(test_data, {"window": 14})
        assert isinstance(rsi_result, pd.Series)
        assert not rsi_result.dropna().empty
        # RSI应该在0-100之间
        rsi_valid = rsi_result.dropna()
        if not rsi_valid.empty:
            assert rsi_valid.min() >= 0
            assert rsi_valid.max() <= 100

        # 测试PE比率
        pe_result = factor_registry._calculate_pe_ratio(test_data, {})
        assert isinstance(pe_result, pd.Series)
        assert not pe_result.dropna().empty

        # 测试PB比率
        pb_result = factor_registry._calculate_pb_ratio(test_data, {})
        assert isinstance(pb_result, pd.Series)
        assert not pb_result.dropna().empty

        # 测试ROE
        roe_result = factor_registry._calculate_roe(test_data, {"period": "ttm"})
        assert isinstance(roe_result, pd.Series)

        # 测试波动率
        volatility_result = factor_registry._calculate_volatility_20d(test_data, {"window": 20, "annualized": True})
        assert isinstance(volatility_result, pd.Series)
        assert not volatility_result.dropna().empty

    def test_factor_calculation_error_handling(self, factor_registry):
        """测试因子计算错误处理"""
        # 测试缺少必需列的情况
        incomplete_data = pd.DataFrame({
            'volume': [1000, 1100, 1200]  # 缺少close列
        })

        with pytest.raises(ValueError, match="缺少close列"):
            factor_registry._calculate_momentum_20d(incomplete_data, {"window": 20})

        with pytest.raises(ValueError, match="缺少close列"):
            factor_registry._calculate_rsi_14d(incomplete_data, {"window": 14})

        # 测试PE计算缺少EPS
        pe_incomplete_data = pd.DataFrame({
            'close': [10, 11, 12]  # 缺少eps列
        })

        with pytest.raises(ValueError, match="缺少close或eps列"):
            factor_registry._calculate_pe_ratio(pe_incomplete_data, {})

    def test_factor_calculation_parameters(self, factor_registry):
        """测试因子计算参数处理"""
        test_data = pd.DataFrame({
            'close': [10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30]
        })

        # 测试不同窗口参数
        momentum_10 = factor_registry._calculate_momentum_20d(test_data, {"window": 10, "method": "return_ratio"})
        momentum_5 = factor_registry._calculate_momentum_20d(test_data, {"window": 5, "method": "return_ratio"})

        assert isinstance(momentum_10, pd.Series)
        assert isinstance(momentum_5, pd.Series)

        # 5日动量应该有更多非空值（因为需要的历史数据更少）
        assert momentum_5.count() >= momentum_10.count()

        # 测试不同计算方法
        momentum_ratio = factor_registry._calculate_momentum_20d(test_data, {"window": 10, "method": "price_ratio"})
        momentum_return = factor_registry._calculate_momentum_20d(test_data, {"window": 10, "method": "return_ratio"})

        assert isinstance(momentum_ratio, pd.Series)
        assert isinstance(momentum_return, pd.Series)

        # 测试未知方法
        with pytest.raises(ValueError, match="未知的动量计算方法"):
            factor_registry._calculate_momentum_20d(test_data, {"window": 10, "method": "unknown_method"})


class TestFactorCalculator:
    """FactorCalculator组件单元测试"""

    @pytest.fixture
    def factor_registry(self):
        """创建FactorRegistry实例"""
        return FactorRegistry()

    @pytest.fixture
    def platform_optimizer(self):
        """创建PlatformOptimizer实例"""
        return PlatformOptimizer()

    @pytest.fixture
    def factor_calculator(self, factor_registry, platform_optimizer):
        """创建FactorCalculator实例"""
        return FactorCalculator(factor_registry, platform_optimizer)

    def test_factor_calculator_initialization(self, factor_calculator):
        """测试FactorCalculator初始化"""
        assert factor_calculator.registry is not None
        assert factor_calculator.optimizer is not None
        assert factor_calculator.calculation_optimizer is not None

    @pytest.mark.asyncio
    async def test_calculate_factor_basic_request(self, factor_calculator):
        """测试基本的因子计算请求"""
        # 创建计算请求
        request = FactorCalculationRequest(
            request_id="test_request_001",
            factor_name="momentum_20d",
            factor_type=FactorType.MOMENTUM,
            stock_codes=["sh600000", "sz000001"],
            start_date=date(2024, 1, 1),
            end_date=date(2024, 1, 31),
            force_recalculate=True
        )

        # 执行计算
        result = await factor_calculator.calculate_factor(request)

        # 验证结果
        assert isinstance(result, FactorCalculationResult)
        assert result.request_id == request.request_id
        assert result.factor_name == request.factor_name
        assert result.factor_type == request.factor_type
        assert result.total_stocks == len(request.stock_codes)
        assert isinstance(result.execution_time_seconds, float)
        assert result.execution_time_seconds > 0

        # 验证计算统计
        assert result.successful_calculations + result.failed_calculations <= result.total_stocks

    @pytest.mark.asyncio
    async def test_calculate_factor_with_real_data(self, factor_calculator):
        """测试使用真实数据的因子计算"""
        # 只使用一个股票代码进行快速测试
        request = FactorCalculationRequest(
            request_id="test_real_data",
            factor_name="volatility_20d",
            factor_type=FactorType.VOLATILITY,
            stock_codes=["sh600000"],
            start_date=date(2024, 1, 1),
            end_date=date(2024, 1, 10),
            force_recalculate=True
        )

        result = await factor_calculator.calculate_factor(request)

        assert isinstance(result, FactorCalculationResult)
        assert result.factor_name == "volatility_20d"
        assert result.total_stocks == 1

        # 如果计算成功，应该有因子数据
        if result.successful_calculations > 0:
            assert len(result.factor_data) > 0

            # 验证因子数据结构
            for factor_data in result.factor_data:
                assert hasattr(factor_data, 'stock_code')
                assert hasattr(factor_data, 'factor_value')
                assert hasattr(factor_data, 'calculation_date')

    @pytest.mark.asyncio
    async def test_calculate_factor_invalid_factor(self, factor_calculator):
        """测试计算不存在的因子"""
        request = FactorCalculationRequest(
            request_id="test_invalid_factor",
            factor_name="nonexistent_factor",
            factor_type=FactorType.MOMENTUM,
            stock_codes=["sh600000"],
            start_date=date(2024, 1, 1),
            end_date=date(2024, 1, 10),
            force_recalculate=True
        )

        with pytest.raises(ValueError, match="未找到因子定义"):
            await factor_calculator.calculate_factor(request)

    @pytest.mark.asyncio
    async def test_calculate_factor_empty_stocks(self, factor_calculator):
        """测试空股票列表的因子计算"""
        request = FactorCalculationRequest(
            request_id="test_empty_stocks",
            factor_name="momentum_20d",
            factor_type=FactorType.MOMENTUM,
            stock_codes=[],
            start_date=date(2024, 1, 1),
            end_date=date(2024, 1, 10),
            force_recalculate=True
        )

        result = await factor_calculator.calculate_factor(request)

        assert isinstance(result, FactorCalculationResult)
        assert result.total_stocks == 0
        assert result.successful_calculations == 0
        assert result.failed_calculations == 0
        assert len(result.factor_data) == 0

    @pytest.mark.asyncio
    async def test_calculate_factor_invalid_date_range(self, factor_calculator):
        """测试无效日期范围的因子计算"""
        # end_date 早于 start_date
        request = FactorCalculationRequest(
            request_id="test_invalid_dates",
            factor_name="momentum_20d",
            factor_type=FactorType.MOMENTUM,
            stock_codes=["sh600000"],
            start_date=date(2024, 1, 31),
            end_date=date(2024, 1, 1),  # 早于start_date
            force_recalculate=True
        )

        # 应该能够处理这种情况，可能返回空结果或调整日期
        result = await factor_calculator.calculate_factor(request)
        assert isinstance(result, FactorCalculationResult)

    @pytest.mark.asyncio
    async def test_calculate_multiple_factors(self, factor_calculator):
        """测试计算多个因子"""
        factor_names = ["momentum_20d", "rsi_14d", "volatility_20d"]
        results = []

        for factor_name in factor_names:
            factor_type = FactorType.MOMENTUM if factor_name in ["momentum_20d", "rsi_14d"] else FactorType.VOLATILITY

            request = FactorCalculationRequest(
                request_id=f"test_multi_{factor_name}",
                factor_name=factor_name,
                factor_type=factor_type,
                stock_codes=["sh600000"],
                start_date=date(2024, 1, 1),
                end_date=date(2024, 1, 5),
                force_recalculate=True
            )

            result = await factor_calculator.calculate_factor(request)
            results.append(result)

        # 验证所有计算都完成
        assert len(results) == len(factor_names)

        for i, result in enumerate(results):
            assert result.factor_name == factor_names[i]
            assert isinstance(result, FactorCalculationResult)

    @pytest.mark.asyncio
    async def test_calculate_factor_performance_optimization(self, factor_calculator):
        """测试因子计算性能优化"""
        # 创建较大的股票列表来测试优化
        stock_codes = [f"sh{600000 + i:06d}" for i in range(20)]

        request = FactorCalculationRequest(
            request_id="test_performance",
            factor_name="momentum_20d",
            factor_type=FactorType.MOMENTUM,
            stock_codes=stock_codes,
            start_date=date(2024, 1, 1),
            end_date=date(2024, 1, 5),
            force_recalculate=True
        )

        # 记录执行时间
        start_time = datetime.now()
        result = await factor_calculator.calculate_factor(request)
        end_time = datetime.now()

        execution_time = (end_time - start_time).total_seconds()

        # 验证结果
        assert isinstance(result, FactorCalculationResult)
        assert result.total_stocks == len(stock_codes)
        assert result.execution_time_seconds == pytest.approx(execution_time, rel=0.1)

        # 验证性能优化生效
        # 对于20只股票，应该在合理时间内完成
        assert execution_time < 60  # 应该在1分钟内完成

    @pytest.mark.asyncio
    async def test_calculate_factor_with_chunking(self, factor_calculator):
        """测试分块计算功能"""
        # 创建足够大的股票列表来触发分块
        stock_codes = [f"sh{600000 + i:06d}" for i in range(10)]

        request = FactorCalculationRequest(
            request_id="test_chunking",
            factor_name="momentum_20d",
            factor_type=FactorType.MOMENTUM,
            stock_codes=stock_codes,
            start_date=date(2024, 1, 1),
            end_date=date(2024, 1, 3),
            chunk_size=3,  # 设置小的分块大小
            force_recalculate=True
        )

        result = await factor_calculator.calculate_factor(request)

        assert isinstance(result, FactorCalculationResult)
        assert result.total_stocks == len(stock_codes)

        # 分块计算应该产生与非分块相同的结果数量
        total_processed = result.successful_calculations + result.failed_calculations
        assert total_processed <= len(stock_codes)

    def test_performance_estimation_integration(self, factor_calculator):
        """测试性能估算集成"""
        # 测试不同规模的性能估算
        small_estimate = factor_calculator.calculation_optimizer.estimate_calculation_time(10, 1)
        medium_estimate = factor_calculator.calculation_optimizer.estimate_calculation_time(100, 1)
        large_estimate = factor_calculator.calculation_optimizer.estimate_calculation_time(1000, 1)

        # 验证估算结果结构
        for estimate in [small_estimate, medium_estimate, large_estimate]:
            assert 'estimated_minutes' in estimate
            assert 'meets_target' in estimate
            assert isinstance(estimate['estimated_minutes'], (int, float))
            assert isinstance(estimate['meets_target'], bool)

        # 验证估算趋势合理
        assert small_estimate['estimated_minutes'] <= medium_estimate['estimated_minutes']
        assert medium_estimate['estimated_minutes'] <= large_estimate['estimated_minutes']

    def test_calculation_strategy_optimization(self, factor_calculator):
        """测试计算策略优化"""
        # 测试不同规模的策略优化
        small_strategy = factor_calculator.calculation_optimizer.optimize_calculation_strategy(10, 1)
        large_strategy = factor_calculator.calculation_optimizer.optimize_calculation_strategy(1000, 1)

        # 验证策略结构
        for strategy in [small_strategy, large_strategy]:
            assert 'worker_count' in strategy
            assert 'chunk_size' in strategy
            assert isinstance(strategy['worker_count'], int)
            assert isinstance(strategy['chunk_size'], int)
            assert strategy['worker_count'] >= 1
            assert strategy['chunk_size'] >= 1

        # 大规模任务应该使用更多工作进程
        assert large_strategy['worker_count'] >= small_strategy['worker_count']


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])