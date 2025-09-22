"""
Hikyuu Models Unit Tests

基于真实Hikyuu框架的数据模型单元测试
不使用mock数据，测试FactorData, FactorCalculationRequest, FactorCalculationResult等核心模型
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, date, timedelta
from typing import Dict, List, Any
from decimal import Decimal

from src.models.hikyuu_models import (
    FactorType, SignalType, PositionType,
    FactorData, FactorCalculationRequest, FactorCalculationResult
)


class TestEnums:
    """枚举类型单元测试"""

    def test_factor_type_enum(self):
        """测试FactorType枚举"""
        # 验证所有因子类型存在
        expected_types = [
            "momentum", "value", "quality", "growth", "volatility",
            "liquidity", "sentiment", "technical", "fundamental", "macro", "custom"
        ]

        for expected_type in expected_types:
            assert any(ft.value == expected_type for ft in FactorType)

        # 验证枚举值
        assert FactorType.MOMENTUM.value == "momentum"
        assert FactorType.VALUE.value == "value"
        assert FactorType.QUALITY.value == "quality"

    def test_signal_type_enum(self):
        """测试SignalType枚举"""
        assert SignalType.BUY.value == "buy"
        assert SignalType.SELL.value == "sell"
        assert SignalType.HOLD.value == "hold"

        # 验证所有信号类型
        signal_values = [st.value for st in SignalType]
        expected_signals = ["buy", "sell", "hold"]
        assert set(signal_values) == set(expected_signals)

    def test_position_type_enum(self):
        """测试PositionType枚举"""
        assert PositionType.LONG.value == "long"
        assert PositionType.SHORT.value == "short"
        assert PositionType.CASH.value == "cash"


class TestFactorData:
    """FactorData模型单元测试"""

    @pytest.fixture
    def basic_factor_data(self):
        """创建基本的FactorData实例"""
        return FactorData(
            factor_name="test_momentum",
            factor_type=FactorType.MOMENTUM,
            stock_code="sh600000",
            calculation_date=date(2024, 1, 15),
            factor_value=0.125,
            factor_score=0.75,
            calculation_method="price_return",
            lookback_period=20,
            data_source="hikyuu",
            quality_score=0.9
        )

    def test_factor_data_initialization(self, basic_factor_data):
        """测试FactorData基本初始化"""
        factor_data = basic_factor_data

        assert factor_data.factor_name == "test_momentum"
        assert factor_data.factor_type == FactorType.MOMENTUM
        assert factor_data.stock_code == "sh600000"
        assert factor_data.calculation_date == date(2024, 1, 15)
        assert factor_data.factor_value == 0.125
        assert factor_data.factor_score == 0.75
        assert factor_data.calculation_method == "price_return"
        assert factor_data.lookback_period == 20
        assert factor_data.data_source == "hikyuu"
        assert factor_data.quality_score == 0.9

    def test_factor_data_post_init_as_of_date(self):
        """测试as_of_date的自动设置"""
        factor_data = FactorData(
            factor_name="test_factor",
            factor_type=FactorType.VALUE,
            stock_code="sz000001",
            calculation_date=date(2024, 1, 10),
            factor_value=1.5
        )

        # as_of_date应该自动设置为calculation_date
        assert factor_data.as_of_date == factor_data.calculation_date

        # 手动设置as_of_date
        custom_as_of_date = date(2024, 1, 8)
        factor_data_with_custom = FactorData(
            factor_name="test_factor",
            factor_type=FactorType.VALUE,
            stock_code="sz000001",
            calculation_date=date(2024, 1, 10),
            factor_value=1.5,
            as_of_date=custom_as_of_date
        )

        assert factor_data_with_custom.as_of_date == custom_as_of_date

    def test_factor_data_invalid_factor_value(self):
        """测试无效因子值的验证"""
        # 测试NaN值
        with pytest.raises(ValueError, match="Invalid factor value"):
            FactorData(
                factor_name="test_factor",
                factor_type=FactorType.MOMENTUM,
                stock_code="sh600000",
                calculation_date=date(2024, 1, 15),
                factor_value=np.nan
            )

        # 测试无穷大值
        with pytest.raises(ValueError, match="Invalid factor value"):
            FactorData(
                factor_name="test_factor",
                factor_type=FactorType.MOMENTUM,
                stock_code="sh600000",
                calculation_date=date(2024, 1, 15),
                factor_value=np.inf
            )

    def test_factor_data_invalid_factor_score(self):
        """测试无效因子分数的验证"""
        # 测试超出范围的分数
        with pytest.raises(ValueError, match="Factor score must be in \\[0,1\\]"):
            FactorData(
                factor_name="test_factor",
                factor_type=FactorType.MOMENTUM,
                stock_code="sh600000",
                calculation_date=date(2024, 1, 15),
                factor_value=0.5,
                factor_score=1.5  # 超出[0,1]范围
            )

        with pytest.raises(ValueError, match="Factor score must be in \\[0,1\\]"):
            FactorData(
                factor_name="test_factor",
                factor_type=FactorType.MOMENTUM,
                stock_code="sh600000",
                calculation_date=date(2024, 1, 15),
                factor_value=0.5,
                factor_score=-0.1  # 小于0
            )

    def test_factor_data_valid_boundary_scores(self):
        """测试边界值因子分数"""
        # 测试边界值
        factor_data_min = FactorData(
            factor_name="test_factor",
            factor_type=FactorType.MOMENTUM,
            stock_code="sh600000",
            calculation_date=date(2024, 1, 15),
            factor_value=0.5,
            factor_score=0.0  # 最小值
        )
        assert factor_data_min.factor_score == 0.0

        factor_data_max = FactorData(
            factor_name="test_factor",
            factor_type=FactorType.MOMENTUM,
            stock_code="sh600000",
            calculation_date=date(2024, 1, 15),
            factor_value=0.5,
            factor_score=1.0  # 最大值
        )
        assert factor_data_max.factor_score == 1.0

    def test_factor_data_without_hikyuu_stock(self, basic_factor_data):
        """测试没有Hikyuu Stock对象的情况"""
        factor_data = basic_factor_data
        factor_data.hikyuu_stock = None

        stock_info = factor_data.get_stock_info()

        assert isinstance(stock_info, dict)
        assert stock_info["stock_code"] == "sh600000"
        assert len(stock_info) == 1  # 只有stock_code

    def test_factor_data_to_dict(self, basic_factor_data):
        """测试转换为字典"""
        factor_dict = basic_factor_data.to_dict()

        expected_keys = [
            "factor_name", "factor_type", "stock_code", "calculation_date",
            "factor_value", "factor_score", "calculation_method", "lookback_period",
            "data_source", "quality_score", "as_of_date"
        ]

        for key in expected_keys:
            assert key in factor_dict

        # 验证值的转换
        assert factor_dict["factor_type"] == "momentum"
        assert factor_dict["calculation_date"] == "2024-01-15"
        assert factor_dict["factor_value"] == 0.125
        assert factor_dict["factor_score"] == 0.75

    def test_factor_data_edge_cases(self):
        """测试边界情况"""
        # 测试极小的因子值
        factor_data_small = FactorData(
            factor_name="small_factor",
            factor_type=FactorType.VOLATILITY,
            stock_code="sh600000",
            calculation_date=date(2024, 1, 15),
            factor_value=1e-10  # 非常小的值
        )
        assert factor_data_small.factor_value == 1e-10

        # 测试极大的因子值
        factor_data_large = FactorData(
            factor_name="large_factor",
            factor_type=FactorType.VOLATILITY,
            stock_code="sh600000",
            calculation_date=date(2024, 1, 15),
            factor_value=1e10  # 非常大的值
        )
        assert factor_data_large.factor_value == 1e10

        # 测试负因子值
        factor_data_negative = FactorData(
            factor_name="negative_factor",
            factor_type=FactorType.MOMENTUM,
            stock_code="sh600000",
            calculation_date=date(2024, 1, 15),
            factor_value=-0.25
        )
        assert factor_data_negative.factor_value == -0.25

    def test_factor_data_different_types(self):
        """测试不同因子类型"""
        factor_types = [
            FactorType.MOMENTUM, FactorType.VALUE, FactorType.QUALITY,
            FactorType.GROWTH, FactorType.VOLATILITY, FactorType.CUSTOM
        ]

        for factor_type in factor_types:
            factor_data = FactorData(
                factor_name=f"test_{factor_type.value}",
                factor_type=factor_type,
                stock_code="sh600000",
                calculation_date=date(2024, 1, 15),
                factor_value=0.5
            )

            assert factor_data.factor_type == factor_type
            assert factor_data.factor_name == f"test_{factor_type.value}"


class TestFactorCalculationRequest:
    """FactorCalculationRequest模型单元测试"""

    @pytest.fixture
    def basic_request(self):
        """创建基本的计算请求"""
        return FactorCalculationRequest(
            request_id="test_request_001",
            factor_name="momentum_20d",
            factor_type=FactorType.MOMENTUM,
            stock_codes=["sh600000", "sz000001", "sh600036"],
            start_date=date(2024, 1, 1),
            end_date=date(2024, 1, 31),
            calculation_params={"window": 20},
            lookback_period=20,
            priority=2,
            max_parallel_workers=8,
            chunk_size=50,
            data_source="hikyuu_test",
            use_cache=True,
            force_recalculate=False,
            created_by="test_user"
        )

    def test_request_initialization(self, basic_request):
        """测试基本初始化"""
        req = basic_request

        assert req.request_id == "test_request_001"
        assert req.factor_name == "momentum_20d"
        assert req.factor_type == FactorType.MOMENTUM
        assert req.stock_codes == ["sh600000", "sz000001", "sh600036"]
        assert req.start_date == date(2024, 1, 1)
        assert req.end_date == date(2024, 1, 31)
        assert req.calculation_params == {"window": 20}
        assert req.lookback_period == 20
        assert req.priority == 2
        assert req.max_parallel_workers == 8
        assert req.chunk_size == 50
        assert req.data_source == "hikyuu_test"
        assert req.use_cache is True
        assert req.force_recalculate is False
        assert req.created_by == "test_user"

    def test_request_default_values(self):
        """测试默认值"""
        req = FactorCalculationRequest(
            request_id="test_default",
            factor_name="test_factor",
            factor_type=FactorType.VALUE,
            stock_codes=["sh600000"],
            start_date=date(2024, 1, 1),
            end_date=date(2024, 1, 10)
        )

        # 验证默认值
        assert req.calculation_params == {}
        assert req.lookback_period is None
        assert req.priority == 1
        assert req.max_parallel_workers == 4
        assert req.chunk_size == 100
        assert req.data_source == "hikyuu_default"
        assert req.use_cache is True
        assert req.force_recalculate is False
        assert req.output_format == "dataframe"
        assert req.normalize_scores is True
        assert req.created_by is None

    def test_request_date_validation(self):
        """测试日期验证"""
        # 测试start_date > end_date的情况
        with pytest.raises(ValueError, match="start_date must be <= end_date"):
            FactorCalculationRequest(
                request_id="test_invalid_dates",
                factor_name="test_factor",
                factor_type=FactorType.MOMENTUM,
                stock_codes=["sh600000"],
                start_date=date(2024, 1, 31),
                end_date=date(2024, 1, 1)  # 早于start_date
            )

    def test_request_empty_stocks_validation(self):
        """测试空股票列表验证"""
        with pytest.raises(ValueError, match="stock_codes cannot be empty"):
            FactorCalculationRequest(
                request_id="test_empty_stocks",
                factor_name="test_factor",
                factor_type=FactorType.MOMENTUM,
                stock_codes=[],  # 空列表
                start_date=date(2024, 1, 1),
                end_date=date(2024, 1, 10)
            )

    def test_request_priority_validation(self):
        """测试优先级验证"""
        # 测试无效优先级
        with pytest.raises(ValueError, match="priority must be 1, 2, or 3"):
            FactorCalculationRequest(
                request_id="test_invalid_priority",
                factor_name="test_factor",
                factor_type=FactorType.MOMENTUM,
                stock_codes=["sh600000"],
                start_date=date(2024, 1, 1),
                end_date=date(2024, 1, 10),
                priority=5  # 无效优先级
            )

        # 测试有效优先级
        for valid_priority in [1, 2, 3]:
            req = FactorCalculationRequest(
                request_id=f"test_priority_{valid_priority}",
                factor_name="test_factor",
                factor_type=FactorType.MOMENTUM,
                stock_codes=["sh600000"],
                start_date=date(2024, 1, 1),
                end_date=date(2024, 1, 10),
                priority=valid_priority
            )
            assert req.priority == valid_priority

    def test_request_to_dict(self, basic_request):
        """测试转换为字典"""
        req_dict = basic_request.to_dict()

        expected_keys = [
            "request_id", "factor_name", "factor_type", "stock_codes",
            "start_date", "end_date", "calculation_params", "lookback_period",
            "priority", "max_parallel_workers", "chunk_size", "data_source",
            "use_cache", "force_recalculate", "output_format", "normalize_scores",
            "created_at", "created_by"
        ]

        for key in expected_keys:
            assert key in req_dict

        # 验证值的转换
        assert req_dict["factor_type"] == "momentum"
        assert req_dict["start_date"] == "2024-01-01"
        assert req_dict["end_date"] == "2024-01-31"
        assert req_dict["stock_codes"] == ["sh600000", "sz000001", "sh600036"]

    def test_request_edge_cases(self):
        """测试边界情况"""
        # 测试单日期范围
        req_single_day = FactorCalculationRequest(
            request_id="test_single_day",
            factor_name="test_factor",
            factor_type=FactorType.MOMENTUM,
            stock_codes=["sh600000"],
            start_date=date(2024, 1, 15),
            end_date=date(2024, 1, 15)  # 同一天
        )
        assert req_single_day.start_date == req_single_day.end_date

        # 测试单股票
        req_single_stock = FactorCalculationRequest(
            request_id="test_single_stock",
            factor_name="test_factor",
            factor_type=FactorType.MOMENTUM,
            stock_codes=["sh600000"],  # 只有一只股票
            start_date=date(2024, 1, 1),
            end_date=date(2024, 1, 10)
        )
        assert len(req_single_stock.stock_codes) == 1

        # 测试大量股票
        many_stocks = [f"sh{600000 + i:06d}" for i in range(100)]
        req_many_stocks = FactorCalculationRequest(
            request_id="test_many_stocks",
            factor_name="test_factor",
            factor_type=FactorType.MOMENTUM,
            stock_codes=many_stocks,
            start_date=date(2024, 1, 1),
            end_date=date(2024, 1, 10)
        )
        assert len(req_many_stocks.stock_codes) == 100

    def test_request_different_factor_types(self):
        """测试不同因子类型的请求"""
        factor_types = [
            FactorType.MOMENTUM, FactorType.VALUE, FactorType.QUALITY,
            FactorType.GROWTH, FactorType.VOLATILITY, FactorType.CUSTOM
        ]

        for factor_type in factor_types:
            req = FactorCalculationRequest(
                request_id=f"test_{factor_type.value}",
                factor_name=f"{factor_type.value}_factor",
                factor_type=factor_type,
                stock_codes=["sh600000"],
                start_date=date(2024, 1, 1),
                end_date=date(2024, 1, 10)
            )

            assert req.factor_type == factor_type
            assert req.factor_name == f"{factor_type.value}_factor"

    def test_request_complex_calculation_params(self):
        """测试复杂的计算参数"""
        complex_params = {
            "window": 20,
            "method": "exponential",
            "alpha": 0.05,
            "min_periods": 10,
            "adjust": True,
            "nested_params": {
                "sub_window": 5,
                "sub_method": "linear"
            }
        }

        req = FactorCalculationRequest(
            request_id="test_complex_params",
            factor_name="complex_factor",
            factor_type=FactorType.MOMENTUM,
            stock_codes=["sh600000"],
            start_date=date(2024, 1, 1),
            end_date=date(2024, 1, 10),
            calculation_params=complex_params
        )

        assert req.calculation_params == complex_params
        assert req.calculation_params["nested_params"]["sub_window"] == 5


class TestFactorCalculationResult:
    """FactorCalculationResult模型单元测试"""

    @pytest.fixture
    def sample_factor_data(self):
        """创建示例因子数据"""
        return [
            FactorData(
                factor_name="test_momentum",
                factor_type=FactorType.MOMENTUM,
                stock_code="sh600000",
                calculation_date=date(2024, 1, 15),
                factor_value=0.125,
                factor_score=0.75
            ),
            FactorData(
                factor_name="test_momentum",
                factor_type=FactorType.MOMENTUM,
                stock_code="sz000001",
                calculation_date=date(2024, 1, 15),
                factor_value=0.08,
                factor_score=0.60
            ),
            FactorData(
                factor_name="test_momentum",
                factor_type=FactorType.MOMENTUM,
                stock_code="sh600036",
                calculation_date=date(2024, 1, 15),
                factor_value=-0.05,
                factor_score=0.30
            )
        ]

    @pytest.fixture
    def basic_result(self, sample_factor_data):
        """创建基本的计算结果"""
        return FactorCalculationResult(
            request_id="test_request_001",
            factor_name="test_momentum",
            factor_type=FactorType.MOMENTUM,
            factor_data=sample_factor_data,
            calculation_date=datetime(2024, 1, 15, 10, 30, 0),
            total_stocks=3,
            successful_calculations=3,
            failed_calculations=0,
            execution_time_seconds=2.5,
            data_quality_score=0.9,
            errors=[],
            warnings=[]
        )

    def test_result_initialization(self, basic_result):
        """测试结果基本初始化"""
        result = basic_result

        assert result.request_id == "test_request_001"
        assert result.factor_name == "test_momentum"
        assert result.factor_type == FactorType.MOMENTUM
        assert len(result.factor_data) == 3
        assert result.total_stocks == 3
        assert result.successful_calculations == 3
        assert result.failed_calculations == 0
        assert result.execution_time_seconds == 2.5
        assert result.data_quality_score == 0.9

    def test_result_post_init_coverage_ratio(self, sample_factor_data):
        """测试覆盖率的自动计算"""
        result = FactorCalculationResult(
            request_id="test_coverage",
            factor_name="test_factor",
            factor_type=FactorType.MOMENTUM,
            factor_data=sample_factor_data,
            calculation_date=datetime.now(),
            total_stocks=5,
            successful_calculations=3,
            failed_calculations=2,
            execution_time_seconds=1.0
        )

        # 覆盖率应该自动计算
        assert result.coverage_ratio == 3/5  # 0.6

    def test_result_factor_statistics_calculation(self, sample_factor_data):
        """测试因子统计信息的自动计算"""
        result = FactorCalculationResult(
            request_id="test_statistics",
            factor_name="test_factor",
            factor_type=FactorType.MOMENTUM,
            factor_data=sample_factor_data,
            calculation_date=datetime.now(),
            total_stocks=3,
            successful_calculations=3,
            failed_calculations=0,
            execution_time_seconds=1.0
        )

        # 统计信息应该自动计算
        assert result.factor_statistics is not None
        assert isinstance(result.factor_statistics, dict)

        # 验证统计指标存在（具体值取决于实现）
        expected_stats = ["mean", "std", "min", "max", "count"]
        for stat in expected_stats:
            assert stat in result.factor_statistics

    def test_result_empty_factor_data(self):
        """测试空因子数据的情况"""
        result = FactorCalculationResult(
            request_id="test_empty",
            factor_name="empty_factor",
            factor_type=FactorType.MOMENTUM,
            factor_data=[],  # 空数据
            calculation_date=datetime.now(),
            total_stocks=1,  # 避免除零错误
            successful_calculations=0,
            failed_calculations=0,
            execution_time_seconds=0.1
        )

        assert len(result.factor_data) == 0
        assert result.coverage_ratio == 0.0  # 0/1=0

    def test_result_with_errors_and_warnings(self, sample_factor_data):
        """测试包含错误和警告的结果"""
        errors = ["Error 1: Invalid data", "Error 2: Calculation failed"]
        warnings = ["Warning 1: Low data quality", "Warning 2: Outliers detected"]

        result = FactorCalculationResult(
            request_id="test_errors_warnings",
            factor_name="test_factor",
            factor_type=FactorType.MOMENTUM,
            factor_data=sample_factor_data,
            calculation_date=datetime.now(),
            total_stocks=5,
            successful_calculations=3,
            failed_calculations=2,
            execution_time_seconds=3.0,
            errors=errors,
            warnings=warnings
        )

        assert result.errors == errors
        assert result.warnings == warnings
        assert len(result.errors) == 2
        assert len(result.warnings) == 2

    def test_result_quality_metrics(self, sample_factor_data):
        """测试质量指标"""
        result = FactorCalculationResult(
            request_id="test_quality",
            factor_name="test_factor",
            factor_type=FactorType.MOMENTUM,
            factor_data=sample_factor_data,
            calculation_date=datetime.now(),
            total_stocks=3,
            successful_calculations=3,
            failed_calculations=0,
            execution_time_seconds=1.5,
            data_quality_score=0.85,
            outlier_ratio=0.1
        )

        assert result.data_quality_score == 0.85
        assert result.outlier_ratio == 0.1
        assert result.coverage_ratio == 1.0  # 3/3

    def test_result_performance_metrics(self, sample_factor_data):
        """测试性能指标"""
        # 快速计算
        fast_result = FactorCalculationResult(
            request_id="test_fast",
            factor_name="fast_factor",
            factor_type=FactorType.MOMENTUM,
            factor_data=sample_factor_data,
            calculation_date=datetime.now(),
            total_stocks=3,
            successful_calculations=3,
            failed_calculations=0,
            execution_time_seconds=0.5
        )

        # 慢速计算
        slow_result = FactorCalculationResult(
            request_id="test_slow",
            factor_name="slow_factor",
            factor_type=FactorType.MOMENTUM,
            factor_data=sample_factor_data,
            calculation_date=datetime.now(),
            total_stocks=3,
            successful_calculations=3,
            failed_calculations=0,
            execution_time_seconds=10.0
        )

        assert fast_result.execution_time_seconds < slow_result.execution_time_seconds
        assert fast_result.execution_time_seconds == 0.5
        assert slow_result.execution_time_seconds == 10.0

    def test_result_with_partial_failures(self):
        """测试部分失败的计算结果"""
        # 只有部分因子数据
        partial_factor_data = [
            FactorData(
                factor_name="partial_factor",
                factor_type=FactorType.MOMENTUM,
                stock_code="sh600000",
                calculation_date=date(2024, 1, 15),
                factor_value=0.1,
                factor_score=0.5
            )
        ]

        result = FactorCalculationResult(
            request_id="test_partial",
            factor_name="partial_factor",
            factor_type=FactorType.MOMENTUM,
            factor_data=partial_factor_data,
            calculation_date=datetime.now(),
            total_stocks=5,  # 总共5只股票
            successful_calculations=1,  # 只成功1只
            failed_calculations=4,  # 失败4只
            execution_time_seconds=2.0,
            errors=["Failed to calculate for 4 stocks"],
            warnings=["Low success rate"]
        )

        assert result.coverage_ratio == 1/5  # 0.2
        assert result.successful_calculations == 1
        assert result.failed_calculations == 4
        assert len(result.factor_data) == 1
        assert len(result.errors) == 1

    def test_result_different_factor_types(self):
        """测试不同因子类型的结果"""
        factor_types = [FactorType.MOMENTUM, FactorType.VALUE, FactorType.VOLATILITY]

        for factor_type in factor_types:
            factor_data = [
                FactorData(
                    factor_name=f"{factor_type.value}_factor",
                    factor_type=factor_type,
                    stock_code="sh600000",
                    calculation_date=date(2024, 1, 15),
                    factor_value=0.1
                )
            ]

            result = FactorCalculationResult(
                request_id=f"test_{factor_type.value}",
                factor_name=f"{factor_type.value}_factor",
                factor_type=factor_type,
                factor_data=factor_data,
                calculation_date=datetime.now(),
                total_stocks=1,
                successful_calculations=1,
                failed_calculations=0,
                execution_time_seconds=1.0
            )

            assert result.factor_type == factor_type


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])