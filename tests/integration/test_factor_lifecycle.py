"""
T026: 因子注册→计算→存储→查询 集成测试

测试完整的因子生命周期管理：
1. 因子注册和元数据管理
2. 平台优化的因子计算
3. 计算结果存储和缓存
4. 因子值查询和检索

这是一个TDD Red-Green-Refactor循环的第一步 - 先创建失败的测试
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta, date
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional
from unittest.mock import Mock, patch, MagicMock

# 导入待实现的模块 (这些导入在Red阶段会失败)
try:
    from src.services.factor_registry import FactorRegistry
    from src.services.factor_calculator import FactorCalculator
    from src.services.factor_storage import FactorStorage
    from src.services.factor_query_engine import FactorQueryEngine
    from src.models.factor_definition import FactorDefinition
    from src.models.factor_metadata import FactorMetadata
    from src.models.factor_calculation_task import FactorCalculationTask
    from src.models.factor_result import FactorResult
    from src.lib.factor_library import BuiltinFactors
except ImportError:
    # TDD Red阶段 - 这些模块还不存在
    FactorRegistry = None
    FactorCalculator = None
    FactorStorage = None
    FactorQueryEngine = None
    FactorDefinition = None
    FactorMetadata = None
    FactorCalculationTask = None
    FactorResult = None
    BuiltinFactors = None


@pytest.mark.integration
@pytest.mark.factor_lifecycle
@pytest.mark.requires_mysql
@pytest.mark.requires_hikyuu
class TestFactorLifecycle:
    """因子注册→计算→存储→查询 集成测试"""

    def setup_method(self):
        """设置测试环境"""
        self.test_stocks = ["sh600000", "sh600001", "sz000001", "sz000002"]
        self.test_date_range = (
            date(2024, 1, 1),
            date(2024, 1, 31)
        )

        # 创建测试用的市场数据
        self.test_market_data = self._create_test_market_data()

        # 定义测试因子
        self.test_factor_definitions = self._create_test_factor_definitions()

    def _create_test_market_data(self) -> pd.DataFrame:
        """创建测试用的市场数据"""
        np.random.seed(42)
        data = []

        for stock in self.test_stocks:
            base_price = 10.0 + np.random.randn() * 2.0
            for i in range(31):  # 31天数据
                current_date = self.test_date_range[0] + timedelta(days=i)

                # 生成相关的OHLC数据
                daily_return = np.random.randn() * 0.02  # 2%波动
                close_price = base_price * (1 + daily_return)

                data.append({
                    'stock_code': stock,
                    'date': current_date,
                    'open': close_price * (1 + np.random.randn() * 0.01),
                    'high': close_price * (1 + abs(np.random.randn() * 0.015)),
                    'low': close_price * (1 - abs(np.random.randn() * 0.015)),
                    'close': close_price,
                    'volume': 1000000 + np.random.randint(0, 500000),
                    'turnover_rate': np.random.uniform(0.5, 5.0),
                })

                base_price = close_price  # 价格连续性

        return pd.DataFrame(data)

    def _create_test_factor_definitions(self) -> List[Dict[str, Any]]:
        """创建测试因子定义"""
        return [
            {
                'factor_id': 'momentum_20d',
                'factor_name': '20日动量因子',
                'description': '过去20个交易日的价格动量',
                'category': 'momentum',
                'formula': '(close_t / close_t-20) - 1',
                'parameters': {
                    'window': 20,
                    'method': 'simple_return'
                },
                'data_requirements': ['close'],
                'output_type': 'numeric',
                'frequency': 'daily',
                'version': '1.0.0'
            },
            {
                'factor_id': 'volatility_20d',
                'factor_name': '20日波动率因子',
                'description': '过去20个交易日的收益率标准差',
                'category': 'risk',
                'formula': 'std(daily_returns, window=20)',
                'parameters': {
                    'window': 20,
                    'annualized': False
                },
                'data_requirements': ['close'],
                'output_type': 'numeric',
                'frequency': 'daily',
                'version': '1.0.0'
            },
            {
                'factor_id': 'turnover_mean_5d',
                'factor_name': '5日均换手率',
                'description': '过去5个交易日的平均换手率',
                'category': 'liquidity',
                'formula': 'mean(turnover_rate, window=5)',
                'parameters': {
                    'window': 5
                },
                'data_requirements': ['turnover_rate'],
                'output_type': 'numeric',
                'frequency': 'daily',
                'version': '1.0.0'
            },
            {
                'factor_id': 'price_relative_strength',
                'factor_name': '相对强度因子',
                'description': '股票相对于市场的强度',
                'category': 'momentum',
                'formula': '(stock_return / market_return) - 1',
                'parameters': {
                    'benchmark': 'market_index',
                    'window': 10
                },
                'data_requirements': ['close', 'market_close'],
                'output_type': 'numeric',
                'frequency': 'daily',
                'version': '1.0.0'
            }
        ]

    @pytest.mark.integration
    def test_complete_factor_lifecycle(self):
        """测试完整的因子生命周期"""
        if FactorRegistry is None:
            pytest.skip("FactorRegistry not implemented yet - TDD Red phase")

        # Step 1: 因子注册
        registry = FactorRegistry()

        registered_factors = []
        for factor_def in self.test_factor_definitions:
            factor_definition = FactorDefinition.from_dict(factor_def)
            registration_result = registry.register_factor(factor_definition)

            assert registration_result['status'] == 'success', \
                f"因子 {factor_def['factor_id']} 注册失败"

            registered_factors.append(registration_result['factor_id'])

        # 验证注册结果
        assert len(registered_factors) == len(self.test_factor_definitions), \
            "所有因子都应该成功注册"

        # Step 2: 因子计算
        calculator = FactorCalculator()

        calculation_results = {}
        for factor_id in registered_factors:
            # 创建计算任务
            calc_task = FactorCalculationTask(
                factor_id=factor_id,
                stocks=self.test_stocks,
                date_range=self.test_date_range,
                priority='normal'
            )

            # 执行计算
            calc_result = calculator.calculate_factor(
                calc_task,
                market_data=self.test_market_data
            )

            assert calc_result['status'] == 'success', \
                f"因子 {factor_id} 计算失败"

            calculation_results[factor_id] = calc_result

        # Step 3: 结果存储
        storage = FactorStorage()

        for factor_id, calc_result in calculation_results.items():
            storage_result = storage.store_factor_values(
                factor_id,
                calc_result['factor_values'],
                calc_result['metadata']
            )

            assert storage_result['status'] == 'success', \
                f"因子 {factor_id} 存储失败"

        # Step 4: 因子查询
        query_engine = FactorQueryEngine()

        for factor_id in registered_factors:
            # 查询因子值
            query_result = query_engine.query_factor_values(
                factor_id=factor_id,
                stocks=self.test_stocks[:2],  # 查询前两只股票
                date_range=(date(2024, 1, 15), date(2024, 1, 25))
            )

            assert query_result is not None, f"因子 {factor_id} 查询失败"
            assert len(query_result) > 0, f"因子 {factor_id} 应该有查询结果"

            # 验证查询结果的数据结构
            assert 'stock_code' in query_result.columns, "查询结果应该包含股票代码"
            assert 'date' in query_result.columns, "查询结果应该包含日期"
            assert 'factor_value' in query_result.columns, "查询结果应该包含因子值"

    @pytest.mark.integration
    def test_factor_registration_validation(self):
        """测试因子注册的验证功能"""
        if FactorRegistry is None:
            pytest.skip("FactorRegistry not implemented yet - TDD Red phase")

        registry = FactorRegistry()

        # 测试有效因子注册
        valid_factor = self.test_factor_definitions[0]
        factor_def = FactorDefinition.from_dict(valid_factor)
        result = registry.register_factor(factor_def)

        assert result['status'] == 'success', "有效因子应该注册成功"

        # 测试重复注册
        duplicate_result = registry.register_factor(factor_def)
        assert duplicate_result['status'] == 'error', "重复注册应该失败"
        assert 'already_exists' in duplicate_result['message'], "应该提示因子已存在"

        # 测试无效因子定义
        invalid_factor = {
            'factor_id': 'invalid_factor',
            # 缺少必要字段
            'description': '无效的因子定义'
        }

        try:
            invalid_def = FactorDefinition.from_dict(invalid_factor)
            invalid_result = registry.register_factor(invalid_def)
            assert invalid_result['status'] == 'error', "无效因子定义应该注册失败"
        except Exception as e:
            # 验证抛出了预期的异常
            assert 'required' in str(e).lower(), "应该提示缺少必要字段"

    @pytest.mark.integration
    def test_factor_calculation_with_platform_optimization(self):
        """测试带平台优化的因子计算"""
        if FactorCalculator is None:
            pytest.skip("FactorCalculator not implemented yet - TDD Red phase")

        calculator = FactorCalculator(enable_platform_optimization=True)

        # 注册一个计算密集型因子
        intensive_factor = {
            'factor_id': 'rolling_correlation_50d',
            'factor_name': '50日滚动相关性',
            'description': '与市场指数的50日滚动相关性',
            'category': 'correlation',
            'formula': 'rolling_correlation(stock_returns, market_returns, window=50)',
            'parameters': {
                'window': 50,
                'min_periods': 30
            },
            'data_requirements': ['close', 'market_close'],
            'output_type': 'numeric',
            'frequency': 'daily',
            'version': '1.0.0'
        }

        calc_task = FactorCalculationTask(
            factor_id=intensive_factor['factor_id'],
            stocks=self.test_stocks,
            date_range=self.test_date_range,
            priority='high',
            platform_optimization=True
        )

        # 执行优化计算
        optimized_result = calculator.calculate_factor(
            calc_task,
            market_data=self.test_market_data
        )

        # 执行非优化计算进行对比
        calc_task.platform_optimization = False
        baseline_result = calculator.calculate_factor(
            calc_task,
            market_data=self.test_market_data
        )

        # 验证优化效果
        assert optimized_result['status'] == 'success', "优化计算应该成功"
        assert baseline_result['status'] == 'success', "基准计算应该成功"

        # 验证计算结果一致性
        optimized_values = optimized_result['factor_values']
        baseline_values = baseline_result['factor_values']

        pd.testing.assert_frame_equal(
            optimized_values.sort_values(['stock_code', 'date']),
            baseline_values.sort_values(['stock_code', 'date']),
            rtol=1e-10,
            atol=1e-10,
            check_exact=False
        )

        # 验证性能提升
        assert optimized_result['execution_time'] <= baseline_result['execution_time'], \
            "优化计算的执行时间应该不超过基准计算"

    @pytest.mark.integration
    def test_factor_storage_and_caching(self):
        """测试因子存储和缓存机制"""
        if FactorStorage is None:
            pytest.skip("FactorStorage not implemented yet - TDD Red phase")

        storage = FactorStorage(enable_cache=True)

        # 创建测试因子数据
        factor_values = pd.DataFrame({
            'stock_code': ['sh600000', 'sh600001'] * 10,
            'date': [date(2024, 1, i) for i in range(1, 11)] * 2,
            'factor_value': np.random.randn(20)
        })

        factor_metadata = {
            'factor_id': 'test_factor',
            'calculation_time': datetime.now(),
            'data_version': '1.0',
            'calculation_params': {'window': 20}
        }

        # 首次存储
        storage_result = storage.store_factor_values(
            'test_factor',
            factor_values,
            factor_metadata
        )

        assert storage_result['status'] == 'success', "因子存储应该成功"

        # 测试缓存命中
        cached_result = storage.get_factor_values(
            'test_factor',
            stocks=['sh600000', 'sh600001'],
            date_range=(date(2024, 1, 1), date(2024, 1, 10))
        )

        assert cached_result is not None, "应该从缓存中获取数据"
        assert len(cached_result) == len(factor_values), "缓存数据应该完整"

        # 测试缓存更新
        updated_values = factor_values.copy()
        updated_values['factor_value'] = updated_values['factor_value'] * 2

        update_result = storage.update_factor_values(
            'test_factor',
            updated_values,
            factor_metadata
        )

        assert update_result['status'] == 'success', "因子更新应该成功"

        # 验证缓存已更新
        updated_cached = storage.get_factor_values(
            'test_factor',
            stocks=['sh600000', 'sh600001'],
            date_range=(date(2024, 1, 1), date(2024, 1, 10))
        )

        assert not updated_cached.equals(factor_values), "缓存应该已更新"

    @pytest.mark.integration
    def test_factor_query_optimization(self):
        """测试因子查询的优化功能"""
        if FactorQueryEngine is None:
            pytest.skip("FactorQueryEngine not implemented yet - TDD Red phase")

        query_engine = FactorQueryEngine()

        # 准备大量测试数据
        large_factor_data = pd.DataFrame({
            'stock_code': np.repeat(self.test_stocks, 100),
            'date': pd.date_range('2023-01-01', periods=100, freq='D').tolist() * len(self.test_stocks),
            'factor_value': np.random.randn(len(self.test_stocks) * 100)
        })

        # 存储测试数据
        storage = FactorStorage()
        storage.store_factor_values(
            'large_test_factor',
            large_factor_data,
            {'factor_id': 'large_test_factor'}
        )

        # 测试不同查询策略的性能
        import time

        # 策略1: 分页查询
        start_time = time.time()
        paginated_results = []
        for page in range(0, len(self.test_stocks), 2):
            page_stocks = self.test_stocks[page:page+2]
            page_result = query_engine.query_factor_values(
                factor_id='large_test_factor',
                stocks=page_stocks,
                date_range=(date(2023, 1, 1), date(2023, 3, 31))
            )
            paginated_results.append(page_result)
        paginated_time = time.time() - start_time

        # 策略2: 批量查询
        start_time = time.time()
        batch_result = query_engine.query_factor_values(
            factor_id='large_test_factor',
            stocks=self.test_stocks,
            date_range=(date(2023, 1, 1), date(2023, 3, 31))
        )
        batch_time = time.time() - start_time

        # 验证查询结果一致性
        paginated_combined = pd.concat(paginated_results, ignore_index=True)
        paginated_combined = paginated_combined.sort_values(['stock_code', 'date'])
        batch_result = batch_result.sort_values(['stock_code', 'date'])

        pd.testing.assert_frame_equal(
            paginated_combined.reset_index(drop=True),
            batch_result.reset_index(drop=True)
        )

        # 记录性能指标 (用于后续优化)
        performance_metrics = {
            'paginated_time': paginated_time,
            'batch_time': batch_time,
            'data_size': len(large_factor_data),
            'query_stocks': len(self.test_stocks)
        }

        assert performance_metrics['batch_time'] > 0, "批量查询应该有执行时间"
        assert performance_metrics['paginated_time'] > 0, "分页查询应该有执行时间"

    @pytest.mark.integration
    def test_factor_versioning_and_migration(self):
        """测试因子版本管理和迁移"""
        if FactorRegistry is None:
            pytest.skip("FactorRegistry not implemented yet - TDD Red phase")

        registry = FactorRegistry()

        # 注册因子v1.0
        factor_v1 = {
            'factor_id': 'momentum_test',
            'factor_name': '动量测试因子',
            'description': '测试版本管理的动量因子',
            'category': 'momentum',
            'formula': '(close_t / close_t-10) - 1',
            'parameters': {'window': 10},
            'data_requirements': ['close'],
            'output_type': 'numeric',
            'frequency': 'daily',
            'version': '1.0.0'
        }

        v1_result = registry.register_factor(FactorDefinition.from_dict(factor_v1))
        assert v1_result['status'] == 'success', "v1.0注册应该成功"

        # 注册因子v2.0 (改进版本)
        factor_v2 = factor_v1.copy()
        factor_v2.update({
            'formula': '(close_t / close_t-20) - 1',  # 改变窗口期
            'parameters': {'window': 20},
            'version': '2.0.0'
        })

        v2_result = registry.register_factor(FactorDefinition.from_dict(factor_v2))
        assert v2_result['status'] == 'success', "v2.0注册应该成功"

        # 测试版本查询
        versions = registry.get_factor_versions('momentum_test')
        assert len(versions) == 2, "应该有两个版本"
        assert '1.0.0' in [v['version'] for v in versions], "应该包含v1.0.0"
        assert '2.0.0' in [v['version'] for v in versions], "应该包含v2.0.0"

        # 测试默认版本
        latest_version = registry.get_latest_version('momentum_test')
        assert latest_version['version'] == '2.0.0', "最新版本应该是v2.0.0"

        # 测试版本迁移
        migration_result = registry.migrate_factor_calculations(
            factor_id='momentum_test',
            from_version='1.0.0',
            to_version='2.0.0',
            date_range=(date(2024, 1, 1), date(2024, 1, 10))
        )

        assert migration_result['status'] == 'success', "版本迁移应该成功"

    @pytest.mark.integration
    def test_factor_dependency_management(self):
        """测试因子依赖关系管理"""
        if FactorRegistry is None:
            pytest.skip("FactorRegistry not implemented yet - TDD Red phase")

        registry = FactorRegistry()

        # 注册基础因子
        base_factor = {
            'factor_id': 'daily_return',
            'factor_name': '日收益率',
            'description': '每日收益率计算',
            'category': 'basic',
            'formula': '(close_t / close_t-1) - 1',
            'parameters': {},
            'data_requirements': ['close'],
            'output_type': 'numeric',
            'frequency': 'daily',
            'version': '1.0.0'
        }

        base_result = registry.register_factor(FactorDefinition.from_dict(base_factor))
        assert base_result['status'] == 'success', "基础因子注册应该成功"

        # 注册依赖因子
        dependent_factor = {
            'factor_id': 'volatility_from_returns',
            'factor_name': '基于收益率的波动率',
            'description': '基于日收益率计算的波动率',
            'category': 'risk',
            'formula': 'std(daily_return, window=20)',
            'parameters': {'window': 20},
            'data_requirements': [],  # 不需要原始数据
            'factor_dependencies': ['daily_return'],  # 依赖其他因子
            'output_type': 'numeric',
            'frequency': 'daily',
            'version': '1.0.0'
        }

        dependent_result = registry.register_factor(FactorDefinition.from_dict(dependent_factor))
        assert dependent_result['status'] == 'success', "依赖因子注册应该成功"

        # 测试依赖关系查询
        dependencies = registry.get_factor_dependencies('volatility_from_returns')
        assert 'daily_return' in dependencies, "应该发现依赖关系"

        # 测试计算顺序
        calc_order = registry.get_calculation_order(['volatility_from_returns', 'daily_return'])
        assert calc_order.index('daily_return') < calc_order.index('volatility_from_returns'), \
            "基础因子应该先于依赖因子计算"

    @pytest.mark.integration
    def test_factor_calculation_error_handling(self):
        """测试因子计算的错误处理"""
        if FactorCalculator is None:
            pytest.skip("FactorCalculator not implemented yet - TDD Red phase")

        calculator = FactorCalculator()

        # 测试数据不足的情况
        insufficient_data = self.test_market_data.head(5)  # 只有5天数据

        calc_task = FactorCalculationTask(
            factor_id='momentum_20d',  # 需要20天数据
            stocks=self.test_stocks,
            date_range=(date(2024, 1, 1), date(2024, 1, 5)),
            priority='normal'
        )

        insufficient_result = calculator.calculate_factor(
            calc_task,
            market_data=insufficient_data
        )

        assert insufficient_result['status'] == 'partial_success', \
            "数据不足应该返回部分成功"
        assert 'warnings' in insufficient_result, "应该包含警告信息"

        # 测试数据缺失的情况
        missing_data = self.test_market_data.copy()
        # 制造缺失数据
        missing_data.loc[missing_data['stock_code'] == 'sh600000', 'close'] = np.nan

        missing_result = calculator.calculate_factor(
            calc_task,
            market_data=missing_data
        )

        assert 'errors' in missing_result, "应该报告数据缺失错误"

        # 测试计算异常处理
        invalid_task = FactorCalculationTask(
            factor_id='nonexistent_factor',
            stocks=self.test_stocks,
            date_range=self.test_date_range,
            priority='normal'
        )

        invalid_result = calculator.calculate_factor(
            invalid_task,
            market_data=self.test_market_data
        )

        assert invalid_result['status'] == 'error', "不存在的因子应该返回错误"

    @pytest.mark.integration
    def test_concurrent_factor_calculations(self):
        """测试并发因子计算"""
        if FactorCalculator is None:
            pytest.skip("FactorCalculator not implemented yet - TDD Red phase")

        import concurrent.futures

        calculator = FactorCalculator()

        # 准备多个计算任务
        tasks = []
        for factor_def in self.test_factor_definitions[:3]:  # 使用前3个因子
            task = FactorCalculationTask(
                factor_id=factor_def['factor_id'],
                stocks=self.test_stocks,
                date_range=self.test_date_range,
                priority='normal'
            )
            tasks.append(task)

        def calculate_single_factor(task):
            """计算单个因子"""
            return calculator.calculate_factor(
                task,
                market_data=self.test_market_data
            )

        # 并发执行计算
        with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
            futures = [
                executor.submit(calculate_single_factor, task)
                for task in tasks
            ]

            results = [f.result() for f in concurrent.futures.as_completed(futures)]

        # 验证并发计算结果
        assert len(results) == 3, "应该完成3个因子计算"

        for result in results:
            assert result['status'] in ['success', 'partial_success'], \
                "每个计算都应该成功或部分成功"

        # 验证结果数据完整性
        factor_ids = {result['factor_id'] for result in results}
        expected_ids = {factor_def['factor_id'] for factor_def in self.test_factor_definitions[:3]}
        assert factor_ids == expected_ids, "应该计算所有预期的因子"