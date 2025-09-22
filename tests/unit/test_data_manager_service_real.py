"""
Data Manager Service Unit Tests

基于真实Hikyuu框架的数据管理服务单元测试
不使用mock数据，测试DataUpdater, DataQualityChecker, DataExceptionHandler功能
"""

import pytest
import asyncio
import pandas as pd
import numpy as np
from datetime import datetime, date, timedelta
from typing import Dict, List, Any

from src.services.data_manager_service import (
    DataUpdater, DataQualityChecker, DataExceptionHandler, DataManagerService
)
from src.models.validation_models import ValidationRule, ValidationResult, ValidationIssue, ValidationSeverity


class TestDataUpdater:
    """DataUpdater组件单元测试"""

    @pytest.fixture
    def data_updater(self):
        """创建DataUpdater实例"""
        config = {
            'data_sources': ['hikyuu'],
            'update_frequency': 'daily',
            'retry_count': 3,
            'max_workers': 5,
            'batch_size': 50
        }
        return DataUpdater(config)

    def test_data_updater_initialization(self, data_updater):
        """测试DataUpdater初始化"""
        assert data_updater.config is not None
        assert data_updater.data_sources == ['hikyuu']
        assert data_updater.update_frequency == 'daily'
        assert data_updater.retry_count == 3

    @pytest.mark.asyncio
    async def test_get_all_stock_codes(self, data_updater):
        """测试获取股票代码列表"""
        stock_codes = await data_updater._get_all_stock_codes()

        assert isinstance(stock_codes, list)
        assert len(stock_codes) > 0

        # 验证股票代码格式
        for code in stock_codes[:10]:  # 检查前10个
            assert isinstance(code, str)
            assert len(code) >= 8  # 例如 sh600000
            assert code[:2] in ['sh', 'sz']

    @pytest.mark.asyncio
    async def test_update_single_stock_success(self, data_updater):
        """测试单个股票数据更新成功情况"""
        stock_code = "sh600000"
        start_date = date(2024, 1, 1)
        end_date = date(2024, 1, 31)
        data_types = ['kdata']

        result = data_updater._update_single_stock(
            stock_code, start_date, end_date, data_types
        )

        assert isinstance(result, dict)
        assert 'success' in result
        assert 'records_count' in result
        assert 'error' in result

        if result['success']:
            assert result['records_count'] >= 0
            assert result['error'] is None

    @pytest.mark.asyncio
    async def test_update_single_stock_invalid_code(self, data_updater):
        """测试无效股票代码的更新"""
        invalid_code = "invalid123"
        start_date = date(2024, 1, 1)
        end_date = date(2024, 1, 31)
        data_types = ['kdata']

        result = data_updater._update_single_stock(
            invalid_code, start_date, end_date, data_types
        )

        assert isinstance(result, dict)
        assert result['success'] is False
        assert result['records_count'] == 0
        assert result['error'] is not None

    @pytest.mark.asyncio
    async def test_update_stock_batch(self, data_updater):
        """测试批量股票数据更新"""
        stock_codes = ["sh600000", "sh600036", "sz000001"]
        start_date = date(2024, 1, 1)
        end_date = date(2024, 1, 10)
        data_types = ['kdata']
        max_workers = 2

        result = await data_updater._update_stock_batch(
            stock_codes, start_date, end_date, data_types, max_workers
        )

        assert isinstance(result, dict)
        assert 'successful_updates' in result
        assert 'failed_updates' in result
        assert 'updated_records' in result
        assert 'errors' in result
        assert 'warnings' in result

        # 验证统计数据
        total_updates = result['successful_updates'] + result['failed_updates']
        assert total_updates <= len(stock_codes)

    @pytest.mark.asyncio
    async def test_update_market_data_full_workflow(self, data_updater):
        """测试完整的市场数据更新工作流"""
        stock_codes = ["sh600000", "sz000001"]
        start_date = date(2024, 1, 1)
        end_date = date(2024, 1, 5)
        data_types = ['kdata']

        result = await data_updater.update_market_data(
            stock_codes=stock_codes,
            start_date=start_date,
            end_date=end_date,
            data_types=data_types
        )

        # 验证返回结果结构
        expected_keys = [
            'total_stocks', 'successful_updates', 'failed_updates',
            'updated_records', 'execution_time_seconds', 'errors', 'warnings'
        ]

        for key in expected_keys:
            assert key in result

        assert result['total_stocks'] == len(stock_codes)
        assert isinstance(result['execution_time_seconds'], float)
        assert result['execution_time_seconds'] > 0

    @pytest.mark.asyncio
    async def test_update_market_data_with_default_params(self, data_updater):
        """测试使用默认参数的市场数据更新"""
        result = await data_updater.update_market_data()

        assert isinstance(result, dict)
        assert 'total_stocks' in result
        assert result['total_stocks'] > 0  # 应该获取到股票列表

    @pytest.mark.asyncio
    async def test_update_market_data_date_validation(self, data_updater):
        """测试日期参数验证"""
        # 测试end_date早于start_date的情况
        start_date = date(2024, 1, 31)
        end_date = date(2024, 1, 1)

        result = await data_updater.update_market_data(
            stock_codes=["sh600000"],
            start_date=start_date,
            end_date=end_date
        )

        # 应该能够处理这种情况，可能返回空结果或调整日期范围
        assert isinstance(result, dict)
        assert 'total_stocks' in result

    @pytest.mark.asyncio
    async def test_concurrent_stock_updates(self, data_updater):
        """测试并发股票更新性能"""
        import time

        stock_codes = [f"sh{600000 + i:06d}" for i in range(10)]
        start_date = date(2024, 1, 1)
        end_date = date(2024, 1, 3)

        start_time = time.time()
        result = await data_updater._update_stock_batch(
            stock_codes, start_date, end_date, ['kdata'], max_workers=5
        )
        end_time = time.time()

        execution_time = end_time - start_time

        # 验证并发执行效果
        assert execution_time < 30  # 应该在30秒内完成
        assert isinstance(result, dict)

        total_attempts = result['successful_updates'] + result['failed_updates']
        assert total_attempts <= len(stock_codes)


class TestDataQualityChecker:
    """DataQualityChecker组件单元测试"""

    @pytest.fixture
    def sample_good_data(self):
        """创建高质量的测试数据"""
        np.random.seed(42)
        dates = pd.date_range('2024-01-01', periods=100, freq='D')

        data = []
        base_price = 10.0
        for i, date in enumerate(dates):
            # 模拟真实的股价波动
            price_change = np.random.randn() * 0.02  # 2%日波动
            base_price *= (1 + price_change)

            data.append({
                'date': date,
                'stock_code': 'sh600000',
                'open': base_price * (1 + np.random.randn() * 0.005),
                'close': base_price,
                'high': base_price * (1 + abs(np.random.randn()) * 0.01),
                'low': base_price * (1 - abs(np.random.randn()) * 0.01),
                'volume': 1000000 + np.random.randint(0, 500000),
                'amount': base_price * (1000000 + np.random.randint(0, 500000))
            })

        return pd.DataFrame(data)

    @pytest.fixture
    def sample_bad_data(self):
        """创建包含质量问题的测试数据"""
        np.random.seed(123)
        dates = pd.date_range('2024-01-01', periods=50, freq='D')

        data = []
        for i, date in enumerate(dates):
            row = {
                'date': date,
                'stock_code': 'sh600000',
                'open': 10.0 + np.random.randn(),
                'close': 10.0 + np.random.randn(),
                'high': 10.0 + np.random.randn(),
                'low': 10.0 + np.random.randn(),
                'volume': 1000000 + np.random.randint(0, 1000000),
                'amount': 10000000 + np.random.randint(0, 5000000)
            }

            # 引入数据质量问题
            if i % 10 == 0:  # 10%缺失值
                row['close'] = None
            if i % 15 == 0:  # 异常价格
                row['high'] = -100  # 负价格
            if i % 8 == 0:  # 价格逻辑错误
                row['low'] = row['high'] + 1  # 最低价高于最高价

        df = pd.DataFrame(data)

        # 添加重复行
        duplicate_rows = df.head(5).copy()
        df = pd.concat([df, duplicate_rows], ignore_index=True)

        return df

    @pytest.fixture
    def quality_checker(self):
        """创建DataQualityChecker实例"""
        return DataQualityChecker()

    def test_quality_checker_initialization(self, quality_checker):
        """测试DataQualityChecker初始化"""
        assert quality_checker.rules is not None
        assert len(quality_checker.rules) > 0

        # 验证默认规则
        rule_ids = [rule.rule_id for rule in quality_checker.rules]
        expected_rules = [
            'dq_missing_values', 'dq_duplicate_values',
            'dq_value_range', 'dq_time_continuity'
        ]

        for expected_rule in expected_rules:
            assert expected_rule in rule_ids

    def test_default_validation_rules(self, quality_checker):
        """测试默认验证规则配置"""
        for rule in quality_checker.rules:
            assert hasattr(rule, 'rule_id')
            assert hasattr(rule, 'rule_name')
            assert hasattr(rule, 'rule_type')
            assert hasattr(rule, 'severity')
            assert rule.rule_id is not None
            assert rule.rule_name is not None

    @pytest.mark.asyncio
    async def test_check_good_data_quality(self, quality_checker, sample_good_data):
        """测试高质量数据的检查"""
        result = await quality_checker.check_data_quality(
            sample_good_data, "good_market_data"
        )

        assert isinstance(result, ValidationResult)
        assert result.validation_id is not None
        assert result.factor_name == "good_market_data"
        assert result.validation_date is not None
        assert isinstance(result.execution_time_seconds, float)
        assert result.execution_time_seconds > 0

        # 高质量数据应该通过大部分检查
        assert result.validation_score >= 0.8  # 至少80分
        assert len(result.issues) <= 2  # 最多2个轻微问题

    @pytest.mark.asyncio
    async def test_check_bad_data_quality(self, quality_checker, sample_bad_data):
        """测试低质量数据的检查"""
        result = await quality_checker.check_data_quality(
            sample_bad_data, "bad_market_data"
        )

        assert isinstance(result, ValidationResult)
        assert result.factor_name == "bad_market_data"

        # 低质量数据应该发现问题
        assert len(result.issues) > 0
        assert result.validation_score < 0.9  # 分数应该较低

        # 检查是否发现了预期的问题类型
        issue_categories = [issue.category for issue in result.issues]
        expected_categories = ['missing_data', 'value_range', 'duplicate_data']

        found_categories = set(issue_categories)
        expected_set = set(expected_categories)

        # 至少应该发现一些预期的问题类型
        assert len(found_categories.intersection(expected_set)) > 0

    @pytest.mark.asyncio
    async def test_missing_values_detection(self, quality_checker):
        """测试缺失值检测功能"""
        # 创建包含缺失值的数据
        data = pd.DataFrame({
            'date': pd.date_range('2024-01-01', periods=10),
            'price': [10, 11, None, 13, None, 15, 16, 17, None, 19],
            'volume': [1000, 1100, 1200, None, 1400, 1500, None, 1700, 1800, 1900]
        })

        result = await quality_checker.check_data_quality(data, "missing_test")

        # 应该检测到缺失值问题
        missing_issues = [issue for issue in result.issues if issue.category == 'missing_data']
        assert len(missing_issues) > 0

        # 验证缺失值比例计算
        missing_rule_result = result.rule_results.get('dq_missing_values')
        if missing_rule_result:
            assert 'actual_value' in missing_rule_result
            assert 'threshold_value' in missing_rule_result
            assert missing_rule_result['actual_value'] > 0

    @pytest.mark.asyncio
    async def test_duplicate_detection(self, quality_checker):
        """测试重复值检测功能"""
        # 创建包含重复行的数据
        base_data = pd.DataFrame({
            'date': pd.date_range('2024-01-01', periods=5),
            'stock_code': ['sh600000'] * 5,
            'price': [10, 11, 12, 13, 14]
        })

        # 添加重复行
        duplicate_data = pd.concat([base_data, base_data.head(2)], ignore_index=True)

        result = await quality_checker.check_data_quality(duplicate_data, "duplicate_test")

        # 应该检测到重复值问题
        duplicate_issues = [issue for issue in result.issues if issue.category == 'duplicate_data']
        assert len(duplicate_issues) > 0

    @pytest.mark.asyncio
    async def test_value_range_detection(self, quality_checker):
        """测试数值范围检测功能"""
        # 创建包含异常值的数据
        data = pd.DataFrame({
            'date': pd.date_range('2024-01-01', periods=10),
            'open_price': [10, 11, -5, 13, 14, 15, 16, 17, 2000, 19],  # 负值和极大值
            'close_price': [10, 11, 12, 13, -10, 15, 16, 17, 18, 1500],  # 负值和极大值
            'volume': [-100, 1100, 1200, 1300, 1400, 1500, 1600, 1700, 1800, 1900]  # 负交易量
        })

        result = await quality_checker.check_data_quality(data, "range_test")

        # 应该检测到值范围问题
        range_issues = [issue for issue in result.issues if issue.category == 'value_range']
        assert len(range_issues) > 0

    @pytest.mark.asyncio
    async def test_time_continuity_detection(self, quality_checker):
        """测试时间连续性检测功能"""
        # 创建时间不连续的数据
        dates = [
            datetime(2024, 1, 1),
            datetime(2024, 1, 2),
            datetime(2024, 1, 10),  # 大gap
            datetime(2024, 1, 11),
            datetime(2024, 1, 20)   # 大gap
        ]

        data = pd.DataFrame({
            'trade_date': dates,
            'stock_code': ['sh600000'] * 5,
            'price': [10, 11, 12, 13, 14]
        })

        result = await quality_checker.check_data_quality(data, "continuity_test")

        # 应该检测到时间连续性问题
        continuity_issues = [issue for issue in result.issues if issue.category == 'time_continuity']
        assert len(continuity_issues) > 0

    @pytest.mark.asyncio
    async def test_empty_dataset_handling(self, quality_checker):
        """测试空数据集处理"""
        empty_data = pd.DataFrame()

        result = await quality_checker.check_data_quality(empty_data, "empty_test")

        assert isinstance(result, ValidationResult)
        assert len(result.issues) > 0

        # 应该有空数据集的关键错误
        critical_issues = [issue for issue in result.issues
                          if issue.severity == ValidationSeverity.CRITICAL]
        assert len(critical_issues) > 0

    @pytest.mark.asyncio
    async def test_multiple_datasets_quality_check(self, quality_checker, sample_good_data, sample_bad_data):
        """测试多数据集质量检查"""
        datasets = {
            'good_data': sample_good_data,
            'bad_data': sample_bad_data
        }

        result = await quality_checker.check_data_quality(datasets, "multi_dataset")

        assert isinstance(result, ValidationResult)
        assert len(result.issues) > 0  # bad_data应该产生问题
        assert len(result.rule_results) > 0


class TestDataExceptionHandler:
    """DataExceptionHandler组件单元测试"""

    @pytest.fixture
    def exception_handler(self):
        """创建DataExceptionHandler实例"""
        config = {
            'strategies': {
                'missing_data': 'interpolate',
                'outlier_data': 'winsorize',
                'duplicate_data': 'deduplicate',
                'inconsistent_data': 'validate'
            }
        }
        return DataExceptionHandler(config)

    @pytest.fixture
    def problematic_data(self):
        """创建有问题的数据"""
        data = pd.DataFrame({
            'date': pd.date_range('2024-01-01', periods=10),
            'stock_code': ['sh600000'] * 10,
            'price': [10, 11, None, 13, None, 15, 16, 17, None, 19],  # 缺失值
            'volume': [1000, 1100, 1200, 1300, 1400, 1500, 1600, 1700, 1800, 1900],
            'amount': [10000, 11000, -12000, 13000, 14000, 15000, 16000, 17000, 18000, 19000]  # 负值
        })

        # 添加重复行
        duplicate_row = data.iloc[0:1].copy()
        data = pd.concat([data, duplicate_row], ignore_index=True)

        return data

    @pytest.fixture
    def sample_validation_result(self, problematic_data):
        """创建示例验证结果"""
        from src.models.validation_models import ValidationRuleType

        result = ValidationResult(
            validation_id="test_validation",
            factor_name="test_data",
            validation_date=datetime.now(),
            passed=False,
            validation_score=0.6
        )

        # 添加一些问题
        result.issues = [
            ValidationIssue(
                issue_id="missing_001",
                rule_id="dq_missing_values",
                rule_name="缺失值检查",
                category="missing_data",
                severity=ValidationSeverity.WARNING,
                description="检测到缺失值"
            ),
            ValidationIssue(
                issue_id="duplicate_001",
                rule_id="dq_duplicate_values",
                rule_name="重复值检查",
                category="duplicate_data",
                severity=ValidationSeverity.ERROR,
                description="检测到重复记录"
            ),
            ValidationIssue(
                issue_id="range_001",
                rule_id="dq_value_range",
                rule_name="数值范围检查",
                category="value_range",
                severity=ValidationSeverity.ERROR,
                description="检测到异常值"
            )
        ]

        return result

    def test_exception_handler_initialization(self, exception_handler):
        """测试DataExceptionHandler初始化"""
        assert exception_handler.strategies is not None
        assert 'missing_data' in exception_handler.strategies
        assert 'duplicate_data' in exception_handler.strategies
        assert 'outlier_data' in exception_handler.strategies

    @pytest.mark.asyncio
    async def test_handle_data_exceptions_complete_workflow(self, exception_handler, problematic_data, sample_validation_result):
        """测试完整的数据异常处理工作流"""
        recovery_config = {
            'missing_data_strategy': 'interpolate',
            'duplicate_strategy': 'drop_duplicates',
            'value_range_strategy': 'clip'
        }

        result = await exception_handler.handle_data_exceptions(
            problematic_data, sample_validation_result, recovery_config
        )

        assert isinstance(result, dict)

        # 验证结果结构
        expected_keys = [
            'success', 'original_data_shape', 'processed_data',
            'fixes_applied', 'issues_resolved', 'issues_remaining',
            'execution_time_seconds', 'warnings'
        ]

        for key in expected_keys:
            assert key in result

        assert isinstance(result['processed_data'], pd.DataFrame)
        assert isinstance(result['fixes_applied'], list)
        assert isinstance(result['issues_resolved'], list)
        assert isinstance(result['execution_time_seconds'], float)
        assert result['execution_time_seconds'] > 0

    @pytest.mark.asyncio
    async def test_fix_missing_data_interpolation(self, exception_handler):
        """测试缺失数据插值修复"""
        # 创建包含缺失值的数据
        data = pd.DataFrame({
            'price': [10.0, 11.0, None, 13.0, None, 15.0],
            'volume': [1000, 1100, None, 1300, 1400, 1500]
        })

        issue = ValidationIssue(
            issue_id="missing_test",
            rule_id="dq_missing_values",
            rule_name="缺失值",
            category="missing_data",
            severity=ValidationSeverity.WARNING,
            description="缺失值测试"
        )

        config = {'missing_data_strategy': 'interpolate'}
        fix_method = await exception_handler._fix_missing_data(data, issue, config)

        assert fix_method == 'linear_interpolation'
        assert data['price'].isna().sum() == 0  # 应该没有缺失值了
        assert data['volume'].isna().sum() == 0

    @pytest.mark.asyncio
    async def test_fix_missing_data_forward_fill(self, exception_handler):
        """测试缺失数据前向填充修复"""
        data = pd.DataFrame({
            'price': [10.0, 11.0, None, None, 14.0, 15.0],
        })

        issue = ValidationIssue(
            issue_id="missing_test",
            rule_id="dq_missing_values",
            rule_name="缺失值",
            category="missing_data",
            severity=ValidationSeverity.WARNING,
            description="缺失值测试"
        )

        config = {'missing_data_strategy': 'forward_fill'}
        fix_method = await exception_handler._fix_missing_data(data, issue, config)

        assert fix_method == 'forward_fill'
        # 验证前向填充效果
        assert data.loc[2, 'price'] == 11.0  # 应该被前值填充
        assert data.loc[3, 'price'] == 11.0

    @pytest.mark.asyncio
    async def test_fix_duplicate_data(self, exception_handler):
        """测试重复数据修复"""
        # 创建包含重复行的数据
        original_data = pd.DataFrame({
            'date': ['2024-01-01', '2024-01-02', '2024-01-03'],
            'price': [10, 11, 12]
        })
        duplicate_data = pd.concat([original_data, original_data.iloc[0:1]], ignore_index=True)

        issue = ValidationIssue(
            issue_id="duplicate_test",
            rule_id="dq_duplicate_values",
            rule_name="重复值",
            category="duplicate_data",
            severity=ValidationSeverity.ERROR,
            description="重复值测试"
        )

        config = {'duplicate_strategy': 'drop_duplicates'}
        fix_method = await exception_handler._fix_duplicate_data(duplicate_data, issue, config)

        assert fix_method == 'drop_duplicates_first'
        assert len(duplicate_data) == 3  # 应该回到原始长度

    @pytest.mark.asyncio
    async def test_fix_value_range_clipping(self, exception_handler):
        """测试数值范围修复 - 截断方法"""
        data = pd.DataFrame({
            'open_price': [-5, 10, 2000, 15, 20],  # 负值和极大值
            'close_price': [8, 11, 12, -10, 18]   # 负值
        })

        issue = ValidationIssue(
            issue_id="range_test",
            rule_id="dq_value_range",
            rule_name="数值范围",
            category="value_range",
            severity=ValidationSeverity.ERROR,
            description="数值范围测试"
        )

        config = {'value_range_strategy': 'clip'}
        fix_method = await exception_handler._fix_value_range(data, issue, config)

        assert fix_method == 'value_clipping'
        # 验证截断效果
        assert data['open_price'].min() >= 0
        assert data['open_price'].max() <= 1000
        assert data['close_price'].min() >= 0

    @pytest.mark.asyncio
    async def test_fix_time_continuity(self, exception_handler):
        """测试时间连续性修复"""
        # 创建时间不连续的数据
        data = pd.DataFrame({
            'trade_date': ['2024-01-01', '2024-01-03', '2024-01-06'],  # 有gaps
            'price': [10, 12, 15]
        })

        issue = ValidationIssue(
            issue_id="continuity_test",
            rule_id="dq_time_continuity",
            rule_name="时间连续性",
            category="time_continuity",
            severity=ValidationSeverity.WARNING,
            description="时间连续性测试"
        )

        config = {'time_continuity_strategy': 'interpolate_dates'}
        fix_method = await exception_handler._fix_time_continuity(data, issue, config)

        assert fix_method == 'date_interpolation'
        # 修复后应该有更多行（插值的日期）
        assert len(data) > 3

    @pytest.mark.asyncio
    async def test_handle_unknown_issue_category(self, exception_handler, problematic_data):
        """测试处理未知问题类别"""
        unknown_issue = ValidationIssue(
            issue_id="unknown_test",
            rule_id="unknown_rule",
            rule_name="未知规则",
            category="unknown_category",
            severity=ValidationSeverity.WARNING,
            description="未知问题类别测试"
        )

        fix_method = await exception_handler._handle_single_issue(
            problematic_data, unknown_issue, {}
        )

        assert fix_method is None

    @pytest.mark.asyncio
    async def test_exception_handling_with_empty_issues(self, exception_handler, problematic_data):
        """测试没有问题时的异常处理"""
        empty_validation_result = ValidationResult(
            validation_id="empty_test",
            factor_name="test_data",
            validation_date=datetime.now(),
            passed=True,
            validation_score=1.0
        )
        empty_validation_result.issues = []

        result = await exception_handler.handle_data_exceptions(
            problematic_data, empty_validation_result, {}
        )

        assert result['success'] is True
        assert len(result['fixes_applied']) == 0
        assert len(result['issues_resolved']) == 0
        assert len(result['issues_remaining']) == 0


class TestDataManagerService:
    """DataManagerService集成测试"""

    @pytest.fixture
    def data_manager_service(self):
        """创建DataManagerService实例"""
        config = {
            'updater': {
                'data_sources': ['hikyuu'],
                'max_workers': 3,
                'batch_size': 10
            },
            'quality_checker': {},
            'exception_handler': {},
            'enable_audit': True
        }
        return DataManagerService(config)

    def test_data_manager_service_initialization(self, data_manager_service):
        """测试DataManagerService初始化"""
        assert data_manager_service.data_updater is not None
        assert data_manager_service.quality_checker is not None
        assert data_manager_service.exception_handler is not None
        assert data_manager_service.enable_audit is True

    def test_get_stock_list_all(self, data_manager_service):
        """测试获取全部股票列表"""
        stocks = data_manager_service.get_stock_list()

        assert isinstance(stocks, list)
        # 即使是空列表也是有效结果
        assert stocks is not None

    def test_get_stock_list_by_market(self, data_manager_service):
        """测试按市场获取股票列表"""
        sh_stocks = data_manager_service.get_stock_list(market='SH')
        sz_stocks = data_manager_service.get_stock_list(market='SZ')

        assert isinstance(sh_stocks, list)
        assert isinstance(sz_stocks, list)

    def test_get_stock_list_by_sector(self, data_manager_service):
        """测试按行业获取股票列表"""
        bank_stocks = data_manager_service.get_stock_list(sector='银行')

        assert isinstance(bank_stocks, list)

    def test_get_market_data_single_stock(self, data_manager_service):
        """测试获取单只股票市场数据"""
        result = data_manager_service.get_market_data(
            'sh600000', '2024-01-01', '2024-01-31'
        )

        assert isinstance(result, dict)
        assert 'stock_code' in result
        assert 'data' in result
        assert 'count' in result
        assert result['stock_code'] == 'sh600000'

    def test_get_market_data_invalid_stock(self, data_manager_service):
        """测试获取无效股票的市场数据"""
        result = data_manager_service.get_market_data(
            'invalid123', '2024-01-01', '2024-01-31'
        )

        assert isinstance(result, dict)
        assert result['stock_code'] == 'invalid123'
        assert result['count'] == 0

    def test_get_market_data_invalid_date_format(self, data_manager_service):
        """测试无效日期格式"""
        result = data_manager_service.get_market_data(
            'sh600000', 'invalid-date', '2024-01-31'
        )

        assert isinstance(result, dict)
        assert 'error' in result

    def test_get_stock_info_valid_stock(self, data_manager_service):
        """测试获取有效股票信息"""
        result = data_manager_service.get_stock_info('sh600000')

        assert isinstance(result, dict)
        assert 'stock_code' in result
        # 即使股票不存在，也应该返回适当的响应

    def test_get_stock_info_invalid_stock(self, data_manager_service):
        """测试获取无效股票信息"""
        result = data_manager_service.get_stock_info('invalid123')

        assert isinstance(result, dict)
        assert result['stock_code'] == 'invalid123'

    @pytest.mark.asyncio
    async def test_execute_data_workflow_complete(self, data_manager_service):
        """测试完整数据工作流执行"""
        workflow_config = {
            'stock_codes': ['sh600000', 'sz000001'],
            'start_date': date(2024, 1, 1),
            'end_date': date(2024, 1, 5),
            'data_types': ['kdata'],
            'update_data': True,
            'check_quality': True,
            'handle_exceptions': True,
            'recovery_config': {
                'missing_data_strategy': 'interpolate',
                'duplicate_strategy': 'drop_duplicates'
            }
        }

        result = await data_manager_service.execute_data_workflow(workflow_config)

        assert isinstance(result, dict)

        # 验证工作流结果结构
        expected_keys = [
            'workflow_id', 'success', 'steps_completed', 'steps_failed',
            'total_execution_time', 'data_summary', 'audit_entries'
        ]

        for key in expected_keys:
            assert key in result

        assert isinstance(result['steps_completed'], list)
        assert isinstance(result['steps_failed'], list)
        assert isinstance(result['total_execution_time'], float)
        assert result['total_execution_time'] > 0

    @pytest.mark.asyncio
    async def test_execute_data_workflow_partial(self, data_manager_service):
        """测试部分数据工作流执行"""
        workflow_config = {
            'update_data': False,  # 跳过数据更新
            'check_quality': True,
            'handle_exceptions': False  # 跳过异常处理
        }

        result = await data_manager_service.execute_data_workflow(workflow_config)

        assert isinstance(result, dict)
        assert result['success'] is True

        # 应该只执行了质量检查
        completed_steps = result['steps_completed']
        failed_steps = result['steps_failed']

        assert 'data_update' not in completed_steps
        assert 'data_update' not in failed_steps
        assert 'exception_handling' not in completed_steps

    @pytest.mark.asyncio
    async def test_get_market_data_batch(self, data_manager_service):
        """测试批量获取市场数据"""
        stock_codes = ['sh600000', 'sz000001', 'sh600036']
        start_date = date(2024, 1, 1)
        end_date = date(2024, 1, 10)

        result = await data_manager_service.get_market_data_batch(
            stock_codes, start_date, end_date
        )

        assert isinstance(result, dict)
        assert 'success' in result
        assert 'data' in result
        assert 'stock_count' in result
        assert 'successful_stocks' in result
        assert 'failed_stocks' in result
        assert 'records_count' in result

        assert result['stock_count'] == len(stock_codes)

        if result['success']:
            assert isinstance(result['data'], pd.DataFrame)
            # 验证数据格式
            if not result['data'].empty:
                expected_columns = ['date', 'stock_code', 'open', 'close', 'high', 'low', 'volume', 'amount']
                for col in expected_columns:
                    assert col in result['data'].columns

    @pytest.mark.asyncio
    async def test_get_market_data_batch_empty_stocks(self, data_manager_service):
        """测试空股票列表的批量数据获取"""
        result = await data_manager_service.get_market_data_batch(
            [], date(2024, 1, 1), date(2024, 1, 10)
        )

        assert isinstance(result, dict)
        assert result['stock_count'] == 0
        assert result['successful_stocks'] == 0
        assert result['failed_stocks'] == 0

    def test_sample_data_generation(self, data_manager_service):
        """测试样本数据生成功能"""
        sample_data = data_manager_service._get_sample_data_for_quality_check()

        assert isinstance(sample_data, pd.DataFrame)
        assert not sample_data.empty
        assert len(sample_data) > 0

        # 验证样本数据格式
        expected_columns = ['date', 'stock_code', 'open', 'close', 'high', 'low', 'volume']
        for col in expected_columns:
            assert col in sample_data.columns


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])