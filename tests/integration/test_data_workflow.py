"""
T025: 数据更新→质量检查→异常处理 集成测试

测试完整的数据管理工作流程：
1. 数据更新流程 (增量/全量)
2. 数据质量检查与验证
3. 异常检测和处理机制
4. 数据完整性报告生成

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
    from src.services.data_updater import DataUpdater
    from src.services.data_quality_checker import DataQualityChecker
    from src.services.data_exception_handler import DataExceptionHandler
    from src.services.data_report_generator import DataReportGenerator
    from src.models.data_quality_rule import DataQualityRule
    from src.models.data_update_task import DataUpdateTask
    from src.models.data_quality_report import DataQualityReport
except ImportError:
    # TDD Red阶段 - 这些模块还不存在
    DataUpdater = None
    DataQualityChecker = None
    DataExceptionHandler = None
    DataReportGenerator = None
    DataQualityRule = None
    DataUpdateTask = None
    DataQualityReport = None


@pytest.mark.integration
@pytest.mark.data_workflow
@pytest.mark.requires_mysql
@pytest.mark.requires_hikyuu
class TestDataWorkflow:
    """数据更新→质量检查→异常处理 集成测试"""

    def setup_method(self):
        """设置测试环境"""
        self.test_stocks = ["sh600000", "sh600001", "sz000001", "sz000002"]
        self.test_date_range = (
            date(2024, 1, 1),
            date(2024, 1, 31)
        )

        # 创建测试数据
        self.valid_test_data = self._create_valid_test_data()
        self.invalid_test_data = self._create_invalid_test_data()

        # 质量检查规则配置
        self.quality_rules = self._create_quality_rules()

    def _create_valid_test_data(self) -> pd.DataFrame:
        """创建有效的测试数据"""
        np.random.seed(42)
        data = []

        for stock in self.test_stocks:
            for i in range(31):  # 31天数据
                current_date = self.test_date_range[0] + timedelta(days=i)
                base_price = 10.0 + np.random.randn() * 0.1

                data.append({
                    'stock_code': stock,
                    'date': current_date,
                    'open': base_price + np.random.randn() * 0.05,
                    'high': base_price + abs(np.random.randn() * 0.1),
                    'low': base_price - abs(np.random.randn() * 0.1),
                    'close': base_price + np.random.randn() * 0.05,
                    'volume': 1000000 + np.random.randint(0, 500000),
                    'amount': 0.0,  # 将通过计算得出
                })

        df = pd.DataFrame(data)
        # 计算成交额
        df['amount'] = df['close'] * df['volume']
        return df

    def _create_invalid_test_data(self) -> pd.DataFrame:
        """创建包含各种问题的测试数据"""
        data = []

        # 问题1: 缺失数据
        data.append({
            'stock_code': 'sh600000',
            'date': date(2024, 1, 15),
            'open': None,  # 缺失开盘价
            'high': 10.5,
            'low': 9.5,
            'close': 10.0,
            'volume': 1000000,
            'amount': 10000000.0
        })

        # 问题2: 价格异常 (high < low)
        data.append({
            'stock_code': 'sh600001',
            'date': date(2024, 1, 16),
            'open': 10.0,
            'high': 9.5,  # 最高价小于最低价
            'low': 10.5,
            'close': 10.0,
            'volume': 1000000,
            'amount': 10000000.0
        })

        # 问题3: 零成交量但有成交额
        data.append({
            'stock_code': 'sz000001',
            'date': date(2024, 1, 17),
            'open': 10.0,
            'high': 10.5,
            'low': 9.5,
            'close': 10.0,
            'volume': 0,  # 零成交量
            'amount': 10000000.0  # 但有成交额
        })

        # 问题4: 价格突变 (超过涨跌停限制)
        data.append({
            'stock_code': 'sz000002',
            'date': date(2024, 1, 18),
            'open': 10.0,
            'high': 15.0,  # 涨幅50%，超过涨停限制
            'low': 9.5,
            'close': 15.0,
            'volume': 1000000,
            'amount': 15000000.0
        })

        return pd.DataFrame(data)

    def _create_quality_rules(self) -> List[Dict[str, Any]]:
        """创建数据质量检查规则"""
        return [
            {
                'rule_name': 'price_consistency',
                'description': '价格一致性检查',
                'rule_type': 'price_validation',
                'parameters': {
                    'check_high_low_consistency': True,
                    'check_ohlc_range': True
                },
                'severity': 'error'
            },
            {
                'rule_name': 'missing_data_check',
                'description': '缺失数据检查',
                'rule_type': 'completeness',
                'parameters': {
                    'required_fields': ['open', 'high', 'low', 'close', 'volume'],
                    'missing_threshold': 0.05  # 5%以下缺失可接受
                },
                'severity': 'warning'
            },
            {
                'rule_name': 'volume_amount_consistency',
                'description': '成交量成交额一致性',
                'rule_type': 'logical_consistency',
                'parameters': {
                    'check_zero_volume_nonzero_amount': True,
                    'amount_volume_ratio_threshold': [0.1, 1000.0]
                },
                'severity': 'error'
            },
            {
                'rule_name': 'price_change_limit',
                'description': '价格变动限制检查',
                'rule_type': 'business_rule',
                'parameters': {
                    'daily_limit_pct': 0.10,  # 10%涨跌停
                    'st_limit_pct': 0.05,     # ST股票5%涨跌停
                },
                'severity': 'warning'
            }
        ]

    @pytest.mark.integration
    def test_complete_data_workflow_success(self):
        """测试完整数据工作流程 - 成功场景"""
        if DataUpdater is None:
            pytest.skip("DataUpdater not implemented yet - TDD Red phase")

        # Step 1: 数据更新
        updater = DataUpdater()
        update_task = updater.create_update_task(
            stocks=self.test_stocks,
            date_range=self.test_date_range,
            update_type='incremental'
        )

        # 执行数据更新
        update_result = updater.execute_update(
            update_task,
            test_data=self.valid_test_data
        )

        # 验证更新结果
        assert update_result['status'] == 'success', "数据更新应该成功"
        assert update_result['updated_records'] > 0, "应该有记录被更新"

        # Step 2: 数据质量检查
        quality_checker = DataQualityChecker(rules=self.quality_rules)
        quality_result = quality_checker.check_data_quality(
            update_result['data'],
            update_task
        )

        # 验证质量检查结果
        assert quality_result is not None, "质量检查应该返回结果"
        assert 'overall_score' in quality_result, "应该包含总体质量分数"
        assert quality_result['overall_score'] >= 0.8, "有效数据的质量分数应该较高"

        # Step 3: 生成报告
        report_generator = DataReportGenerator()
        report = report_generator.generate_quality_report(
            update_result,
            quality_result
        )

        # 验证报告内容
        assert report is not None, "应该生成质量报告"
        assert 'summary' in report, "报告应该包含摘要"
        assert 'details' in report, "报告应该包含详细信息"

    @pytest.mark.integration
    def test_complete_data_workflow_with_exceptions(self):
        """测试包含异常处理的完整数据工作流程"""
        if DataUpdater is None:
            pytest.skip("DataUpdater not implemented yet - TDD Red phase")

        # Step 1: 数据更新 (使用包含错误的数据)
        updater = DataUpdater()
        update_task = updater.create_update_task(
            stocks=self.test_stocks,
            date_range=self.test_date_range,
            update_type='incremental'
        )

        # 合并有效和无效数据
        combined_data = pd.concat([
            self.valid_test_data,
            self.invalid_test_data
        ], ignore_index=True)

        update_result = updater.execute_update(
            update_task,
            test_data=combined_data
        )

        # Step 2: 数据质量检查 (应该发现问题)
        quality_checker = DataQualityChecker(rules=self.quality_rules)
        quality_result = quality_checker.check_data_quality(
            update_result['data'],
            update_task
        )

        # 验证发现了质量问题
        assert quality_result['issues_found'] > 0, "应该发现数据质量问题"
        assert quality_result['overall_score'] < 0.8, "有问题的数据质量分数应该较低"

        # Step 3: 异常处理
        exception_handler = DataExceptionHandler()
        handled_result = exception_handler.handle_quality_issues(
            quality_result,
            update_result['data'],
            auto_fix=True
        )

        # 验证异常处理结果
        assert handled_result['fixed_issues'] > 0, "应该修复了一些问题"
        assert handled_result['remaining_issues'] >= 0, "可能还有未修复的问题"

        # Step 4: 生成异常报告
        report_generator = DataReportGenerator()
        exception_report = report_generator.generate_exception_report(
            quality_result,
            handled_result
        )

        # 验证异常报告
        assert exception_report is not None, "应该生成异常报告"
        assert 'exception_summary' in exception_report, "应该包含异常摘要"
        assert 'fix_actions' in exception_report, "应该包含修复操作记录"

    @pytest.mark.integration
    def test_incremental_vs_full_update(self):
        """测试增量更新与全量更新的对比"""
        if DataUpdater is None:
            pytest.skip("DataUpdater not implemented yet - TDD Red phase")

        updater = DataUpdater()

        # 全量更新
        full_update_task = updater.create_update_task(
            stocks=self.test_stocks,
            date_range=self.test_date_range,
            update_type='full'
        )

        full_result = updater.execute_update(
            full_update_task,
            test_data=self.valid_test_data
        )

        # 增量更新 (模拟只更新最近几天)
        recent_date_range = (
            date(2024, 1, 25),
            date(2024, 1, 31)
        )

        incremental_task = updater.create_update_task(
            stocks=self.test_stocks,
            date_range=recent_date_range,
            update_type='incremental'
        )

        # 创建增量数据
        incremental_data = self.valid_test_data[
            self.valid_test_data['date'] >= recent_date_range[0]
        ].copy()

        incremental_result = updater.execute_update(
            incremental_task,
            test_data=incremental_data
        )

        # 验证更新类型差异
        assert full_result['updated_records'] > incremental_result['updated_records'], \
            "全量更新的记录数应该大于增量更新"

        assert incremental_result['execution_time'] < full_result['execution_time'], \
            "增量更新的执行时间应该更短"

    @pytest.mark.integration
    def test_data_quality_rule_customization(self):
        """测试数据质量规则的自定义配置"""
        if DataQualityChecker is None:
            pytest.skip("DataQualityChecker not implemented yet - TDD Red phase")

        # 创建自定义规则
        custom_rules = [
            {
                'rule_name': 'custom_volume_check',
                'description': '自定义成交量检查',
                'rule_type': 'custom',
                'parameters': {
                    'min_volume': 100000,
                    'max_volume': 100000000,
                    'zero_volume_allowed': False
                },
                'severity': 'error'
            }
        ]

        quality_checker = DataQualityChecker(rules=custom_rules)

        # 创建测试数据 - 违反自定义规则
        test_data = pd.DataFrame([
            {
                'stock_code': 'sh600000',
                'date': date(2024, 1, 15),
                'open': 10.0,
                'high': 10.5,
                'low': 9.5,
                'close': 10.0,
                'volume': 50000,  # 低于最小值
                'amount': 500000.0
            }
        ])

        quality_result = quality_checker.check_data_quality(test_data, None)

        # 验证自定义规则被应用
        assert quality_result['issues_found'] > 0, "应该发现违反自定义规则的问题"
        assert any(
            issue['rule_name'] == 'custom_volume_check'
            for issue in quality_result['issues']
        ), "应该发现自定义成交量检查的问题"

    @pytest.mark.integration
    def test_exception_handling_strategies(self):
        """测试不同的异常处理策略"""
        if DataExceptionHandler is None:
            pytest.skip("DataExceptionHandler not implemented yet - TDD Red phase")

        exception_handler = DataExceptionHandler()

        # 策略1: 自动修复
        auto_fix_result = exception_handler.handle_quality_issues(
            {'issues': [{'type': 'missing_data', 'field': 'open', 'action': 'interpolate'}]},
            self.invalid_test_data,
            strategy='auto_fix'
        )

        assert auto_fix_result['strategy'] == 'auto_fix', "应该使用自动修复策略"

        # 策略2: 数据隔离
        quarantine_result = exception_handler.handle_quality_issues(
            {'issues': [{'type': 'price_anomaly', 'severity': 'high'}]},
            self.invalid_test_data,
            strategy='quarantine'
        )

        assert quarantine_result['strategy'] == 'quarantine', "应该使用隔离策略"

        # 策略3: 人工审核
        manual_review_result = exception_handler.handle_quality_issues(
            {'issues': [{'type': 'business_rule_violation', 'severity': 'critical'}]},
            self.invalid_test_data,
            strategy='manual_review'
        )

        assert manual_review_result['strategy'] == 'manual_review', "应该使用人工审核策略"

    @pytest.mark.integration
    def test_data_workflow_monitoring(self):
        """测试数据工作流程的监控功能"""
        if DataUpdater is None:
            pytest.skip("DataUpdater not implemented yet - TDD Red phase")

        updater = DataUpdater()

        # 启用监控
        update_task = updater.create_update_task(
            stocks=self.test_stocks,
            date_range=self.test_date_range,
            update_type='incremental',
            enable_monitoring=True
        )

        # 执行更新并监控
        with updater.monitor_execution(update_task) as monitor:
            update_result = updater.execute_update(
                update_task,
                test_data=self.valid_test_data
            )

        # 验证监控数据
        monitoring_data = monitor.get_monitoring_data()

        assert 'start_time' in monitoring_data, "应该记录开始时间"
        assert 'end_time' in monitoring_data, "应该记录结束时间"
        assert 'execution_time' in monitoring_data, "应该记录执行时间"
        assert 'memory_usage' in monitoring_data, "应该记录内存使用"
        assert 'cpu_usage' in monitoring_data, "应该记录CPU使用"

    @pytest.mark.integration
    def test_data_workflow_rollback(self):
        """测试数据工作流程的回滚功能"""
        if DataUpdater is None:
            pytest.skip("DataUpdater not implemented yet - TDD Red phase")

        updater = DataUpdater()

        # 创建更新任务
        update_task = updater.create_update_task(
            stocks=self.test_stocks,
            date_range=self.test_date_range,
            update_type='incremental',
            enable_rollback=True
        )

        # 执行更新
        update_result = updater.execute_update(
            update_task,
            test_data=self.valid_test_data
        )

        # 假设发现严重问题，需要回滚
        rollback_result = updater.rollback_update(update_task)

        # 验证回滚结果
        assert rollback_result['status'] == 'success', "回滚应该成功"
        assert rollback_result['rollback_records'] == update_result['updated_records'], \
            "回滚的记录数应该等于更新的记录数"

    @pytest.mark.integration
    def test_concurrent_data_operations(self):
        """测试并发数据操作的安全性"""
        if DataUpdater is None:
            pytest.skip("DataUpdater not implemented yet - TDD Red phase")

        import concurrent.futures

        updater = DataUpdater()

        def run_update_task(stock_subset):
            """执行单个更新任务"""
            task = updater.create_update_task(
                stocks=stock_subset,
                date_range=self.test_date_range,
                update_type='incremental'
            )

            subset_data = self.valid_test_data[
                self.valid_test_data['stock_code'].isin(stock_subset)
            ]

            return updater.execute_update(task, test_data=subset_data)

        # 并发执行多个更新任务
        stock_subsets = [
            ["sh600000", "sh600001"],
            ["sz000001", "sz000002"]
        ]

        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
            futures = [
                executor.submit(run_update_task, subset)
                for subset in stock_subsets
            ]

            results = [f.result() for f in concurrent.futures.as_completed(futures)]

        # 验证并发执行结果
        assert len(results) == 2, "两个并发任务都应该完成"
        for result in results:
            assert result['status'] == 'success', "每个任务都应该成功"

        # 验证数据一致性
        total_updated = sum(result['updated_records'] for result in results)
        expected_total = len(self.valid_test_data)
        assert total_updated == expected_total, "并发更新的总记录数应该正确"