"""
T027: 验证配置→因子验证→报告生成 集成测试

测试完整的因子验证工作流程：
1. 配置验证规则和标准
2. 执行因子统计验证和业务验证
3. 生成详细的验证报告
4. 确保验证结果的准确性和可追溯性

这是一个TDD Red-Green-Refactor循环的第一步 - 先创建失败的测试
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Any, Optional
from unittest.mock import Mock, patch

# 导入待实现的模块 (这些导入在Red阶段会失败)
try:
    from src.lib.validation_config import ValidationConfig
    from src.lib.factor_validator import FactorValidator
    from src.lib.validation_reporter import ValidationReporter
    from src.services.validation_service import ValidationService
    from src.models.validation_result import ValidationResult
    from src.models.validation_rule import ValidationRule
except ImportError:
    # TDD Red阶段 - 这些模块还不存在
    ValidationConfig = None
    FactorValidator = None
    ValidationReporter = None
    ValidationService = None
    ValidationResult = None
    ValidationRule = None


@pytest.mark.integration
@pytest.mark.validation
@pytest.mark.requires_hikyuu
class TestValidationWorkflow:
    """验证配置→因子验证→报告生成 集成测试"""

    def setup_method(self):
        """设置测试环境"""
        self.test_factors = self._create_test_factors()
        self.validation_config = self._create_validation_config()
        self.expected_validation_rules = [
            "statistical_distribution",
            "correlation_analysis",
            "stability_check",
            "outlier_detection",
            "business_logic_validation"
        ]

    def _create_test_factors(self) -> Dict[str, pd.DataFrame]:
        """创建测试用因子数据"""
        np.random.seed(42)
        dates = pd.date_range('2024-01-01', periods=252, freq='D')
        stocks = [f"sh{600000 + i:06d}" for i in range(50)]

        factors = {}

        # 正常因子数据
        normal_factor_data = []
        for stock in stocks:
            for date in dates:
                normal_factor_data.append({
                    'stock_code': stock,
                    'date': date,
                    'factor_value': np.random.normal(0, 1),  # 正态分布
                    'factor_name': 'momentum_20d'
                })
        factors['normal_factor'] = pd.DataFrame(normal_factor_data)

        # 异常因子数据 (包含极值)
        anomaly_factor_data = []
        for stock in stocks:
            for date in dates:
                # 10%概率产生极值
                if np.random.random() < 0.1:
                    value = np.random.choice([-999, 999])  # 极值
                else:
                    value = np.random.normal(0, 1)

                anomaly_factor_data.append({
                    'stock_code': stock,
                    'date': date,
                    'factor_value': value,
                    'factor_name': 'anomaly_factor'
                })
        factors['anomaly_factor'] = pd.DataFrame(anomaly_factor_data)

        # 高相关性因子数据
        base_values = np.random.normal(0, 1, len(stocks) * len(dates))
        correlated_factor_data = []
        idx = 0
        for stock in stocks:
            for date in dates:
                # 与基准高度相关 (0.95相关性)
                value = 0.95 * base_values[idx] + 0.05 * np.random.normal(0, 1)
                correlated_factor_data.append({
                    'stock_code': stock,
                    'date': date,
                    'factor_value': value,
                    'factor_name': 'correlated_factor'
                })
                idx += 1
        factors['correlated_factor'] = pd.DataFrame(correlated_factor_data)

        return factors

    def _create_validation_config(self) -> Dict[str, Any]:
        """创建验证配置"""
        return {
            'statistical_validation': {
                'distribution_tests': ['normality', 'skewness', 'kurtosis'],
                'outlier_threshold': 3.0,
                'missing_data_threshold': 0.05,
                'zero_variance_threshold': 1e-6
            },
            'correlation_validation': {
                'max_correlation': 0.8,
                'correlation_window': 252,
                'correlation_methods': ['pearson', 'spearman']
            },
            'stability_validation': {
                'rolling_window': 63,  # 季度
                'stability_threshold': 0.7,
                'regime_change_detection': True
            },
            'business_validation': {
                'factor_range_checks': True,
                'economic_intuition_tests': True,
                'sector_consistency_checks': True
            },
            'performance_validation': {
                'ic_threshold': 0.02,
                'ic_ir_threshold': 0.5,
                'turnover_threshold': 0.3
            }
        }

    @pytest.mark.integration
    def test_complete_validation_workflow(self):
        """测试完整的验证工作流程"""
        # 这个测试在Red阶段应该失败，因为相关类还没有实现
        if ValidationConfig is None:
            pytest.skip("ValidationConfig not implemented yet - TDD Red phase")

        # Step 1: 配置验证规则
        config = ValidationConfig()
        validation_rules = config.load_validation_rules(self.validation_config)

        # 验证配置加载结果
        assert validation_rules is not None, "验证规则不能为空"
        assert len(validation_rules) >= len(self.expected_validation_rules), \
            f"验证规则数量不足，期望至少{len(self.expected_validation_rules)}个"

        for rule_name in self.expected_validation_rules:
            assert any(rule.rule_type == rule_name for rule in validation_rules), \
                f"缺少必需的验证规则: {rule_name}"

        # Step 2: 执行因子验证
        validator = FactorValidator(validation_rules)
        validation_results = {}

        for factor_name, factor_data in self.test_factors.items():
            result = validator.validate_factor(factor_name, factor_data)
            validation_results[factor_name] = result

            # 验证结果格式
            assert result is not None, f"因子{factor_name}验证结果不能为空"
            assert hasattr(result, 'factor_name'), "验证结果必须包含因子名称"
            assert hasattr(result, 'validation_score'), "验证结果必须包含验证分数"
            assert hasattr(result, 'rule_results'), "验证结果必须包含规则结果"
            assert hasattr(result, 'issues'), "验证结果必须包含问题列表"

        # Step 3: 生成验证报告
        reporter = ValidationReporter()
        report = reporter.generate_comprehensive_report(validation_results)

        # 验证报告内容
        assert report is not None, "验证报告不能为空"
        assert 'summary' in report, "报告必须包含摘要"
        assert 'factor_details' in report, "报告必须包含因子详情"
        assert 'recommendations' in report, "报告必须包含建议"
        assert 'validation_timestamp' in report, "报告必须包含验证时间戳"

        # 验证摘要信息
        summary = report['summary']
        assert 'total_factors' in summary, "摘要必须包含因子总数"
        assert 'passed_factors' in summary, "摘要必须包含通过验证的因子数"
        assert 'failed_factors' in summary, "摘要必须包含未通过验证的因子数"
        assert summary['total_factors'] == len(self.test_factors), "因子总数应该正确"

    @pytest.mark.integration
    def test_statistical_validation_accuracy(self):
        """测试统计验证的准确性"""
        if FactorValidator is None:
            pytest.skip("FactorValidator not implemented yet - TDD Red phase")

        config = ValidationConfig()
        statistical_rules = config.get_statistical_rules(self.validation_config)
        validator = FactorValidator(statistical_rules)

        # 测试正常因子（应该通过验证）
        normal_result = validator.validate_factor(
            'normal_factor',
            self.test_factors['normal_factor']
        )

        assert normal_result.validation_score > 0.7, \
            "正常因子的验证分数应该较高"

        statistical_issues = [issue for issue in normal_result.issues
                            if issue.category == 'statistical']
        assert len(statistical_issues) == 0, \
            "正常因子不应该有统计问题"

        # 测试异常因子（应该检测出问题）
        anomaly_result = validator.validate_factor(
            'anomaly_factor',
            self.test_factors['anomaly_factor']
        )

        assert anomaly_result.validation_score < 0.5, \
            "异常因子的验证分数应该较低"

        outlier_issues = [issue for issue in anomaly_result.issues
                         if 'outlier' in issue.description.lower()]
        assert len(outlier_issues) > 0, \
            "异常因子应该检测出极值问题"

    @pytest.mark.integration
    def test_correlation_validation_detection(self):
        """测试相关性验证检测"""
        if FactorValidator is None:
            pytest.skip("FactorValidator not implemented yet - TDD Red phase")

        config = ValidationConfig()
        correlation_rules = config.get_correlation_rules(self.validation_config)
        validator = FactorValidator(correlation_rules)

        # 同时验证正常因子和高相关因子
        all_factors = pd.concat([
            self.test_factors['normal_factor'],
            self.test_factors['correlated_factor']
        ])

        correlation_result = validator.validate_factor_correlations(all_factors)

        # 验证相关性检测结果
        assert correlation_result is not None, "相关性验证结果不能为空"

        high_correlation_issues = [
            issue for issue in correlation_result.issues
            if 'correlation' in issue.description.lower()
        ]
        assert len(high_correlation_issues) > 0, \
            "应该检测到高相关性问题"

    @pytest.mark.integration
    def test_stability_validation_over_time(self):
        """测试时间序列稳定性验证"""
        if FactorValidator is None:
            pytest.skip("FactorValidator not implemented yet - TDD Red phase")

        config = ValidationConfig()
        stability_rules = config.get_stability_rules(self.validation_config)
        validator = FactorValidator(stability_rules)

        # 创建不稳定的因子数据（前半年和后半年分布不同）
        unstable_factor_data = []
        dates = pd.date_range('2024-01-01', periods=252, freq='D')
        stocks = [f"sh{600000 + i:06d}" for i in range(20)]

        for i, date in enumerate(dates):
            for stock in stocks:
                # 前半年: 均值0，标准差1
                # 后半年: 均值2，标准差0.5 (分布发生变化)
                if i < 126:
                    value = np.random.normal(0, 1)
                else:
                    value = np.random.normal(2, 0.5)

                unstable_factor_data.append({
                    'stock_code': stock,
                    'date': date,
                    'factor_value': value,
                    'factor_name': 'unstable_factor'
                })

        unstable_factor_df = pd.DataFrame(unstable_factor_data)

        stability_result = validator.validate_factor_stability(
            'unstable_factor',
            unstable_factor_df
        )

        # 验证稳定性检测结果
        assert stability_result.validation_score < 0.6, \
            "不稳定因子的稳定性分数应该较低"

        stability_issues = [
            issue for issue in stability_result.issues
            if 'stability' in issue.description.lower() or 'regime' in issue.description.lower()
        ]
        assert len(stability_issues) > 0, \
            "应该检测到稳定性问题"

    @pytest.mark.integration
    def test_business_logic_validation(self):
        """测试业务逻辑验证"""
        if FactorValidator is None:
            pytest.skip("FactorValidator not implemented yet - TDD Red phase")

        config = ValidationConfig()
        business_rules = config.get_business_rules(self.validation_config)
        validator = FactorValidator(business_rules)

        # 创建违反业务逻辑的因子数据
        business_violation_data = []
        dates = pd.date_range('2024-01-01', periods=50, freq='D')
        stocks = [f"sh{600000 + i:06d}" for i in range(10)]

        for stock in stocks:
            for date in dates:
                # 故意创建不合理的因子值 (如PE比率为负数)
                business_violation_data.append({
                    'stock_code': stock,
                    'date': date,
                    'factor_value': -np.random.uniform(1, 100),  # PE不应该为负
                    'factor_name': 'pe_ratio'
                })

        business_violation_df = pd.DataFrame(business_violation_data)

        business_result = validator.validate_business_logic(
            'pe_ratio',
            business_violation_df
        )

        # 验证业务逻辑检测结果
        assert business_result.validation_score < 0.3, \
            "违反业务逻辑的因子分数应该很低"

        business_issues = [
            issue for issue in business_result.issues
            if 'business' in issue.description.lower()
        ]
        assert len(business_issues) > 0, \
            "应该检测到业务逻辑问题"

    @pytest.mark.integration
    def test_validation_report_generation(self):
        """测试验证报告生成功能"""
        if ValidationReporter is None:
            pytest.skip("ValidationReporter not implemented yet - TDD Red phase")

        # 模拟验证结果
        mock_validation_results = {
            'factor_a': Mock(
                factor_name='factor_a',
                validation_score=0.85,
                passed=True,
                issues=[],
                rule_results={'statistical': 0.9, 'correlation': 0.8}
            ),
            'factor_b': Mock(
                factor_name='factor_b',
                validation_score=0.45,
                passed=False,
                issues=[Mock(category='statistical', description='High outlier rate')],
                rule_results={'statistical': 0.3, 'correlation': 0.6}
            )
        }

        reporter = ValidationReporter()

        # 测试生成HTML报告
        html_report = reporter.generate_html_report(mock_validation_results)
        assert html_report is not None, "HTML报告不能为空"
        assert '<html>' in html_report, "应该生成有效的HTML"
        assert 'factor_a' in html_report, "报告应该包含因子A信息"
        assert 'factor_b' in html_report, "报告应该包含因子B信息"

        # 测试生成PDF报告
        pdf_path = Path("/tmp/test_validation_report.pdf")
        reporter.generate_pdf_report(mock_validation_results, pdf_path)
        assert pdf_path.exists(), "PDF报告文件应该被创建"

        # 测试生成Excel报告
        excel_path = Path("/tmp/test_validation_report.xlsx")
        reporter.generate_excel_report(mock_validation_results, excel_path)
        assert excel_path.exists(), "Excel报告文件应该被创建"

        # 清理临时文件
        for path in [pdf_path, excel_path]:
            if path.exists():
                path.unlink()

    @pytest.mark.integration
    def test_validation_service_integration(self):
        """测试验证服务集成"""
        if ValidationService is None:
            pytest.skip("ValidationService not implemented yet - TDD Red phase")

        service = ValidationService()

        # 测试批量验证
        batch_results = service.validate_factor_batch(
            self.test_factors,
            self.validation_config
        )

        assert batch_results is not None, "批量验证结果不能为空"
        assert len(batch_results) == len(self.test_factors), \
            "验证结果数量应该与输入因子数量一致"

        for factor_name in self.test_factors.keys():
            assert factor_name in batch_results, \
                f"验证结果应该包含因子{factor_name}"

        # 测试验证结果持久化
        persistence_result = service.save_validation_results(
            batch_results,
            storage_path="/tmp/validation_results"
        )

        assert persistence_result is True, "验证结果保存应该成功"

    @pytest.mark.integration
    def test_validation_performance_benchmarks(self):
        """测试验证性能基准"""
        if ValidationService is None:
            pytest.skip("ValidationService not implemented yet - TDD Red phase")

        service = ValidationService()

        # 创建大规模测试数据
        large_factor_data = []
        dates = pd.date_range('2020-01-01', '2024-01-01', freq='D')  # 4年数据
        stocks = [f"sh{600000 + i:06d}" for i in range(1000)]  # 1000只股票

        start_time = datetime.now()

        # 模拟大规模验证 (实际实现中应该优化性能)
        for i in range(min(10, len(stocks))):  # 仅测试前10只股票以控制测试时间
            stock = stocks[i]
            for date in dates[:100]:  # 仅测试前100天
                large_factor_data.append({
                    'stock_code': stock,
                    'date': date,
                    'factor_value': np.random.normal(0, 1),
                    'factor_name': 'large_test_factor'
                })

        large_factor_df = pd.DataFrame(large_factor_data)
        validation_result = service.validate_single_factor(
            'large_test_factor',
            large_factor_df,
            self.validation_config
        )

        end_time = datetime.now()
        execution_time = (end_time - start_time).total_seconds()

        # 验证性能要求 (根据实际需求调整)
        assert execution_time < 30, \
            f"大规模验证时间过长: {execution_time}秒，应少于30秒"
        assert validation_result is not None, "大规模验证应该返回结果"

    @pytest.mark.integration
    def test_validation_error_handling(self):
        """测试验证过程中的错误处理"""
        if ValidationService is None:
            pytest.skip("ValidationService not implemented yet - TDD Red phase")

        service = ValidationService()

        # 测试空数据处理
        empty_df = pd.DataFrame()
        result = service.validate_single_factor(
            'empty_factor',
            empty_df,
            self.validation_config
        )

        assert result is not None, "空数据验证应该返回结果"
        assert result.passed is False, "空数据验证应该失败"
        assert any('empty' in issue.description.lower() for issue in result.issues), \
            "应该报告空数据问题"

        # 测试缺失列处理
        invalid_df = pd.DataFrame({
            'wrong_column': [1, 2, 3],
            'another_wrong_column': [4, 5, 6]
        })

        result = service.validate_single_factor(
            'invalid_factor',
            invalid_df,
            self.validation_config
        )

        assert result is not None, "无效数据验证应该返回结果"
        assert result.passed is False, "无效数据验证应该失败"

    @pytest.mark.integration
    def test_validation_config_flexibility(self):
        """测试验证配置的灵活性"""
        if ValidationConfig is None:
            pytest.skip("ValidationConfig not implemented yet - TDD Red phase")

        config = ValidationConfig()

        # 测试自定义验证配置
        custom_config = {
            'statistical_validation': {
                'outlier_threshold': 2.5,  # 更严格的极值检测
                'distribution_tests': ['normality']  # 仅检测正态性
            },
            'correlation_validation': {
                'max_correlation': 0.9  # 更宽松的相关性限制
            }
        }

        custom_rules = config.load_validation_rules(custom_config)
        default_rules = config.load_validation_rules(self.validation_config)

        # 验证自定义配置生效
        assert len(custom_rules) != len(default_rules), \
            "自定义配置应该产生不同的规则集"

        # 测试配置覆盖
        override_config = config.override_config(
            base_config=self.validation_config,
            overrides={'statistical_validation.outlier_threshold': 4.0}
        )

        assert override_config['statistical_validation']['outlier_threshold'] == 4.0, \
            "配置覆盖应该生效"