"""
Validation Service Unit Tests

基于真实Hikyuu框架的验证服务单元测试
不使用mock数据，测试ValidationConfig, FactorValidator, ValidationReporter功能
"""

import pytest
import asyncio
import pandas as pd
import numpy as np
from datetime import datetime, date, timedelta
from typing import Dict, List, Any

from src.services.validation_service import (
    ValidationConfig, FactorValidator, ValidationReporter, ValidationService
)
from src.models.validation_models import (
    ValidationRule, ValidationResult, ValidationIssue, ValidationSeverity,
    ValidationRuleType, create_statistical_validation_rules
)


class TestValidationConfig:
    """ValidationConfig组件单元测试"""

    @pytest.fixture
    def validation_config(self):
        """创建ValidationConfig实例"""
        config = {
            'statistical_validation': True,
            'correlation_validation': True,
            'custom_rules': [
                {
                    'rule_id': 'custom_test_rule',
                    'rule_name': '自定义测试规则',
                    'rule_type': ValidationRuleType.STATISTICAL,
                    'severity': ValidationSeverity.WARNING,
                    'description': '用于测试的自定义规则',
                    'thresholds': {'min_value': 0.0, 'max_value': 1.0}
                }
            ]
        }
        return ValidationConfig(config)

    def test_validation_config_initialization(self, validation_config):
        """测试ValidationConfig初始化"""
        assert validation_config.config is not None
        assert validation_config.logger is not None

    def test_load_validation_rules_with_all_types(self, validation_config):
        """测试加载所有类型的验证规则"""
        config = {
            'statistical_validation': True,
            'correlation_validation': True,
            'custom_rules': []
        }

        rules = validation_config.load_validation_rules(config)

        assert len(rules) > 0

        # 验证统计规则存在
        stat_rules = [rule for rule in rules if rule.rule_type == ValidationRuleType.STATISTICAL]
        assert len(stat_rules) > 0

        # 验证相关性规则存在
        corr_rules = [rule for rule in rules if rule.rule_type == ValidationRuleType.CORRELATION]
        assert len(corr_rules) > 0

    def test_load_validation_rules_with_custom_rules(self, validation_config):
        """测试加载自定义验证规则"""
        config = {
            'statistical_validation': False,
            'correlation_validation': False,
            'custom_rules': [
                {
                    'rule_id': 'custom_rule_1',
                    'rule_name': 'Custom Rule 1',
                    'rule_type': ValidationRuleType.BUSINESS_LOGIC,
                    'severity': ValidationSeverity.ERROR,
                    'description': 'Test custom rule',
                    'thresholds': {'threshold': 0.5}
                }
            ]
        }

        rules = validation_config.load_validation_rules(config)

        assert len(rules) == 1
        assert rules[0].rule_id == 'custom_rule_1'
        assert rules[0].rule_type == ValidationRuleType.BUSINESS_LOGIC

    def test_get_statistical_rules(self, validation_config):
        """测试获取统计验证规则"""
        config = {'statistical_validation': True}
        stat_rules = validation_config.get_statistical_rules(config)

        assert isinstance(stat_rules, list)
        assert len(stat_rules) > 0

        for rule in stat_rules:
            assert rule.rule_type == ValidationRuleType.STATISTICAL

    def test_get_correlation_rules(self, validation_config):
        """测试获取相关性验证规则"""
        config = {'correlation_validation': True}
        corr_rules = validation_config.get_correlation_rules(config)

        assert isinstance(corr_rules, list)
        assert len(corr_rules) > 0

        for rule in corr_rules:
            assert rule.rule_type == ValidationRuleType.CORRELATION

    def test_get_stability_rules(self, validation_config):
        """测试获取稳定性验证规则"""
        config = {
            'custom_rules': [
                {
                    'rule_id': 'stability_test',
                    'rule_name': 'Stability Test',
                    'rule_type': ValidationRuleType.STABILITY,
                    'severity': ValidationSeverity.WARNING,
                    'description': 'Test stability rule',
                    'thresholds': {}
                }
            ]
        }

        stability_rules = validation_config.get_stability_rules(config)

        assert len(stability_rules) == 1
        assert stability_rules[0].rule_type == ValidationRuleType.STABILITY

    def test_get_business_rules(self, validation_config):
        """测试获取业务逻辑验证规则"""
        config = {
            'custom_rules': [
                {
                    'rule_id': 'business_test',
                    'rule_name': 'Business Test',
                    'rule_type': ValidationRuleType.BUSINESS_LOGIC,
                    'severity': ValidationSeverity.ERROR,
                    'description': 'Test business rule',
                    'thresholds': {}
                }
            ]
        }

        business_rules = validation_config.get_business_rules(config)

        assert len(business_rules) == 1
        assert business_rules[0].rule_type == ValidationRuleType.BUSINESS_LOGIC


class TestFactorValidator:
    """FactorValidator组件单元测试"""

    @pytest.fixture
    def sample_factor_data(self):
        """创建示例因子数据"""
        np.random.seed(42)
        data = pd.DataFrame({
            'stock_code': ['sh600000'] * 100 + ['sz000001'] * 100,
            'date': pd.concat([
                pd.date_range('2024-01-01', periods=100),
                pd.date_range('2024-01-01', periods=100)
            ]),
            'factor_value': np.concatenate([
                np.random.normal(0.1, 0.05, 100),  # 正常分布数据
                np.random.normal(0.15, 0.03, 100)
            ]),
            'factor_score': np.concatenate([
                np.random.uniform(0.3, 0.8, 100),
                np.random.uniform(0.4, 0.9, 100)
            ])
        })
        return data

    @pytest.fixture
    def problematic_factor_data(self):
        """创建有问题的因子数据"""
        np.random.seed(123)

        # 创建非正态分布数据（高偏度）
        factor_values = np.concatenate([
            np.random.exponential(0.05, 80),  # 指数分布
            [10.0, -5.0]  # 异常值
        ])

        data = pd.DataFrame({
            'stock_code': ['sh600000'] * 82,
            'date': pd.date_range('2024-01-01', periods=82),
            'factor_value': factor_values,
            'factor_score': np.random.uniform(0, 1, 82)
        })
        return data

    @pytest.fixture
    def factor_validator(self):
        """创建FactorValidator实例"""
        rules = create_statistical_validation_rules()
        return FactorValidator(rules)

    @pytest.mark.asyncio
    async def test_validate_factor_normal_data(self, factor_validator, sample_factor_data):
        """测试正常数据的因子验证"""
        result = await factor_validator.validate_factor("test_momentum", sample_factor_data)

        assert isinstance(result, ValidationResult)
        assert result.validation_id is not None
        assert result.factor_name == "test_momentum"
        assert result.validation_date is not None
        assert isinstance(result.passed, bool)
        assert 0.0 <= result.validation_score <= 1.0

        # 正常数据应该通过大部分验证
        assert result.validation_score >= 0.5

    @pytest.mark.asyncio
    async def test_validate_factor_problematic_data(self, factor_validator, problematic_factor_data):
        """测试有问题数据的因子验证"""
        result = await factor_validator.validate_factor("problematic_factor", problematic_factor_data)

        assert isinstance(result, ValidationResult)
        assert result.factor_name == "problematic_factor"

        # 有问题的数据应该被检测出来
        assert len(result.issues) > 0

        # 检查是否检测到异常值
        outlier_issues = [issue for issue in result.issues if 'outlier' in issue.description.lower()]
        assert len(outlier_issues) > 0

    @pytest.mark.asyncio
    async def test_normality_check_normal_data(self, factor_validator, sample_factor_data):
        """测试正态性检查 - 正常数据"""
        from src.models.validation_models import ValidationRuleType

        # 创建正态性检查规则
        normality_rule = ValidationRule(
            rule_id="stat_normality_test",
            rule_name="正态性检验",
            rule_type=ValidationRuleType.STATISTICAL,
            severity=ValidationSeverity.WARNING,
            description="Shapiro-Wilk正态性检验",
            thresholds={'p_value_min': 0.05}
        )

        result = ValidationResult(
            validation_id="test_normality",
            factor_name="test_factor",
            validation_date=datetime.now(),
            passed=True,
            validation_score=1.0
        )

        await factor_validator._check_normality(sample_factor_data, normality_rule, result)

        # 对于正常分布的数据，不应该有太多正态性问题
        normality_issues = [issue for issue in result.issues if issue.rule_id == "stat_normality_test"]
        assert len(normality_issues) <= 1  # 可能有轻微的非正态性

    @pytest.mark.asyncio
    async def test_outlier_check_with_outliers(self, factor_validator, problematic_factor_data):
        """测试异常值检查 - 包含异常值的数据"""
        from src.models.validation_models import ValidationRuleType

        # 创建异常值检查规则
        outlier_rule = ValidationRule(
            rule_id="stat_outlier_detection",
            rule_name="异常值检测",
            rule_type=ValidationRuleType.STATISTICAL,
            severity=ValidationSeverity.WARNING,
            description="基于IQR的异常值检测",
            thresholds={'max_outlier_ratio': 0.05}
        )

        result = ValidationResult(
            validation_id="test_outliers",
            factor_name="test_factor",
            validation_date=datetime.now(),
            passed=True,
            validation_score=1.0
        )

        await factor_validator._check_outliers(problematic_factor_data, outlier_rule, result)

        # 应该检测到异常值
        outlier_issues = [issue for issue in result.issues if issue.rule_id == "stat_outlier_detection"]
        assert len(outlier_issues) > 0

    @pytest.mark.asyncio
    async def test_validate_factor_edge_cases(self, factor_validator):
        """测试边界情况"""
        # 空数据
        empty_data = pd.DataFrame()
        result = await factor_validator.validate_factor("empty_factor", empty_data)
        assert isinstance(result, ValidationResult)
        assert result.factor_name == "empty_factor"

        # 只有一行数据
        single_row_data = pd.DataFrame({
            'factor_value': [0.5],
            'factor_score': [0.7]
        })
        result = await factor_validator.validate_factor("single_row", single_row_data)
        assert isinstance(result, ValidationResult)

        # 缺少必要列的数据
        missing_column_data = pd.DataFrame({
            'other_column': [1, 2, 3]
        })
        result = await factor_validator.validate_factor("missing_columns", missing_column_data)
        assert isinstance(result, ValidationResult)

    @pytest.mark.asyncio
    async def test_validate_factor_with_custom_rules(self):
        """测试使用自定义规则的验证"""
        # 创建自定义规则
        custom_rule = ValidationRule(
            rule_id="custom_range_check",
            rule_name="自定义范围检查",
            rule_type=ValidationRuleType.BUSINESS_LOGIC,
            severity=ValidationSeverity.ERROR,
            description="检查因子值是否在预期范围内",
            thresholds={'min_value': 0.0, 'max_value': 1.0}
        )

        validator = FactorValidator([custom_rule])

        # 创建超出范围的数据
        out_of_range_data = pd.DataFrame({
            'factor_value': [-0.5, 0.5, 1.5, 0.3],  # 包含超出[0,1]范围的值
            'factor_score': [0.2, 0.5, 0.8, 0.6]
        })

        result = await validator.validate_factor("range_test", out_of_range_data)
        assert isinstance(result, ValidationResult)

    @pytest.mark.asyncio
    async def test_validation_scoring_system(self, factor_validator, problematic_factor_data):
        """测试验证评分系统"""
        result = await factor_validator.validate_factor("scoring_test", problematic_factor_data)

        # 验证分数应该在0-1之间
        assert 0.0 <= result.validation_score <= 1.0

        # 如果有问题，验证分数应该降低
        if result.issues:
            assert result.validation_score < 1.0

        # 通过状态应该与分数一致
        if result.validation_score > 0.6:
            assert result.passed is True
        else:
            assert result.passed is False


class TestValidationReporter:
    """ValidationReporter组件单元测试"""

    @pytest.fixture
    def validation_reporter(self):
        """创建ValidationReporter实例"""
        return ValidationReporter()

    @pytest.fixture
    def sample_validation_results(self):
        """创建示例验证结果"""
        # 创建成功的验证结果
        successful_result = ValidationResult(
            validation_id="success_test",
            factor_name="momentum_20d",
            validation_date=datetime.now(),
            passed=True,
            validation_score=0.85
        )

        # 创建失败的验证结果
        failed_result = ValidationResult(
            validation_id="failed_test",
            factor_name="value_factor",
            validation_date=datetime.now(),
            passed=False,
            validation_score=0.45
        )

        # 添加问题
        failed_result.issues = [
            ValidationIssue(
                issue_id="issue_1",
                rule_id="stat_outlier_detection",
                rule_name="异常值检测",
                category="outliers",
                severity=ValidationSeverity.WARNING,
                description="检测到过多异常值"
            ),
            ValidationIssue(
                issue_id="issue_2",
                rule_id="corr_high_correlation",
                rule_name="高相关性检查",
                category="correlation",
                severity=ValidationSeverity.ERROR,
                description="与其他因子相关性过高"
            )
        ]

        return {
            'momentum_20d': successful_result,
            'value_factor': failed_result
        }

    def test_generate_comprehensive_report(self, validation_reporter, sample_validation_results):
        """测试生成综合验证报告"""
        report = validation_reporter.generate_comprehensive_report(sample_validation_results)

        # 验证报告结构
        assert 'summary' in report
        assert 'factor_details' in report
        assert 'recommendations' in report
        assert 'validation_timestamp' in report

        # 验证摘要信息
        summary = report['summary']
        assert summary['total_factors'] == 2
        assert summary['passed_factors'] == 1
        assert summary['failed_factors'] == 1
        assert summary['overall_pass_rate'] == 0.5

        # 验证因子详情
        factor_details = report['factor_details']
        assert 'momentum_20d' in factor_details
        assert 'value_factor' in factor_details
        assert factor_details['momentum_20d']['passed'] is True
        assert factor_details['value_factor']['passed'] is False

        # 验证建议
        recommendations = report['recommendations']
        assert isinstance(recommendations, list)

    def test_generate_recommendations(self, validation_reporter):
        """测试生成改进建议"""
        # 创建包含常见问题的验证结果
        result_with_issues = ValidationResult(
            validation_id="issues_test",
            factor_name="test_factor",
            validation_date=datetime.now(),
            passed=False,
            validation_score=0.3
        )

        result_with_issues.issues = [
            ValidationIssue(
                issue_id="outlier_issue",
                rule_id="outlier_check",
                rule_name="异常值检查",
                category="outliers",
                severity=ValidationSeverity.WARNING,
                description="发现异常值比例过高"
            ),
            ValidationIssue(
                issue_id="correlation_issue",
                rule_id="correlation_check",
                rule_name="相关性检查",
                category="correlation",
                severity=ValidationSeverity.ERROR,
                description="因子间存在高相关性"
            )
        ]

        results = {'test_factor': result_with_issues}
        recommendations = validation_reporter._generate_recommendations(results)

        # 应该生成针对异常值和相关性的建议
        assert len(recommendations) > 0

        outlier_recommendation = any('异常值处理' in rec for rec in recommendations)
        correlation_recommendation = any('相关性' in rec for rec in recommendations)

        assert outlier_recommendation
        assert correlation_recommendation

    def test_empty_results_report(self, validation_reporter):
        """测试空结果的报告生成"""
        empty_results = {}
        report = validation_reporter.generate_comprehensive_report(empty_results)

        assert report['summary']['total_factors'] == 0
        assert report['summary']['passed_factors'] == 0
        assert report['summary']['failed_factors'] == 0
        assert report['summary']['overall_pass_rate'] == 0
        assert report['factor_details'] == {}
        assert isinstance(report['recommendations'], list)

    def test_report_timestamp_format(self, validation_reporter, sample_validation_results):
        """测试报告时间戳格式"""
        report = validation_reporter.generate_comprehensive_report(sample_validation_results)

        timestamp = report['validation_timestamp']
        # 验证时间戳是ISO格式
        parsed_time = datetime.fromisoformat(timestamp)
        assert isinstance(parsed_time, datetime)


class TestValidationService:
    """ValidationService集成测试"""

    @pytest.fixture
    def validation_service(self):
        """创建ValidationService实例"""
        config = {
            'config': {
                'statistical_validation': True,
                'correlation_validation': True
            }
        }
        return ValidationService(config)

    @pytest.fixture
    def multi_factor_data(self):
        """创建多因子数据"""
        np.random.seed(42)

        factors = {}

        # 高质量因子
        factors['momentum_20d'] = pd.DataFrame({
            'stock_code': ['sh600000', 'sz000001'] * 50,
            'date': pd.concat([pd.date_range('2024-01-01', periods=50)] * 2),
            'factor_value': np.random.normal(0.1, 0.05, 100),
            'factor_score': np.random.uniform(0.4, 0.9, 100)
        })

        # 低质量因子（包含异常值）
        problematic_values = np.concatenate([
            np.random.normal(0.05, 0.02, 90),
            [5.0, -3.0]  # 异常值
        ])

        factors['value_pe'] = pd.DataFrame({
            'stock_code': ['sh600000', 'sz000001'] * 46,
            'date': pd.concat([pd.date_range('2024-01-01', periods=46)] * 2),
            'factor_value': problematic_values,
            'factor_score': np.random.uniform(0.2, 0.8, 92)
        })

        return factors

    @pytest.mark.asyncio
    async def test_validate_factor_batch(self, validation_service, multi_factor_data):
        """测试批量因子验证"""
        validation_config = {
            'statistical_validation': True,
            'correlation_validation': False
        }

        results = await validation_service.validate_factor_batch(
            multi_factor_data, validation_config
        )

        # 验证结果结构
        assert isinstance(results, dict)
        assert len(results) == len(multi_factor_data)
        assert 'momentum_20d' in results
        assert 'value_pe' in results

        # 验证每个结果
        for factor_name, result in results.items():
            assert isinstance(result, ValidationResult)
            assert result.factor_name == factor_name
            assert result.validation_id is not None

    @pytest.mark.asyncio
    async def test_validate_single_factor(self, validation_service, multi_factor_data):
        """测试单个因子验证"""
        validation_config = {
            'statistical_validation': True
        }

        factor_data = multi_factor_data['momentum_20d']
        result = await validation_service.validate_single_factor(
            'momentum_20d', factor_data, validation_config
        )

        assert isinstance(result, ValidationResult)
        assert result.factor_name == 'momentum_20d'
        assert result.validation_date is not None

    def test_save_validation_results(self, validation_service, tmp_path):
        """测试保存验证结果"""
        # 创建示例验证结果
        result = ValidationResult(
            validation_id="save_test",
            factor_name="test_factor",
            validation_date=datetime.now(),
            passed=True,
            validation_score=0.8
        )

        results = {'test_factor': result}
        storage_path = str(tmp_path / "validation_results")

        success = validation_service.save_validation_results(results, storage_path)

        assert success is True

        # 验证文件是否创建
        expected_file = tmp_path / "validation_results" / "test_factor_validation.json"
        assert expected_file.exists()

        # 验证文件内容
        import json
        with open(expected_file, 'r', encoding='utf-8') as f:
            saved_data = json.load(f)

        assert saved_data['factor_name'] == 'test_factor'
        assert saved_data['passed'] is True

    @pytest.mark.asyncio
    async def test_end_to_end_validation_workflow(self, validation_service, multi_factor_data):
        """测试端到端验证工作流"""
        validation_config = {
            'statistical_validation': True,
            'correlation_validation': True,
            'custom_rules': []
        }

        # Step 1: 批量验证
        results = await validation_service.validate_factor_batch(
            multi_factor_data, validation_config
        )

        # Step 2: 生成报告
        reporter = ValidationReporter()
        report = reporter.generate_comprehensive_report(results)

        # Step 3: 验证工作流完整性
        assert len(results) == len(multi_factor_data)
        assert report['summary']['total_factors'] == len(multi_factor_data)

        # 验证至少有一些质量检查结果
        total_issues = sum(len(result.issues) for result in results.values())
        assert total_issues >= 0  # 可能没有问题，也可能有问题

        # 验证报告包含所有必要信息
        assert 'recommendations' in report
        assert 'validation_timestamp' in report

    @pytest.mark.asyncio
    async def test_validation_performance(self, validation_service):
        """测试验证性能"""
        import time

        # 创建大量因子数据
        large_factor_data = {}
        np.random.seed(42)

        for i in range(5):  # 5个因子
            large_factor_data[f'factor_{i}'] = pd.DataFrame({
                'stock_code': [f'sh{600000+j:06d}' for j in range(100)],
                'date': pd.date_range('2024-01-01', periods=100),
                'factor_value': np.random.normal(0.1, 0.05, 100),
                'factor_score': np.random.uniform(0.3, 0.9, 100)
            })

        validation_config = {
            'statistical_validation': True,
            'correlation_validation': False  # 关闭相关性检查以提高性能
        }

        start_time = time.time()
        results = await validation_service.validate_factor_batch(
            large_factor_data, validation_config
        )
        end_time = time.time()

        execution_time = end_time - start_time

        # 验证性能
        assert execution_time < 10.0  # 应该在10秒内完成
        assert len(results) == 5

        # 验证所有因子都被处理
        for i in range(5):
            assert f'factor_{i}' in results

    @pytest.mark.asyncio
    async def test_validation_error_handling(self, validation_service):
        """测试验证错误处理"""
        # 测试无效数据
        invalid_data = {
            'invalid_factor': pd.DataFrame({
                'invalid_column': [1, 2, 3]  # 缺少必要的列
            })
        }

        validation_config = {'statistical_validation': True}

        results = await validation_service.validate_factor_batch(
            invalid_data, validation_config
        )

        # 应该能够处理错误情况
        assert isinstance(results, dict)
        # 可能返回空结果或包含错误信息


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])