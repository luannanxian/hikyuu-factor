"""
Validation Workflow Integration Test (T027)
集成测试: 验证配置→因子验证→报告生成
"""

import asyncio
import pytest
import requests
import time
from typing import Dict, Any, List

from tests.integration.test_environment import IntegrationTestBase, integration_test_environment


class TestValidationWorkflowIntegration:
    """验证工作流集成测试"""

    @pytest.mark.asyncio
    async def test_complete_validation_workflow(self):
        """T027: 完整因子验证工作流"""
        async with integration_test_environment() as agent_manager:
            test_base = IntegrationTestBase(agent_manager)

            # Step 1: 准备测试因子
            test_factor_id = f"validation_test_{int(time.time())}"

            factor_definition = {
                "factor_id": test_factor_id,
                "name": "验证测试因子",
                "description": "用于验证工作流测试的动量因子",
                "category": "momentum",
                "formula": "CLOSE / REF(CLOSE, 20) - 1",
                "hikyuu_formula": {
                    "expression": "CLOSE / REF(CLOSE, 20) - 1",
                    "parameters": {"lookback_period": 20}
                },
                "version": "1.0.0"
            }

            # 注册因子
            register_response = requests.post(
                test_base.get_api_url("factor", "/api/v1/factors"),
                json=factor_definition
            )
            assert register_response.status_code == 201
            print(f"Test factor registered: {test_factor_id}")

            # 计算因子数据
            calculation_request = {
                "factor_id": test_factor_id,
                "stock_universe": ["sh000001", "sh000002", "sz000001", "sz000002"],
                "date_range": {
                    "start_date": "2010-01-01",
                    "end_date": "2023-12-31"
                },
                "calculation_mode": "full"
            }

            calc_response = requests.post(
                test_base.get_api_url("factor", f"/api/v1/factors/{test_factor_id}/calculate"),
                json=calculation_request
            )
            assert calc_response.status_code == 202

            calc_task_id = calc_response.json()["task_id"]
            calc_result = await test_base.wait_for_task_completion("factor", calc_task_id, timeout=600)
            assert calc_result["status"] == "completed"
            print("Factor calculation completed for validation")

            # Step 2: 配置验证参数
            validation_config = {
                "factor_id": test_factor_id,
                "validation_config": {
                    "train_start": "2010-01-01",
                    "train_end": "2016-12-31",
                    "test_start": "2017-01-01",
                    "test_end": "2020-12-31",
                    "validation_start": "2021-01-01",
                    "validation_end": "2023-12-31"
                },
                "validation_methods": [
                    "ic_analysis",
                    "layered_returns",
                    "turnover_analysis",
                    "risk_analysis"
                ],
                "benchmark": "000300.SH",  # 沪深300
                "universe": "custom",
                "stock_universe": ["sh000001", "sh000002", "sz000001", "sz000002"],
                "frequency": "daily",
                "layers": 5,
                "long_short": True,
                "neutralization": ["industry"],
                "rebalance_frequency": "monthly"
            }

            print("Validation configuration prepared")

            # Step 3: 启动因子验证
            validation_response = requests.post(
                test_base.get_api_url("validation", "/api/v1/validation/factor"),
                json=validation_config
            )
            assert validation_response.status_code == 202

            validation_data = validation_response.json()
            validation_id = validation_data["validation_id"]
            assert validation_data["status"] == "running"

            print(f"Factor validation started: {validation_id}")

            # Step 4: 监控验证进度
            validation_result = await test_base.wait_for_task_completion("validation", validation_id, timeout=1200)  # 20分钟
            assert validation_result["status"] == "completed"

            print("Factor validation completed")

            # Step 5: 获取验证报告
            report_response = requests.get(
                test_base.get_api_url("validation", f"/api/v1/validation/report/{validation_id}")
            )
            assert report_response.status_code == 200

            validation_report = report_response.json()
            assert validation_report["validation_id"] == validation_id
            assert validation_report["status"] == "completed"
            assert "report" in validation_report

            report = validation_report["report"]

            # Step 6: 验证报告内容完整性
            # 验证综合评分
            assert "summary" in report
            summary = report["summary"]
            assert "overall_score" in summary
            assert "risk_level" in summary
            assert "recommendation" in summary
            assert 0 <= summary["overall_score"] <= 100

            print(f"Validation summary - Score: {summary['overall_score']}, Risk: {summary['risk_level']}, Recommendation: {summary['recommendation']}")

            # 验证IC分析结果
            assert "ic_analysis" in report
            ic_analysis = report["ic_analysis"]
            assert "mean_ic" in ic_analysis
            assert "ic_std" in ic_analysis
            assert "ic_ir" in ic_analysis
            assert "ic_hit_rate" in ic_analysis
            assert 0 <= ic_analysis["ic_hit_rate"] <= 1

            print(f"IC Analysis - Mean IC: {ic_analysis['mean_ic']:.4f}, IR: {ic_analysis['ic_ir']:.4f}, Hit Rate: {ic_analysis['ic_hit_rate']:.2%}")

            # 验证分层收益分析
            assert "layered_returns" in report
            layered_returns = report["layered_returns"]
            assert "layers" in layered_returns
            assert layered_returns["layers"] == validation_config["layers"]
            assert "returns_by_layer" in layered_returns
            assert "long_short_spread" in layered_returns
            assert len(layered_returns["returns_by_layer"]) == validation_config["layers"]

            print(f"Layered Returns - Long-Short Spread: {layered_returns['long_short_spread']:.2%}")

            # 验证换手率分析
            assert "turnover_analysis" in report
            turnover_analysis = report["turnover_analysis"]
            assert "mean_turnover" in turnover_analysis
            assert "turnover_cost_impact" in turnover_analysis
            assert 0 <= turnover_analysis["mean_turnover"] <= 1

            print(f"Turnover Analysis - Mean Turnover: {turnover_analysis['mean_turnover']:.2%}")

            # 验证风险指标
            assert "risk_metrics" in report
            risk_metrics = report["risk_metrics"]
            assert "volatility" in risk_metrics
            assert "beta" in risk_metrics
            assert "tracking_error" in risk_metrics
            assert risk_metrics["volatility"] >= 0
            assert risk_metrics["tracking_error"] >= 0

            print(f"Risk Metrics - Volatility: {risk_metrics['volatility']:.2%}, Beta: {risk_metrics['beta']:.2f}")

            # Step 7: 测试不同格式的报告
            # 获取PDF格式报告
            pdf_response = requests.get(
                test_base.get_api_url("validation", f"/api/v1/validation/report/{validation_id}"),
                params={"format": "pdf"}
            )
            assert pdf_response.status_code == 200
            assert "application/pdf" in pdf_response.headers.get("Content-Type", "")

            # 获取Excel格式报告
            excel_response = requests.get(
                test_base.get_api_url("validation", f"/api/v1/validation/report/{validation_id}"),
                params={"format": "excel"}
            )
            assert excel_response.status_code == 200
            assert "spreadsheet" in excel_response.headers.get("Content-Type", "") or "excel" in excel_response.headers.get("Content-Type", "")

            # 获取摘要报告
            summary_response = requests.get(
                test_base.get_api_url("validation", f"/api/v1/validation/report/{validation_id}"),
                params={"summary_only": "true"}
            )
            assert summary_response.status_code == 200

            summary_report = summary_response.json()
            assert "report" in summary_report
            assert "summary" in summary_report["report"]
            # 摘要模式应该只包含关键指标
            summary_keys = set(summary_report["report"].keys())
            assert "summary" in summary_keys
            print("Different report formats validated successfully")

            # Step 8: 清理测试数据
            cleanup_response = requests.delete(
                test_base.get_api_url("factor", f"/api/v1/factors/{test_factor_id}")
            )
            assert cleanup_response.status_code == 200

            print("Validation workflow test completed successfully")

    @pytest.mark.asyncio
    async def test_validation_error_scenarios(self):
        """测试验证工作流的错误场景"""
        async with integration_test_environment() as agent_manager:
            test_base = IntegrationTestBase(agent_manager)

            # 测试不存在的因子验证
            invalid_validation_config = {
                "factor_id": "nonexistent_factor_12345",
                "validation_config": {
                    "train_start": "2010-01-01",
                    "train_end": "2016-12-31",
                    "test_start": "2017-01-01",
                    "test_end": "2020-12-31"
                }
            }

            error_response = requests.post(
                test_base.get_api_url("validation", "/api/v1/validation/factor"),
                json=invalid_validation_config
            )
            assert error_response.status_code == 404

            error_data = error_response.json()
            assert "error" in error_data
            assert "not found" in error_data["error"]["message"].lower()

            # 测试无效的日期配置
            invalid_date_config = {
                "factor_id": "test_factor",
                "validation_config": {
                    "train_start": "2020-01-01",  # 训练期开始晚于结束
                    "train_end": "2016-12-31",
                    "test_start": "2017-01-01",
                    "test_end": "2021-12-31"
                }
            }

            date_error_response = requests.post(
                test_base.get_api_url("validation", "/api/v1/validation/factor"),
                json=invalid_date_config
            )
            assert date_error_response.status_code == 400

            date_error_data = date_error_response.json()
            assert "error" in date_error_data
            print("Error scenarios handled correctly")

    @pytest.mark.asyncio
    async def test_custom_validation_configuration(self):
        """测试自定义验证配置"""
        async with integration_test_environment() as agent_manager:
            test_base = IntegrationTestBase(agent_manager)

            # 创建自定义验证配置
            custom_config = {
                "factor_id": f"custom_validation_{int(time.time())}",
                "validation_config": {
                    "train_start": "2015-01-01",
                    "train_end": "2018-12-31",
                    "test_start": "2019-01-01",
                    "test_end": "2021-12-31",
                    "validation_start": "2022-01-01",
                    "validation_end": "2023-06-30"
                },
                "validation_methods": ["ic_analysis"],  # 只进行IC分析
                "universe": "top100",
                "layers": 10,
                "rebalance_frequency": "weekly",
                "commission_rate": 0.001,
                "impact_cost": 0.0005
            }

            # 首先注册测试因子
            factor_def = {
                "factor_id": custom_config["factor_id"],
                "name": "自定义验证测试因子",
                "category": "test",
                "formula": "RSI(CLOSE, 14)",
                "version": "1.0.0"
            }

            requests.post(test_base.get_api_url("factor", "/api/v1/factors"), json=factor_def)

            # 启动自定义验证
            validation_response = requests.post(
                test_base.get_api_url("validation", "/api/v1/validation/factor"),
                json=custom_config
            )

            if validation_response.status_code == 202:
                validation_id = validation_response.json()["validation_id"]

                # 验证自定义配置是否正确应用
                status_response = requests.get(
                    test_base.get_api_url("validation", f"/api/v1/validation/report/{validation_id}")
                )

                if status_response.status_code == 200:
                    config_data = status_response.json()
                    if "metadata" in config_data:
                        metadata = config_data["metadata"]
                        assert metadata["validation_period"]["start"] == custom_config["validation_config"]["validation_start"]
                        print("Custom validation configuration applied successfully")

                # 清理
                requests.delete(test_base.get_api_url("factor", f"/api/v1/factors/{custom_config['factor_id']}"))

    @pytest.mark.asyncio
    async def test_concurrent_validations(self):
        """测试并发验证处理"""
        async with integration_test_environment() as agent_manager:
            test_base = IntegrationTestBase(agent_manager)

            # 准备多个测试因子
            factor_ids = []
            for i in range(3):
                factor_id = f"concurrent_test_{i}_{int(time.time())}"
                factor_def = {
                    "factor_id": factor_id,
                    "name": f"并发测试因子{i}",
                    "category": "test",
                    "formula": f"MA(CLOSE, {10 + i * 5})",
                    "version": "1.0.0"
                }

                register_response = requests.post(
                    test_base.get_api_url("factor", "/api/v1/factors"),
                    json=factor_def
                )
                if register_response.status_code == 201:
                    factor_ids.append(factor_id)

            # 启动并发验证
            validation_ids = []
            for factor_id in factor_ids:
                validation_config = {
                    "factor_id": factor_id,
                    "validation_config": {
                        "train_start": "2020-01-01",
                        "train_end": "2021-12-31",
                        "test_start": "2022-01-01",
                        "test_end": "2023-06-30"
                    },
                    "validation_methods": ["ic_analysis"],
                    "layers": 3  # 减少计算量
                }

                response = requests.post(
                    test_base.get_api_url("validation", "/api/v1/validation/factor"),
                    json=validation_config
                )

                if response.status_code == 202:
                    validation_ids.append(response.json()["validation_id"])
                elif response.status_code == 429:
                    print(f"Validation request rate limited (expected behavior)")
                    break

            print(f"Started {len(validation_ids)} concurrent validations")

            # 清理测试因子
            for factor_id in factor_ids:
                requests.delete(test_base.get_api_url("factor", f"/api/v1/factors/{factor_id}"))

            if len(validation_ids) > 0:
                print("Concurrent validation handling verified")