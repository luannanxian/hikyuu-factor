"""
Data Workflow Integration Test (T025)
集成测试: 数据更新→质量检查→异常处理
"""

import asyncio
import pytest
import requests
import time
from typing import Dict, Any, List

from tests.integration.test_environment import IntegrationTestBase, integration_test_environment


class TestDataWorkflowIntegration:
    """数据工作流集成测试"""

    @pytest.mark.asyncio
    async def test_data_update_to_quality_validation_workflow(self):
        """T025: 完整数据更新到质量验证工作流"""
        async with integration_test_environment() as agent_manager:
            test_base = IntegrationTestBase(agent_manager)

            # Step 1: 触发数据更新任务
            update_request = {
                "data_types": ["kdata", "financial"],
                "stock_codes": ["sh000001", "sh000002", "sz000001", "sz000002"],
                "start_date": "2023-12-01",
                "end_date": "2023-12-31",
                "force_update": False,
                "batch_size": 50,
                "priority": "normal"
            }

            update_response = requests.post(
                test_base.get_api_url("data", "/api/v1/data/update"),
                json=update_request
            )
            assert update_response.status_code == 202  # 异步处理

            update_data = update_response.json()
            update_task_id = update_data["task_id"]
            assert update_data["status"] == "running"

            print(f"Data update task started: {update_task_id}")

            # Step 2: 监控数据更新进度
            update_result = await test_base.wait_for_task_completion("data", update_task_id, timeout=300)
            assert update_result["status"] == "completed"

            update_summary = update_result["result"]
            assert "processed_count" in update_summary
            assert "success_count" in update_summary
            assert "error_count" in update_summary

            print(f"Data update completed: {update_summary['success_count']}/{update_summary['processed_count']} successful")

            # Step 3: 执行数据质量检查
            quality_request = {
                "stock_codes": update_request["stock_codes"],
                "start_date": update_request["start_date"],
                "end_date": update_request["end_date"],
                "check_types": [
                    "price_consistency",
                    "volume_validity",
                    "change_threshold",
                    "missing_data",
                    "duplicate_records"
                ],
                "thresholds": {
                    "max_price_change": 0.2,  # 20%单日涨跌幅限制
                    "min_volume": 0,
                    "max_missing_ratio": 0.05  # 最多5%缺失数据
                },
                "detailed_report": True
            }

            quality_response = requests.post(
                test_base.get_api_url("data", "/api/v1/data/quality/check"),
                json=quality_request
            )
            assert quality_response.status_code == 202

            quality_data = quality_response.json()
            quality_task_id = quality_data["task_id"]

            # Step 4: 等待质量检查完成
            quality_result = await test_base.wait_for_task_completion("data", quality_task_id, timeout=120)
            assert quality_result["status"] == "completed"

            quality_report = quality_result["result"]
            assert "overall_score" in quality_report
            assert "check_results" in quality_report
            assert "issues_found" in quality_report

            print(f"Data quality score: {quality_report['overall_score']}")

            # Step 5: 处理检测到的异常
            if quality_report["issues_found"] > 0:
                issues = quality_report["detailed_issues"]

                # 分类处理不同类型的问题
                critical_issues = [issue for issue in issues if issue["severity"] == "critical"]
                warning_issues = [issue for issue in issues if issue["severity"] == "warning"]

                print(f"Found {len(critical_issues)} critical issues, {len(warning_issues)} warnings")

                # 对于严重问题，启动数据修复
                if critical_issues:
                    repair_request = {
                        "issues": critical_issues,
                        "repair_strategies": ["auto_fix", "flag_for_review"],
                        "backup_before_repair": True
                    }

                    repair_response = requests.post(
                        test_base.get_api_url("data", "/api/v1/data/repair"),
                        json=repair_request
                    )
                    assert repair_response.status_code == 202

                    repair_data = repair_response.json()
                    repair_task_id = repair_data["task_id"]

                    # 等待修复完成
                    repair_result = await test_base.wait_for_task_completion("data", repair_task_id, timeout=180)
                    assert repair_result["status"] == "completed"

                    repair_summary = repair_result["result"]
                    assert "repaired_count" in repair_summary
                    assert "flagged_count" in repair_summary

                    print(f"Data repair completed: {repair_summary['repaired_count']} repaired, {repair_summary['flagged_count']} flagged")

            # Step 6: 生成最终报告
            report_request = {
                "workflow_id": f"data_workflow_{int(time.time())}",
                "tasks": [
                    {"task_id": update_task_id, "task_type": "data_update"},
                    {"task_id": quality_task_id, "task_type": "quality_check"}
                ],
                "include_metrics": True,
                "include_recommendations": True
            }

            report_response = requests.post(
                test_base.get_api_url("data", "/api/v1/data/workflow/report"),
                json=report_request
            )
            assert report_response.status_code == 200

            workflow_report = report_response.json()
            assert "workflow_summary" in workflow_report
            assert "performance_metrics" in workflow_report
            assert "data_quality_assessment" in workflow_report
            assert "recommendations" in workflow_report

            # 验证整体工作流成功
            workflow_summary = workflow_report["workflow_summary"]
            assert workflow_summary["status"] == "completed"
            assert workflow_summary["overall_success"] is True

            print("Data workflow completed successfully with comprehensive quality validation")