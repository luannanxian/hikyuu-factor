"""
Factor Lifecycle Integration Test (T026)
集成测试: 因子注册→计算→存储→查询
"""

import asyncio
import pytest
import requests
import time
from typing import Dict, Any, List

from tests.integration.test_environment import IntegrationTestBase, integration_test_environment


class TestFactorLifecycleIntegration:
    """因子生命周期集成测试"""

    @pytest.mark.asyncio
    async def test_complete_factor_lifecycle_workflow(self):
        """T026: 完整因子生命周期工作流"""
        async with integration_test_environment() as agent_manager:
            test_base = IntegrationTestBase(agent_manager)

            # Step 1: 注册新因子
            factor_definition = {
                "factor_id": f"test_momentum_{int(time.time())}",
                "name": "测试动量因子",
                "description": "基于20日动量计算的测试因子",
                "category": "momentum",
                "formula": "MA(CLOSE, 20) / MA(CLOSE, 60) - 1",
                "hikyuu_formula": {
                    "expression": "CLOSE / REF(CLOSE, 20) - 1",
                    "parameters": {
                        "lookback_period": 20,
                        "normalization": "zscore"
                    }
                },
                "data_dependencies": ["kdata"],
                "update_frequency": "daily",
                "neutralization": ["industry", "market_cap"],
                "version": "1.0.0",
                "author": "integration_test",
                "tags": ["momentum", "test", "integration"]
            }

            register_response = requests.post(
                test_base.get_api_url("factor", "/api/v1/factors"),
                json=factor_definition
            )
            assert register_response.status_code == 201

            register_data = register_response.json()
            factor_id = register_data["factor_id"]
            assert factor_id == factor_definition["factor_id"]
            assert register_data["status"] == "registered"

            print(f"Factor registered: {factor_id}")

            # Step 2: 验证因子注册
            factor_info_response = requests.get(
                test_base.get_api_url("factor", f"/api/v1/factors/{factor_id}")
            )
            assert factor_info_response.status_code == 200

            factor_info = factor_info_response.json()
            assert factor_info["factor_id"] == factor_id
            assert factor_info["status"] == "registered"
            assert "hikyuu_formula" in factor_info

            # Step 3: 执行因子计算
            calculation_request = {
                "factor_id": factor_id,
                "stock_universe": ["sh000001", "sh000002", "sz000001", "sz000002", "sh000300"],
                "date_range": {
                    "start_date": "2023-11-01",
                    "end_date": "2023-11-30"
                },
                "calculation_mode": "full",
                "platform_optimization": True,
                "cache_results": True,
                "neutralization_enabled": True
            }

            calculation_response = requests.post(
                test_base.get_api_url("factor", f"/api/v1/factors/{factor_id}/calculate"),
                json=calculation_request
            )
            assert calculation_response.status_code == 202

            calc_data = calculation_response.json()
            calc_task_id = calc_data["task_id"]
            assert calc_data["status"] == "running"

            print(f"Factor calculation started: {calc_task_id}")

            # Step 4: 监控计算进度
            calc_result = await test_base.wait_for_task_completion("factor", calc_task_id, timeout=1800)  # 30分钟
            assert calc_result["status"] == "completed"

            calc_summary = calc_result["result"]
            assert "processed_stocks" in calc_summary
            assert "calculated_points" in calc_summary
            assert "storage_location" in calc_summary
            assert "performance_metrics" in calc_summary

            processed_stocks = calc_summary["processed_stocks"]
            calculated_points = calc_summary["calculated_points"]

            print(f"Factor calculation completed: {processed_stocks} stocks, {calculated_points} data points")

            # 验证性能目标 - 全市场计算应在30分钟内完成
            performance_metrics = calc_summary["performance_metrics"]
            execution_time_ms = performance_metrics["execution_time_ms"]
            assert execution_time_ms <= 1800000, f"Calculation took too long: {execution_time_ms}ms"

            # Step 5: 查询计算结果
            query_request = {
                "factor_id": factor_id,
                "stock_codes": ["sh000001", "sz000001"],
                "start_date": "2023-11-01",
                "end_date": "2023-11-30",
                "format": "json",
                "include_metadata": True
            }

            query_response = requests.get(
                test_base.get_api_url("factor", f"/api/v1/factors/{factor_id}/values"),
                params=query_request
            )
            assert query_response.status_code == 200

            query_data = query_response.json()
            assert "factor_values" in query_data
            assert "metadata" in query_data

            factor_values = query_data["factor_values"]
            assert len(factor_values) > 0

            # 验证数据格式和完整性
            for value in factor_values[:5]:  # 检查前5个数据点
                assert "stock_code" in value
                assert "trade_date" in value
                assert "factor_value" in value
                assert value["factor_value"] is not None

            print(f"Factor values retrieved: {len(factor_values)} data points")

            # Step 6: 验证数据存储一致性
            # 重新查询相同数据，验证缓存和存储一致性
            second_query_response = requests.get(
                test_base.get_api_url("factor", f"/api/v1/factors/{factor_id}/values"),
                params=query_request
            )
            assert second_query_response.status_code == 200

            second_query_data = second_query_response.json()
            assert len(second_query_data["factor_values"]) == len(factor_values)

            # 验证数值一致性
            for i, (original, cached) in enumerate(zip(factor_values, second_query_data["factor_values"])):
                assert original["stock_code"] == cached["stock_code"]
                assert original["trade_date"] == cached["trade_date"]
                assert abs(original["factor_value"] - cached["factor_value"]) < 1e-10, f"Value mismatch at index {i}"

            print("Factor lifecycle test completed successfully")