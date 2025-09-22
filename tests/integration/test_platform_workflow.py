"""
Platform Workflow Integration Test (T024)
集成测试: 平台检测→优化配置→性能验证
"""

import asyncio
import pytest
import requests
import time
from typing import Dict, Any

from tests.integration.test_environment import IntegrationTestBase, integration_test_environment


class TestPlatformWorkflowIntegration:
    """平台工作流集成测试"""

    @pytest.mark.asyncio
    async def test_platform_detection_to_performance_verification(self):
        """T024: 完整平台检测到性能验证工作流"""
        async with integration_test_environment() as agent_manager:
            test_base = IntegrationTestBase(agent_manager)

            # Step 1: 检测当前平台
            platform_response = requests.get(
                test_base.get_api_url("data", "/api/v1/system/platform")
            )
            assert platform_response.status_code == 200

            platform_data = platform_response.json()
            assert "platform_type" in platform_data
            assert "simd_support" in platform_data
            assert "optimization_available" in platform_data

            detected_platform = platform_data["platform_type"]
            print(f"Detected platform: {detected_platform}")

            # Step 2: 基于检测结果配置优化
            optimization_config = {
                "platform_type": detected_platform,
                "enable_simd": platform_data["simd_support"]["available"],
                "optimization_level": "aggressive",
                "memory_optimization": True,
                "batch_processing": True
            }

            # 如果是Apple Silicon，启用ARM NEON优化
            if detected_platform == "apple_silicon":
                optimization_config.update({
                    "arm_neon_enabled": True,
                    "metal_acceleration": True,
                    "unified_memory": True
                })

            # 如果是x86_64，启用AVX/SSE优化
            elif detected_platform == "x86_64":
                optimization_config.update({
                    "avx2_enabled": platform_data["simd_support"]["avx2"],
                    "sse4_enabled": platform_data["simd_support"]["sse4"],
                    "cache_optimization": True
                })

            config_response = requests.post(
                test_base.get_api_url("data", "/api/v1/system/optimization/config"),
                json=optimization_config
            )
            assert config_response.status_code == 200

            config_result = config_response.json()
            assert config_result["status"] == "applied"
            assert "performance_improvement" in config_result

            # Step 3: 执行性能基准测试
            benchmark_request = {
                "test_type": "factor_calculation",
                "factor_id": "momentum_20d",
                "stock_universe": ["sh000001", "sh000002", "sz000001", "sz000002"],
                "date_range": {
                    "start_date": "2023-01-01",
                    "end_date": "2023-01-31"
                },
                "benchmark_iterations": 3
            }

            benchmark_response = requests.post(
                test_base.get_api_url("factor", "/api/v1/factor/benchmark"),
                json=benchmark_request
            )
            assert benchmark_response.status_code == 202  # 异步处理

            benchmark_data = benchmark_response.json()
            benchmark_id = benchmark_data["benchmark_id"]

            # Step 4: 等待基准测试完成
            benchmark_result = await test_base.wait_for_task_completion("factor", benchmark_id, timeout=120)
            assert benchmark_result["status"] == "completed"

            performance_metrics = benchmark_result["result"]["performance_metrics"]

            # Step 5: 验证性能提升
            assert "execution_time_ms" in performance_metrics
            assert "memory_usage_mb" in performance_metrics
            assert "cpu_utilization" in performance_metrics
            assert "optimization_effectiveness" in performance_metrics

            # 验证优化效果
            optimization_effectiveness = performance_metrics["optimization_effectiveness"]
            assert optimization_effectiveness["enabled"] is True

            if detected_platform == "apple_silicon":
                # Apple Silicon应该有NEON优化
                assert "arm_neon" in optimization_effectiveness
                assert optimization_effectiveness["arm_neon"]["used"] is True
                assert optimization_effectiveness["arm_neon"]["speedup"] > 1.0

            elif detected_platform == "x86_64":
                # x86_64应该有AVX/SSE优化
                if platform_data["simd_support"]["avx2"]:
                    assert "avx2" in optimization_effectiveness
                    assert optimization_effectiveness["avx2"]["used"] is True
                    assert optimization_effectiveness["avx2"]["speedup"] > 1.0

            # 验证整体性能目标
            execution_time = performance_metrics["execution_time_ms"]
            assert execution_time < 10000, f"Performance target not met: {execution_time}ms > 10s"

            print(f"Platform workflow completed successfully:")
            print(f"  Platform: {detected_platform}")
            print(f"  Execution time: {execution_time}ms")
            print(f"  Optimization effectiveness: {optimization_effectiveness}")

    @pytest.mark.asyncio
    async def test_platform_optimization_error_handling(self):
        """测试平台优化错误处理"""
        async with integration_test_environment() as agent_manager:
            test_base = IntegrationTestBase(agent_manager)

            # 测试无效配置
            invalid_config = {
                "platform_type": "invalid_platform",
                "enable_simd": True,
                "optimization_level": "invalid_level"
            }

            response = requests.post(
                test_base.get_api_url("data", "/api/v1/system/optimization/config"),
                json=invalid_config
            )
            assert response.status_code == 400

            error_data = response.json()
            assert "error" in error_data
            assert "invalid_platform" in error_data["error"]["message"].lower()

    @pytest.mark.asyncio
    async def test_cross_platform_compatibility(self):
        """测试跨平台兼容性"""
        async with integration_test_environment() as agent_manager:
            test_base = IntegrationTestBase(agent_manager)

            # 测试所有支持的平台类型配置
            supported_platforms = ["apple_silicon", "x86_64", "generic"]

            for platform_type in supported_platforms:
                config = {
                    "platform_type": platform_type,
                    "optimization_level": "balanced",
                    "auto_detect": False  # 强制使用指定平台
                }

                response = requests.post(
                    test_base.get_api_url("data", "/api/v1/system/optimization/config"),
                    json=config
                )

                # 所有平台配置都应该被接受
                assert response.status_code == 200, f"Platform {platform_type} config failed"

                result = response.json()
                assert result["status"] == "applied"
                assert result["platform_type"] == platform_type