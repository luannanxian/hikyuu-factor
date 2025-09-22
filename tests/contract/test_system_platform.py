"""
系统平台管理API契约测试 - System Platform Management API Contract Tests

测试 GET /api/v1/system/platform 端点
根据 system-api.yaml 合约规范，验证平台信息返回和schema
"""
import pytest
import requests
from tests.utils import assert_json_schema, assert_response_time


class TestSystemPlatformContract:
    """
    系统平台API契约测试

    这些测试必须在实现前编写并失败，遵循TDD原则
    """

    def setup_method(self):
        """测试setup - 配置测试环境"""
        self.base_url = "http://localhost:8000"
        self.platform_endpoint = f"{self.base_url}/api/v1/system/platform"

    def test_platform_info_endpoint_exists(self):
        """
        测试: GET /api/v1/system/platform 端点存在

        期望: 端点应该存在且返回200状态码
        当前状态: 应该失败 (端点尚未实现)
        """
        response = requests.get(self.platform_endpoint)

        # 这个测试现在应该失败，因为端点尚未实现
        assert response.status_code == 200, f"Platform endpoint should exist, got {response.status_code}"

    def test_platform_info_response_schema(self):
        """
        测试: 平台信息响应schema验证

        期望返回格式:
        {
          "status": "success",
          "data": {
            "platform_type": "apple_silicon|x86_64|arm64_linux|generic",
            "cpu_architecture": "arm64|x86_64|aarch64",
            "cpu_count": int,
            "total_memory_gb": float,
            "available_memory_gb": float,
            "simd_support": ["ARM_NEON"] | ["SSE2", "SSE3", "AVX", "AVX2"]
          }
        }
        """
        response = requests.get(self.platform_endpoint)
        assert response.status_code == 200

        data = response.json()

        # 验证顶级响应结构
        assert "status" in data
        assert "data" in data
        assert data["status"] == "success"

        platform_data = data["data"]

        # 验证平台信息字段
        required_fields = [
            "platform_type", "cpu_architecture", "cpu_count",
            "total_memory_gb", "available_memory_gb", "simd_support"
        ]

        for field in required_fields:
            assert field in platform_data, f"Missing required field: {field}"

        # 验证字段类型和取值
        assert platform_data["platform_type"] in [
            "apple_silicon", "x86_64", "arm64_linux", "generic"
        ]
        assert isinstance(platform_data["cpu_count"], int)
        assert platform_data["cpu_count"] > 0
        assert isinstance(platform_data["total_memory_gb"], (int, float))
        assert isinstance(platform_data["available_memory_gb"], (int, float))
        assert isinstance(platform_data["simd_support"], list)
        assert len(platform_data["simd_support"]) > 0

    def test_platform_type_specific_simd_support(self):
        """
        测试: 不同平台类型的SIMD指令集支持

        Apple Silicon应该支持ARM_NEON
        x86_64应该支持SSE/AVX系列指令
        """
        response = requests.get(self.platform_endpoint)
        assert response.status_code == 200

        data = response.json()["data"]
        platform_type = data["platform_type"]
        simd_support = data["simd_support"]

        if platform_type == "apple_silicon":
            assert "ARM_NEON" in simd_support, "Apple Silicon should support ARM_NEON"
        elif platform_type == "x86_64":
            # x86_64应该至少支持SSE2
            x86_instructions = ["SSE2", "SSE3", "SSE41", "AVX", "AVX2"]
            assert any(instr in simd_support for instr in x86_instructions), \
                f"x86_64 should support at least one SSE/AVX instruction, got: {simd_support}"

    def test_response_time_performance(self):
        """
        测试: API响应时间应该 < 200ms (性能要求)
        """
        response = requests.get(self.platform_endpoint)
        assert response.status_code == 200

        # 检查响应时间
        assert_response_time(response, max_time_ms=200)

    def test_content_type_header(self):
        """
        测试: Content-Type头应该是application/json
        """
        response = requests.get(self.platform_endpoint)
        assert response.status_code == 200
        assert response.headers.get("Content-Type") == "application/json"

    def test_apple_silicon_optimization_detection(self):
        """
        测试: Apple Silicon平台的优化检测

        如果运行在Apple Silicon上，应该检测到相应的优化配置
        """
        response = requests.get(self.platform_endpoint)
        assert response.status_code == 200

        data = response.json()["data"]

        # 如果是Apple Silicon，验证特定配置
        if data["platform_type"] == "apple_silicon":
            assert data["cpu_architecture"] in ["arm64", "aarch64"]
            assert "ARM_NEON" in data["simd_support"]
            # Apple Silicon通常内存较大
            assert data["total_memory_gb"] >= 8

    def test_x86_optimization_detection(self):
        """
        测试: x86_64平台的优化检测
        """
        response = requests.get(self.platform_endpoint)
        assert response.status_code == 200

        data = response.json()["data"]

        # 如果是x86_64，验证特定配置
        if data["platform_type"] == "x86_64":
            assert data["cpu_architecture"] in ["x86_64", "amd64"]
            # x86应该支持某些SIMD指令
            simd_support = data["simd_support"]
            assert any(instr in simd_support for instr in ["SSE2", "AVX", "AVX2"])

    def test_error_handling_for_invalid_requests(self):
        """
        测试: 错误请求的处理

        测试无效HTTP方法应该返回405 Method Not Allowed
        """
        # POST应该不被支持
        response = requests.post(self.platform_endpoint)
        assert response.status_code == 405

        # PUT应该不被支持
        response = requests.put(self.platform_endpoint)
        assert response.status_code == 405

        # DELETE应该不被支持
        response = requests.delete(self.platform_endpoint)
        assert response.status_code == 405

    def test_cors_headers_present(self):
        """
        测试: CORS头应该存在 (跨域支持)
        """
        response = requests.get(self.platform_endpoint)
        assert response.status_code == 200

        # 检查CORS头
        assert "Access-Control-Allow-Origin" in response.headers
        assert "Access-Control-Allow-Methods" in response.headers


class TestSystemPlatformIntegration:
    """
    系统平台集成测试 - 测试与其他组件的集成
    """

    def setup_method(self):
        self.base_url = "http://localhost:8000"
        self.platform_endpoint = f"{self.base_url}/api/v1/system/platform"

    def test_platform_detection_consistency(self):
        """
        测试: 平台检测的一致性

        多次调用应该返回一致的平台信息
        """
        responses = []
        for _ in range(3):
            response = requests.get(self.platform_endpoint)
            assert response.status_code == 200
            responses.append(response.json())

        # 验证所有响应一致
        first_response = responses[0]["data"]
        for response_data in responses[1:]:
            data = response_data["data"]
            assert data["platform_type"] == first_response["platform_type"]
            assert data["cpu_architecture"] == first_response["cpu_architecture"]
            assert data["cpu_count"] == first_response["cpu_count"]
            assert data["simd_support"] == first_response["simd_support"]

    def test_memory_info_reasonable_bounds(self):
        """
        测试: 内存信息的合理性边界检查
        """
        response = requests.get(self.platform_endpoint)
        assert response.status_code == 200

        data = response.json()["data"]

        # 内存信息应该合理
        total_memory = data["total_memory_gb"]
        available_memory = data["available_memory_gb"]

        assert total_memory > 0, "Total memory should be positive"
        assert available_memory > 0, "Available memory should be positive"
        assert available_memory <= total_memory, "Available memory should not exceed total memory"

        # 现代系统应该至少有1GB内存
        assert total_memory >= 1.0, "System should have at least 1GB memory"

        # 可用内存不应该超过总内存的95% (预留系统使用)
        assert available_memory <= total_memory * 0.95, "Available memory seems suspiciously high"


# 这个测试文件应该运行失败，因为端点尚未实现
# 这是TDD的正确流程：先写测试，测试失败，然后实现功能使测试通过