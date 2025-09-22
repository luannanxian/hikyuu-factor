"""
系统健康检查API契约测试 - System Health Check API Contract Tests

测试 GET /api/v1/system/health 端点
根据 system-api.yaml 合约规范，验证健康状态检查
"""
import pytest
import requests
import time
from tests.utils import assert_json_schema, assert_response_time


class TestSystemHealthContract:
    """
    系统健康检查API契约测试

    这些测试必须在实现前编写并失败，遵循TDD原则
    """

    def setup_method(self):
        """测试setup - 配置测试环境"""
        self.base_url = "http://localhost:8000"
        self.health_endpoint = f"{self.base_url}/api/v1/system/health"

    def test_health_check_endpoint_exists(self):
        """
        测试: GET /api/v1/system/health 端点存在

        期望: 端点应该存在且返回健康状态
        当前状态: 应该失败 (端点尚未实现)
        """
        response = requests.get(self.health_endpoint)

        # 这个测试现在应该失败，因为端点尚未实现
        assert response.status_code == 200, f"Health endpoint should exist, got {response.status_code}"

    def test_health_check_response_schema(self):
        """
        测试: 健康检查响应schema验证

        期望返回格式:
        {
          "status": "healthy|degraded|unhealthy",
          "timestamp": "ISO datetime",
          "version": "1.0.0",
          "uptime_seconds": int,
          "components": {
            "database": {
              "status": "healthy|unhealthy",
              "response_time_ms": float,
              "details": "..."
            },
            "hikyuu": {
              "status": "healthy|unhealthy",
              "version": "2.6.8",
              "details": "..."
            },
            "agents": {
              "data_manager": {"status": "healthy|unhealthy", "port": 8001},
              "factor_calculation": {"status": "healthy|unhealthy", "port": 8002},
              "validation": {"status": "healthy|unhealthy", "port": 8003},
              "signal_generation": {"status": "healthy|unhealthy", "port": 8004}
            }
          }
        }
        """
        response = requests.get(self.health_endpoint)
        assert response.status_code == 200

        data = response.json()

        # 验证顶级字段
        required_top_fields = ["status", "timestamp", "version", "uptime_seconds", "components"]
        for field in required_top_fields:
            assert field in data, f"Missing required field: {field}"

        # 验证status值
        assert data["status"] in ["healthy", "degraded", "unhealthy"]

        # 验证类型
        assert isinstance(data["uptime_seconds"], int)
        assert data["uptime_seconds"] >= 0

        # 验证components结构
        components = data["components"]
        assert "database" in components
        assert "hikyuu" in components
        assert "agents" in components

        # 验证数据库组件
        db_component = components["database"]
        assert "status" in db_component
        assert "response_time_ms" in db_component
        assert db_component["status"] in ["healthy", "unhealthy"]

        # 验证Hikyuu组件
        hikyuu_component = components["hikyuu"]
        assert "status" in hikyuu_component
        assert "version" in hikyuu_component
        assert hikyuu_component["status"] in ["healthy", "unhealthy"]

        # 验证Agent组件
        agents = components["agents"]
        expected_agents = ["data_manager", "factor_calculation", "validation", "signal_generation"]
        for agent_name in expected_agents:
            assert agent_name in agents, f"Missing agent: {agent_name}"
            agent = agents[agent_name]
            assert "status" in agent
            assert "port" in agent
            assert agent["status"] in ["healthy", "unhealthy"]

    def test_health_check_agent_status_validation(self):
        """
        测试: Agent状态验证

        验证4个核心Agent的状态检查
        """
        response = requests.get(self.health_endpoint)
        assert response.status_code == 200

        data = response.json()
        agents = data["components"]["agents"]

        # 验证Agent端口映射
        expected_agent_ports = {
            "data_manager": 8001,
            "factor_calculation": 8002,
            "validation": 8003,
            "signal_generation": 8004
        }

        for agent_name, expected_port in expected_agent_ports.items():
            assert agent_name in agents
            agent = agents[agent_name]
            assert agent["port"] == expected_port, \
                f"{agent_name} should be on port {expected_port}, got {agent['port']}"

    def test_overall_health_status_logic(self):
        """
        测试: 整体健康状态逻辑

        - 所有组件healthy → 整体healthy
        - 任何组件unhealthy → 整体unhealthy
        - 部分组件问题但核心可用 → 整体degraded
        """
        response = requests.get(self.health_endpoint)
        assert response.status_code == 200

        data = response.json()
        overall_status = data["status"]
        components = data["components"]

        # 收集所有组件状态
        component_statuses = []

        # 数据库状态
        component_statuses.append(components["database"]["status"])

        # Hikyuu状态
        component_statuses.append(components["hikyuu"]["status"])

        # Agent状态
        for agent in components["agents"].values():
            component_statuses.append(agent["status"])

        # 验证整体状态逻辑
        if all(status == "healthy" for status in component_statuses):
            assert overall_status == "healthy", "All components healthy should result in overall healthy"
        elif any(status == "unhealthy" for status in component_statuses):
            # 如果有组件不健康，整体应该是degraded或unhealthy
            assert overall_status in ["degraded", "unhealthy"], \
                "Unhealthy components should result in degraded or unhealthy overall status"

    def test_database_connectivity_check(self):
        """
        测试: 数据库连接检查

        数据库组件应该包含连接状态和响应时间
        """
        response = requests.get(self.health_endpoint)
        assert response.status_code == 200

        data = response.json()
        db_component = data["components"]["database"]

        # 验证数据库健康检查字段
        assert "status" in db_component
        assert "response_time_ms" in db_component

        # 如果数据库健康，响应时间应该合理
        if db_component["status"] == "healthy":
            response_time = db_component["response_time_ms"]
            assert isinstance(response_time, (int, float))
            assert response_time > 0
            assert response_time < 5000, "Database response time should be < 5 seconds"

    def test_hikyuu_framework_check(self):
        """
        测试: Hikyuu框架状态检查

        验证Hikyuu框架的初始化状态和版本信息
        """
        response = requests.get(self.health_endpoint)
        assert response.status_code == 200

        data = response.json()
        hikyuu_component = data["components"]["hikyuu"]

        # 验证Hikyuu组件字段
        assert "status" in hikyuu_component
        assert "version" in hikyuu_component

        # 如果Hikyuu健康，版本应该符合要求
        if hikyuu_component["status"] == "healthy":
            version = hikyuu_component["version"]
            assert isinstance(version, str)
            # 验证版本格式 (应该是2.6.8或更高版本)
            version_parts = version.split(".")
            assert len(version_parts) >= 3, f"Invalid version format: {version}"
            major, minor, patch = map(int, version_parts[:3])
            assert (major, minor, patch) >= (2, 6, 8), \
                f"Hikyuu version {version} should be >= 2.6.8"

    def test_health_check_response_time(self):
        """
        测试: 健康检查响应时间

        健康检查应该快速响应 (< 1秒)
        """
        start_time = time.time()
        response = requests.get(self.health_endpoint)
        elapsed_time = time.time() - start_time

        assert response.status_code == 200
        assert elapsed_time < 1.0, f"Health check took too long: {elapsed_time:.2f}s"

    def test_health_check_content_type(self):
        """
        测试: Content-Type头验证
        """
        response = requests.get(self.health_endpoint)
        assert response.status_code == 200
        assert response.headers.get("Content-Type") == "application/json"

    def test_health_check_caching_headers(self):
        """
        测试: 缓存头设置

        健康检查不应该被缓存
        """
        response = requests.get(self.health_endpoint)
        assert response.status_code == 200

        # 验证禁用缓存的头
        cache_control = response.headers.get("Cache-Control", "")
        assert "no-cache" in cache_control or "no-store" in cache_control

    def test_health_check_timestamp_format(self):
        """
        测试: 时间戳格式验证

        时间戳应该是ISO 8601格式
        """
        response = requests.get(self.health_endpoint)
        assert response.status_code == 200

        data = response.json()
        timestamp = data["timestamp"]

        # 验证ISO 8601格式
        from datetime import datetime
        try:
            parsed_time = datetime.fromisoformat(timestamp.replace("Z", "+00:00"))
            # 时间戳应该是最近的 (不超过10秒前)
            now = datetime.now(parsed_time.tzinfo)
            time_diff = abs((now - parsed_time).total_seconds())
            assert time_diff < 10, f"Timestamp seems stale: {timestamp}"
        except ValueError:
            pytest.fail(f"Invalid timestamp format: {timestamp}")

    def test_uptime_tracking(self):
        """
        测试: 系统运行时间跟踪

        运行时间应该是递增的
        """
        # 第一次调用
        response1 = requests.get(self.health_endpoint)
        assert response1.status_code == 200
        uptime1 = response1.json()["uptime_seconds"]

        # 等待1秒
        time.sleep(1)

        # 第二次调用
        response2 = requests.get(self.health_endpoint)
        assert response2.status_code == 200
        uptime2 = response2.json()["uptime_seconds"]

        # 运行时间应该增加
        assert uptime2 > uptime1, "Uptime should increase between calls"
        assert uptime2 - uptime1 >= 1, "Uptime should increase by at least 1 second"

    def test_error_handling_for_invalid_methods(self):
        """
        测试: 无效HTTP方法的错误处理
        """
        # POST应该不被支持
        response = requests.post(self.health_endpoint)
        assert response.status_code == 405

        # PUT应该不被支持
        response = requests.put(self.health_endpoint)
        assert response.status_code == 405

        # DELETE应该不被支持
        response = requests.delete(self.health_endpoint)
        assert response.status_code == 405


class TestSystemHealthIntegration:
    """
    系统健康检查集成测试
    """

    def setup_method(self):
        self.base_url = "http://localhost:8000"
        self.health_endpoint = f"{self.base_url}/api/v1/system/health"

    def test_health_check_agent_cascade_failure(self):
        """
        测试: Agent级联故障检测

        如果多个Agent不可用，整体状态应该反映严重程度
        """
        response = requests.get(self.health_endpoint)
        assert response.status_code == 200

        data = response.json()
        agents = data["components"]["agents"]

        # 统计不健康的Agent数量
        unhealthy_agents = [
            agent_name for agent_name, agent_data in agents.items()
            if agent_data["status"] == "unhealthy"
        ]

        overall_status = data["status"]

        # 如果超过一半Agent不健康，整体应该是unhealthy
        if len(unhealthy_agents) > len(agents) // 2:
            assert overall_status == "unhealthy", \
                f"Too many unhealthy agents ({len(unhealthy_agents)}/{len(agents)}) should result in unhealthy status"

    def test_dependency_health_validation(self):
        """
        测试: 依赖服务健康验证

        验证关键依赖服务的健康状态
        """
        response = requests.get(self.health_endpoint)
        assert response.status_code == 200

        data = response.json()
        components = data["components"]

        # 数据库是关键依赖
        if components["database"]["status"] == "unhealthy":
            # 数据库不健康时，整体状态不应该是healthy
            assert data["status"] != "healthy", \
                "Unhealthy database should prevent overall healthy status"

        # Hikyuu是关键依赖
        if components["hikyuu"]["status"] == "unhealthy":
            # Hikyuu不健康时，整体状态不应该是healthy
            assert data["status"] != "healthy", \
                "Unhealthy Hikyuu should prevent overall healthy status"


# 这个测试文件应该运行失败，因为端点尚未实现
# 这是TDD的正确流程：先写测试，测试失败，然后实现功能使测试通过