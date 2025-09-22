"""
因子列表API契约测试 - Factors List API Contract Tests

测试 GET /api/v1/factors 端点
根据 factor_calculation_api.yaml 合约规范，验证因子列表获取功能
"""
import pytest
import requests
from datetime import datetime, date
from tests.utils import assert_json_schema, assert_response_time


class TestFactorsListContract:
    """
    因子列表API契约测试

    这些测试必须在实现前编写并失败，遵循TDD原则
    """

    def setup_method(self):
        """测试setup - 配置测试环境"""
        self.base_url = "http://localhost:8002"  # Factor Calculation Agent端口
        self.factors_endpoint = f"{self.base_url}/api/v1/factors"

    def test_factors_list_endpoint_exists(self):
        """
        测试: GET /api/v1/factors 端点存在

        期望: 端点应该存在且返回因子列表
        当前状态: 应该失败 (端点尚未实现)
        """
        response = requests.get(self.factors_endpoint)

        # 这个测试现在应该失败，因为端点尚未实现
        assert response.status_code == 200, f"Factors list endpoint should exist, got {response.status_code}"

    def test_factors_list_response_schema(self):
        """
        测试: 因子列表响应schema验证

        期望返回格式:
        {
          "status": "success",
          "data": {
            "factors": [
              {
                "id": "momentum_20d_1234",
                "name": "20日动量因子",
                "category": "momentum",
                "hikyuu_formula": "MA(CLOSE(), 5) / MA(CLOSE(), 20) - 1",
                "economic_logic": "短期均线相对长期均线的偏离程度",
                "version": "1.0.0",
                "created_by": "factor_engineer_01",
                "created_at": "2025-01-15T10:30:00Z",
                "status": "active",
                "hikyuu_dependencies": ["CLOSE", "MA"],
                "parameters": {
                  "fast_period": 5,
                  "slow_period": 20
                },
                "performance_metrics": {
                  "ic_mean": 0.08,
                  "ic_ir": 1.2,
                  "sharpe_ratio": 0.85
                }
              }
            ],
            "total_count": 50,
            "filtered_count": 25,
            "page": 1,
            "page_size": 20
          }
        }
        """
        response = requests.get(self.factors_endpoint)
        assert response.status_code == 200

        data = response.json()

        # 验证顶级响应结构
        assert "status" in data
        assert "data" in data
        assert data["status"] == "success"

        factors_data = data["data"]

        # 验证数据结构
        required_fields = ["factors", "total_count", "filtered_count", "page", "page_size"]
        for field in required_fields:
            assert field in factors_data, f"Missing required field: {field}"

        # 验证因子列表
        factors = factors_data["factors"]
        assert isinstance(factors, list)

        # 验证单个因子对象结构
        if factors:
            factor = factors[0]
            factor_required_fields = [
                "id", "name", "category", "hikyuu_formula", "economic_logic",
                "version", "created_by", "created_at", "status",
                "hikyuu_dependencies", "parameters"
            ]
            for field in factor_required_fields:
                assert field in factor, f"Missing factor field: {field}"

            # 验证字段类型
            assert isinstance(factor["id"], str)
            assert isinstance(factor["name"], str)
            assert isinstance(factor["hikyuu_dependencies"], list)
            assert isinstance(factor["parameters"], dict)
            assert factor["status"] in ["active", "inactive", "deprecated"]

    def test_factors_category_validation(self):
        """
        测试: 因子类别验证

        支持的因子类别: momentum, value, quality, growth, risk, technical
        """
        valid_categories = ["momentum", "value", "quality", "growth", "risk", "technical"]

        response = requests.get(self.factors_endpoint)
        assert response.status_code == 200

        data = response.json()
        factors = data["data"]["factors"]

        # 验证所有因子的类别都在有效范围内
        for factor in factors:
            assert factor["category"] in valid_categories, \
                f"Invalid factor category: {factor['category']}"

    def test_factors_pagination(self):
        """
        测试: 分页功能

        验证page和page_size参数
        """
        # 测试第一页
        response = requests.get(self.factors_endpoint, params={"page": 1, "page_size": 10})
        assert response.status_code == 200

        data = response.json()
        factors_data = data["data"]

        assert factors_data["page"] == 1
        assert factors_data["page_size"] == 10
        assert len(factors_data["factors"]) <= 10

        # 测试第二页
        response = requests.get(self.factors_endpoint, params={"page": 2, "page_size": 10})
        assert response.status_code == 200

        data2 = response.json()
        factors_data2 = data2["data"]

        assert factors_data2["page"] == 2
        assert factors_data2["page_size"] == 10

        # 验证不同页的因子不重复
        page1_ids = {factor["id"] for factor in factors_data["factors"]}
        page2_ids = {factor["id"] for factor in factors_data2["factors"]}

        overlap = page1_ids.intersection(page2_ids)
        assert len(overlap) == 0, f"Pages should not have overlapping factors: {overlap}"

    def test_factors_category_filter(self):
        """
        测试: 按类别过滤因子
        """
        # 测试动量因子过滤
        response = requests.get(self.factors_endpoint, params={"category": "momentum"})
        assert response.status_code == 200

        data = response.json()
        factors = data["data"]["factors"]

        # 验证只返回动量因子
        for factor in factors:
            assert factor["category"] == "momentum", \
                f"Should only return momentum factors, got {factor['category']}"

        # 测试价值因子过滤
        response = requests.get(self.factors_endpoint, params={"category": "value"})
        assert response.status_code == 200

        data = response.json()
        factors = data["data"]["factors"]

        # 验证只返回价值因子
        for factor in factors:
            assert factor["category"] == "value", \
                f"Should only return value factors, got {factor['category']}"

    def test_factors_status_filter(self):
        """
        测试: 按状态过滤因子

        默认只返回active因子，可选择包含其他状态
        """
        # 默认只返回活跃因子
        response = requests.get(self.factors_endpoint)
        assert response.status_code == 200

        data = response.json()
        factors = data["data"]["factors"]

        # 验证所有因子都是活跃的
        for factor in factors:
            assert factor["status"] == "active", \
                f"Should only return active factors by default, got {factor['status']}"

        # 测试包含所有状态
        response = requests.get(self.factors_endpoint, params={"status": "all"})
        assert response.status_code == 200

        data = response.json()
        all_factors = data["data"]["factors"]

        # 现在应该包含不同状态的因子
        statuses = {factor["status"] for factor in all_factors}
        assert len(statuses) >= 1  # 至少有一种状态

    def test_factors_search_functionality(self):
        """
        测试: 搜索功能

        支持按因子名称或ID搜索
        """
        # 按名称搜索
        response = requests.get(self.factors_endpoint, params={"search": "动量"})
        assert response.status_code == 200

        data = response.json()
        factors = data["data"]["factors"]

        # 验证搜索结果包含查询关键词
        if factors:
            found_match = any("动量" in factor["name"] or "动量" in factor["economic_logic"]
                            for factor in factors)
            assert found_match, "Search results should contain the search term"

        # 按ID前缀搜索
        response = requests.get(self.factors_endpoint, params={"search": "momentum"})
        assert response.status_code == 200

        data = response.json()
        factors = data["data"]["factors"]

        if factors:
            found_match = any("momentum" in factor["id"].lower()
                            for factor in factors)
            assert found_match, "Search results should contain the search term in ID"

    def test_factors_sorting(self):
        """
        测试: 排序功能

        支持按不同字段排序
        """
        # 测试按创建时间排序
        response = requests.get(self.factors_endpoint, params={
            "sort_by": "created_at",
            "sort_order": "desc",
            "page_size": 10
        })
        assert response.status_code == 200

        data = response.json()
        factors = data["data"]["factors"]

        # 验证按创建时间降序排列
        if len(factors) > 1:
            for i in range(1, len(factors)):
                time1 = datetime.fromisoformat(factors[i-1]["created_at"].replace("Z", "+00:00"))
                time2 = datetime.fromisoformat(factors[i]["created_at"].replace("Z", "+00:00"))
                assert time1 >= time2, "Factors should be sorted by created_at in descending order"

        # 测试按名称排序
        response = requests.get(self.factors_endpoint, params={
            "sort_by": "name",
            "sort_order": "asc",
            "page_size": 10
        })
        assert response.status_code == 200

        data = response.json()
        factors = data["data"]["factors"]

        # 验证按名称升序排列
        if len(factors) > 1:
            for i in range(1, len(factors)):
                assert factors[i-1]["name"] <= factors[i]["name"], \
                    "Factors should be sorted by name in ascending order"

    def test_factors_performance_filter(self):
        """
        测试: 按表现指标过滤因子

        支持按IC、IR、夏普比率等指标过滤
        """
        # 测试高IC因子过滤
        response = requests.get(self.factors_endpoint, params={
            "min_ic": 0.05,
            "include_performance": True
        })
        assert response.status_code == 200

        data = response.json()
        factors = data["data"]["factors"]

        # 验证返回的因子都有表现指标
        for factor in factors:
            if "performance_metrics" in factor:
                ic_mean = factor["performance_metrics"].get("ic_mean")
                if ic_mean is not None:
                    assert ic_mean >= 0.05, f"Factor IC should be >= 0.05, got {ic_mean}"

    def test_factors_hikyuu_dependencies_validation(self):
        """
        测试: Hikyuu依赖验证

        验证因子的Hikyuu依赖正确性
        """
        response = requests.get(self.factors_endpoint)
        assert response.status_code == 200

        data = response.json()
        factors = data["data"]["factors"]

        # 常见的Hikyuu指标
        valid_hikyuu_indicators = [
            "CLOSE", "OPEN", "HIGH", "LOW", "VOLUME", "AMOUNT",
            "MA", "EMA", "SMA", "MACD", "RSI", "BOLL", "KDJ",
            "ATR", "CCI", "WR", "BIAS", "ROC", "DMI", "TRIX"
        ]

        for factor in factors:
            dependencies = factor["hikyuu_dependencies"]
            for dep in dependencies:
                # 验证依赖项是有效的Hikyuu指标或基础数据
                # 注意：这里可能需要根据实际的Hikyuu API调整
                assert isinstance(dep, str), f"Hikyuu dependency should be string: {dep}"
                assert len(dep) > 0, "Hikyuu dependency should not be empty"

    def test_invalid_parameters_handling(self):
        """
        测试: 无效参数处理

        验证各种无效参数的错误处理
        """
        invalid_params_tests = [
            # 无效页码
            ({"page": 0}, 400),
            ({"page": -1}, 400),
            # 无效页面大小
            ({"page_size": 0}, 400),
            ({"page_size": 1001}, 400),  # 假设最大1000
            # 无效类别
            ({"category": "invalid_category"}, 400),
            # 无效状态
            ({"status": "invalid_status"}, 400),
            # 无效排序字段
            ({"sort_by": "invalid_field"}, 400),
            # 无效排序顺序
            ({"sort_order": "invalid"}, 400),
        ]

        for params, expected_status in invalid_params_tests:
            response = requests.get(self.factors_endpoint, params=params)
            assert response.status_code == expected_status, \
                f"Invalid params {params} should return {expected_status}, got {response.status_code}"

    def test_response_time_performance(self):
        """
        测试: API响应时间应该 < 200ms (性能要求)
        """
        response = requests.get(self.factors_endpoint, params={"page_size": 50})
        assert response.status_code == 200

        # 检查响应时间
        assert_response_time(response, max_time_ms=200)

    def test_factors_version_management(self):
        """
        测试: 因子版本管理

        验证因子版本信息的正确性
        """
        response = requests.get(self.factors_endpoint)
        assert response.status_code == 200

        data = response.json()
        factors = data["data"]["factors"]

        for factor in factors:
            version = factor["version"]
            # 验证版本格式（语义化版本）
            assert isinstance(version, str)
            version_parts = version.split(".")
            assert len(version_parts) >= 2, f"Version should have at least major.minor: {version}"

            # 验证版本部分都是数字
            for part in version_parts:
                assert part.isdigit(), f"Version parts should be numeric: {version}"

    def test_factors_metadata_completeness(self):
        """
        测试: 因子元数据完整性

        验证因子必需的元数据字段都存在且有效
        """
        response = requests.get(self.factors_endpoint)
        assert response.status_code == 200

        data = response.json()
        factors = data["data"]["factors"]

        for factor in factors:
            # 验证必需字段不为空
            assert factor["name"].strip(), "Factor name should not be empty"
            assert factor["hikyuu_formula"].strip(), "Hikyuu formula should not be empty"
            assert factor["economic_logic"].strip(), "Economic logic should not be empty"
            assert factor["created_by"].strip(), "Created by should not be empty"

            # 验证创建时间格式
            try:
                datetime.fromisoformat(factor["created_at"].replace("Z", "+00:00"))
            except ValueError:
                pytest.fail(f"Invalid created_at format: {factor['created_at']}")

            # 验证参数字典
            assert isinstance(factor["parameters"], dict)

    def test_content_type_and_headers(self):
        """
        测试: Content-Type和响应头
        """
        response = requests.get(self.factors_endpoint)
        assert response.status_code == 200
        assert response.headers.get("Content-Type") == "application/json"


class TestFactorsListFiltering:
    """
    因子列表高级过滤测试
    """

    def setup_method(self):
        self.base_url = "http://localhost:8002"
        self.factors_endpoint = f"{self.base_url}/api/v1/factors"

    def test_multiple_filter_combination(self):
        """
        测试: 多重过滤条件组合

        验证多个过滤条件可以同时使用
        """
        params = {
            "category": "momentum",
            "status": "active",
            "min_ic": 0.03,
            "page_size": 20
        }

        response = requests.get(self.factors_endpoint, params=params)
        assert response.status_code == 200

        data = response.json()
        factors = data["data"]["factors"]

        # 验证所有过滤条件都被满足
        for factor in factors:
            assert factor["category"] == "momentum"
            assert factor["status"] == "active"
            if "performance_metrics" in factor:
                ic_mean = factor["performance_metrics"].get("ic_mean")
                if ic_mean is not None:
                    assert ic_mean >= 0.03

    def test_empty_result_handling(self):
        """
        测试: 空结果处理

        当过滤条件导致无结果时的处理
        """
        # 使用不可能的过滤条件
        params = {
            "category": "momentum",
            "search": "不存在的因子名称12345",
            "min_ic": 0.99  # 不太可能有这么高的IC
        }

        response = requests.get(self.factors_endpoint, params=params)
        assert response.status_code == 200

        data = response.json()
        factors_data = data["data"]

        # 验证空结果的正确处理
        assert factors_data["factors"] == []
        assert factors_data["filtered_count"] == 0
        assert factors_data["page"] == 1


# 这个测试文件应该运行失败，因为端点尚未实现
# 这是TDD的正确流程：先写测试，测试失败，然后实现功能使测试通过