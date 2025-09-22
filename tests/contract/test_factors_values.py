"""
因子值获取API契约测试 - Factors Values API Contract Tests

测试 GET /api/v1/factors/{id}/values 端点
根据 factor_calculation_api.yaml 合约规范，验证因子值查询功能
"""
import pytest
import requests
from datetime import datetime, date, timedelta
from tests.utils import assert_json_schema, assert_response_time


class TestFactorsValuesContract:
    """
    因子值获取API契约测试

    这些测试必须在实现前编写并失败，遵循TDD原则
    """

    def setup_method(self):
        """测试setup - 配置测试环境"""
        self.base_url = "http://localhost:8002"  # Factor Calculation Agent端口
        self.test_factor_id = "momentum_20d_test"
        self.values_endpoint = f"{self.base_url}/api/v1/factors/{self.test_factor_id}/values"

    def test_factor_values_endpoint_exists(self):
        """
        测试: GET /api/v1/factors/{id}/values 端点存在

        期望: 端点应该存在且返回因子值数据
        当前状态: 应该失败 (端点尚未实现)
        """
        response = requests.get(self.values_endpoint)

        # 这个测试现在应该失败，因为端点尚未实现
        assert response.status_code == 200, f"Factor values endpoint should exist, got {response.status_code}"

    def test_factor_values_response_schema(self):
        """
        测试: 因子值响应schema验证

        期望返回格式:
        {
          "status": "success",
          "data": {
            "factor_id": "momentum_20d_test",
            "factor_name": "20日动量因子",
            "values": [
              {
                "datetime": "2024-01-02",
                "stock_code": "sh000001",
                "factor_value": 0.0523,
                "is_valid": true,
                "data_quality_score": 1.0,
                "calculation_metadata": {
                  "hikyuu_context": "MA(CLOSE(), 5) / MA(CLOSE(), 20) - 1",
                  "dependencies_available": true,
                  "calculation_time": "2025-01-15T10:30:00Z"
                }
              }
            ],
            "query_info": {
              "stock_codes": ["sh000001", "sz000002"],
              "date_range": {
                "start_date": "2024-01-01",
                "end_date": "2024-12-31"
              },
              "total_records": 500,
              "valid_records": 485,
              "data_coverage": 0.97
            },
            "performance_metrics": {
              "query_time_ms": 125,
              "data_source": "cache",
              "cache_hit_rate": 0.95
            }
          }
        }
        """
        response = requests.get(self.values_endpoint)
        assert response.status_code == 200

        data = response.json()

        # 验证顶级响应结构
        assert "status" in data
        assert "data" in data
        assert data["status"] == "success"

        values_data = data["data"]

        # 验证数据结构
        required_fields = ["factor_id", "factor_name", "values", "query_info"]
        for field in required_fields:
            assert field in values_data, f"Missing required field: {field}"

        # 验证因子信息
        assert values_data["factor_id"] == self.test_factor_id
        assert isinstance(values_data["factor_name"], str)

        # 验证因子值列表
        values = values_data["values"]
        assert isinstance(values, list)

        # 验证单个因子值记录结构
        if values:
            value_record = values[0]
            value_required_fields = [
                "datetime", "stock_code", "factor_value", "is_valid"
            ]
            for field in value_required_fields:
                assert field in value_record, f"Missing value field: {field}"

            # 验证字段类型
            assert isinstance(value_record["datetime"], str)
            assert isinstance(value_record["stock_code"], str)
            assert isinstance(value_record["factor_value"], (int, float, type(None)))
            assert isinstance(value_record["is_valid"], bool)

        # 验证查询信息
        query_info = values_data["query_info"]
        query_required_fields = ["total_records", "valid_records", "data_coverage"]
        for field in query_required_fields:
            assert field in query_info, f"Missing query info field: {field}"

    def test_stock_codes_parameter(self):
        """
        测试: 股票代码参数过滤

        支持按股票代码过滤因子值
        """
        # 测试单只股票
        params = {"stock_codes": "sh000001"}
        response = requests.get(self.values_endpoint, params=params)
        assert response.status_code == 200

        data = response.json()["data"]
        values = data["values"]

        # 验证只返回指定股票的数据
        for value in values:
            assert value["stock_code"] == "sh000001", \
                f"Should only return data for sh000001, got {value['stock_code']}"

        # 测试多只股票
        params = {"stock_codes": "sh000001,sz000002,sh600519"}
        response = requests.get(self.values_endpoint, params=params)
        assert response.status_code == 200

        data = response.json()["data"]
        values = data["values"]

        # 验证只返回指定股票的数据
        expected_stocks = {"sh000001", "sz000002", "sh600519"}
        actual_stocks = {value["stock_code"] for value in values}
        assert actual_stocks.issubset(expected_stocks), \
            f"Returned stocks {actual_stocks} should be subset of {expected_stocks}"

    def test_date_range_parameters(self):
        """
        测试: 日期范围参数过滤

        支持按日期范围过滤因子值
        """
        # 测试开始日期
        params = {"start_date": "2024-06-01"}
        response = requests.get(self.values_endpoint, params=params)
        assert response.status_code == 200

        data = response.json()["data"]
        values = data["values"]

        # 验证日期范围
        for value in values:
            value_date = datetime.fromisoformat(value["datetime"]).date()
            assert value_date >= date(2024, 6, 1), \
                f"Date {value_date} should be >= 2024-06-01"

        # 测试结束日期
        params = {"end_date": "2024-06-30"}
        response = requests.get(self.values_endpoint, params=params)
        assert response.status_code == 200

        data = response.json()["data"]
        values = data["values"]

        # 验证日期范围
        for value in values:
            value_date = datetime.fromisoformat(value["datetime"]).date()
            assert value_date <= date(2024, 6, 30), \
                f"Date {value_date} should be <= 2024-06-30"

        # 测试日期范围组合
        params = {
            "start_date": "2024-01-01",
            "end_date": "2024-03-31"
        }
        response = requests.get(self.values_endpoint, params=params)
        assert response.status_code == 200

        data = response.json()["data"]
        values = data["values"]

        # 验证日期范围
        for value in values:
            value_date = datetime.fromisoformat(value["datetime"]).date()
            assert date(2024, 1, 1) <= value_date <= date(2024, 3, 31), \
                f"Date {value_date} should be in range [2024-01-01, 2024-03-31]"

    def test_query_range_parameter(self):
        """
        测试: Hikyuu查询范围参数

        支持Hikyuu风格的查询范围语法
        """
        # 测试最近N个交易日
        params = {"query_range": "Query(-100)"}  # 最近100个交易日
        response = requests.get(self.values_endpoint, params=params)
        assert response.status_code == 200

        data = response.json()["data"]
        values = data["values"]

        # 验证返回的数据量合理
        if values:
            # 应该有数据，但具体数量取决于股票数量和实际交易日
            assert len(values) > 0

        # 测试特定日期范围的Hikyuu查询
        params = {"query_range": "Query(Datetime(2024, 1, 1), Datetime(2024, 12, 31))"}
        response = requests.get(self.values_endpoint, params=params)
        assert response.status_code == 200

    def test_only_valid_parameter(self):
        """
        测试: 只返回有效值参数

        支持过滤掉无效的因子值
        """
        # 测试包含所有值
        params = {"only_valid": "false"}
        response = requests.get(self.values_endpoint, params=params)
        assert response.status_code == 200

        data_all = response.json()["data"]
        values_all = data_all["values"]

        # 测试只返回有效值
        params = {"only_valid": "true"}
        response = requests.get(self.values_endpoint, params=params)
        assert response.status_code == 200

        data_valid = response.json()["data"]
        values_valid = data_valid["values"]

        # 验证有效值过滤
        for value in values_valid:
            assert value["is_valid"] is True, "Should only return valid values"

        # 有效值数量应该 <= 总数量
        assert len(values_valid) <= len(values_all)

    def test_output_format_parameter(self):
        """
        测试: 输出格式参数

        支持不同的输出格式
        """
        # 测试JSON格式（默认）
        params = {"output_format": "json"}
        response = requests.get(self.values_endpoint, params=params)
        assert response.status_code == 200
        assert response.headers.get("Content-Type") == "application/json"

        # 测试CSV格式
        params = {"output_format": "csv"}
        response = requests.get(self.values_endpoint, params=params)
        assert response.status_code == 200
        # CSV应该返回text/csv或application/csv
        content_type = response.headers.get("Content-Type", "")
        assert "csv" in content_type.lower()

        # 验证CSV内容格式
        csv_content = response.text
        lines = csv_content.strip().split('\n')
        if len(lines) > 0:
            # 第一行应该是header
            header = lines[0]
            assert "datetime" in header
            assert "stock_code" in header
            assert "factor_value" in header

    def test_pagination_parameters(self):
        """
        测试: 分页参数

        支持大数据集的分页查询
        """
        # 测试第一页
        params = {"page": 1, "page_size": 50}
        response = requests.get(self.values_endpoint, params=params)
        assert response.status_code == 200

        data = response.json()["data"]
        values = data["values"]
        query_info = data["query_info"]

        # 验证分页信息
        assert len(values) <= 50
        assert "total_records" in query_info
        assert "page" in query_info or "page" in data
        assert "page_size" in query_info or "page_size" in data

        # 测试第二页
        params = {"page": 2, "page_size": 50}
        response = requests.get(self.values_endpoint, params=params)
        assert response.status_code == 200

        data2 = response.json()["data"]
        values2 = data2["values"]

        # 验证不同页的数据不重复
        if values and values2:
            page1_records = {(v["datetime"], v["stock_code"]) for v in values}
            page2_records = {(v["datetime"], v["stock_code"]) for v in values2}
            overlap = page1_records.intersection(page2_records)
            assert len(overlap) == 0, "Pages should not have overlapping records"

    def test_include_metadata_parameter(self):
        """
        测试: 包含元数据参数

        控制是否返回计算元数据
        """
        # 测试包含元数据
        params = {"include_metadata": "true"}
        response = requests.get(self.values_endpoint, params=params)
        assert response.status_code == 200

        data = response.json()["data"]
        values = data["values"]

        if values:
            value = values[0]
            # 应该包含计算元数据
            assert "calculation_metadata" in value

        # 测试不包含元数据
        params = {"include_metadata": "false"}
        response = requests.get(self.values_endpoint, params=params)
        assert response.status_code == 200

        data = response.json()["data"]
        values = data["values"]

        if values:
            value = values[0]
            # 不应该包含计算元数据，或者元数据为空
            assert "calculation_metadata" not in value or value["calculation_metadata"] is None

    def test_data_quality_filtering(self):
        """
        测试: 数据质量过滤

        支持按数据质量分数过滤
        """
        # 测试最小质量分数
        params = {"min_quality_score": 0.8}
        response = requests.get(self.values_endpoint, params=params)
        assert response.status_code == 200

        data = response.json()["data"]
        values = data["values"]

        # 验证质量分数过滤
        for value in values:
            if "data_quality_score" in value:
                score = value["data_quality_score"]
                assert score >= 0.8, f"Quality score {score} should be >= 0.8"

    def test_aggregation_parameters(self):
        """
        测试: 聚合参数

        支持数据聚合选项
        """
        # 测试月度聚合
        params = {"aggregate": "monthly"}
        response = requests.get(self.values_endpoint, params=params)
        # 聚合功能可能是可选的
        assert response.status_code in [200, 400]

        if response.status_code == 200:
            data = response.json()["data"]
            values = data["values"]

            # 验证聚合结果
            if values:
                # 检查日期是否为月末或月初
                dates = [datetime.fromisoformat(v["datetime"]).date() for v in values]
                # 月度聚合后，日期应该较少且规律

    def test_factor_id_validation(self):
        """
        测试: 因子ID验证

        验证不存在的因子ID处理
        """
        # 测试不存在的因子ID
        nonexistent_factor_id = "nonexistent_factor_12345"
        nonexistent_endpoint = f"{self.base_url}/api/v1/factors/{nonexistent_factor_id}/values"

        response = requests.get(nonexistent_endpoint)
        assert response.status_code == 404

        # 验证错误响应格式
        data = response.json()
        assert "status" in data
        assert data["status"] == "error"

    def test_invalid_parameters_handling(self):
        """
        测试: 无效参数处理

        验证各种无效参数的错误处理
        """
        invalid_params_tests = [
            # 无效股票代码
            ({"stock_codes": "invalid_code"}, 400),
            # 无效日期格式
            ({"start_date": "2024/01/01"}, 400),
            ({"end_date": "invalid-date"}, 400),
            # 无效日期范围
            ({"start_date": "2024-12-31", "end_date": "2024-01-01"}, 400),
            # 无效页码
            ({"page": 0}, 400),
            ({"page": -1}, 400),
            # 无效页面大小
            ({"page_size": 0}, 400),
            ({"page_size": 10001}, 400),  # 假设最大10000
            # 无效输出格式
            ({"output_format": "invalid"}, 400),
            # 无效查询范围
            ({"query_range": "invalid_query"}, 400),
        ]

        for params, expected_status in invalid_params_tests:
            response = requests.get(self.values_endpoint, params=params)
            assert response.status_code == expected_status, \
                f"Invalid params {params} should return {expected_status}, got {response.status_code}"

    def test_data_consistency_validation(self):
        """
        测试: 数据一致性验证

        验证返回数据的业务逻辑一致性
        """
        response = requests.get(self.values_endpoint)
        assert response.status_code == 200

        data = response.json()["data"]
        values = data["values"]

        for value in values:
            # 日期格式验证
            try:
                datetime.fromisoformat(value["datetime"])
            except ValueError:
                pytest.fail(f"Invalid datetime format: {value['datetime']}")

            # 股票代码格式验证
            stock_code = value["stock_code"]
            assert len(stock_code) == 8, f"Stock code should be 8 characters: {stock_code}"
            assert stock_code.startswith(("sh", "sz")), f"Stock code should start with sh/sz: {stock_code}"

            # 因子值验证
            if value["is_valid"]:
                factor_value = value["factor_value"]
                assert factor_value is not None, "Valid records should have non-null factor values"
                assert isinstance(factor_value, (int, float)), \
                    f"Factor value should be numeric: {factor_value}"

    def test_response_time_performance(self):
        """
        测试: API响应时间应该 < 200ms (性能要求)
        """
        # 测试小量数据查询
        params = {"stock_codes": "sh000001", "query_range": "Query(-10)"}
        response = requests.get(self.values_endpoint, params=params)
        assert response.status_code == 200

        # 检查响应时间
        assert_response_time(response, max_time_ms=200)

    def test_cache_performance_indicators(self):
        """
        测试: 缓存性能指标

        验证响应中的缓存性能信息
        """
        response = requests.get(self.values_endpoint)
        assert response.status_code == 200

        data = response.json()["data"]

        # 验证性能指标存在
        if "performance_metrics" in data:
            performance = data["performance_metrics"]

            if "query_time_ms" in performance:
                query_time = performance["query_time_ms"]
                assert isinstance(query_time, (int, float))
                assert query_time > 0

            if "cache_hit_rate" in performance:
                hit_rate = performance["cache_hit_rate"]
                assert isinstance(hit_rate, (int, float))
                assert 0 <= hit_rate <= 1

    def test_empty_result_handling(self):
        """
        测试: 空结果处理

        当查询条件导致无结果时的处理
        """
        # 查询不存在的股票
        params = {"stock_codes": "sh999999"}  # 不存在的股票代码
        response = requests.get(self.values_endpoint, params=params)
        assert response.status_code == 200

        data = response.json()["data"]
        values = data["values"]

        # 验证空结果的正确处理
        assert values == []
        assert data["query_info"]["total_records"] == 0

    def test_content_type_headers(self):
        """
        测试: Content-Type和响应头
        """
        response = requests.get(self.values_endpoint)
        assert response.status_code == 200
        assert response.headers.get("Content-Type") == "application/json"

        # 测试缓存相关头
        # 因子值可能需要缓存控制
        cache_control = response.headers.get("Cache-Control")
        if cache_control:
            # 验证缓存策略合理
            assert "max-age" in cache_control or "no-cache" in cache_control


# 这个测试文件应该运行失败，因为端点尚未实现
# 这是TDD的正确流程：先写测试，测试失败，然后实现功能使测试通过