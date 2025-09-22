"""
数据质量检查API契约测试 - Data Quality Check API Contract Tests

测试 POST /api/v1/data/quality/check 端点
根据 data_manager_api.yaml 合约规范，验证数据质量检查功能 (FR-007)
"""
import pytest
import requests
from datetime import datetime, date
from tests.utils import assert_json_schema, assert_response_time


class TestDataQualityContract:
    """
    数据质量检查API契约测试

    这些测试必须在实现前编写并失败，遵循TDD原则
    """

    def setup_method(self):
        """测试setup - 配置测试环境"""
        self.base_url = "http://localhost:8001"  # Data Manager Agent端口
        self.quality_endpoint = f"{self.base_url}/api/v1/data/quality/check"

    def test_quality_check_endpoint_exists(self):
        """
        测试: POST /api/v1/data/quality/check 端点存在

        期望: 端点应该存在且处理质量检查请求
        当前状态: 应该失败 (端点尚未实现)
        """
        check_request = {
            "check_date": "2025-01-15",
            "check_types": ["price_consistency"],
            "threshold_config": {}
        }

        response = requests.post(self.quality_endpoint, json=check_request)

        # 这个测试现在应该失败，因为端点尚未实现
        assert response.status_code in [200, 202], f"Quality check endpoint should exist, got {response.status_code}"

    def test_quality_check_request_schema(self):
        """
        测试: 数据质量检查请求schema验证

        有效请求格式:
        {
          "check_date": "2025-01-15",
          "check_types": ["price_consistency", "volume_validity", "change_threshold"],
          "stock_codes": ["sh000001", "sz000002"],  // 可选
          "threshold_config": {
            "max_price_change_pct": 0.3,
            "min_volume": 100,
            "max_volume": 1000000000
          },
          "detailed_report": true  // 可选
        }
        """
        valid_requests = [
            # 基本请求
            {
                "check_date": "2025-01-15",
                "check_types": ["price_consistency"],
                "threshold_config": {}
            },
            # 完整请求
            {
                "check_date": "2025-01-15",
                "check_types": ["price_consistency", "volume_validity", "change_threshold"],
                "stock_codes": ["sh000001", "sz000002"],
                "threshold_config": {
                    "max_price_change_pct": 0.3,
                    "min_volume": 100,
                    "max_volume": 1000000000
                },
                "detailed_report": True
            },
            # 单一检查类型
            {
                "check_date": "2025-01-14",
                "check_types": ["volume_validity"],
                "threshold_config": {
                    "min_volume": 1000
                }
            }
        ]

        for request_data in valid_requests:
            response = requests.post(self.quality_endpoint, json=request_data)
            assert response.status_code in [200, 202], \
                f"Valid request should be accepted: {request_data}"

    def test_quality_check_response_schema(self):
        """
        测试: 数据质量检查响应schema验证

        期望响应格式:
        {
          "status": "success",
          "data": {
            "check_id": "quality_20250115_123456",
            "check_date": "2025-01-15",
            "total_stocks_checked": 5000,
            "issues_found": 25,
            "overall_quality_score": 99.5,
            "check_results": {
              "price_consistency": {
                "passed": 4975,
                "failed": 25,
                "issues": [
                  {
                    "stock_code": "sh600000",
                    "issue_type": "invalid_high_price",
                    "description": "High price lower than close price",
                    "severity": "high"
                  }
                ]
              },
              "volume_validity": {
                "passed": 5000,
                "failed": 0,
                "issues": []
              }
            },
            "generated_at": "2025-01-15T10:30:00Z"
          }
        }
        """
        check_request = {
            "check_date": "2025-01-15",
            "check_types": ["price_consistency", "volume_validity"],
            "threshold_config": {
                "max_price_change_pct": 0.3
            }
        }

        response = requests.post(self.quality_endpoint, json=check_request)
        assert response.status_code in [200, 202]

        data = response.json()

        # 验证顶级响应结构
        assert "status" in data
        assert "data" in data
        assert data["status"] == "success"

        check_data = data["data"]

        # 验证检查结果结构
        required_fields = [
            "check_id", "check_date", "total_stocks_checked",
            "issues_found", "overall_quality_score", "check_results", "generated_at"
        ]
        for field in required_fields:
            assert field in check_data, f"Missing required field: {field}"

        # 验证数据类型
        assert isinstance(check_data["total_stocks_checked"], int)
        assert isinstance(check_data["issues_found"], int)
        assert isinstance(check_data["overall_quality_score"], (int, float))
        assert 0 <= check_data["overall_quality_score"] <= 100

        # 验证检查结果
        check_results = check_data["check_results"]
        assert isinstance(check_results, dict)

        # 验证每个检查类型的结果
        for check_type in ["price_consistency", "volume_validity"]:
            if check_type in check_results:
                result = check_results[check_type]
                assert "passed" in result
                assert "failed" in result
                assert "issues" in result
                assert isinstance(result["passed"], int)
                assert isinstance(result["failed"], int)
                assert isinstance(result["issues"], list)

    def test_check_types_validation(self):
        """
        测试: 检查类型参数验证

        支持的检查类型: price_consistency, volume_validity, change_threshold,
                      missing_data, duplicate_records, timestamp_validity
        """
        # 测试有效检查类型
        valid_check_types = [
            ["price_consistency"],
            ["volume_validity"],
            ["change_threshold"],
            ["missing_data"],
            ["duplicate_records"],
            ["timestamp_validity"],
            ["price_consistency", "volume_validity"],
            ["price_consistency", "volume_validity", "change_threshold"]
        ]

        for check_types in valid_check_types:
            check_request = {
                "check_date": "2025-01-15",
                "check_types": check_types,
                "threshold_config": {}
            }

            response = requests.post(self.quality_endpoint, json=check_request)
            assert response.status_code in [200, 202], \
                f"Valid check types should be accepted: {check_types}"

        # 测试无效检查类型
        invalid_check_types = [
            ["invalid_check"],
            ["price_consistency", "invalid_check"],
            [],  # 空列表
            ["PRICE_CONSISTENCY"]  # 大小写敏感
        ]

        for check_types in invalid_check_types:
            check_request = {
                "check_date": "2025-01-15",
                "check_types": check_types,
                "threshold_config": {}
            }

            response = requests.post(self.quality_endpoint, json=check_request)
            assert response.status_code == 400, \
                f"Invalid check types should be rejected: {check_types}"

    def test_price_consistency_check(self):
        """
        测试: 价格一致性检查 (FR-007)

        验证价格数据的逻辑一致性：
        - high >= max(open, close, low)
        - low <= min(open, close, high)
        - close > 0
        """
        check_request = {
            "check_date": "2025-01-15",
            "check_types": ["price_consistency"],
            "threshold_config": {},
            "detailed_report": True
        }

        response = requests.post(self.quality_endpoint, json=check_request)
        assert response.status_code in [200, 202]

        data = response.json()["data"]

        # 验证价格一致性检查结果存在
        assert "price_consistency" in data["check_results"]

        price_check = data["check_results"]["price_consistency"]
        assert "passed" in price_check
        assert "failed" in price_check
        assert "issues" in price_check

        # 如果有问题，验证问题格式
        if price_check["issues"]:
            issue = price_check["issues"][0]
            required_issue_fields = ["stock_code", "issue_type", "description", "severity"]
            for field in required_issue_fields:
                assert field in issue, f"Missing issue field: {field}"

            assert issue["severity"] in ["low", "medium", "high", "critical"]

    def test_volume_validity_check(self):
        """
        测试: 成交量有效性检查

        验证成交量数据的合理性：
        - volume >= 0
        - amount >= 0
        - 成交量不应该为异常大值
        """
        check_request = {
            "check_date": "2025-01-15",
            "check_types": ["volume_validity"],
            "threshold_config": {
                "min_volume": 0,
                "max_volume": 1000000000
            }
        }

        response = requests.post(self.quality_endpoint, json=check_request)
        assert response.status_code in [200, 202]

        data = response.json()["data"]

        # 验证成交量检查结果存在
        assert "volume_validity" in data["check_results"]

        volume_check = data["check_results"]["volume_validity"]
        assert isinstance(volume_check["passed"], int)
        assert isinstance(volume_check["failed"], int)

    def test_change_threshold_check(self):
        """
        测试: 价格变动阈值检查

        验证价格变动的合理性，检测异常波动
        """
        check_request = {
            "check_date": "2025-01-15",
            "check_types": ["change_threshold"],
            "threshold_config": {
                "max_price_change_pct": 0.3  # 30%最大涨跌幅
            }
        }

        response = requests.post(self.quality_endpoint, json=check_request)
        assert response.status_code in [200, 202]

        data = response.json()["data"]

        # 验证价格变动检查结果存在
        assert "change_threshold" in data["check_results"]

        change_check = data["check_results"]["change_threshold"]
        assert "passed" in change_check
        assert "failed" in change_check

        # 如果有异常变动，验证问题描述
        if change_check["issues"]:
            issue = change_check["issues"][0]
            assert "excessive_price_change" in issue["issue_type"] or "price_change" in issue["description"]

    def test_threshold_config_validation(self):
        """
        测试: 阈值配置参数验证

        验证各种阈值参数的有效性
        """
        # 测试有效阈值配置
        valid_configs = [
            {
                "max_price_change_pct": 0.3,
                "min_volume": 100,
                "max_volume": 1000000000
            },
            {
                "max_price_change_pct": 0.1  # 更严格的阈值
            },
            {}  # 空配置，使用默认值
        ]

        for threshold_config in valid_configs:
            check_request = {
                "check_date": "2025-01-15",
                "check_types": ["price_consistency"],
                "threshold_config": threshold_config
            }

            response = requests.post(self.quality_endpoint, json=check_request)
            assert response.status_code in [200, 202], \
                f"Valid threshold config should be accepted: {threshold_config}"

        # 测试无效阈值配置
        invalid_configs = [
            {"max_price_change_pct": -0.1},  # 负值
            {"max_price_change_pct": 2.0},   # 过大值 (200%)
            {"min_volume": -1},              # 负的最小成交量
            {"max_volume": 0}                # 无效的最大成交量
        ]

        for threshold_config in invalid_configs:
            check_request = {
                "check_date": "2025-01-15",
                "check_types": ["price_consistency"],
                "threshold_config": threshold_config
            }

            response = requests.post(self.quality_endpoint, json=check_request)
            assert response.status_code == 400, \
                f"Invalid threshold config should be rejected: {threshold_config}"

    def test_stock_codes_filter(self):
        """
        测试: 指定股票代码过滤

        可以只检查特定股票的数据质量
        """
        specific_stocks = ["sh000001", "sz000002", "sh600519"]

        check_request = {
            "check_date": "2025-01-15",
            "check_types": ["price_consistency"],
            "stock_codes": specific_stocks,
            "threshold_config": {}
        }

        response = requests.post(self.quality_endpoint, json=check_request)
        assert response.status_code in [200, 202]

        data = response.json()["data"]

        # 检查的股票数量应该等于或少于指定的股票数量
        assert data["total_stocks_checked"] <= len(specific_stocks)

        # 如果有详细报告，验证只检查了指定的股票
        if "detailed_report" in data:
            for issue in data["check_results"]["price_consistency"]["issues"]:
                assert issue["stock_code"] in specific_stocks

    def test_quality_score_calculation(self):
        """
        测试: 质量评分计算

        质量评分应该基于检查结果合理计算
        """
        check_request = {
            "check_date": "2025-01-15",
            "check_types": ["price_consistency", "volume_validity"],
            "threshold_config": {}
        }

        response = requests.post(self.quality_endpoint, json=check_request)
        assert response.status_code in [200, 202]

        data = response.json()["data"]

        quality_score = data["overall_quality_score"]
        total_stocks = data["total_stocks_checked"]
        issues_found = data["issues_found"]

        # 验证质量评分合理性
        if total_stocks > 0:
            expected_min_score = ((total_stocks - issues_found) / total_stocks) * 100
            # 质量评分应该反映问题比例
            assert quality_score <= 100
            assert quality_score >= 0

            # 如果没有问题，质量评分应该很高
            if issues_found == 0:
                assert quality_score >= 99.0

    def test_missing_data_check(self):
        """
        测试: 缺失数据检查

        检测数据缺失的情况
        """
        check_request = {
            "check_date": "2025-01-15",
            "check_types": ["missing_data"],
            "threshold_config": {}
        }

        response = requests.post(self.quality_endpoint, json=check_request)
        assert response.status_code in [200, 202]

        data = response.json()["data"]

        # 验证缺失数据检查结果存在
        assert "missing_data" in data["check_results"]

        missing_check = data["check_results"]["missing_data"]
        assert "passed" in missing_check
        assert "failed" in missing_check

    def test_duplicate_records_check(self):
        """
        测试: 重复记录检查

        检测是否存在重复的数据记录
        """
        check_request = {
            "check_date": "2025-01-15",
            "check_types": ["duplicate_records"],
            "threshold_config": {}
        }

        response = requests.post(self.quality_endpoint, json=check_request)
        assert response.status_code in [200, 202]

        data = response.json()["data"]

        # 验证重复记录检查结果存在
        assert "duplicate_records" in data["check_results"]

    def test_date_parameter_validation(self):
        """
        测试: 检查日期参数验证
        """
        # 测试有效日期
        valid_dates = [
            "2025-01-15",
            "2024-12-31",
            datetime.now().date().isoformat()
        ]

        for date_str in valid_dates:
            check_request = {
                "check_date": date_str,
                "check_types": ["price_consistency"],
                "threshold_config": {}
            }

            response = requests.post(self.quality_endpoint, json=check_request)
            assert response.status_code in [200, 202], \
                f"Valid date should be accepted: {date_str}"

        # 测试无效日期
        invalid_dates = [
            "2025-13-01",  # 无效月份
            "invalid-date",
            "2025/01/15",  # 错误格式
            ""
        ]

        for date_str in invalid_dates:
            check_request = {
                "check_date": date_str,
                "check_types": ["price_consistency"],
                "threshold_config": {}
            }

            response = requests.post(self.quality_endpoint, json=check_request)
            assert response.status_code == 400, \
                f"Invalid date should be rejected: {date_str}"

    def test_detailed_report_option(self):
        """
        测试: 详细报告选项

        详细报告应该包含更多问题详情
        """
        # 请求详细报告
        check_request = {
            "check_date": "2025-01-15",
            "check_types": ["price_consistency"],
            "threshold_config": {},
            "detailed_report": True
        }

        response = requests.post(self.quality_endpoint, json=check_request)
        assert response.status_code in [200, 202]

        detailed_data = response.json()["data"]

        # 请求简单报告
        check_request["detailed_report"] = False
        response = requests.post(self.quality_endpoint, json=check_request)
        assert response.status_code in [200, 202]

        simple_data = response.json()["data"]

        # 详细报告应该包含更多信息
        # (具体差异取决于实现)

    def test_response_time_performance(self):
        """
        测试: API响应时间应该 < 200ms (性能要求)

        注意：这是启动检查任务的时间，不是检查完成时间
        """
        check_request = {
            "check_date": "2025-01-15",
            "check_types": ["price_consistency"],
            "threshold_config": {}
        }

        response = requests.post(self.quality_endpoint, json=check_request)
        assert response.status_code in [200, 202]

        # 检查响应时间
        assert_response_time(response, max_time_ms=200)

    def test_error_handling(self):
        """
        测试: 错误处理

        验证各种错误情况的处理
        """
        # 缺少必需字段
        invalid_requests = [
            {},  # 完全空的请求
            {"check_date": "2025-01-15"},  # 缺少check_types
            {"check_types": ["price_consistency"]},  # 缺少check_date
        ]

        for invalid_request in invalid_requests:
            response = requests.post(self.quality_endpoint, json=invalid_request)
            assert response.status_code == 400, \
                f"Invalid request should be rejected: {invalid_request}"

            # 验证错误响应格式
            data = response.json()
            assert "status" in data
            assert data["status"] == "error"


# 这个测试文件应该运行失败，因为端点尚未实现
# 这是TDD的正确流程：先写测试，测试失败，然后实现功能使测试通过