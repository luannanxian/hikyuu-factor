"""
因子计算API契约测试 - Factors Calculate API Contract Tests

测试 POST /api/v1/factors/{id}/calculate 端点
根据 factor_calculation_api.yaml 合约规范，验证因子计算功能 (FR-008)
"""
import pytest
import requests
from datetime import datetime, date
from tests.utils import assert_json_schema, assert_response_time


class TestFactorsCalculateContract:
    """
    因子计算API契约测试

    这些测试必须在实现前编写并失败，遵循TDD原则
    """

    def setup_method(self):
        """测试setup - 配置测试环境"""
        self.base_url = "http://localhost:8002"  # Factor Calculation Agent端口
        self.test_factor_id = "momentum_20d_test"
        self.calculate_endpoint = f"{self.base_url}/api/v1/factors/{self.test_factor_id}/calculate"

    def test_factor_calculate_endpoint_exists(self):
        """
        测试: POST /api/v1/factors/{id}/calculate 端点存在

        期望: 端点应该存在且处理因子计算请求
        当前状态: 应该失败 (端点尚未实现)
        """
        calculate_request = {
            "stock_universe": ["sh000001", "sz000002"],
            "date_range": {
                "start_date": "2024-01-01",
                "end_date": "2024-12-31"
            }
        }

        response = requests.post(self.calculate_endpoint, json=calculate_request)

        # 这个测试现在应该失败，因为端点尚未实现
        assert response.status_code in [200, 202], f"Factor calculate endpoint should exist, got {response.status_code}"

    def test_factor_calculate_request_schema(self):
        """
        测试: 因子计算请求schema验证

        有效请求格式:
        {
          "stock_universe": ["sh000001", "sz000002", "sh600519"],
          "date_range": {
            "start_date": "2024-01-01",
            "end_date": "2024-12-31"
          },
          "optimization_config": {  // 可选，平台优化配置
            "process_count": 8,
            "low_precision_mode": true,
            "memory_limit_gb": 16.0
          },
          "calculation_mode": "batch",  // 可选: "batch", "streaming"
          "output_format": "dataframe",  // 可选: "dataframe", "json", "csv"
          "cache_results": true  // 可选
        }
        """
        valid_requests = [
            # 基本请求
            {
                "stock_universe": ["sh000001", "sz000002"],
                "date_range": {
                    "start_date": "2024-01-01",
                    "end_date": "2024-12-31"
                }
            },
            # 完整请求
            {
                "stock_universe": ["sh000001", "sz000002", "sh600519", "sz000858"],
                "date_range": {
                    "start_date": "2023-01-01",
                    "end_date": "2024-12-31"
                },
                "optimization_config": {
                    "process_count": 8,
                    "low_precision_mode": True,
                    "memory_limit_gb": 16.0
                },
                "calculation_mode": "batch",
                "output_format": "dataframe",
                "cache_results": True
            },
            # 流式计算
            {
                "stock_universe": ["sh000001"],
                "date_range": {
                    "start_date": "2024-11-01",
                    "end_date": "2024-12-31"
                },
                "calculation_mode": "streaming",
                "output_format": "json"
            }
        ]

        for request_data in valid_requests:
            response = requests.post(self.calculate_endpoint, json=request_data)
            assert response.status_code in [200, 202], \
                f"Valid request should be accepted: {request_data}"

    def test_factor_calculate_response_schema(self):
        """
        测试: 因子计算响应schema验证

        期望响应格式 (异步任务):
        {
          "status": "success",
          "task_id": "calc_momentum_20d_20250115_123456",
          "message": "Factor calculation task started",
          "estimated_completion": "2025-01-15T11:05:00Z",
          "data": {
            "factor_id": "momentum_20d_test",
            "stock_count": 4500,
            "date_range": {
              "start_date": "2024-01-01",
              "end_date": "2024-12-31"
            },
            "total_calculations": 1125000,  // stock_count * trading_days
            "optimization_used": {
              "platform_type": "apple_silicon",
              "process_count": 8,
              "low_precision_mode": true
            },
            "estimated_duration_minutes": 25
          }
        }
        """
        calculate_request = {
            "stock_universe": ["sh000001", "sz000002"],
            "date_range": {
                "start_date": "2024-01-01",
                "end_date": "2024-12-31"
            }
        }

        response = requests.post(self.calculate_endpoint, json=calculate_request)
        assert response.status_code in [200, 202]

        data = response.json()

        # 验证顶级响应结构
        required_fields = ["status", "task_id", "message", "data"]
        for field in required_fields:
            assert field in data, f"Missing required field: {field}"

        assert data["status"] == "success"
        assert isinstance(data["task_id"], str)
        assert len(data["task_id"]) > 0

        # 验证任务数据
        task_data = data["data"]
        task_required_fields = [
            "factor_id", "stock_count", "date_range", "total_calculations"
        ]
        for field in task_required_fields:
            assert field in task_data, f"Missing task data field: {field}"

        # 验证数据类型
        assert task_data["factor_id"] == self.test_factor_id
        assert isinstance(task_data["stock_count"], int)
        assert isinstance(task_data["total_calculations"], int)
        assert task_data["stock_count"] > 0
        assert task_data["total_calculations"] > 0

    def test_stock_universe_validation(self):
        """
        测试: 股票池参数验证

        验证股票代码格式和数量限制
        """
        base_request = {
            "date_range": {
                "start_date": "2024-01-01",
                "end_date": "2024-12-31"
            }
        }

        # 测试有效股票池
        valid_stock_universes = [
            ["sh000001"],  # 单只股票
            ["sh000001", "sz000002"],  # 少量股票
            ["sh000001", "sz000002", "sh600519", "sz000858", "sh600036"],  # 多只股票
            # 可以测试更大的股票池，但要注意测试时间
        ]

        for stock_universe in valid_stock_universes:
            request_data = base_request.copy()
            request_data["stock_universe"] = stock_universe

            response = requests.post(self.calculate_endpoint, json=request_data)
            assert response.status_code in [200, 202], \
                f"Valid stock universe should be accepted: {len(stock_universe)} stocks"

        # 测试无效股票池
        invalid_stock_universes = [
            [],  # 空股票池
            ["invalid_code"],  # 无效股票代码
            ["sh00001"],  # 长度错误
            ["ab000001"],  # 无效前缀
            ["sh000001", "duplicate", "sh000001"],  # 重复股票
        ]

        for stock_universe in invalid_stock_universes:
            request_data = base_request.copy()
            request_data["stock_universe"] = stock_universe

            response = requests.post(self.calculate_endpoint, json=request_data)
            assert response.status_code == 400, \
                f"Invalid stock universe should be rejected: {stock_universe}"

    def test_date_range_validation(self):
        """
        测试: 日期范围参数验证

        验证日期格式和范围的合理性
        """
        base_request = {
            "stock_universe": ["sh000001", "sz000002"]
        }

        # 测试有效日期范围
        valid_date_ranges = [
            # 短期范围
            {
                "start_date": "2024-12-01",
                "end_date": "2024-12-31"
            },
            # 中期范围
            {
                "start_date": "2024-01-01",
                "end_date": "2024-12-31"
            },
            # 长期范围
            {
                "start_date": "2020-01-01",
                "end_date": "2024-12-31"
            }
        ]

        for date_range in valid_date_ranges:
            request_data = base_request.copy()
            request_data["date_range"] = date_range

            response = requests.post(self.calculate_endpoint, json=request_data)
            assert response.status_code in [200, 202], \
                f"Valid date range should be accepted: {date_range}"

        # 测试无效日期范围
        invalid_date_ranges = [
            # 无效日期格式
            {
                "start_date": "2024/01/01",
                "end_date": "2024-12-31"
            },
            # 结束日期早于开始日期
            {
                "start_date": "2024-12-31",
                "end_date": "2024-01-01"
            },
            # 无效日期
            {
                "start_date": "2024-13-01",
                "end_date": "2024-12-31"
            },
            # 未来日期（如果不允许）
            {
                "start_date": "2030-01-01",
                "end_date": "2030-12-31"
            }
        ]

        for date_range in invalid_date_ranges:
            request_data = base_request.copy()
            request_data["date_range"] = date_range

            response = requests.post(self.calculate_endpoint, json=request_data)
            assert response.status_code == 400, \
                f"Invalid date range should be rejected: {date_range}"

    def test_optimization_config_validation(self):
        """
        测试: 优化配置参数验证

        验证平台优化配置的正确性
        """
        base_request = {
            "stock_universe": ["sh000001", "sz000002"],
            "date_range": {
                "start_date": "2024-01-01",
                "end_date": "2024-12-31"
            }
        }

        # 测试有效优化配置
        valid_optimization_configs = [
            # Apple Silicon优化
            {
                "process_count": 8,
                "low_precision_mode": True,
                "memory_limit_gb": 16.0
            },
            # x86_64优化
            {
                "process_count": 16,
                "low_precision_mode": False,
                "memory_limit_gb": 32.0
            },
            # 最小配置
            {
                "process_count": 4
            }
        ]

        for optimization_config in valid_optimization_configs:
            request_data = base_request.copy()
            request_data["optimization_config"] = optimization_config

            response = requests.post(self.calculate_endpoint, json=request_data)
            assert response.status_code in [200, 202], \
                f"Valid optimization config should be accepted: {optimization_config}"

        # 测试无效优化配置
        invalid_optimization_configs = [
            {"process_count": 0},  # 无效进程数
            {"process_count": -1},  # 负值
            {"process_count": 100},  # 过大值
            {"memory_limit_gb": 0},  # 无效内存限制
            {"memory_limit_gb": -1},  # 负值
        ]

        for optimization_config in invalid_optimization_configs:
            request_data = base_request.copy()
            request_data["optimization_config"] = optimization_config

            response = requests.post(self.calculate_endpoint, json=request_data)
            assert response.status_code == 400, \
                f"Invalid optimization config should be rejected: {optimization_config}"

    def test_calculation_mode_validation(self):
        """
        测试: 计算模式验证

        支持批量和流式两种计算模式
        """
        base_request = {
            "stock_universe": ["sh000001", "sz000002"],
            "date_range": {
                "start_date": "2024-01-01",
                "end_date": "2024-12-31"
            }
        }

        # 测试有效计算模式
        valid_modes = ["batch", "streaming"]

        for mode in valid_modes:
            request_data = base_request.copy()
            request_data["calculation_mode"] = mode

            response = requests.post(self.calculate_endpoint, json=request_data)
            assert response.status_code in [200, 202], \
                f"Valid calculation mode should be accepted: {mode}"

        # 测试无效计算模式
        invalid_modes = ["invalid", "BATCH", "parallel", ""]

        for mode in invalid_modes:
            request_data = base_request.copy()
            request_data["calculation_mode"] = mode

            response = requests.post(self.calculate_endpoint, json=request_data)
            assert response.status_code == 400, \
                f"Invalid calculation mode should be rejected: {mode}"

    def test_output_format_validation(self):
        """
        测试: 输出格式验证

        支持多种输出格式
        """
        base_request = {
            "stock_universe": ["sh000001", "sz000002"],
            "date_range": {
                "start_date": "2024-01-01",
                "end_date": "2024-12-31"
            }
        }

        # 测试有效输出格式
        valid_formats = ["dataframe", "json", "csv"]

        for format_type in valid_formats:
            request_data = base_request.copy()
            request_data["output_format"] = format_type

            response = requests.post(self.calculate_endpoint, json=request_data)
            assert response.status_code in [200, 202], \
                f"Valid output format should be accepted: {format_type}"

        # 测试无效输出格式
        invalid_formats = ["xml", "excel", "parquet", ""]

        for format_type in invalid_formats:
            request_data = base_request.copy()
            request_data["output_format"] = format_type

            response = requests.post(self.calculate_endpoint, json=request_data)
            assert response.status_code == 400, \
                f"Invalid output format should be rejected: {format_type}"

    def test_large_scale_calculation_estimation(self):
        """
        测试: 大规模计算估算

        验证系统对大规模计算的处理和估算
        """
        # 模拟全市场计算
        large_scale_request = {
            "stock_universe": [f"sh{600000 + i:06d}" for i in range(100)],  # 100只股票
            "date_range": {
                "start_date": "2020-01-01",
                "end_date": "2024-12-31"
            }
        }

        response = requests.post(self.calculate_endpoint, json=large_scale_request)
        assert response.status_code in [200, 202]

        data = response.json()["data"]

        # 验证大规模计算的估算信息
        assert data["stock_count"] == 100
        assert data["total_calculations"] > 100000  # 应该是股票数 × 交易日数

        # 验证估算时间存在且合理
        if "estimated_duration_minutes" in data:
            duration = data["estimated_duration_minutes"]
            assert isinstance(duration, (int, float))
            assert duration > 0
            # 应该少于30分钟（性能要求）
            assert duration <= 30, f"Estimated duration {duration} minutes exceeds 30-minute target"

    def test_factor_id_validation(self):
        """
        测试: 因子ID验证

        验证请求中的因子ID是否存在
        """
        # 测试不存在的因子ID
        nonexistent_factor_id = "nonexistent_factor_12345"
        nonexistent_endpoint = f"{self.base_url}/api/v1/factors/{nonexistent_factor_id}/calculate"

        calculate_request = {
            "stock_universe": ["sh000001"],
            "date_range": {
                "start_date": "2024-01-01",
                "end_date": "2024-12-31"
            }
        }

        response = requests.post(nonexistent_endpoint, json=calculate_request)
        assert response.status_code == 404, "Nonexistent factor should return 404"

        # 验证错误响应格式
        if response.status_code == 404:
            data = response.json()
            assert "status" in data
            assert data["status"] == "error"

    def test_concurrent_calculation_handling(self):
        """
        测试: 并发计算处理

        验证同时提交多个计算任务的处理
        """
        calculate_request = {
            "stock_universe": ["sh000001", "sz000002"],
            "date_range": {
                "start_date": "2024-01-01",
                "end_date": "2024-12-31"
            }
        }

        # 提交多个并发请求
        responses = []
        for i in range(3):
            response = requests.post(self.calculate_endpoint, json=calculate_request)
            responses.append(response)

        # 验证所有请求都被接受
        for i, response in enumerate(responses):
            assert response.status_code in [200, 202], \
                f"Concurrent request {i} should be accepted"

        # 验证任务ID都不同
        task_ids = [resp.json()["task_id"] for resp in responses]
        assert len(set(task_ids)) == len(task_ids), "All task IDs should be unique"

    def test_missing_required_fields(self):
        """
        测试: 缺少必需字段的错误处理
        """
        # 缺少stock_universe
        request1 = {
            "date_range": {
                "start_date": "2024-01-01",
                "end_date": "2024-12-31"
            }
        }

        response1 = requests.post(self.calculate_endpoint, json=request1)
        assert response1.status_code == 400

        # 缺少date_range
        request2 = {
            "stock_universe": ["sh000001", "sz000002"]
        }

        response2 = requests.post(self.calculate_endpoint, json=request2)
        assert response2.status_code == 400

        # 缺少start_date
        request3 = {
            "stock_universe": ["sh000001"],
            "date_range": {
                "end_date": "2024-12-31"
            }
        }

        response3 = requests.post(self.calculate_endpoint, json=request3)
        assert response3.status_code == 400

    def test_response_time_performance(self):
        """
        测试: API响应时间应该 < 200ms (性能要求)

        注意：这是启动计算任务的时间，不是计算完成时间
        """
        calculate_request = {
            "stock_universe": ["sh000001", "sz000002"],
            "date_range": {
                "start_date": "2024-01-01",
                "end_date": "2024-12-31"
            }
        }

        response = requests.post(self.calculate_endpoint, json=calculate_request)
        assert response.status_code in [200, 202]

        # 检查响应时间
        assert_response_time(response, max_time_ms=200)

    def test_content_type_validation(self):
        """
        测试: Content-Type验证

        应该只接受application/json
        """
        calculate_request = {
            "stock_universe": ["sh000001"],
            "date_range": {
                "start_date": "2024-01-01",
                "end_date": "2024-12-31"
            }
        }

        # 正确的Content-Type
        response = requests.post(
            self.calculate_endpoint,
            json=calculate_request,
            headers={"Content-Type": "application/json"}
        )
        assert response.status_code in [200, 202]

        # 错误的Content-Type
        response = requests.post(
            self.calculate_endpoint,
            data=str(calculate_request),
            headers={"Content-Type": "text/plain"}
        )
        assert response.status_code == 415


class TestFactorCalculationStatusContract:
    """
    因子计算状态查询契约测试
    """

    def setup_method(self):
        self.base_url = "http://localhost:8002"
        self.test_factor_id = "momentum_20d_test"
        self.calculate_endpoint = f"{self.base_url}/api/v1/factors/{self.test_factor_id}/calculate"

    def test_calculation_status_endpoint(self):
        """
        测试: GET /api/v1/tasks/{task_id}/status 端点

        期望响应格式:
        {
          "status": "success",
          "data": {
            "task_id": "calc_momentum_20d_20250115_123456",
            "status": "pending|running|completed|failed",
            "progress": 75.5,
            "current_stage": "Calculating batch 3 of 4",
            "started_at": "2025-01-15T10:30:00Z",
            "completed_at": null,
            "optimization_metrics": {
              "platform_used": "apple_silicon",
              "process_count": 8,
              "memory_usage_gb": 12.5,
              "cpu_usage_percent": 85.2
            },
            "result_summary": {...}  // 只在completed时存在
          }
        }
        """
        # 先创建一个计算任务
        calculate_request = {
            "stock_universe": ["sh000001", "sz000002"],
            "date_range": {
                "start_date": "2024-01-01",
                "end_date": "2024-12-31"
            }
        }

        create_response = requests.post(self.calculate_endpoint, json=calculate_request)
        assert create_response.status_code in [200, 202]

        task_id = create_response.json()["task_id"]

        # 查询任务状态
        status_endpoint = f"{self.base_url}/api/v1/tasks/{task_id}/status"
        status_response = requests.get(status_endpoint)

        assert status_response.status_code == 200

        data = status_response.json()
        assert "status" in data
        assert "data" in data

        task_data = data["data"]
        required_fields = ["task_id", "status", "progress", "started_at"]
        for field in required_fields:
            assert field in task_data, f"Missing field: {field}"

        assert task_data["task_id"] == task_id
        assert task_data["status"] in ["pending", "running", "completed", "failed"]
        assert isinstance(task_data["progress"], (int, float))
        assert 0 <= task_data["progress"] <= 100


# 这个测试文件应该运行失败，因为端点尚未实现
# 这是TDD的正确流程：先写测试，测试失败，然后实现功能使测试通过