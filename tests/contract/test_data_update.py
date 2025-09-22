"""
数据更新API契约测试 - Data Update API Contract Tests

测试 POST /api/v1/data/update 端点
根据 data_manager_api.yaml 合约规范，验证数据更新功能
"""
import pytest
import requests
import time
from datetime import datetime, date
from tests.utils import assert_json_schema, assert_response_time


class TestDataUpdateContract:
    """
    数据更新API契约测试

    这些测试必须在实现前编写并失败，遵循TDD原则
    """

    def setup_method(self):
        """测试setup - 配置测试环境"""
        self.base_url = "http://localhost:8001"  # Data Manager Agent端口
        self.update_endpoint = f"{self.base_url}/api/v1/data/update"
        self.status_endpoint = f"{self.base_url}/api/v1/data/update"

    def test_data_update_endpoint_exists(self):
        """
        测试: POST /api/v1/data/update 端点存在

        期望: 端点应该存在且处理数据更新请求
        当前状态: 应该失败 (端点尚未实现)
        """
        update_request = {
            "date": "2025-01-15",
            "data_types": ["market_data"],
            "force_update": False
        }

        response = requests.post(self.update_endpoint, json=update_request)

        # 这个测试现在应该失败，因为端点尚未实现
        assert response.status_code in [200, 202], f"Data update endpoint should exist, got {response.status_code}"

    def test_data_update_request_schema(self):
        """
        测试: 数据更新请求schema验证

        有效请求格式:
        {
          "date": "2025-01-15",
          "data_types": ["market_data", "financial_data", "stock_list"],
          "force_update": false,
          "stock_codes": ["sh000001", "sz000002"],  // 可选
          "batch_size": 1000,  // 可选
          "priority": "normal"  // 可选: "low", "normal", "high"
        }
        """
        valid_requests = [
            # 基本请求
            {
                "date": "2025-01-15",
                "data_types": ["market_data"],
                "force_update": False
            },
            # 完整请求
            {
                "date": "2025-01-15",
                "data_types": ["market_data", "financial_data", "stock_list"],
                "force_update": True,
                "stock_codes": ["sh000001", "sz000002"],
                "batch_size": 500,
                "priority": "high"
            },
            # 多种数据类型
            {
                "date": "2025-01-14",
                "data_types": ["financial_data"],
                "force_update": False,
                "priority": "low"
            }
        ]

        for request_data in valid_requests:
            response = requests.post(self.update_endpoint, json=request_data)
            assert response.status_code in [200, 202], \
                f"Valid request should be accepted: {request_data}"

    def test_data_update_response_schema(self):
        """
        测试: 数据更新响应schema验证

        期望响应格式 (异步任务):
        {
          "status": "success",
          "task_id": "update_20250115_123456",
          "message": "Data update task started",
          "estimated_completion": "2025-01-15T11:00:00Z",
          "data": {
            "request_date": "2025-01-15",
            "data_types": ["market_data"],
            "total_stocks": 5000,
            "batch_size": 1000,
            "priority": "normal"
          }
        }
        """
        update_request = {
            "date": "2025-01-15",
            "data_types": ["market_data"],
            "force_update": False
        }

        response = requests.post(self.update_endpoint, json=update_request)
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
        task_required_fields = ["request_date", "data_types", "total_stocks", "batch_size", "priority"]
        for field in task_required_fields:
            assert field in task_data, f"Missing task data field: {field}"

        # 验证数据类型
        assert isinstance(task_data["total_stocks"], int)
        assert isinstance(task_data["batch_size"], int)
        assert task_data["priority"] in ["low", "normal", "high"]

    def test_data_update_task_id_format(self):
        """
        测试: 任务ID格式验证

        任务ID应该包含时间戳和唯一标识
        """
        update_request = {
            "date": "2025-01-15",
            "data_types": ["market_data"],
            "force_update": False
        }

        response = requests.post(self.update_endpoint, json=update_request)
        assert response.status_code in [200, 202]

        data = response.json()
        task_id = data["task_id"]

        # 验证任务ID格式: update_YYYYMMDD_HHMMSS 或类似
        assert "update" in task_id.lower(), "Task ID should contain 'update'"
        assert len(task_id) >= 10, "Task ID should be sufficiently long for uniqueness"

        # 验证任务ID唯一性
        response2 = requests.post(self.update_endpoint, json=update_request)
        assert response2.status_code in [200, 202]
        task_id2 = response2.json()["task_id"]

        assert task_id != task_id2, "Task IDs should be unique for different requests"

    def test_data_types_validation(self):
        """
        测试: 数据类型参数验证

        支持的数据类型: market_data, financial_data, stock_list
        """
        # 测试有效数据类型
        valid_data_types = [
            ["market_data"],
            ["financial_data"],
            ["stock_list"],
            ["market_data", "financial_data"],
            ["market_data", "financial_data", "stock_list"]
        ]

        for data_types in valid_data_types:
            update_request = {
                "date": "2025-01-15",
                "data_types": data_types,
                "force_update": False
            }

            response = requests.post(self.update_endpoint, json=update_request)
            assert response.status_code in [200, 202], \
                f"Valid data types should be accepted: {data_types}"

        # 测试无效数据类型
        invalid_data_types = [
            ["invalid_type"],
            ["market_data", "invalid_type"],
            [],  # 空列表
            ["MARKET_DATA"]  # 大小写敏感
        ]

        for data_types in invalid_data_types:
            update_request = {
                "date": "2025-01-15",
                "data_types": data_types,
                "force_update": False
            }

            response = requests.post(self.update_endpoint, json=update_request)
            assert response.status_code == 400, \
                f"Invalid data types should be rejected: {data_types}"

    def test_date_parameter_validation(self):
        """
        测试: 日期参数验证

        日期应该是有效的ISO格式，且不能是未来日期
        """
        # 测试有效日期
        valid_dates = [
            "2025-01-15",
            "2024-12-31",
            datetime.now().date().isoformat()
        ]

        for date_str in valid_dates:
            update_request = {
                "date": date_str,
                "data_types": ["market_data"],
                "force_update": False
            }

            response = requests.post(self.update_endpoint, json=update_request)
            # 注意：如果是未来日期，可能会被拒绝
            assert response.status_code in [200, 202, 400]

        # 测试无效日期格式
        invalid_dates = [
            "2025-13-01",  # 无效月份
            "2025-01-32",  # 无效日期
            "invalid-date",
            "2025/01/15",  # 错误格式
            ""
        ]

        for date_str in invalid_dates:
            update_request = {
                "date": date_str,
                "data_types": ["market_data"],
                "force_update": False
            }

            response = requests.post(self.update_endpoint, json=update_request)
            assert response.status_code == 400, \
                f"Invalid date should be rejected: {date_str}"

    def test_stock_codes_parameter(self):
        """
        测试: 股票代码参数验证

        支持指定特定股票代码进行更新
        """
        # 测试有效股票代码
        valid_stock_codes = [
            ["sh000001"],
            ["sz000002"],
            ["sh000001", "sz000002", "sh600519"]
        ]

        for stock_codes in valid_stock_codes:
            update_request = {
                "date": "2025-01-15",
                "data_types": ["market_data"],
                "force_update": False,
                "stock_codes": stock_codes
            }

            response = requests.post(self.update_endpoint, json=update_request)
            assert response.status_code in [200, 202], \
                f"Valid stock codes should be accepted: {stock_codes}"

        # 测试无效股票代码
        invalid_stock_codes = [
            ["invalid"],
            ["sh00001"],  # 长度不对
            ["ab000001"],  # 无效前缀
            []  # 空列表
        ]

        for stock_codes in invalid_stock_codes:
            update_request = {
                "date": "2025-01-15",
                "data_types": ["market_data"],
                "force_update": False,
                "stock_codes": stock_codes
            }

            response = requests.post(self.update_endpoint, json=update_request)
            assert response.status_code == 400, \
                f"Invalid stock codes should be rejected: {stock_codes}"

    def test_batch_size_parameter(self):
        """
        测试: 批处理大小参数验证
        """
        # 测试有效批处理大小
        valid_batch_sizes = [100, 500, 1000, 2000]

        for batch_size in valid_batch_sizes:
            update_request = {
                "date": "2025-01-15",
                "data_types": ["market_data"],
                "force_update": False,
                "batch_size": batch_size
            }

            response = requests.post(self.update_endpoint, json=update_request)
            assert response.status_code in [200, 202], \
                f"Valid batch size should be accepted: {batch_size}"

        # 测试无效批处理大小
        invalid_batch_sizes = [0, -1, 10001]  # 假设最大10000

        for batch_size in invalid_batch_sizes:
            update_request = {
                "date": "2025-01-15",
                "data_types": ["market_data"],
                "force_update": False,
                "batch_size": batch_size
            }

            response = requests.post(self.update_endpoint, json=update_request)
            assert response.status_code == 400, \
                f"Invalid batch size should be rejected: {batch_size}"

    def test_priority_parameter(self):
        """
        测试: 优先级参数验证
        """
        valid_priorities = ["low", "normal", "high"]

        for priority in valid_priorities:
            update_request = {
                "date": "2025-01-15",
                "data_types": ["market_data"],
                "force_update": False,
                "priority": priority
            }

            response = requests.post(self.update_endpoint, json=update_request)
            assert response.status_code in [200, 202], \
                f"Valid priority should be accepted: {priority}"

        # 测试无效优先级
        invalid_priorities = ["urgent", "LOW", "invalid", ""]

        for priority in invalid_priorities:
            update_request = {
                "date": "2025-01-15",
                "data_types": ["market_data"],
                "force_update": False,
                "priority": priority
            }

            response = requests.post(self.update_endpoint, json=update_request)
            assert response.status_code == 400, \
                f"Invalid priority should be rejected: {priority}"

    def test_force_update_parameter(self):
        """
        测试: 强制更新参数

        force_update=true 应该覆盖已存在的数据
        """
        update_request_normal = {
            "date": "2025-01-15",
            "data_types": ["market_data"],
            "force_update": False
        }

        update_request_force = {
            "date": "2025-01-15",
            "data_types": ["market_data"],
            "force_update": True
        }

        # 两种请求都应该被接受
        response1 = requests.post(self.update_endpoint, json=update_request_normal)
        assert response1.status_code in [200, 202]

        response2 = requests.post(self.update_endpoint, json=update_request_force)
        assert response2.status_code in [200, 202]

        # 响应中应该反映force_update的设置
        data1 = response1.json()
        data2 = response2.json()

        # 任务ID应该不同
        assert data1["task_id"] != data2["task_id"]

    def test_duplicate_request_handling(self):
        """
        测试: 重复请求处理

        相同的数据更新请求应该有适当的处理逻辑
        """
        update_request = {
            "date": "2025-01-15",
            "data_types": ["market_data"],
            "force_update": False
        }

        # 发送第一个请求
        response1 = requests.post(self.update_endpoint, json=update_request)
        assert response1.status_code in [200, 202]
        task_id1 = response1.json()["task_id"]

        # 立即发送相同请求
        response2 = requests.post(self.update_endpoint, json=update_request)

        # 应该被接受，但可能返回不同的处理策略
        assert response2.status_code in [200, 202, 409]

        if response2.status_code != 409:  # 如果不是冲突
            task_id2 = response2.json()["task_id"]
            # 可能是相同任务ID (去重) 或不同任务ID (允许重复)
            # 这取决于具体的业务逻辑实现

    def test_missing_required_fields(self):
        """
        测试: 缺少必需字段的错误处理
        """
        # 缺少date
        request1 = {
            "data_types": ["market_data"],
            "force_update": False
        }

        response1 = requests.post(self.update_endpoint, json=request1)
        assert response1.status_code == 400

        # 缺少data_types
        request2 = {
            "date": "2025-01-15",
            "force_update": False
        }

        response2 = requests.post(self.update_endpoint, json=request2)
        assert response2.status_code == 400

        # 缺少force_update (可能有默认值)
        request3 = {
            "date": "2025-01-15",
            "data_types": ["market_data"]
        }

        response3 = requests.post(self.update_endpoint, json=request3)
        # 这个可能被接受，如果force_update有默认值
        assert response3.status_code in [200, 202, 400]

    def test_response_time_performance(self):
        """
        测试: API响应时间应该 < 200ms (性能要求)

        注意：这是启动异步任务的时间，不是任务完成时间
        """
        update_request = {
            "date": "2025-01-15",
            "data_types": ["market_data"],
            "force_update": False
        }

        response = requests.post(self.update_endpoint, json=update_request)
        assert response.status_code in [200, 202]

        # 检查响应时间
        assert_response_time(response, max_time_ms=200)

    def test_content_type_validation(self):
        """
        测试: Content-Type验证

        应该只接受application/json
        """
        update_request = {
            "date": "2025-01-15",
            "data_types": ["market_data"],
            "force_update": False
        }

        # 正确的Content-Type
        response = requests.post(
            self.update_endpoint,
            json=update_request,
            headers={"Content-Type": "application/json"}
        )
        assert response.status_code in [200, 202]

        # 错误的Content-Type
        response = requests.post(
            self.update_endpoint,
            data=str(update_request),
            headers={"Content-Type": "text/plain"}
        )
        assert response.status_code == 415


class TestDataUpdateStatusContract:
    """
    数据更新状态查询契约测试
    """

    def setup_method(self):
        self.base_url = "http://localhost:8001"
        self.update_endpoint = f"{self.base_url}/api/v1/data/update"

    def test_update_status_endpoint(self):
        """
        测试: GET /api/v1/data/update/{task_id}/status 端点

        期望响应格式:
        {
          "status": "success",
          "data": {
            "task_id": "update_20250115_123456",
            "status": "pending|running|completed|failed",
            "progress": 75.5,
            "message": "Processing batch 3 of 4",
            "started_at": "2025-01-15T10:30:00Z",
            "completed_at": null,
            "result": {...}  // 只在completed时存在
          }
        }
        """
        # 先创建一个更新任务
        update_request = {
            "date": "2025-01-15",
            "data_types": ["market_data"],
            "force_update": False
        }

        create_response = requests.post(self.update_endpoint, json=update_request)
        assert create_response.status_code in [200, 202]

        task_id = create_response.json()["task_id"]

        # 查询任务状态
        status_endpoint = f"{self.update_endpoint}/{task_id}/status"
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