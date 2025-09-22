"""
Factor Validation API Contract Tests (T021)
测试因子样本外验证API的合约 (FR-005)
"""

import pytest
import requests
from typing import Dict, Any

from tests.utils import (
    get_test_base_url,
    assert_json_schema,
    assert_response_time,
    assert_api_error_response,
    create_api_test_schema
)


class TestFactorValidationAPI:
    """因子验证API合约测试"""

    def setup_method(self):
        self.base_url = get_test_base_url("validation")  # Port 8083
        self.validation_endpoint = f"{self.base_url}/api/v1/validation/factor"

    def test_factor_validation_success_contract(self):
        """T021-1: 因子验证成功响应合约"""
        # 构造有效的验证请求
        request_data = {
            "factor_id": "momentum_20d",
            "validation_config": {
                "train_start": "2010-01-01",
                "train_end": "2016-12-31",
                "test_start": "2017-01-01",
                "test_end": "2020-12-31",
                "validation_start": "2021-01-01",
                "validation_end": "2023-12-31"
            },
            "validation_methods": ["ic_analysis", "layered_returns", "turnover_analysis"],
            "benchmark": "000300.SH",  # 沪深300基准
            "universe": "hs300",
            "frequency": "daily",
            "layers": 10,
            "long_short": True
        }

        # 定义期望的响应schema
        expected_schema = create_api_test_schema({
            "validation_id": {"type": "string", "pattern": "^val_[a-f0-9]{32}$"},
            "factor_id": {"type": "string"},
            "status": {"type": "string", "enum": ["running", "completed", "failed"]},
            "config": {
                "type": "object",
                "properties": {
                    "train_period": {"type": "object"},
                    "test_period": {"type": "object"},
                    "validation_period": {"type": "object"},
                    "methods": {"type": "array", "items": {"type": "string"}},
                    "benchmark": {"type": "string"},
                    "universe": {"type": "string"},
                    "layers": {"type": "integer", "minimum": 2, "maximum": 20}
                },
                "required": ["train_period", "test_period", "validation_period"]
            },
            "created_at": {"type": "string", "format": "date-time"},
            "estimated_duration": {"type": "integer", "minimum": 60}  # 至少1分钟
        })

        # 执行请求
        response = requests.post(
            self.validation_endpoint,
            json=request_data,
            headers={"Content-Type": "application/json"}
        )

        # 验证合约
        assert response.status_code == 202  # Accepted异步处理
        assert_response_time(response, 200)
        assert_json_schema(response.json(), expected_schema)

        # 验证业务逻辑
        data = response.json()
        assert data["factor_id"] == request_data["factor_id"]
        assert data["status"] == "running"
        assert data["validation_id"].startswith("val_")
        assert data["estimated_duration"] >= 60

    def test_factor_validation_parameter_validation(self):
        """T021-2: 因子验证参数校验合约"""
        # 测试缺失必要参数
        invalid_requests = [
            # 缺失factor_id
            {
                "validation_config": {
                    "train_start": "2010-01-01",
                    "train_end": "2016-12-31"
                }
            },
            # 无效日期范围 - 训练期晚于测试期
            {
                "factor_id": "momentum_20d",
                "validation_config": {
                    "train_start": "2018-01-01",
                    "train_end": "2020-12-31",
                    "test_start": "2017-01-01",
                    "test_end": "2019-12-31"
                }
            },
            # 无效的验证方法
            {
                "factor_id": "momentum_20d",
                "validation_config": {
                    "train_start": "2010-01-01",
                    "train_end": "2016-12-31",
                    "test_start": "2017-01-01",
                    "test_end": "2020-12-31"
                },
                "validation_methods": ["invalid_method"]
            },
            # 分层数量超出范围
            {
                "factor_id": "momentum_20d",
                "validation_config": {
                    "train_start": "2010-01-01",
                    "train_end": "2016-12-31",
                    "test_start": "2017-01-01",
                    "test_end": "2020-12-31"
                },
                "layers": 25  # 超过最大值20
            }
        ]

        for invalid_request in invalid_requests:
            response = requests.post(
                self.validation_endpoint,
                json=invalid_request,
                headers={"Content-Type": "application/json"}
            )

            assert_api_error_response(response, 400)
            assert_response_time(response, 200)

    def test_factor_validation_not_found(self):
        """T021-3: 因子不存在错误合约"""
        request_data = {
            "factor_id": "nonexistent_factor_12345",
            "validation_config": {
                "train_start": "2010-01-01",
                "train_end": "2016-12-31",
                "test_start": "2017-01-01",
                "test_end": "2020-12-31"
            }
        }

        response = requests.post(
            self.validation_endpoint,
            json=request_data,
            headers={"Content-Type": "application/json"}
        )

        assert_api_error_response(response, 404)
        assert_response_time(response, 200)

        error_data = response.json()
        assert "factor_id" in error_data["error"]["details"]
        assert "not found" in error_data["error"]["message"].lower()

    def test_factor_validation_concurrent_limit(self):
        """T021-4: 并发验证限制合约"""
        # 模拟同时提交多个验证任务
        request_data = {
            "factor_id": "momentum_20d",
            "validation_config": {
                "train_start": "2010-01-01",
                "train_end": "2016-12-31",
                "test_start": "2017-01-01",
                "test_end": "2020-12-31"
            }
        }

        # 假设系统限制为最多5个并发验证
        responses = []
        for i in range(7):  # 超过限制数量
            response = requests.post(
                self.validation_endpoint,
                json=request_data,
                headers={"Content-Type": "application/json"}
            )
            responses.append(response)

        # 前几个请求应该成功
        successful_count = sum(1 for r in responses if r.status_code == 202)
        rejected_count = sum(1 for r in responses if r.status_code == 429)  # Too Many Requests

        assert successful_count >= 1  # 至少有一个成功
        assert rejected_count >= 1   # 超出限制的请求被拒绝

        # 验证被拒绝请求的错误格式
        for response in responses:
            if response.status_code == 429:
                assert_api_error_response(response, 429)
                error_data = response.json()
                assert "concurrent" in error_data["error"]["message"].lower()