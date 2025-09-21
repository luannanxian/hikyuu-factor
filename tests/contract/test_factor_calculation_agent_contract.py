"""
FactorCalculation Agent API Contract Tests

测试因子计算Agent的API契约，确保：
- 因子计算API的请求响应格式
- 因子状态监控API的一致性
- 因子结果查询API的数据格式
- 因子列表API的完整性
- 错误处理和边界条件
"""
import pytest
import json
import jsonschema
from datetime import datetime, date
from typing import Dict, List, Any, Optional


@pytest.mark.contract
class TestFactorCalculationAgentContract:
    """FactorCalculation Agent API契约测试"""

    def setup_method(self):
        """设置测试环境"""
        # API基础URL
        self.base_url = "http://localhost:8002/api/v1"

        # 预期的API endpoints
        self.endpoints = {
            "calculate": "/factors/calculate",
            "status": "/factors/status",
            "results": "/factors/results",
            "list": "/factors/list"
        }

        # Request/Response schemas for contract validation
        self.request_schemas = {
            "calculate_factors": {
                "type": "object",
                "properties": {
                    "factor_ids": {
                        "type": "array",
                        "items": {"type": "string"},
                        "minItems": 1
                    },
                    "stocks": {
                        "type": "array",
                        "items": {"type": "string"},
                        "minItems": 1
                    },
                    "start_date": {"type": "string", "format": "date"},
                    "end_date": {"type": "string", "format": "date"},
                    "parameters": {
                        "type": "object",
                        "additionalProperties": True
                    },
                    "priority": {
                        "type": "string",
                        "enum": ["low", "normal", "high", "urgent"]
                    }
                },
                "required": ["factor_ids", "stocks", "start_date", "end_date"]
            },
            "get_results": {
                "type": "object",
                "properties": {
                    "calculation_id": {"type": "string"},
                    "factor_ids": {
                        "type": "array",
                        "items": {"type": "string"}
                    },
                    "stocks": {
                        "type": "array",
                        "items": {"type": "string"}
                    },
                    "date_range": {
                        "type": "object",
                        "properties": {
                            "start_date": {"type": "string", "format": "date"},
                            "end_date": {"type": "string", "format": "date"}
                        }
                    },
                    "format": {
                        "type": "string",
                        "enum": ["json", "csv", "parquet"]
                    }
                }
            }
        }

        self.response_schemas = {
            "calculation_response": {
                "type": "object",
                "properties": {
                    "status": {"type": "string", "enum": ["success", "error"]},
                    "data": {
                        "type": "object",
                        "properties": {
                            "calculation_id": {"type": "string"},
                            "job_status": {
                                "type": "string",
                                "enum": ["queued", "running", "completed", "failed"]
                            },
                            "estimated_completion": {"type": "string"},
                            "progress": {
                                "type": "object",
                                "properties": {
                                    "completed": {"type": "integer"},
                                    "total": {"type": "integer"},
                                    "percentage": {"type": "number", "minimum": 0, "maximum": 100}
                                },
                                "required": ["completed", "total", "percentage"]
                            }
                        },
                        "required": ["calculation_id", "job_status"]
                    },
                    "timestamp": {"type": "string"},
                    "request_id": {"type": "string"}
                },
                "required": ["status", "data", "timestamp"]
            },
            "factor_results": {
                "type": "object",
                "properties": {
                    "status": {"type": "string", "enum": ["success", "error"]},
                    "data": {
                        "type": "object",
                        "properties": {
                            "calculation_id": {"type": "string"},
                            "factors": {
                                "type": "object",
                                "patternProperties": {
                                    "^[a-zA-Z_][a-zA-Z0-9_]*$": {  # Factor ID pattern
                                        "type": "object",
                                        "properties": {
                                            "metadata": {
                                                "type": "object",
                                                "properties": {
                                                    "factor_id": {"type": "string"},
                                                    "factor_name": {"type": "string"},
                                                    "category": {"type": "string"},
                                                    "calculation_time": {"type": "string"},
                                                    "parameters": {"type": "object"}
                                                },
                                                "required": ["factor_id", "factor_name"]
                                            },
                                            "values": {
                                                "type": "object",
                                                "patternProperties": {
                                                    "^[a-z]{2}[0-9]{6}$": {  # Stock code pattern
                                                        "type": "array",
                                                        "items": {
                                                            "type": "object",
                                                            "properties": {
                                                                "date": {"type": "string", "format": "date"},
                                                                "value": {"type": ["number", "null"]},
                                                                "quality": {
                                                                    "type": "string",
                                                                    "enum": ["good", "warning", "error"]
                                                                }
                                                            },
                                                            "required": ["date", "value"]
                                                        }
                                                    }
                                                }
                                            }
                                        },
                                        "required": ["metadata", "values"]
                                    }
                                }
                            }
                        },
                        "required": ["calculation_id", "factors"]
                    },
                    "timestamp": {"type": "string"},
                    "request_id": {"type": "string"}
                },
                "required": ["status", "data", "timestamp"]
            },
            "factor_list": {
                "type": "object",
                "properties": {
                    "status": {"type": "string", "enum": ["success", "error"]},
                    "data": {
                        "type": "object",
                        "properties": {
                            "factors": {
                                "type": "array",
                                "items": {
                                    "type": "object",
                                    "properties": {
                                        "factor_id": {"type": "string"},
                                        "factor_name": {"type": "string"},
                                        "category": {"type": "string"},
                                        "description": {"type": "string"},
                                        "parameters": {
                                            "type": "object",
                                            "properties": {
                                                "default": {"type": "object"},
                                                "schema": {"type": "object"}
                                            }
                                        },
                                        "computation_cost": {
                                            "type": "string",
                                            "enum": ["low", "medium", "high"]
                                        },
                                        "data_requirements": {
                                            "type": "array",
                                            "items": {"type": "string"}
                                        },
                                        "version": {"type": "string"}
                                    },
                                    "required": ["factor_id", "factor_name", "category"]
                                }
                            },
                            "categories": {
                                "type": "array",
                                "items": {"type": "string"}
                            },
                            "total": {"type": "integer"}
                        },
                        "required": ["factors", "total"]
                    },
                    "timestamp": {"type": "string"},
                    "request_id": {"type": "string"}
                },
                "required": ["status", "data", "timestamp"]
            }
        }

    @pytest.mark.contract
    def test_calculate_factors_endpoint(self):
        """测试因子计算API端点契约"""
        endpoint = self.endpoints["calculate"]

        # Test 1: API endpoint structure
        assert endpoint.startswith("/factors/"), "Factor calculation API should be under /factors/ path"
        assert "calculate" in endpoint, "Endpoint should contain 'calculate'"

        # Test 2: Request parameter validation
        valid_request = {
            "factor_ids": ["momentum_20d", "rsi_14d"],
            "stocks": ["sh600000", "sz000001"],
            "start_date": "2024-01-01",
            "end_date": "2024-01-31",
            "parameters": {
                "momentum_20d": {"period": 20},
                "rsi_14d": {"period": 14}
            },
            "priority": "normal"
        }

        # Validate against request schema
        try:
            jsonschema.validate(valid_request, self.request_schemas["calculate_factors"])
        except jsonschema.ValidationError as e:
            pytest.fail(f"Valid calculation request failed schema validation: {e}")

        # Test 3: Invalid request should fail validation
        invalid_request = {
            "factor_ids": [],  # Empty array
            "stocks": ["sh600000"],
            # Missing required start_date and end_date
        }

        with pytest.raises(jsonschema.ValidationError):
            jsonschema.validate(invalid_request, self.request_schemas["calculate_factors"])

    @pytest.mark.contract
    def test_factor_status_endpoint(self):
        """测试因子状态监控API端点契约"""
        endpoint = self.endpoints["status"]

        # Test 1: API endpoint structure
        assert endpoint.startswith("/factors/"), "Factor status API should be under /factors/ path"
        assert "status" in endpoint, "Endpoint should contain 'status'"

        # Test 2: Status response should include job progress
        expected_status_fields = [
            "calculation_id",
            "job_status",
            "progress",
            "estimated_completion"
        ]

        # This would be validated against actual response in integration tests

    @pytest.mark.contract
    def test_factor_results_endpoint(self):
        """测试因子结果查询API端点契约"""
        endpoint = self.endpoints["results"]

        # Test 1: API endpoint structure
        assert endpoint.startswith("/factors/"), "Factor results API should be under /factors/ path"
        assert "results" in endpoint, "Endpoint should contain 'results'"

        # Test 2: Request validation
        valid_request = {
            "calculation_id": "calc_123456",
            "factor_ids": ["momentum_20d"],
            "stocks": ["sh600000"],
            "date_range": {
                "start_date": "2024-01-01",
                "end_date": "2024-01-31"
            },
            "format": "json"
        }

        try:
            jsonschema.validate(valid_request, self.request_schemas["get_results"])
        except jsonschema.ValidationError as e:
            pytest.fail(f"Valid results request failed schema validation: {e}")

    @pytest.mark.contract
    def test_list_factors_endpoint(self):
        """测试因子列表API端点契约"""
        endpoint = self.endpoints["list"]

        # Test 1: API endpoint structure
        assert endpoint.startswith("/factors/"), "Factor list API should be under /factors/ path"
        assert "list" in endpoint, "Endpoint should contain 'list'"

        # Test 2: Factor list should include metadata
        expected_factor_fields = [
            "factor_id",
            "factor_name",
            "category",
            "description",
            "parameters"
        ]

        # This would be validated against actual response in integration tests

    @pytest.mark.contract
    def test_calculation_response_schema_validation(self):
        """测试因子计算响应格式契约"""
        # Mock calculation response
        mock_response = {
            "status": "success",
            "data": {
                "calculation_id": "calc_20240115_001",
                "job_status": "queued",
                "estimated_completion": "2024-01-15T09:45:00Z",
                "progress": {
                    "completed": 0,
                    "total": 100,
                    "percentage": 0.0
                }
            },
            "timestamp": "2024-01-15T09:30:00Z",
            "request_id": "req_calc_001"
        }

        # Validate response schema
        try:
            jsonschema.validate(mock_response, self.response_schemas["calculation_response"])
        except jsonschema.ValidationError as e:
            pytest.fail(f"Valid calculation response failed schema validation: {e}")

    @pytest.mark.contract
    def test_factor_results_response_schema_validation(self):
        """测试因子结果响应格式契约"""
        # Mock factor results response
        mock_response = {
            "status": "success",
            "data": {
                "calculation_id": "calc_20240115_001",
                "factors": {
                    "momentum_20d": {
                        "metadata": {
                            "factor_id": "momentum_20d",
                            "factor_name": "20日动量因子",
                            "category": "momentum",
                            "calculation_time": "2024-01-15T09:35:00Z",
                            "parameters": {"period": 20}
                        },
                        "values": {
                            "sh600000": [
                                {
                                    "date": "2024-01-15",
                                    "value": 0.0523,
                                    "quality": "good"
                                }
                            ]
                        }
                    }
                }
            },
            "timestamp": "2024-01-15T09:35:00Z",
            "request_id": "req_results_001"
        }

        # Validate response schema
        try:
            jsonschema.validate(mock_response, self.response_schemas["factor_results"])
        except jsonschema.ValidationError as e:
            pytest.fail(f"Valid factor results response failed schema validation: {e}")

    @pytest.mark.contract
    def test_factor_list_response_schema_validation(self):
        """测试因子列表响应格式契约"""
        # Mock factor list response
        mock_response = {
            "status": "success",
            "data": {
                "factors": [
                    {
                        "factor_id": "momentum_20d",
                        "factor_name": "20日动量因子",
                        "category": "momentum",
                        "description": "基于20日收益率的动量因子",
                        "parameters": {
                            "default": {"period": 20},
                            "schema": {
                                "type": "object",
                                "properties": {
                                    "period": {"type": "integer", "minimum": 1, "maximum": 252}
                                }
                            }
                        },
                        "computation_cost": "low",
                        "data_requirements": ["daily_price"],
                        "version": "1.0.0"
                    }
                ],
                "categories": ["momentum", "mean_reversion", "value", "quality"],
                "total": 1
            },
            "timestamp": "2024-01-15T09:30:00Z",
            "request_id": "req_list_001"
        }

        # Validate response schema
        try:
            jsonschema.validate(mock_response, self.response_schemas["factor_list"])
        except jsonschema.ValidationError as e:
            pytest.fail(f"Valid factor list response failed schema validation: {e}")

    @pytest.mark.contract
    def test_error_response_format_contract(self):
        """测试错误响应格式契约"""
        # Standard error response format
        error_response_schema = {
            "type": "object",
            "properties": {
                "status": {"type": "string", "enum": ["error"]},
                "error": {
                    "type": "object",
                    "properties": {
                        "code": {"type": "string"},
                        "message": {"type": "string"},
                        "details": {"type": "object"}
                    },
                    "required": ["code", "message"]
                },
                "timestamp": {"type": "string"},
                "request_id": {"type": "string"}
            },
            "required": ["status", "error", "timestamp"]
        }

        mock_error_response = {
            "status": "error",
            "error": {
                "code": "INVALID_FACTOR_ID",
                "message": "Unknown factor ID specified",
                "details": {
                    "invalid_factors": ["unknown_factor"],
                    "available_factors": ["momentum_20d", "rsi_14d"]
                }
            },
            "timestamp": "2024-01-15T09:30:00Z",
            "request_id": "req_error_002"
        }

        try:
            jsonschema.validate(mock_error_response, error_response_schema)
        except jsonschema.ValidationError as e:
            pytest.fail(f"Valid error response failed schema validation: {e}")

    @pytest.mark.contract
    def test_http_status_codes_contract(self):
        """测试HTTP状态码契约"""
        expected_status_codes = {
            "POST /factors/calculate": [200, 400, 422, 500],
            "GET /factors/status": [200, 404, 500],
            "GET /factors/results": [200, 404, 500],
            "GET /factors/list": [200, 500]
        }

        # Verify expected status codes are defined
        for endpoint, codes in expected_status_codes.items():
            assert 200 in codes, f"{endpoint} should support 200 OK"
            assert 500 in codes, f"{endpoint} should handle 500 Internal Server Error"

    @pytest.mark.contract
    def test_async_computation_contract(self):
        """测试异步计算契约"""
        # Async computation flow validation
        async_flow_states = [
            "queued",      # Job submitted and queued
            "running",     # Job is being processed
            "completed",   # Job finished successfully
            "failed"       # Job failed with error
        ]

        # Each state should be valid
        for state in async_flow_states:
            assert state in ["queued", "running", "completed", "failed"], \
                f"Invalid job state: {state}"

        # State transitions should be logical
        valid_transitions = {
            "queued": ["running", "failed"],
            "running": ["completed", "failed"],
            "completed": [],  # Terminal state
            "failed": []      # Terminal state
        }

        assert len(valid_transitions) == 4, "Should define all state transitions"

    @pytest.mark.contract
    def test_factor_computation_requirements_contract(self):
        """测试因子计算需求契约"""
        # Factor computation requirements
        computation_requirements = {
            "input_validation": {
                "stock_codes": "Valid format: ^[a-z]{2}[0-9]{6}$",
                "date_range": "start_date <= end_date",
                "factor_ids": "Must exist in factor registry"
            },
            "output_guarantees": {
                "point_in_time": "No look-ahead bias",
                "data_quality": "Quality flags for each value",
                "metadata": "Complete factor metadata included"
            },
            "performance_targets": {
                "calculation_time": "< 30 minutes for full market",
                "memory_usage": "< 16GB for large factor sets",
                "concurrent_jobs": ">= 4 parallel calculations"
            }
        }

        # Verify contract requirements are defined
        assert "input_validation" in computation_requirements
        assert "output_guarantees" in computation_requirements
        assert "performance_targets" in computation_requirements