"""
验证Agent API契约测试

测试Validation Agent的RESTful API接口契约，确保：
1. API端点正确性
2. 请求/响应格式符合规范
3. 错误处理机制
4. 数据验证规则

Contract: specs/002-spec-md/contracts/validation-agent-api.yaml
"""

import pytest
import jsonschema
from typing import Dict, Any


@pytest.mark.contract
@pytest.mark.requires_hikyuu
@pytest.mark.agent_validation
class TestValidationAgentContract:
    """验证Agent API契约测试"""

    def setup_method(self):
        """设置测试方法"""
        # 定义验证Agent的API契约
        self.validation_agent_contract = {
            "openapi": "3.0.3",
            "info": {
                "title": "Validation Agent API",
                "version": "1.0.0",
                "description": "A股因子验证与回测Agent API"
            },
            "paths": {
                "/api/v1/factors/{factor_id}/validate": {
                    "post": {
                        "summary": "验证因子计算结果",
                        "parameters": [
                            {
                                "name": "factor_id",
                                "in": "path",
                                "required": True,
                                "schema": {"type": "string"}
                            }
                        ],
                        "requestBody": {
                            "required": True,
                            "content": {
                                "application/json": {
                                    "schema": {
                                        "type": "object",
                                        "properties": {
                                            "stocks": {
                                                "type": "array",
                                                "items": {"type": "string"},
                                                "description": "股票代码列表"
                                            },
                                            "start_date": {
                                                "type": "string",
                                                "format": "date",
                                                "description": "验证开始日期"
                                            },
                                            "end_date": {
                                                "type": "string",
                                                "format": "date",
                                                "description": "验证结束日期"
                                            },
                                            "validation_rules": {
                                                "type": "array",
                                                "items": {
                                                    "type": "object",
                                                    "properties": {
                                                        "rule_type": {
                                                            "type": "string",
                                                            "enum": ["missing_check", "outlier_check", "consistency_check", "correlation_check"]
                                                        },
                                                        "parameters": {"type": "object"}
                                                    },
                                                    "required": ["rule_type"]
                                                },
                                                "description": "验证规则配置"
                                            }
                                        },
                                        "required": ["stocks", "start_date", "end_date"]
                                    }
                                }
                            }
                        },
                        "responses": {
                            "200": {
                                "description": "验证成功",
                                "content": {
                                    "application/json": {
                                        "schema": {
                                            "type": "object",
                                            "properties": {
                                                "validation_id": {"type": "string"},
                                                "factor_id": {"type": "string"},
                                                "status": {
                                                    "type": "string",
                                                    "enum": ["passed", "failed", "warning"]
                                                },
                                                "summary": {
                                                    "type": "object",
                                                    "properties": {
                                                        "total_checks": {"type": "integer"},
                                                        "passed_checks": {"type": "integer"},
                                                        "failed_checks": {"type": "integer"},
                                                        "warning_checks": {"type": "integer"}
                                                    }
                                                },
                                                "results": {
                                                    "type": "array",
                                                    "items": {
                                                        "type": "object",
                                                        "properties": {
                                                            "rule_type": {"type": "string"},
                                                            "status": {"type": "string"},
                                                            "message": {"type": "string"},
                                                            "details": {"type": "object"}
                                                        }
                                                    }
                                                },
                                                "created_at": {
                                                    "type": "string",
                                                    "format": "date-time"
                                                }
                                            },
                                            "required": ["validation_id", "factor_id", "status", "summary", "results"]
                                        }
                                    }
                                }
                            },
                            "400": {
                                "description": "请求参数错误",
                                "content": {
                                    "application/json": {
                                        "schema": {
                                            "type": "object",
                                            "properties": {
                                                "error": {"type": "string"},
                                                "message": {"type": "string"},
                                                "details": {"type": "object"}
                                            }
                                        }
                                    }
                                }
                            },
                            "404": {
                                "description": "因子不存在"
                            },
                            "500": {
                                "description": "服务器内部错误"
                            }
                        }
                    }
                },
                "/api/v1/backtests": {
                    "post": {
                        "summary": "创建因子回测任务",
                        "requestBody": {
                            "required": True,
                            "content": {
                                "application/json": {
                                    "schema": {
                                        "type": "object",
                                        "properties": {
                                            "factor_ids": {
                                                "type": "array",
                                                "items": {"type": "string"},
                                                "description": "因子ID列表"
                                            },
                                            "universe": {
                                                "type": "object",
                                                "properties": {
                                                    "type": {
                                                        "type": "string",
                                                        "enum": ["all", "index", "custom"]
                                                    },
                                                    "stocks": {
                                                        "type": "array",
                                                        "items": {"type": "string"}
                                                    },
                                                    "index_code": {"type": "string"}
                                                },
                                                "required": ["type"]
                                            },
                                            "period": {
                                                "type": "object",
                                                "properties": {
                                                    "start_date": {"type": "string", "format": "date"},
                                                    "end_date": {"type": "string", "format": "date"}
                                                },
                                                "required": ["start_date", "end_date"]
                                            },
                                            "strategy": {
                                                "type": "object",
                                                "properties": {
                                                    "type": {
                                                        "type": "string",
                                                        "enum": ["long_short", "long_only", "factor_ic"]
                                                    },
                                                    "rebalance_freq": {
                                                        "type": "string",
                                                        "enum": ["daily", "weekly", "monthly"]
                                                    },
                                                    "parameters": {"type": "object"}
                                                },
                                                "required": ["type", "rebalance_freq"]
                                            }
                                        },
                                        "required": ["factor_ids", "universe", "period", "strategy"]
                                    }
                                }
                            }
                        },
                        "responses": {
                            "201": {
                                "description": "回测任务创建成功",
                                "content": {
                                    "application/json": {
                                        "schema": {
                                            "type": "object",
                                            "properties": {
                                                "backtest_id": {"type": "string"},
                                                "status": {
                                                    "type": "string",
                                                    "enum": ["created", "running", "completed", "failed"]
                                                },
                                                "created_at": {"type": "string", "format": "date-time"},
                                                "estimated_duration": {"type": "integer"}
                                            },
                                            "required": ["backtest_id", "status", "created_at"]
                                        }
                                    }
                                }
                            }
                        }
                    }
                },
                "/api/v1/backtests/{backtest_id}": {
                    "get": {
                        "summary": "获取回测结果",
                        "parameters": [
                            {
                                "name": "backtest_id",
                                "in": "path",
                                "required": True,
                                "schema": {"type": "string"}
                            }
                        ],
                        "responses": {
                            "200": {
                                "description": "回测结果",
                                "content": {
                                    "application/json": {
                                        "schema": {
                                            "type": "object",
                                            "properties": {
                                                "backtest_id": {"type": "string"},
                                                "status": {"type": "string"},
                                                "progress": {"type": "number", "minimum": 0, "maximum": 100},
                                                "results": {
                                                    "type": "object",
                                                    "properties": {
                                                        "performance": {
                                                            "type": "object",
                                                            "properties": {
                                                                "total_return": {"type": "number"},
                                                                "annual_return": {"type": "number"},
                                                                "volatility": {"type": "number"},
                                                                "sharpe_ratio": {"type": "number"},
                                                                "max_drawdown": {"type": "number"}
                                                            }
                                                        },
                                                        "factor_analysis": {
                                                            "type": "object",
                                                            "properties": {
                                                                "ic_mean": {"type": "number"},
                                                                "ic_std": {"type": "number"},
                                                                "ic_ir": {"type": "number"},
                                                                "turnover": {"type": "number"}
                                                            }
                                                        }
                                                    }
                                                },
                                                "created_at": {"type": "string", "format": "date-time"},
                                                "completed_at": {"type": "string", "format": "date-time"}
                                            },
                                            "required": ["backtest_id", "status", "progress"]
                                        }
                                    }
                                }
                            }
                        }
                    }
                },
                "/api/v1/status": {
                    "get": {
                        "summary": "获取验证Agent状态",
                        "responses": {
                            "200": {
                                "description": "Agent状态信息",
                                "content": {
                                    "application/json": {
                                        "schema": {
                                            "type": "object",
                                            "properties": {
                                                "agent_id": {"type": "string"},
                                                "status": {
                                                    "type": "string",
                                                    "enum": ["healthy", "busy", "error"]
                                                },
                                                "version": {"type": "string"},
                                                "active_validations": {"type": "integer"},
                                                "active_backtests": {"type": "integer"},
                                                "resource_usage": {
                                                    "type": "object",
                                                    "properties": {
                                                        "cpu_percent": {"type": "number"},
                                                        "memory_mb": {"type": "number"}
                                                    }
                                                },
                                                "last_health_check": {"type": "string", "format": "date-time"}
                                            },
                                            "required": ["agent_id", "status", "version"]
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }

        # JSON Schema验证器
        self.validator = jsonschema.Draft7Validator({})

    def test_validation_agent_contract_structure(self):
        """测试验证Agent契约结构完整性"""
        required_paths = [
            "/api/v1/factors/{factor_id}/validate",
            "/api/v1/backtests",
            "/api/v1/backtests/{backtest_id}",
            "/api/v1/status"
        ]

        contract_paths = list(self.validation_agent_contract["paths"].keys())

        for path in required_paths:
            assert path in contract_paths, f"缺少必要的API路径: {path}"

    def test_factor_validation_endpoint_contract(self):
        """测试因子验证端点契约"""
        path_config = self.validation_agent_contract["paths"]["/api/v1/factors/{factor_id}/validate"]["post"]

        # 验证请求body schema
        request_schema = path_config["requestBody"]["content"]["application/json"]["schema"]

        # 测试有效请求
        valid_request = {
            "stocks": ["sh000001", "sz000002"],
            "start_date": "2024-01-01",
            "end_date": "2024-12-31",
            "validation_rules": [
                {
                    "rule_type": "missing_check",
                    "parameters": {"threshold": 0.05}
                },
                {
                    "rule_type": "outlier_check",
                    "parameters": {"method": "iqr", "factor": 3.0}
                }
            ]
        }

        try:
            jsonschema.validate(valid_request, request_schema)
        except jsonschema.ValidationError as e:
            pytest.fail(f"有效的因子验证请求未通过schema验证: {e}")

        # 验证响应schema
        response_schema = path_config["responses"]["200"]["content"]["application/json"]["schema"]

        valid_response = {
            "validation_id": "val_20241201_001",
            "factor_id": "momentum_20d",
            "status": "passed",
            "summary": {
                "total_checks": 4,
                "passed_checks": 3,
                "failed_checks": 0,
                "warning_checks": 1
            },
            "results": [
                {
                    "rule_type": "missing_check",
                    "status": "passed",
                    "message": "数据完整性检查通过",
                    "details": {"missing_rate": 0.02}
                }
            ],
            "created_at": "2024-12-01T10:00:00Z"
        }

        try:
            jsonschema.validate(valid_response, response_schema)
        except jsonschema.ValidationError as e:
            pytest.fail(f"有效的因子验证响应未通过schema验证: {e}")

    def test_backtest_creation_endpoint_contract(self):
        """测试回测创建端点契约"""
        path_config = self.validation_agent_contract["paths"]["/api/v1/backtests"]["post"]

        request_schema = path_config["requestBody"]["content"]["application/json"]["schema"]

        # 测试有效回测请求
        valid_request = {
            "factor_ids": ["momentum_20d", "reversal_5d"],
            "universe": {
                "type": "index",
                "index_code": "000300"
            },
            "period": {
                "start_date": "2023-01-01",
                "end_date": "2024-12-31"
            },
            "strategy": {
                "type": "long_short",
                "rebalance_freq": "monthly",
                "parameters": {
                    "long_ratio": 0.3,
                    "short_ratio": 0.3
                }
            }
        }

        try:
            jsonschema.validate(valid_request, request_schema)
        except jsonschema.ValidationError as e:
            pytest.fail(f"有效的回测创建请求未通过schema验证: {e}")

        # 验证响应schema
        response_schema = path_config["responses"]["201"]["content"]["application/json"]["schema"]

        valid_response = {
            "backtest_id": "bt_20241201_001",
            "status": "created",
            "created_at": "2024-12-01T10:00:00Z",
            "estimated_duration": 300
        }

        try:
            jsonschema.validate(valid_response, response_schema)
        except jsonschema.ValidationError as e:
            pytest.fail(f"有效的回测创建响应未通过schema验证: {e}")

    def test_backtest_results_endpoint_contract(self):
        """测试回测结果端点契约"""
        path_config = self.validation_agent_contract["paths"]["/api/v1/backtests/{backtest_id}"]["get"]

        response_schema = path_config["responses"]["200"]["content"]["application/json"]["schema"]

        # 测试完成的回测结果
        valid_response = {
            "backtest_id": "bt_20241201_001",
            "status": "completed",
            "progress": 100.0,
            "results": {
                "performance": {
                    "total_return": 0.25,
                    "annual_return": 0.12,
                    "volatility": 0.18,
                    "sharpe_ratio": 0.67,
                    "max_drawdown": -0.08
                },
                "factor_analysis": {
                    "ic_mean": 0.05,
                    "ic_std": 0.15,
                    "ic_ir": 0.33,
                    "turnover": 0.4
                }
            },
            "created_at": "2024-12-01T10:00:00Z",
            "completed_at": "2024-12-01T10:05:00Z"
        }

        try:
            jsonschema.validate(valid_response, response_schema)
        except jsonschema.ValidationError as e:
            pytest.fail(f"有效的回测结果响应未通过schema验证: {e}")

    def test_status_endpoint_contract(self):
        """测试状态端点契约"""
        path_config = self.validation_agent_contract["paths"]["/api/v1/status"]["get"]

        response_schema = path_config["responses"]["200"]["content"]["application/json"]["schema"]

        valid_response = {
            "agent_id": "validation-agent-001",
            "status": "healthy",
            "version": "1.0.0",
            "active_validations": 2,
            "active_backtests": 1,
            "resource_usage": {
                "cpu_percent": 45.2,
                "memory_mb": 1024.5
            },
            "last_health_check": "2024-12-01T10:00:00Z"
        }

        try:
            jsonschema.validate(valid_response, response_schema)
        except jsonschema.ValidationError as e:
            pytest.fail(f"有效的状态响应未通过schema验证: {e}")

    def test_error_response_contracts(self):
        """测试错误响应契约"""
        # 测试400错误响应
        error_schema = self.validation_agent_contract["paths"]["/api/v1/factors/{factor_id}/validate"]["post"]["responses"]["400"]["content"]["application/json"]["schema"]

        error_response = {
            "error": "ValidationError",
            "message": "股票代码格式无效",
            "details": {
                "invalid_stocks": ["invalid_code"],
                "valid_format": "sh000001或sz000002"
            }
        }

        try:
            jsonschema.validate(error_response, error_schema)
        except jsonschema.ValidationError as e:
            pytest.fail(f"有效的错误响应未通过schema验证: {e}")

    def test_validation_rules_enum_constraints(self):
        """测试验证规则枚举约束"""
        request_schema = self.validation_agent_contract["paths"]["/api/v1/factors/{factor_id}/validate"]["post"]["requestBody"]["content"]["application/json"]["schema"]

        # 测试无效的validation rule type
        invalid_request = {
            "stocks": ["sh000001"],
            "start_date": "2024-01-01",
            "end_date": "2024-12-31",
            "validation_rules": [
                {
                    "rule_type": "invalid_rule_type"  # 不在枚举中
                }
            ]
        }

        with pytest.raises(jsonschema.ValidationError):
            jsonschema.validate(invalid_request, request_schema)

    def test_status_enum_constraints(self):
        """测试状态枚举约束"""
        response_schema = self.validation_agent_contract["paths"]["/api/v1/status"]["get"]["responses"]["200"]["content"]["application/json"]["schema"]

        # 测试无效状态
        invalid_response = {
            "agent_id": "validation-agent-001",
            "status": "invalid_status",  # 不在枚举中
            "version": "1.0.0"
        }

        with pytest.raises(jsonschema.ValidationError):
            jsonschema.validate(invalid_response, response_schema)

    def test_required_fields_validation(self):
        """测试必填字段验证"""
        request_schema = self.validation_agent_contract["paths"]["/api/v1/factors/{factor_id}/validate"]["post"]["requestBody"]["content"]["application/json"]["schema"]

        # 测试缺少必填字段的请求
        incomplete_request = {
            "stocks": ["sh000001"],
            "start_date": "2024-01-01"
            # 缺少 end_date
        }

        with pytest.raises(jsonschema.ValidationError):
            jsonschema.validate(incomplete_request, request_schema)

    def test_date_format_validation(self):
        """测试日期格式验证"""
        request_schema = self.validation_agent_contract["paths"]["/api/v1/factors/{factor_id}/validate"]["post"]["requestBody"]["content"]["application/json"]["schema"]

        # 测试有效日期格式
        valid_date_request = {
            "stocks": ["sh000001"],
            "start_date": "2024-01-01",  # 正确格式 YYYY-MM-DD
            "end_date": "2024-12-31"
        }

        try:
            jsonschema.validate(valid_date_request, request_schema)
        except jsonschema.ValidationError as e:
            pytest.fail(f"Valid date format failed validation: {e}")

        # Note: Date format validation depends on jsonschema version and format checker
        # For comprehensive date validation, additional custom validation would be needed

    def test_array_items_validation(self):
        """测试数组项目验证"""
        request_schema = self.validation_agent_contract["paths"]["/api/v1/backtests"]["post"]["requestBody"]["content"]["application/json"]["schema"]

        # 测试空的factor_ids数组
        empty_factors_request = {
            "factor_ids": [],  # 空数组
            "universe": {"type": "all"},
            "period": {
                "start_date": "2024-01-01",
                "end_date": "2024-12-31"
            },
            "strategy": {
                "type": "long_only",
                "rebalance_freq": "monthly"
            }
        }

        # 这应该通过验证（空数组是有效的）
        try:
            jsonschema.validate(empty_factors_request, request_schema)
        except jsonschema.ValidationError as e:
            pytest.fail(f"空factor_ids数组应该是有效的: {e}")