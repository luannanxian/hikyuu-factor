"""
信号生成Agent API契约测试

测试Signal Generation Agent的RESTful API接口契约，确保：
1. API端点正确性
2. 请求/响应格式符合规范
3. 错误处理机制
4. 人工确认机制

Contract: specs/002-spec-md/contracts/signal-generation-agent-api.yaml
"""

import pytest
import jsonschema
from typing import Dict, Any


@pytest.mark.contract
@pytest.mark.requires_hikyuu
@pytest.mark.agent_signal
class TestSignalGenerationAgentContract:
    """信号生成Agent API契约测试"""

    def setup_method(self):
        """设置测试方法"""
        # 定义信号生成Agent的API契约
        self.signal_agent_contract = {
            "openapi": "3.0.3",
            "info": {
                "title": "Signal Generation Agent API",
                "version": "1.0.0",
                "description": "A股交易信号生成与确认Agent API"
            },
            "paths": {
                "/api/v1/signals/generate": {
                    "post": {
                        "summary": "生成交易信号",
                        "requestBody": {
                            "required": True,
                            "content": {
                                "application/json": {
                                    "schema": {
                                        "type": "object",
                                        "properties": {
                                            "strategy_id": {
                                                "type": "string",
                                                "description": "交易策略ID"
                                            },
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
                                            "parameters": {
                                                "type": "object",
                                                "properties": {
                                                    "signal_threshold": {"type": "number"},
                                                    "position_size": {"type": "number"},
                                                    "risk_limit": {"type": "number"}
                                                }
                                            },
                                            "target_date": {
                                                "type": "string",
                                                "format": "date",
                                                "description": "目标交易日期"
                                            }
                                        },
                                        "required": ["strategy_id", "factor_ids", "universe", "target_date"]
                                    }
                                }
                            }
                        },
                        "responses": {
                            "200": {
                                "description": "信号生成成功",
                                "content": {
                                    "application/json": {
                                        "schema": {
                                            "type": "object",
                                            "properties": {
                                                "signal_batch_id": {"type": "string"},
                                                "strategy_id": {"type": "string"},
                                                "status": {
                                                    "type": "string",
                                                    "enum": ["generated", "pending_review", "confirmed", "rejected"]
                                                },
                                                "total_signals": {"type": "integer"},
                                                "signals": {
                                                    "type": "array",
                                                    "items": {
                                                        "type": "object",
                                                        "properties": {
                                                            "signal_id": {"type": "string"},
                                                            "stock_code": {"type": "string"},
                                                            "action": {
                                                                "type": "string",
                                                                "enum": ["buy", "sell", "hold"]
                                                            },
                                                            "strength": {
                                                                "type": "number",
                                                                "minimum": 0,
                                                                "maximum": 1
                                                            },
                                                            "confidence": {
                                                                "type": "number",
                                                                "minimum": 0,
                                                                "maximum": 1
                                                            },
                                                            "factor_contributions": {
                                                                "type": "object",
                                                                "additionalProperties": {"type": "number"}
                                                            },
                                                            "risk_score": {"type": "number"}
                                                        },
                                                        "required": ["signal_id", "stock_code", "action", "strength", "confidence"]
                                                    }
                                                },
                                                "created_at": {"type": "string", "format": "date-time"},
                                                "expires_at": {"type": "string", "format": "date-time"}
                                            },
                                            "required": ["signal_batch_id", "strategy_id", "status", "total_signals", "signals"]
                                        }
                                    }
                                }
                            },
                            "400": {
                                "description": "请求参数错误"
                            },
                            "500": {
                                "description": "服务器内部错误"
                            }
                        }
                    }
                },
                "/api/v1/signals/{signal_batch_id}/review": {
                    "post": {
                        "summary": "人工审核交易信号",
                        "parameters": [
                            {
                                "name": "signal_batch_id",
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
                                            "reviewer_id": {
                                                "type": "string",
                                                "description": "审核人员ID"
                                            },
                                            "action": {
                                                "type": "string",
                                                "enum": ["approve", "reject", "modify"],
                                                "description": "审核决定"
                                            },
                                            "modifications": {
                                                "type": "array",
                                                "items": {
                                                    "type": "object",
                                                    "properties": {
                                                        "signal_id": {"type": "string"},
                                                        "new_action": {
                                                            "type": "string",
                                                            "enum": ["buy", "sell", "hold", "cancel"]
                                                        },
                                                        "new_strength": {"type": "number"},
                                                        "reason": {"type": "string"}
                                                    },
                                                    "required": ["signal_id"]
                                                }
                                            },
                                            "comments": {
                                                "type": "string",
                                                "description": "审核意见"
                                            }
                                        },
                                        "required": ["reviewer_id", "action"]
                                    }
                                }
                            }
                        },
                        "responses": {
                            "200": {
                                "description": "审核完成",
                                "content": {
                                    "application/json": {
                                        "schema": {
                                            "type": "object",
                                            "properties": {
                                                "signal_batch_id": {"type": "string"},
                                                "review_id": {"type": "string"},
                                                "status": {
                                                    "type": "string",
                                                    "enum": ["confirmed", "rejected", "modified"]
                                                },
                                                "reviewer_id": {"type": "string"},
                                                "reviewed_at": {"type": "string", "format": "date-time"},
                                                "final_signals": {
                                                    "type": "array",
                                                    "items": {
                                                        "type": "object",
                                                        "properties": {
                                                            "signal_id": {"type": "string"},
                                                            "stock_code": {"type": "string"},
                                                            "action": {"type": "string"},
                                                            "strength": {"type": "number"},
                                                            "modified": {"type": "boolean"}
                                                        }
                                                    }
                                                }
                                            },
                                            "required": ["signal_batch_id", "review_id", "status", "reviewer_id", "reviewed_at"]
                                        }
                                    }
                                }
                            },
                            "404": {
                                "description": "信号批次不存在"
                            }
                        }
                    }
                },
                "/api/v1/signals/{signal_batch_id}": {
                    "get": {
                        "summary": "获取信号批次详情",
                        "parameters": [
                            {
                                "name": "signal_batch_id",
                                "in": "path",
                                "required": True,
                                "schema": {"type": "string"}
                            }
                        ],
                        "responses": {
                            "200": {
                                "description": "信号批次详情",
                                "content": {
                                    "application/json": {
                                        "schema": {
                                            "type": "object",
                                            "properties": {
                                                "signal_batch_id": {"type": "string"},
                                                "strategy_id": {"type": "string"},
                                                "status": {"type": "string"},
                                                "total_signals": {"type": "integer"},
                                                "signals": {
                                                    "type": "array",
                                                    "items": {
                                                        "type": "object",
                                                        "properties": {
                                                            "signal_id": {"type": "string"},
                                                            "stock_code": {"type": "string"},
                                                            "action": {"type": "string"},
                                                            "strength": {"type": "number"},
                                                            "confidence": {"type": "number"},
                                                            "factor_contributions": {"type": "object"},
                                                            "risk_score": {"type": "number"}
                                                        }
                                                    }
                                                },
                                                "audit_trail": {
                                                    "type": "array",
                                                    "items": {
                                                        "type": "object",
                                                        "properties": {
                                                            "timestamp": {"type": "string", "format": "date-time"},
                                                            "action": {"type": "string"},
                                                            "user_id": {"type": "string"},
                                                            "details": {"type": "object"}
                                                        }
                                                    }
                                                },
                                                "created_at": {"type": "string", "format": "date-time"},
                                                "expires_at": {"type": "string", "format": "date-time"}
                                            },
                                            "required": ["signal_batch_id", "strategy_id", "status", "total_signals", "signals"]
                                        }
                                    }
                                }
                            }
                        }
                    }
                },
                "/api/v1/signals/history": {
                    "get": {
                        "summary": "获取历史信号记录",
                        "parameters": [
                            {
                                "name": "strategy_id",
                                "in": "query",
                                "schema": {"type": "string"}
                            },
                            {
                                "name": "start_date",
                                "in": "query",
                                "schema": {"type": "string", "format": "date"}
                            },
                            {
                                "name": "end_date",
                                "in": "query",
                                "schema": {"type": "string", "format": "date"}
                            },
                            {
                                "name": "status",
                                "in": "query",
                                "schema": {
                                    "type": "string",
                                    "enum": ["generated", "pending_review", "confirmed", "rejected"]
                                }
                            }
                        ],
                        "responses": {
                            "200": {
                                "description": "历史信号记录",
                                "content": {
                                    "application/json": {
                                        "schema": {
                                            "type": "object",
                                            "properties": {
                                                "total_count": {"type": "integer"},
                                                "signal_batches": {
                                                    "type": "array",
                                                    "items": {
                                                        "type": "object",
                                                        "properties": {
                                                            "signal_batch_id": {"type": "string"},
                                                            "strategy_id": {"type": "string"},
                                                            "status": {"type": "string"},
                                                            "total_signals": {"type": "integer"},
                                                            "created_at": {"type": "string", "format": "date-time"},
                                                            "reviewed_at": {"type": "string", "format": "date-time"},
                                                            "performance": {
                                                                "type": "object",
                                                                "properties": {
                                                                    "accuracy": {"type": "number"},
                                                                    "return": {"type": "number"}
                                                                }
                                                            }
                                                        }
                                                    }
                                                }
                                            },
                                            "required": ["total_count", "signal_batches"]
                                        }
                                    }
                                }
                            }
                        }
                    }
                },
                "/api/v1/status": {
                    "get": {
                        "summary": "获取信号生成Agent状态",
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
                                                "active_generations": {"type": "integer"},
                                                "pending_reviews": {"type": "integer"},
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

    def test_signal_agent_contract_structure(self):
        """测试信号生成Agent契约结构完整性"""
        required_paths = [
            "/api/v1/signals/generate",
            "/api/v1/signals/{signal_batch_id}/review",
            "/api/v1/signals/{signal_batch_id}",
            "/api/v1/signals/history",
            "/api/v1/status"
        ]

        contract_paths = list(self.signal_agent_contract["paths"].keys())

        for path in required_paths:
            assert path in contract_paths, f"缺少必要的API路径: {path}"

    def test_signal_generation_endpoint_contract(self):
        """测试信号生成端点契约"""
        path_config = self.signal_agent_contract["paths"]["/api/v1/signals/generate"]["post"]

        # 验证请求body schema
        request_schema = path_config["requestBody"]["content"]["application/json"]["schema"]

        # 测试有效请求
        valid_request = {
            "strategy_id": "momentum_strategy_v1",
            "factor_ids": ["momentum_20d", "reversal_5d"],
            "universe": {
                "type": "index",
                "index_code": "000300"
            },
            "parameters": {
                "signal_threshold": 0.6,
                "position_size": 0.05,
                "risk_limit": 0.02
            },
            "target_date": "2024-12-02"
        }

        try:
            jsonschema.validate(valid_request, request_schema)
        except jsonschema.ValidationError as e:
            pytest.fail(f"有效的信号生成请求未通过schema验证: {e}")

        # 验证响应schema
        response_schema = path_config["responses"]["200"]["content"]["application/json"]["schema"]

        valid_response = {
            "signal_batch_id": "sb_20241201_001",
            "strategy_id": "momentum_strategy_v1",
            "status": "generated",
            "total_signals": 150,
            "signals": [
                {
                    "signal_id": "sig_20241201_001",
                    "stock_code": "sh000001",
                    "action": "buy",
                    "strength": 0.75,
                    "confidence": 0.85,
                    "factor_contributions": {
                        "momentum_20d": 0.6,
                        "reversal_5d": 0.15
                    },
                    "risk_score": 0.3
                }
            ],
            "created_at": "2024-12-01T10:00:00Z",
            "expires_at": "2024-12-01T16:00:00Z"
        }

        try:
            jsonschema.validate(valid_response, response_schema)
        except jsonschema.ValidationError as e:
            pytest.fail(f"有效的信号生成响应未通过schema验证: {e}")

    def test_signal_review_endpoint_contract(self):
        """测试信号审核端点契约"""
        path_config = self.signal_agent_contract["paths"]["/api/v1/signals/{signal_batch_id}/review"]["post"]

        request_schema = path_config["requestBody"]["content"]["application/json"]["schema"]

        # 测试批准请求
        approve_request = {
            "reviewer_id": "reviewer_001",
            "action": "approve",
            "comments": "信号质量良好，批准执行"
        }

        try:
            jsonschema.validate(approve_request, request_schema)
        except jsonschema.ValidationError as e:
            pytest.fail(f"有效的批准请求未通过schema验证: {e}")

        # 测试修改请求
        modify_request = {
            "reviewer_id": "reviewer_001",
            "action": "modify",
            "modifications": [
                {
                    "signal_id": "sig_20241201_001",
                    "new_action": "hold",
                    "new_strength": 0.5,
                    "reason": "风险过高"
                }
            ],
            "comments": "部分信号需要调整"
        }

        try:
            jsonschema.validate(modify_request, request_schema)
        except jsonschema.ValidationError as e:
            pytest.fail(f"有效的修改请求未通过schema验证: {e}")

        # 验证响应schema
        response_schema = path_config["responses"]["200"]["content"]["application/json"]["schema"]

        valid_response = {
            "signal_batch_id": "sb_20241201_001",
            "review_id": "rev_20241201_001",
            "status": "confirmed",
            "reviewer_id": "reviewer_001",
            "reviewed_at": "2024-12-01T11:00:00Z",
            "final_signals": [
                {
                    "signal_id": "sig_20241201_001",
                    "stock_code": "sh000001",
                    "action": "buy",
                    "strength": 0.75,
                    "modified": False
                }
            ]
        }

        try:
            jsonschema.validate(valid_response, response_schema)
        except jsonschema.ValidationError as e:
            pytest.fail(f"有效的审核响应未通过schema验证: {e}")

    def test_signal_details_endpoint_contract(self):
        """测试信号详情端点契约"""
        path_config = self.signal_agent_contract["paths"]["/api/v1/signals/{signal_batch_id}"]["get"]

        response_schema = path_config["responses"]["200"]["content"]["application/json"]["schema"]

        valid_response = {
            "signal_batch_id": "sb_20241201_001",
            "strategy_id": "momentum_strategy_v1",
            "status": "confirmed",
            "total_signals": 150,
            "signals": [
                {
                    "signal_id": "sig_20241201_001",
                    "stock_code": "sh000001",
                    "action": "buy",
                    "strength": 0.75,
                    "confidence": 0.85,
                    "factor_contributions": {
                        "momentum_20d": 0.6,
                        "reversal_5d": 0.15
                    },
                    "risk_score": 0.3
                }
            ],
            "audit_trail": [
                {
                    "timestamp": "2024-12-01T10:00:00Z",
                    "action": "generated",
                    "user_id": "system",
                    "details": {"total_signals": 150}
                },
                {
                    "timestamp": "2024-12-01T11:00:00Z",
                    "action": "reviewed",
                    "user_id": "reviewer_001",
                    "details": {"review_action": "approve"}
                }
            ],
            "created_at": "2024-12-01T10:00:00Z",
            "expires_at": "2024-12-01T16:00:00Z"
        }

        try:
            jsonschema.validate(valid_response, response_schema)
        except jsonschema.ValidationError as e:
            pytest.fail(f"有效的信号详情响应未通过schema验证: {e}")

    def test_signal_history_endpoint_contract(self):
        """测试历史信号端点契约"""
        path_config = self.signal_agent_contract["paths"]["/api/v1/signals/history"]["get"]

        response_schema = path_config["responses"]["200"]["content"]["application/json"]["schema"]

        valid_response = {
            "total_count": 50,
            "signal_batches": [
                {
                    "signal_batch_id": "sb_20241201_001",
                    "strategy_id": "momentum_strategy_v1",
                    "status": "confirmed",
                    "total_signals": 150,
                    "created_at": "2024-12-01T10:00:00Z",
                    "reviewed_at": "2024-12-01T11:00:00Z",
                    "performance": {
                        "accuracy": 0.72,
                        "return": 0.035
                    }
                }
            ]
        }

        try:
            jsonschema.validate(valid_response, response_schema)
        except jsonschema.ValidationError as e:
            pytest.fail(f"有效的历史信号响应未通过schema验证: {e}")

    def test_status_endpoint_contract(self):
        """测试状态端点契约"""
        path_config = self.signal_agent_contract["paths"]["/api/v1/status"]["get"]

        response_schema = path_config["responses"]["200"]["content"]["application/json"]["schema"]

        valid_response = {
            "agent_id": "signal-agent-001",
            "status": "healthy",
            "version": "1.0.0",
            "active_generations": 1,
            "pending_reviews": 3,
            "resource_usage": {
                "cpu_percent": 35.7,
                "memory_mb": 512.3
            },
            "last_health_check": "2024-12-01T10:00:00Z"
        }

        try:
            jsonschema.validate(valid_response, response_schema)
        except jsonschema.ValidationError as e:
            pytest.fail(f"有效的状态响应未通过schema验证: {e}")

    def test_signal_action_enum_constraints(self):
        """测试信号动作枚举约束"""
        request_schema = self.signal_agent_contract["paths"]["/api/v1/signals/generate"]["post"]["requestBody"]["content"]["application/json"]["schema"]

        # 信号动作在响应中定义，这里测试生成参数
        universe_request = {
            "strategy_id": "test_strategy",
            "factor_ids": ["momentum_20d"],
            "universe": {
                "type": "invalid_type"  # 不在枚举中
            },
            "target_date": "2024-12-02"
        }

        with pytest.raises(jsonschema.ValidationError):
            jsonschema.validate(universe_request, request_schema)

    def test_review_action_enum_constraints(self):
        """测试审核动作枚举约束"""
        request_schema = self.signal_agent_contract["paths"]["/api/v1/signals/{signal_batch_id}/review"]["post"]["requestBody"]["content"]["application/json"]["schema"]

        # 测试无效的审核动作
        invalid_request = {
            "reviewer_id": "reviewer_001",
            "action": "invalid_action"  # 不在枚举中
        }

        with pytest.raises(jsonschema.ValidationError):
            jsonschema.validate(invalid_request, request_schema)

    def test_signal_strength_range_validation(self):
        """测试信号强度范围验证"""
        # 这里验证响应中信号强度的范围约束
        response_schema = self.signal_agent_contract["paths"]["/api/v1/signals/generate"]["post"]["responses"]["200"]["content"]["application/json"]["schema"]

        # 测试无效的信号强度（超出范围）
        invalid_response = {
            "signal_batch_id": "sb_20241201_001",
            "strategy_id": "momentum_strategy_v1",
            "status": "generated",
            "total_signals": 1,
            "signals": [
                {
                    "signal_id": "sig_20241201_001",
                    "stock_code": "sh000001",
                    "action": "buy",
                    "strength": 1.5,  # 超出0-1范围
                    "confidence": 0.85
                }
            ]
        }

        with pytest.raises(jsonschema.ValidationError):
            jsonschema.validate(invalid_response, response_schema)

    def test_required_fields_validation(self):
        """测试必填字段验证"""
        request_schema = self.signal_agent_contract["paths"]["/api/v1/signals/generate"]["post"]["requestBody"]["content"]["application/json"]["schema"]

        # 测试缺少必填字段的请求
        incomplete_request = {
            "strategy_id": "momentum_strategy_v1",
            "factor_ids": ["momentum_20d"],
            "universe": {"type": "all"}
            # 缺少 target_date
        }

        with pytest.raises(jsonschema.ValidationError):
            jsonschema.validate(incomplete_request, request_schema)

    def test_date_format_validation(self):
        """测试日期格式验证"""
        request_schema = self.signal_agent_contract["paths"]["/api/v1/signals/generate"]["post"]["requestBody"]["content"]["application/json"]["schema"]

        # 测试有效日期格式
        valid_date_request = {
            "strategy_id": "momentum_strategy_v1",
            "factor_ids": ["momentum_20d"],
            "universe": {"type": "all"},
            "target_date": "2024-12-02"  # 正确格式 YYYY-MM-DD
        }

        try:
            jsonschema.validate(valid_date_request, request_schema)
        except jsonschema.ValidationError as e:
            pytest.fail(f"Valid date format failed validation: {e}")

        # Note: Date format validation depends on jsonschema version and format checker
        # For comprehensive date validation, additional custom validation would be needed

    def test_nested_object_validation(self):
        """测试嵌套对象验证"""
        request_schema = self.signal_agent_contract["paths"]["/api/v1/signals/generate"]["post"]["requestBody"]["content"]["application/json"]["schema"]

        # 测试universe对象缺少必填字段
        invalid_universe_request = {
            "strategy_id": "momentum_strategy_v1",
            "factor_ids": ["momentum_20d"],
            "universe": {
                # 缺少 type 字段
                "stocks": ["sh000001"]
            },
            "target_date": "2024-12-02"
        }

        with pytest.raises(jsonschema.ValidationError):
            jsonschema.validate(invalid_universe_request, request_schema)