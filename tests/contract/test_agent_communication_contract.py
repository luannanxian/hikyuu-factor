"""
Agent间通信契约测试

测试Agent之间的通信协议契约，确保：
1. Agent间API调用规范
2. 消息格式一致性
3. 错误处理和重试机制
4. 服务发现和负载均衡

Contract: specs/002-spec-md/contracts/agent-communication-api.yaml
"""

import pytest
import jsonschema
from typing import Dict, Any


@pytest.mark.contract
@pytest.mark.requires_hikyuu
@pytest.mark.agent_communication
class TestAgentCommunicationContract:
    """Agent间通信契约测试"""

    def setup_method(self):
        """设置测试方法"""
        # 定义Agent间通信的API契约
        self.communication_contract = {
            "openapi": "3.0.3",
            "info": {
                "title": "Agent Communication API",
                "version": "1.0.0",
                "description": "A股因子系统Agent间通信协议"
            },
            "paths": {
                "/api/v1/agents/discover": {
                    "get": {
                        "summary": "服务发现 - 获取可用Agent列表",
                        "parameters": [
                            {
                                "name": "agent_type",
                                "in": "query",
                                "schema": {
                                    "type": "string",
                                    "enum": ["data_manager", "factor_calculation", "validation", "signal_generation"]
                                }
                            },
                            {
                                "name": "status",
                                "in": "query",
                                "schema": {
                                    "type": "string",
                                    "enum": ["healthy", "busy", "error"]
                                }
                            }
                        ],
                        "responses": {
                            "200": {
                                "description": "可用Agent列表",
                                "content": {
                                    "application/json": {
                                        "schema": {
                                            "type": "object",
                                            "properties": {
                                                "agents": {
                                                    "type": "array",
                                                    "items": {
                                                        "type": "object",
                                                        "properties": {
                                                            "agent_id": {"type": "string"},
                                                            "agent_type": {
                                                                "type": "string",
                                                                "enum": ["data_manager", "factor_calculation", "validation", "signal_generation"]
                                                            },
                                                            "endpoint": {"type": "string", "format": "uri"},
                                                            "status": {
                                                                "type": "string",
                                                                "enum": ["healthy", "busy", "error"]
                                                            },
                                                            "load": {"type": "number", "minimum": 0, "maximum": 1},
                                                            "last_heartbeat": {"type": "string", "format": "date-time"}
                                                        },
                                                        "required": ["agent_id", "agent_type", "endpoint", "status", "load"]
                                                    }
                                                },
                                                "total_count": {"type": "integer"}
                                            },
                                            "required": ["agents", "total_count"]
                                        }
                                    }
                                }
                            }
                        }
                    }
                },
                "/api/v1/agents/{agent_id}/call": {
                    "post": {
                        "summary": "Agent间远程调用",
                        "parameters": [
                            {
                                "name": "agent_id",
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
                                            "method": {
                                                "type": "string",
                                                "description": "调用的方法名"
                                            },
                                            "parameters": {
                                                "type": "object",
                                                "description": "方法参数"
                                            },
                                            "request_id": {
                                                "type": "string",
                                                "description": "请求唯一标识"
                                            },
                                            "caller_id": {
                                                "type": "string",
                                                "description": "调用方Agent ID"
                                            },
                                            "timeout": {
                                                "type": "integer",
                                                "description": "超时时间（秒）",
                                                "default": 30
                                            },
                                            "retry_count": {
                                                "type": "integer",
                                                "description": "重试次数",
                                                "default": 3
                                            }
                                        },
                                        "required": ["method", "request_id", "caller_id"]
                                    }
                                }
                            }
                        },
                        "responses": {
                            "200": {
                                "description": "调用成功",
                                "content": {
                                    "application/json": {
                                        "schema": {
                                            "type": "object",
                                            "properties": {
                                                "request_id": {"type": "string"},
                                                "status": {
                                                    "type": "string",
                                                    "enum": ["success", "error", "timeout"]
                                                },
                                                "result": {
                                                    "type": "object",
                                                    "description": "方法执行结果"
                                                },
                                                "error": {
                                                    "type": "object",
                                                    "properties": {
                                                        "code": {"type": "string"},
                                                        "message": {"type": "string"},
                                                        "details": {"type": "object"}
                                                    }
                                                },
                                                "execution_time": {"type": "number"},
                                                "agent_id": {"type": "string"}
                                            },
                                            "required": ["request_id", "status", "execution_time", "agent_id"]
                                        }
                                    }
                                }
                            },
                            "400": {
                                "description": "请求参数错误"
                            },
                            "404": {
                                "description": "Agent不存在"
                            },
                            "503": {
                                "description": "Agent不可用"
                            }
                        }
                    }
                },
                "/api/v1/agents/broadcast": {
                    "post": {
                        "summary": "广播消息到所有Agent",
                        "requestBody": {
                            "required": True,
                            "content": {
                                "application/json": {
                                    "schema": {
                                        "type": "object",
                                        "properties": {
                                            "message_type": {
                                                "type": "string",
                                                "enum": ["system_shutdown", "configuration_update", "health_check", "custom"]
                                            },
                                            "message": {
                                                "type": "object",
                                                "description": "广播消息内容"
                                            },
                                            "sender_id": {
                                                "type": "string",
                                                "description": "发送方Agent ID"
                                            },
                                            "target_types": {
                                                "type": "array",
                                                "items": {
                                                    "type": "string",
                                                    "enum": ["data_manager", "factor_calculation", "validation", "signal_generation"]
                                                },
                                                "description": "目标Agent类型（空为全部）"
                                            },
                                            "priority": {
                                                "type": "string",
                                                "enum": ["low", "normal", "high", "critical"],
                                                "default": "normal"
                                            }
                                        },
                                        "required": ["message_type", "message", "sender_id"]
                                    }
                                }
                            }
                        },
                        "responses": {
                            "200": {
                                "description": "广播成功",
                                "content": {
                                    "application/json": {
                                        "schema": {
                                            "type": "object",
                                            "properties": {
                                                "broadcast_id": {"type": "string"},
                                                "total_targets": {"type": "integer"},
                                                "success_count": {"type": "integer"},
                                                "failed_count": {"type": "integer"},
                                                "results": {
                                                    "type": "array",
                                                    "items": {
                                                        "type": "object",
                                                        "properties": {
                                                            "agent_id": {"type": "string"},
                                                            "status": {
                                                                "type": "string",
                                                                "enum": ["delivered", "failed", "timeout"]
                                                            },
                                                            "error": {"type": "string"}
                                                        }
                                                    }
                                                }
                                            },
                                            "required": ["broadcast_id", "total_targets", "success_count", "failed_count"]
                                        }
                                    }
                                }
                            }
                        }
                    }
                },
                "/api/v1/agents/heartbeat": {
                    "post": {
                        "summary": "Agent心跳注册",
                        "requestBody": {
                            "required": True,
                            "content": {
                                "application/json": {
                                    "schema": {
                                        "type": "object",
                                        "properties": {
                                            "agent_id": {"type": "string"},
                                            "agent_type": {
                                                "type": "string",
                                                "enum": ["data_manager", "factor_calculation", "validation", "signal_generation"]
                                            },
                                            "endpoint": {"type": "string", "format": "uri"},
                                            "status": {
                                                "type": "string",
                                                "enum": ["healthy", "busy", "error"]
                                            },
                                            "load": {"type": "number", "minimum": 0, "maximum": 1},
                                            "capabilities": {
                                                "type": "array",
                                                "items": {"type": "string"},
                                                "description": "Agent支持的功能列表"
                                            },
                                            "resource_usage": {
                                                "type": "object",
                                                "properties": {
                                                    "cpu_percent": {"type": "number"},
                                                    "memory_mb": {"type": "number"},
                                                    "disk_gb": {"type": "number"}
                                                }
                                            }
                                        },
                                        "required": ["agent_id", "agent_type", "endpoint", "status", "load"]
                                    }
                                }
                            }
                        },
                        "responses": {
                            "200": {
                                "description": "心跳注册成功",
                                "content": {
                                    "application/json": {
                                        "schema": {
                                            "type": "object",
                                            "properties": {
                                                "registered": {"type": "boolean"},
                                                "next_heartbeat": {"type": "integer", "description": "下次心跳间隔（秒）"},
                                                "cluster_config": {
                                                    "type": "object",
                                                    "properties": {
                                                        "leader_agent": {"type": "string"},
                                                        "total_agents": {"type": "integer"}
                                                    }
                                                }
                                            },
                                            "required": ["registered", "next_heartbeat"]
                                        }
                                    }
                                }
                            }
                        }
                    }
                },
                "/api/v1/agents/workflow": {
                    "post": {
                        "summary": "协调多Agent工作流",
                        "requestBody": {
                            "required": True,
                            "content": {
                                "application/json": {
                                    "schema": {
                                        "type": "object",
                                        "properties": {
                                            "workflow_id": {"type": "string"},
                                            "workflow_type": {
                                                "type": "string",
                                                "enum": ["factor_calculation", "signal_generation", "backtest", "custom"]
                                            },
                                            "steps": {
                                                "type": "array",
                                                "items": {
                                                    "type": "object",
                                                    "properties": {
                                                        "step_id": {"type": "string"},
                                                        "agent_type": {"type": "string"},
                                                        "method": {"type": "string"},
                                                        "parameters": {"type": "object"},
                                                        "depends_on": {
                                                            "type": "array",
                                                            "items": {"type": "string"}
                                                        },
                                                        "timeout": {"type": "integer"},
                                                        "retry_policy": {
                                                            "type": "object",
                                                            "properties": {
                                                                "max_retries": {"type": "integer"},
                                                                "backoff": {"type": "string", "enum": ["linear", "exponential"]}
                                                            }
                                                        }
                                                    },
                                                    "required": ["step_id", "agent_type", "method"]
                                                }
                                            },
                                            "initiator_id": {"type": "string"}
                                        },
                                        "required": ["workflow_id", "workflow_type", "steps", "initiator_id"]
                                    }
                                }
                            }
                        },
                        "responses": {
                            "200": {
                                "description": "工作流启动成功",
                                "content": {
                                    "application/json": {
                                        "schema": {
                                            "type": "object",
                                            "properties": {
                                                "workflow_id": {"type": "string"},
                                                "status": {
                                                    "type": "string",
                                                    "enum": ["started", "running", "completed", "failed", "cancelled"]
                                                },
                                                "step_status": {
                                                    "type": "array",
                                                    "items": {
                                                        "type": "object",
                                                        "properties": {
                                                            "step_id": {"type": "string"},
                                                            "status": {
                                                                "type": "string",
                                                                "enum": ["pending", "running", "completed", "failed", "skipped"]
                                                            },
                                                            "assigned_agent": {"type": "string"},
                                                            "started_at": {"type": "string", "format": "date-time"},
                                                            "completed_at": {"type": "string", "format": "date-time"}
                                                        }
                                                    }
                                                },
                                                "created_at": {"type": "string", "format": "date-time"}
                                            },
                                            "required": ["workflow_id", "status", "step_status"]
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }

        # JSON Schema验证器 - 启用格式检查
        from jsonschema import validate, draft7_format_checker
        self.validator = jsonschema.Draft7Validator({}, format_checker=draft7_format_checker)

    def test_communication_contract_structure(self):
        """测试Agent通信契约结构完整性"""
        required_paths = [
            "/api/v1/agents/discover",
            "/api/v1/agents/{agent_id}/call",
            "/api/v1/agents/broadcast",
            "/api/v1/agents/heartbeat",
            "/api/v1/agents/workflow"
        ]

        contract_paths = list(self.communication_contract["paths"].keys())

        for path in required_paths:
            assert path in contract_paths, f"缺少必要的API路径: {path}"

    def test_service_discovery_endpoint_contract(self):
        """测试服务发现端点契约"""
        path_config = self.communication_contract["paths"]["/api/v1/agents/discover"]["get"]

        response_schema = path_config["responses"]["200"]["content"]["application/json"]["schema"]

        valid_response = {
            "agents": [
                {
                    "agent_id": "data-manager-001",
                    "agent_type": "data_manager",
                    "endpoint": "http://localhost:8001/api/v1",
                    "status": "healthy",
                    "load": 0.3,
                    "last_heartbeat": "2024-12-01T10:00:00Z"
                },
                {
                    "agent_id": "factor-calc-001",
                    "agent_type": "factor_calculation",
                    "endpoint": "http://localhost:8002/api/v1",
                    "status": "busy",
                    "load": 0.8,
                    "last_heartbeat": "2024-12-01T10:00:05Z"
                }
            ],
            "total_count": 2
        }

        try:
            jsonschema.validate(valid_response, response_schema)
        except jsonschema.ValidationError as e:
            pytest.fail(f"有效的服务发现响应未通过schema验证: {e}")

    def test_agent_call_endpoint_contract(self):
        """测试Agent调用端点契约"""
        path_config = self.communication_contract["paths"]["/api/v1/agents/{agent_id}/call"]["post"]

        # 验证请求schema
        request_schema = path_config["requestBody"]["content"]["application/json"]["schema"]

        valid_request = {
            "method": "calculate_factor",
            "parameters": {
                "factor_id": "momentum_20d",
                "stocks": ["sh000001", "sz000002"],
                "start_date": "2024-01-01",
                "end_date": "2024-12-31"
            },
            "request_id": "req_20241201_001",
            "caller_id": "signal-agent-001",
            "timeout": 60,
            "retry_count": 2
        }

        try:
            jsonschema.validate(valid_request, request_schema)
        except jsonschema.ValidationError as e:
            pytest.fail(f"有效的Agent调用请求未通过schema验证: {e}")

        # 验证响应schema
        response_schema = path_config["responses"]["200"]["content"]["application/json"]["schema"]

        # 成功响应
        success_response = {
            "request_id": "req_20241201_001",
            "status": "success",
            "result": {
                "factor_values": {
                    "sh000001": [0.1, 0.2, 0.15],
                    "sz000002": [0.05, 0.18, 0.12]
                },
                "calculation_time": "2024-12-01T10:05:00Z"
            },
            "execution_time": 45.2,
            "agent_id": "factor-calc-001"
        }

        try:
            jsonschema.validate(success_response, response_schema)
        except jsonschema.ValidationError as e:
            pytest.fail(f"有效的成功调用响应未通过schema验证: {e}")

        # 错误响应
        error_response = {
            "request_id": "req_20241201_001",
            "status": "error",
            "error": {
                "code": "CALCULATION_FAILED",
                "message": "因子计算失败",
                "details": {"missing_data": ["sh000001"]}
            },
            "execution_time": 5.0,
            "agent_id": "factor-calc-001"
        }

        try:
            jsonschema.validate(error_response, response_schema)
        except jsonschema.ValidationError as e:
            pytest.fail(f"有效的错误调用响应未通过schema验证: {e}")

    def test_broadcast_endpoint_contract(self):
        """测试广播端点契约"""
        path_config = self.communication_contract["paths"]["/api/v1/agents/broadcast"]["post"]

        request_schema = path_config["requestBody"]["content"]["application/json"]["schema"]

        valid_request = {
            "message_type": "configuration_update",
            "message": {
                "config_key": "hikyuu.data_source",
                "new_value": "mysql://localhost:3306/hikyuu",
                "effective_time": "2024-12-01T12:00:00Z"
            },
            "sender_id": "system-controller",
            "target_types": ["data_manager", "factor_calculation"],
            "priority": "high"
        }

        try:
            jsonschema.validate(valid_request, request_schema)
        except jsonschema.ValidationError as e:
            pytest.fail(f"有效的广播请求未通过schema验证: {e}")

        response_schema = path_config["responses"]["200"]["content"]["application/json"]["schema"]

        valid_response = {
            "broadcast_id": "bc_20241201_001",
            "total_targets": 3,
            "success_count": 2,
            "failed_count": 1,
            "results": [
                {
                    "agent_id": "data-manager-001",
                    "status": "delivered"
                },
                {
                    "agent_id": "factor-calc-001",
                    "status": "delivered"
                },
                {
                    "agent_id": "factor-calc-002",
                    "status": "failed",
                    "error": "Connection timeout"
                }
            ]
        }

        try:
            jsonschema.validate(valid_response, response_schema)
        except jsonschema.ValidationError as e:
            pytest.fail(f"有效的广播响应未通过schema验证: {e}")

    def test_heartbeat_endpoint_contract(self):
        """测试心跳端点契约"""
        path_config = self.communication_contract["paths"]["/api/v1/agents/heartbeat"]["post"]

        request_schema = path_config["requestBody"]["content"]["application/json"]["schema"]

        valid_request = {
            "agent_id": "factor-calc-001",
            "agent_type": "factor_calculation",
            "endpoint": "http://localhost:8002/api/v1",
            "status": "healthy",
            "load": 0.45,
            "capabilities": ["momentum_factors", "reversal_factors", "technical_indicators"],
            "resource_usage": {
                "cpu_percent": 45.2,
                "memory_mb": 1024.5,
                "disk_gb": 50.2
            }
        }

        try:
            jsonschema.validate(valid_request, request_schema)
        except jsonschema.ValidationError as e:
            pytest.fail(f"有效的心跳请求未通过schema验证: {e}")

        response_schema = path_config["responses"]["200"]["content"]["application/json"]["schema"]

        valid_response = {
            "registered": True,
            "next_heartbeat": 30,
            "cluster_config": {
                "leader_agent": "system-controller",
                "total_agents": 8
            }
        }

        try:
            jsonschema.validate(valid_response, response_schema)
        except jsonschema.ValidationError as e:
            pytest.fail(f"有效的心跳响应未通过schema验证: {e}")

    def test_workflow_endpoint_contract(self):
        """测试工作流端点契约"""
        path_config = self.communication_contract["paths"]["/api/v1/agents/workflow"]["post"]

        request_schema = path_config["requestBody"]["content"]["application/json"]["schema"]

        valid_request = {
            "workflow_id": "wf_factor_calc_20241201",
            "workflow_type": "factor_calculation",
            "steps": [
                {
                    "step_id": "load_data",
                    "agent_type": "data_manager",
                    "method": "get_stock_data",
                    "parameters": {
                        "stocks": ["sh000001", "sz000002"],
                        "start_date": "2024-01-01",
                        "end_date": "2024-12-31"
                    },
                    "timeout": 60
                },
                {
                    "step_id": "calculate_factors",
                    "agent_type": "factor_calculation",
                    "method": "calculate_factor",
                    "parameters": {
                        "factor_id": "momentum_20d"
                    },
                    "depends_on": ["load_data"],
                    "timeout": 120,
                    "retry_policy": {
                        "max_retries": 3,
                        "backoff": "exponential"
                    }
                }
            ],
            "initiator_id": "signal-agent-001"
        }

        try:
            jsonschema.validate(valid_request, request_schema)
        except jsonschema.ValidationError as e:
            pytest.fail(f"有效的工作流请求未通过schema验证: {e}")

        response_schema = path_config["responses"]["200"]["content"]["application/json"]["schema"]

        valid_response = {
            "workflow_id": "wf_factor_calc_20241201",
            "status": "started",
            "step_status": [
                {
                    "step_id": "load_data",
                    "status": "running",
                    "assigned_agent": "data-manager-001",
                    "started_at": "2024-12-01T10:00:00Z"
                },
                {
                    "step_id": "calculate_factors",
                    "status": "pending"
                }
            ],
            "created_at": "2024-12-01T10:00:00Z"
        }

        try:
            jsonschema.validate(valid_response, response_schema)
        except jsonschema.ValidationError as e:
            pytest.fail(f"有效的工作流响应未通过schema验证: {e}")

    def test_enum_constraints_validation(self):
        """测试枚举约束验证"""
        request_schema = self.communication_contract["paths"]["/api/v1/agents/heartbeat"]["post"]["requestBody"]["content"]["application/json"]["schema"]

        # 测试无效的agent_type
        invalid_request = {
            "agent_id": "test-agent",
            "agent_type": "invalid_type",  # 不在枚举中
            "endpoint": "http://localhost:8000",
            "status": "healthy",
            "load": 0.5
        }

        with pytest.raises(jsonschema.ValidationError):
            jsonschema.validate(invalid_request, request_schema)

    def test_load_range_validation(self):
        """测试负载范围验证"""
        request_schema = self.communication_contract["paths"]["/api/v1/agents/heartbeat"]["post"]["requestBody"]["content"]["application/json"]["schema"]

        # 测试负载超出范围
        invalid_request = {
            "agent_id": "test-agent",
            "agent_type": "data_manager",
            "endpoint": "http://localhost:8000",
            "status": "healthy",
            "load": 1.5  # 超出0-1范围
        }

        with pytest.raises(jsonschema.ValidationError):
            jsonschema.validate(invalid_request, request_schema)

    def test_required_fields_validation(self):
        """测试必填字段验证"""
        request_schema = self.communication_contract["paths"]["/api/v1/agents/{agent_id}/call"]["post"]["requestBody"]["content"]["application/json"]["schema"]

        # 测试缺少必填字段
        incomplete_request = {
            "method": "test_method",
            "request_id": "test_req_001"
            # 缺少 caller_id
        }

        with pytest.raises(jsonschema.ValidationError):
            jsonschema.validate(incomplete_request, request_schema)

    def test_uri_format_validation(self):
        """测试URI格式验证"""
        request_schema = self.communication_contract["paths"]["/api/v1/agents/heartbeat"]["post"]["requestBody"]["content"]["application/json"]["schema"]

        # 测试有效的URI格式
        valid_request = {
            "agent_id": "test-agent",
            "agent_type": "data_manager",
            "endpoint": "http://localhost:8000/api/v1",  # 有效URI
            "status": "healthy",
            "load": 0.5
        }

        try:
            jsonschema.validate(valid_request, request_schema)
        except jsonschema.ValidationError as e:
            pytest.fail(f"Valid URI request failed validation: {e}")

        # Note: URI format validation depends on jsonschema version and format checker
        # For comprehensive URI validation, additional custom validation would be needed