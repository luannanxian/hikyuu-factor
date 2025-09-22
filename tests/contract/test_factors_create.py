"""
因子创建API契约测试 - Factors Create API Contract Tests

测试 POST /api/v1/factors 端点
根据 factor_calculation_api.yaml 合约规范，验证因子注册功能 (FR-013)
"""
import pytest
import requests
from datetime import datetime
from tests.utils import assert_json_schema, assert_response_time


class TestFactorsCreateContract:
    """
    因子创建API契约测试

    这些测试必须在实现前编写并失败，遵循TDD原则
    """

    def setup_method(self):
        """测试setup - 配置测试环境"""
        self.base_url = "http://localhost:8002"  # Factor Calculation Agent端口
        self.factors_endpoint = f"{self.base_url}/api/v1/factors"

    def test_factors_create_endpoint_exists(self):
        """
        测试: POST /api/v1/factors 端点存在

        期望: 端点应该存在且处理因子注册请求
        当前状态: 应该失败 (端点尚未实现)
        """
        factor_data = {
            "name": "测试动量因子",
            "category": "momentum",
            "hikyuu_formula": "MA(CLOSE(), 5) / MA(CLOSE(), 20) - 1",
            "economic_logic": "短期均线相对长期均线的偏离程度，反映股价的短期动量特征",
            "parameters": {
                "fast_period": 5,
                "slow_period": 20
            }
        }

        response = requests.post(self.factors_endpoint, json=factor_data)

        # 这个测试现在应该失败，因为端点尚未实现
        assert response.status_code in [200, 201], f"Factor create endpoint should exist, got {response.status_code}"

    def test_factors_create_request_schema(self):
        """
        测试: 因子创建请求schema验证

        有效请求格式:
        {
          "name": "20日动量因子",
          "category": "momentum",
          "hikyuu_formula": "MA(CLOSE(), 5) / MA(CLOSE(), 20) - 1",
          "economic_logic": "短期均线相对长期均线的偏离程度",
          "parameters": {
            "fast_period": 5,
            "slow_period": 20
          },
          "description": "详细描述",  // 可选
          "tags": ["技术分析", "动量"],  // 可选
          "public": true  // 可选，默认false
        }
        """
        valid_requests = [
            # 基本请求
            {
                "name": "简单动量因子",
                "category": "momentum",
                "hikyuu_formula": "CLOSE() / MA(CLOSE(), 20) - 1",
                "economic_logic": "价格相对均线的偏离",
                "parameters": {"period": 20}
            },
            # 完整请求
            {
                "name": "高级价值因子",
                "category": "value",
                "hikyuu_formula": "1 / PE_RATIO()",
                "economic_logic": "市盈率的倒数，反映估值吸引力",
                "parameters": {
                    "method": "ttm",
                    "min_pe": 5,
                    "max_pe": 100
                },
                "description": "基于市盈率的价值因子，用于识别低估值股票",
                "tags": ["基本面", "价值投资"],
                "public": True
            },
            # 技术因子
            {
                "name": "RSI反转因子",
                "category": "technical",
                "hikyuu_formula": "IF(RSI(14) > 70, -1, IF(RSI(14) < 30, 1, 0))",
                "economic_logic": "基于RSI指标的超买超卖反转信号",
                "parameters": {
                    "period": 14,
                    "overbought": 70,
                    "oversold": 30
                }
            }
        ]

        for request_data in valid_requests:
            response = requests.post(self.factors_endpoint, json=request_data)
            assert response.status_code in [200, 201], \
                f"Valid request should be accepted: {request_data['name']}"

    def test_factors_create_response_schema(self):
        """
        测试: 因子创建响应schema验证

        期望响应格式:
        {
          "status": "success",
          "data": {
            "id": "momentum_20d_1234",
            "name": "20日动量因子",
            "category": "momentum",
            "version": "1.0.0",
            "status": "active",
            "created_at": "2025-01-15T10:30:00Z",
            "created_by": "factor_engineer_01",
            "validation_required": true,
            "estimated_calculation_time": "5 minutes"
          },
          "message": "Factor registered successfully"
        }
        """
        factor_data = {
            "name": "测试动量因子",
            "category": "momentum",
            "hikyuu_formula": "MA(CLOSE(), 5) / MA(CLOSE(), 20) - 1",
            "economic_logic": "短期均线相对长期均线的偏离程度",
            "parameters": {
                "fast_period": 5,
                "slow_period": 20
            }
        }

        response = requests.post(self.factors_endpoint, json=factor_data)
        assert response.status_code in [200, 201]

        data = response.json()

        # 验证顶级响应结构
        assert "status" in data
        assert "data" in data
        assert "message" in data
        assert data["status"] == "success"

        factor_result = data["data"]

        # 验证因子结果字段
        required_fields = [
            "id", "name", "category", "version", "status",
            "created_at", "created_by"
        ]
        for field in required_fields:
            assert field in factor_result, f"Missing required field: {field}"

        # 验证字段类型和值
        assert isinstance(factor_result["id"], str)
        assert len(factor_result["id"]) > 0
        assert factor_result["name"] == factor_data["name"]
        assert factor_result["category"] == factor_data["category"]
        assert factor_result["status"] in ["active", "pending_validation"]

        # 验证版本号格式
        version = factor_result["version"]
        version_parts = version.split(".")
        assert len(version_parts) >= 2, f"Version should be in semantic format: {version}"

    def test_factor_id_generation(self):
        """
        测试: 因子ID生成规则

        因子ID应该包含类别、特征和唯一标识
        """
        factor_data = {
            "name": "动量测试因子",
            "category": "momentum",
            "hikyuu_formula": "MA(CLOSE(), 10)",
            "economic_logic": "10日均线",
            "parameters": {"period": 10}
        }

        response = requests.post(self.factors_endpoint, json=factor_data)
        assert response.status_code in [200, 201]

        data = response.json()
        factor_id = data["data"]["id"]

        # 验证ID格式：应该包含类别信息
        assert "momentum" in factor_id.lower() or factor_id.startswith("momentum"), \
            f"Factor ID should contain category: {factor_id}"

        # 验证ID唯一性
        response2 = requests.post(self.factors_endpoint, json=factor_data)
        assert response2.status_code in [200, 201]
        factor_id2 = response2.json()["data"]["id"]

        assert factor_id != factor_id2, "Factor IDs should be unique"

    def test_factor_name_validation(self):
        """
        测试: 因子名称验证

        因子名称应该有适当的长度和字符限制
        """
        base_factor = {
            "category": "momentum",
            "hikyuu_formula": "CLOSE()",
            "economic_logic": "测试逻辑",
            "parameters": {}
        }

        # 测试有效名称
        valid_names = [
            "简单测试因子",
            "Test Factor",
            "20日动量因子_v1",
            "MA_Cross_Signal",
            "价值因子-PE修正版"
        ]

        for name in valid_names:
            factor_data = base_factor.copy()
            factor_data["name"] = name

            response = requests.post(self.factors_endpoint, json=factor_data)
            assert response.status_code in [200, 201], \
                f"Valid name should be accepted: {name}"

        # 测试无效名称
        invalid_names = [
            "",  # 空名称
            "a",  # 太短
            "x" * 201,  # 太长 (假设限制200字符)
            "因子@#$%^&*()",  # 特殊字符过多
        ]

        for name in invalid_names:
            factor_data = base_factor.copy()
            factor_data["name"] = name

            response = requests.post(self.factors_endpoint, json=factor_data)
            assert response.status_code == 400, \
                f"Invalid name should be rejected: {name}"

    def test_factor_category_validation(self):
        """
        测试: 因子类别验证

        只允许预定义的因子类别
        """
        base_factor = {
            "name": "测试因子",
            "hikyuu_formula": "CLOSE()",
            "economic_logic": "测试逻辑",
            "parameters": {}
        }

        # 测试有效类别
        valid_categories = ["momentum", "value", "quality", "growth", "risk", "technical"]

        for category in valid_categories:
            factor_data = base_factor.copy()
            factor_data["category"] = category

            response = requests.post(self.factors_endpoint, json=factor_data)
            assert response.status_code in [200, 201], \
                f"Valid category should be accepted: {category}"

        # 测试无效类别
        invalid_categories = ["invalid", "MOMENTUM", "custom", "", "size"]

        for category in invalid_categories:
            factor_data = base_factor.copy()
            factor_data["category"] = category

            response = requests.post(self.factors_endpoint, json=factor_data)
            assert response.status_code == 400, \
                f"Invalid category should be rejected: {category}"

    def test_hikyuu_formula_validation(self):
        """
        测试: Hikyuu公式验证

        验证Hikyuu公式的语法正确性
        """
        base_factor = {
            "name": "测试因子",
            "category": "momentum",
            "economic_logic": "测试逻辑",
            "parameters": {}
        }

        # 测试有效公式
        valid_formulas = [
            "CLOSE()",
            "MA(CLOSE(), 20)",
            "MA(CLOSE(), 5) / MA(CLOSE(), 20) - 1",
            "RSI(14)",
            "MACD().macd",
            "IF(CLOSE() > MA(CLOSE(), 20), 1, 0)",
            "HIGH() - LOW()",
            "VOLUME() / MA(VOLUME(), 10)"
        ]

        for formula in valid_formulas:
            factor_data = base_factor.copy()
            factor_data["hikyuu_formula"] = formula

            response = requests.post(self.factors_endpoint, json=factor_data)
            assert response.status_code in [200, 201], \
                f"Valid formula should be accepted: {formula}"

        # 测试无效公式
        invalid_formulas = [
            "",  # 空公式
            "INVALID_FUNCTION()",
            "CLOSE() +",  # 语法错误
            "python code",  # 非Hikyuu语法
            "CLOSE() / 0",  # 可能的除零错误
        ]

        for formula in invalid_formulas:
            factor_data = base_factor.copy()
            factor_data["hikyuu_formula"] = formula

            response = requests.post(self.factors_endpoint, json=formula_data)
            assert response.status_code == 400, \
                f"Invalid formula should be rejected: {formula}"

    def test_parameters_validation(self):
        """
        测试: 参数验证

        验证因子参数的类型和取值范围
        """
        base_factor = {
            "name": "测试因子",
            "category": "momentum",
            "hikyuu_formula": "MA(CLOSE(), period)",
            "economic_logic": "测试逻辑"
        }

        # 测试有效参数
        valid_parameters = [
            {"period": 20},
            {"fast_period": 5, "slow_period": 20},
            {"threshold": 0.05, "min_value": 0.01},
            {"method": "simple", "normalize": True},
            {}  # 空参数也应该被接受
        ]

        for params in valid_parameters:
            factor_data = base_factor.copy()
            factor_data["parameters"] = params

            response = requests.post(self.factors_endpoint, json=factor_data)
            assert response.status_code in [200, 201], \
                f"Valid parameters should be accepted: {params}"

        # 测试无效参数
        invalid_parameters = [
            {"period": 0},  # 无效周期
            {"period": -5},  # 负值
            {"period": 1000},  # 过大值
            {"threshold": "invalid"},  # 类型错误
        ]

        for params in invalid_parameters:
            factor_data = base_factor.copy()
            factor_data["parameters"] = params

            response = requests.post(self.factors_endpoint, json=factor_data)
            assert response.status_code == 400, \
                f"Invalid parameters should be rejected: {params}"

    def test_duplicate_factor_handling(self):
        """
        测试: 重复因子处理

        同名因子或相似公式的处理策略
        """
        factor_data = {
            "name": "重复测试因子",
            "category": "momentum",
            "hikyuu_formula": "MA(CLOSE(), 20)",
            "economic_logic": "20日均线",
            "parameters": {"period": 20}
        }

        # 创建第一个因子
        response1 = requests.post(self.factors_endpoint, json=factor_data)
        assert response1.status_code in [200, 201]

        # 尝试创建同名因子
        response2 = requests.post(self.factors_endpoint, json=factor_data)

        # 应该被拒绝或创建新版本
        assert response2.status_code in [200, 201, 409]

        if response2.status_code == 409:
            # 冲突响应应该包含错误信息
            data = response2.json()
            assert "status" in data
            assert data["status"] == "error"
            assert "duplicate" in data.get("message", "").lower()

    def test_economic_logic_validation(self):
        """
        测试: 经济逻辑验证

        经济逻辑描述应该有适当的长度
        """
        base_factor = {
            "name": "测试因子",
            "category": "momentum",
            "hikyuu_formula": "CLOSE()",
            "parameters": {}
        }

        # 测试有效经济逻辑
        valid_logic = [
            "短期均线相对长期均线的偏离程度，反映股价的短期动量特征。",
            "This factor captures momentum signals.",
            "基于RSI指标的反转信号，当RSI超过70时做空，低于30时做多。"
        ]

        for logic in valid_logic:
            factor_data = base_factor.copy()
            factor_data["economic_logic"] = logic

            response = requests.post(self.factors_endpoint, json=factor_data)
            assert response.status_code in [200, 201], \
                f"Valid economic logic should be accepted"

        # 测试无效经济逻辑
        invalid_logic = [
            "",  # 空描述
            "短",  # 太短
            "x" * 2001,  # 太长 (假设限制2000字符)
        ]

        for logic in invalid_logic:
            factor_data = base_factor.copy()
            factor_data["economic_logic"] = logic

            response = requests.post(self.factors_endpoint, json=factor_data)
            assert response.status_code == 400, \
                f"Invalid economic logic should be rejected"

    def test_tags_and_description_optional_fields(self):
        """
        测试: 可选字段处理

        tags和description字段应该是可选的
        """
        factor_data = {
            "name": "可选字段测试",
            "category": "technical",
            "hikyuu_formula": "RSI(14)",
            "economic_logic": "RSI指标",
            "parameters": {"period": 14},
            "description": "这是一个详细的描述，包含了因子的使用场景和注意事项。",
            "tags": ["技术分析", "RSI", "反转"],
            "public": True
        }

        response = requests.post(self.factors_endpoint, json=factor_data)
        assert response.status_code in [200, 201]

        # 验证可选字段在响应中
        result = response.json()["data"]
        # 注意：具体的响应字段取决于实现

    def test_missing_required_fields(self):
        """
        测试: 缺少必需字段的错误处理
        """
        # 缺少name
        factor1 = {
            "category": "momentum",
            "hikyuu_formula": "CLOSE()",
            "economic_logic": "测试",
            "parameters": {}
        }

        response1 = requests.post(self.factors_endpoint, json=factor1)
        assert response1.status_code == 400

        # 缺少category
        factor2 = {
            "name": "测试因子",
            "hikyuu_formula": "CLOSE()",
            "economic_logic": "测试",
            "parameters": {}
        }

        response2 = requests.post(self.factors_endpoint, json=factor2)
        assert response2.status_code == 400

        # 缺少hikyuu_formula
        factor3 = {
            "name": "测试因子",
            "category": "momentum",
            "economic_logic": "测试",
            "parameters": {}
        }

        response3 = requests.post(self.factors_endpoint, json=factor3)
        assert response3.status_code == 400

    def test_response_time_performance(self):
        """
        测试: API响应时间应该 < 200ms (性能要求)
        """
        factor_data = {
            "name": "性能测试因子",
            "category": "momentum",
            "hikyuu_formula": "MA(CLOSE(), 20)",
            "economic_logic": "20日均线",
            "parameters": {"period": 20}
        }

        response = requests.post(self.factors_endpoint, json=factor_data)
        assert response.status_code in [200, 201]

        # 检查响应时间
        assert_response_time(response, max_time_ms=200)

    def test_content_type_validation(self):
        """
        测试: Content-Type验证

        应该只接受application/json
        """
        factor_data = {
            "name": "Content-Type测试",
            "category": "momentum",
            "hikyuu_formula": "CLOSE()",
            "economic_logic": "测试",
            "parameters": {}
        }

        # 正确的Content-Type
        response = requests.post(
            self.factors_endpoint,
            json=factor_data,
            headers={"Content-Type": "application/json"}
        )
        assert response.status_code in [200, 201]

        # 错误的Content-Type
        response = requests.post(
            self.factors_endpoint,
            data=str(factor_data),
            headers={"Content-Type": "text/plain"}
        )
        assert response.status_code == 415


# 这个测试文件应该运行失败，因为端点尚未实现
# 这是TDD的正确流程：先写测试，测试失败，然后实现功能使测试通过