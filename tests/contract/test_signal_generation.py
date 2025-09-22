"""
Signal Generation API Contract Tests (T023)
测试信号生成API的合约 (FR-003, FR-004, FR-012)
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


class TestSignalGenerationAPI:
    """信号生成API合约测试"""

    def setup_method(self):
        self.base_url = get_test_base_url("signal")  # Port 8084
        self.generate_endpoint = f"{self.base_url}/api/v1/signal/generate"
        self.confirm_endpoint = f"{self.base_url}/api/v1/signal/confirm"
        self.explanation_endpoint = f"{self.base_url}/api/v1/signal/explanation"

    def test_signal_generation_success_contract(self):
        """T023-1: 信号生成成功响应合约 (FR-003)"""
        # 构造信号生成请求
        request_data = {
            "strategy_id": "momentum_v1",
            "factors": ["momentum_20d", "reversal_5d", "volume_factor"],
            "universe": "hs300",
            "trade_date": "2023-12-01",
            "position_size": 1000000,  # 100万资金
            "max_positions": 50,
            "sector_neutral": True,
            "risk_budget": 0.15,
            "transaction_cost": 0.003,
            "human_confirmation_required": True  # FR-003强制人工确认
        }

        # 定义期望的响应schema
        expected_schema = create_api_test_schema({
            "signal_id": {"type": "string", "pattern": "^sig_[a-f0-9]{32}$"},
            "strategy_id": {"type": "string"},
            "status": {"type": "string", "enum": ["pending_confirmation", "confirmed", "rejected"]},
            "trade_date": {"type": "string", "format": "date"},
            "signals": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "stock_code": {"type": "string", "pattern": "^(sh|sz)[0-9]{6}$"},
                        "stock_name": {"type": "string"},
                        "action": {"type": "string", "enum": ["buy", "sell", "hold"]},
                        "target_weight": {"type": "number", "minimum": 0, "maximum": 1},
                        "current_weight": {"type": "number", "minimum": 0, "maximum": 1},
                        "weight_change": {"type": "number"},
                        "confidence": {"type": "number", "minimum": 0, "maximum": 1},
                        "expected_return": {"type": "number"},
                        "risk_contribution": {"type": "number", "minimum": 0},
                        "sector": {"type": "string"},
                        "market_cap": {"type": "number", "minimum": 0}
                    },
                    "required": ["stock_code", "action", "target_weight", "confidence"]
                }
            },
            "portfolio_metrics": {
                "type": "object",
                "properties": {
                    "total_positions": {"type": "integer", "minimum": 0},
                    "active_positions": {"type": "integer", "minimum": 0},
                    "turnover": {"type": "number", "minimum": 0, "maximum": 1},
                    "expected_return": {"type": "number"},
                    "expected_risk": {"type": "number", "minimum": 0},
                    "sharpe_ratio": {"type": "number"},
                    "max_weight": {"type": "number", "minimum": 0, "maximum": 1},
                    "sector_concentration": {"type": "object"}
                },
                "required": ["total_positions", "turnover", "expected_return"]
            },
            "risk_analysis": {
                "type": "object",
                "properties": {
                    "total_risk": {"type": "number", "minimum": 0},
                    "factor_risk": {"type": "number", "minimum": 0},
                    "specific_risk": {"type": "number", "minimum": 0},
                    "risk_decomposition": {"type": "object"},
                    "stress_test": {"type": "object"},
                    "var_95": {"type": "number"},
                    "expected_shortfall": {"type": "number"}
                },
                "required": ["total_risk", "factor_risk", "specific_risk"]
            },
            "confirmation_required": {"type": "boolean"},
            "expires_at": {"type": "string", "format": "date-time"},
            "created_at": {"type": "string", "format": "date-time"}
        })

        # 执行请求
        response = requests.post(
            self.generate_endpoint,
            json=request_data,
            headers={"Content-Type": "application/json"}
        )

        # 验证合约 (FR-003: 强制人工确认)
        assert response.status_code == 201  # Created
        assert_response_time(response, 900000)  # 15分钟内完成 (FR-012)
        assert_json_schema(response.json(), expected_schema)

        # 验证业务逻辑
        data = response.json()
        assert data["status"] == "pending_confirmation"  # FR-003
        assert data["confirmation_required"] is True
        assert data["signal_id"].startswith("sig_")
        assert len(data["signals"]) <= request_data["max_positions"]

        # 验证A股股票代码格式
        for signal in data["signals"]:
            stock_code = signal["stock_code"]
            assert stock_code.startswith(("sh", "sz"))
            assert len(stock_code) == 8  # sh/sz + 6位数字

    def test_signal_confirmation_contract(self):
        """T023-2: 信号确认合约 (FR-003)"""
        signal_id = "sig_a1b2c3d4e5f67890abcdef1234567890"

        # 确认信号请求
        confirmation_data = {
            "signal_id": signal_id,
            "action": "confirm",
            "user_id": "trader_001",
            "confirmation_notes": "已验证因子有效性和风险控制",
            "position_adjustments": [
                {
                    "stock_code": "sh000001",
                    "target_weight": 0.025,  # 人工调整权重
                    "reason": "降低集中度风险"
                }
            ]
        }

        response = requests.post(
            self.confirm_endpoint,
            json=confirmation_data,
            headers={"Content-Type": "application/json"}
        )

        # 验证确认响应
        assert response.status_code == 200
        assert_response_time(response, 200)

        data = response.json()
        assert data["signal_id"] == signal_id
        assert data["status"] == "confirmed"
        assert "confirmed_at" in data
        assert "confirmed_by" in data
        assert data["confirmed_by"] == confirmation_data["user_id"]

    def test_signal_rejection_contract(self):
        """T023-3: 信号拒绝合约 (FR-003)"""
        signal_id = "sig_b2c3d4e5f67890abcdef1234567890a1"

        # 拒绝信号请求
        rejection_data = {
            "signal_id": signal_id,
            "action": "reject",
            "user_id": "trader_001",
            "rejection_reason": "市场环境不适合当前策略",
            "rejection_notes": "近期波动率异常，暂停交易"
        }

        response = requests.post(
            self.confirm_endpoint,
            json=rejection_data,
            headers={"Content-Type": "application/json"}
        )

        assert response.status_code == 200
        assert_response_time(response, 200)

        data = response.json()
        assert data["signal_id"] == signal_id
        assert data["status"] == "rejected"
        assert "rejected_at" in data
        assert "rejected_by" in data
        assert data["rejection_reason"] == rejection_data["rejection_reason"]

    def test_signal_explanation_contract(self):
        """T023-4: 可解释信号合约 (FR-004)"""
        signal_id = "sig_c3d4e5f67890abcdef1234567890a1b2"

        response = requests.get(f"{self.explanation_endpoint}/{signal_id}")

        # 定义可解释性响应schema
        expected_schema = create_api_test_schema({
            "signal_id": {"type": "string"},
            "explanation": {
                "type": "object",
                "properties": {
                    "factor_contributions": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "factor_name": {"type": "string"},
                                "contribution": {"type": "number"},
                                "importance": {"type": "number", "minimum": 0, "maximum": 1},
                                "direction": {"type": "string", "enum": ["positive", "negative", "neutral"]},
                                "confidence": {"type": "number", "minimum": 0, "maximum": 1}
                            },
                            "required": ["factor_name", "contribution", "importance"]
                        }
                    },
                    "decision_tree": {
                        "type": "object",
                        "properties": {
                            "root_condition": {"type": "string"},
                            "branches": {"type": "array"},
                            "leaf_decisions": {"type": "array"}
                        }
                    },
                    "similar_cases": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "date": {"type": "string", "format": "date"},
                                "similarity": {"type": "number", "minimum": 0, "maximum": 1},
                                "outcome": {"type": "string"},
                                "performance": {"type": "number"}
                            }
                        }
                    },
                    "risk_factors": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "risk_type": {"type": "string"},
                                "probability": {"type": "number", "minimum": 0, "maximum": 1},
                                "impact": {"type": "string", "enum": ["low", "medium", "high"]},
                                "mitigation": {"type": "string"}
                            }
                        }
                    }
                },
                "required": ["factor_contributions", "decision_tree", "risk_factors"]
            },
            "generation_metadata": {
                "type": "object",
                "properties": {
                    "model_version": {"type": "string"},
                    "explanation_method": {"type": "string"},
                    "computation_time": {"type": "number", "minimum": 0},
                    "data_quality_score": {"type": "number", "minimum": 0, "maximum": 1}
                }
            }
        })

        assert response.status_code == 200
        assert_response_time(response, 500)  # 解释生成可能较慢
        assert_json_schema(response.json(), expected_schema)

        # 验证解释的合理性
        data = response.json()
        explanation = data["explanation"]

        # 因子贡献度总和应该接近1
        total_importance = sum(
            factor["importance"] for factor in explanation["factor_contributions"]
        )
        assert 0.8 <= total_importance <= 1.2  # 允许一定误差

    def test_daily_signal_update_contract(self):
        """T023-5: 每日信号更新合约 (FR-012)"""
        update_data = {
            "update_date": "2023-12-01",
            "strategies": ["momentum_v1", "mean_reversion_v2"],
            "force_update": False,
            "notification_enabled": True,
            "batch_size": 100
        }

        daily_update_endpoint = f"{self.base_url}/api/v1/signal/daily-update"

        response = requests.post(
            daily_update_endpoint,
            json=update_data,
            headers={"Content-Type": "application/json"}
        )

        # 每日更新应该在15分钟内完成 (FR-012)
        assert response.status_code == 202  # Accepted异步处理
        assert_response_time(response, 200)

        data = response.json()
        assert "update_id" in data
        assert "estimated_completion" in data
        assert data["status"] == "running"

        # 验证估计完成时间在15分钟内
        import datetime
        estimated_time = datetime.datetime.fromisoformat(
            data["estimated_completion"].replace("Z", "+00:00")
        )
        now = datetime.datetime.now(datetime.timezone.utc)
        time_diff = estimated_time - now
        assert time_diff.total_seconds() <= 900  # 15分钟

    def test_signal_generation_parameter_validation(self):
        """T023-6: 信号生成参数校验合约"""
        # 测试各种无效参数
        invalid_requests = [
            # 缺失必要参数
            {
                "strategy_id": "momentum_v1"
                # 缺失factors
            },
            # 无效的股票池
            {
                "strategy_id": "momentum_v1",
                "factors": ["momentum_20d"],
                "universe": "invalid_universe"
            },
            # 无效的日期格式
            {
                "strategy_id": "momentum_v1",
                "factors": ["momentum_20d"],
                "universe": "hs300",
                "trade_date": "2023-13-01"  # 无效月份
            },
            # 风险预算超出范围
            {
                "strategy_id": "momentum_v1",
                "factors": ["momentum_20d"],
                "universe": "hs300",
                "trade_date": "2023-12-01",
                "risk_budget": 1.5  # 超过100%
            }
        ]

        for invalid_request in invalid_requests:
            response = requests.post(
                self.generate_endpoint,
                json=invalid_request,
                headers={"Content-Type": "application/json"}
            )

            assert_api_error_response(response, 400)
            assert_response_time(response, 200)

    def test_signal_not_found_contract(self):
        """T023-7: 信号不存在错误合约"""
        nonexistent_signal_id = "sig_nonexistent1234567890abcdef123"

        response = requests.get(f"{self.explanation_endpoint}/{nonexistent_signal_id}")

        assert_api_error_response(response, 404)
        assert_response_time(response, 100)

        error_data = response.json()
        assert "signal_id" in error_data["error"]["details"]
        assert nonexistent_signal_id in error_data["error"]["message"]