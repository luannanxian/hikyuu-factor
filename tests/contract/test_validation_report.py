"""
Validation Report API Contract Tests (T022)
测试验证报告API的合约 (FR-006)
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


class TestValidationReportAPI:
    """验证报告API合约测试"""

    def setup_method(self):
        self.base_url = get_test_base_url("validation")  # Port 8083
        self.report_endpoint = f"{self.base_url}/api/v1/validation/report"

    def test_validation_report_success_contract(self):
        """T022-1: 验证报告成功响应合约"""
        validation_id = "val_a1b2c3d4e5f67890abcdef1234567890"

        # 定义期望的综合验证报告schema (FR-006)
        expected_schema = create_api_test_schema({
            "validation_id": {"type": "string"},
            "factor_id": {"type": "string"},
            "status": {"type": "string", "enum": ["completed", "running", "failed"]},
            "report": {
                "type": "object",
                "properties": {
                    "summary": {
                        "type": "object",
                        "properties": {
                            "overall_score": {"type": "number", "minimum": 0, "maximum": 100},
                            "risk_level": {"type": "string", "enum": ["low", "medium", "high"]},
                            "recommendation": {"type": "string", "enum": ["strong_buy", "buy", "hold", "sell", "strong_sell"]},
                            "confidence": {"type": "number", "minimum": 0, "maximum": 1}
                        },
                        "required": ["overall_score", "risk_level", "recommendation"]
                    },
                    "ic_analysis": {
                        "type": "object",
                        "properties": {
                            "mean_ic": {"type": "number"},
                            "ic_std": {"type": "number", "minimum": 0},
                            "ic_ir": {"type": "number"},  # IC信息比率
                            "ic_hit_rate": {"type": "number", "minimum": 0, "maximum": 1},
                            "rank_ic": {"type": "number"},
                            "monthly_ic": {"type": "array", "items": {"type": "number"}}
                        },
                        "required": ["mean_ic", "ic_std", "ic_ir", "ic_hit_rate"]
                    },
                    "layered_returns": {
                        "type": "object",
                        "properties": {
                            "layers": {"type": "integer", "minimum": 2},
                            "returns_by_layer": {"type": "array", "items": {"type": "number"}},
                            "cumulative_returns": {"type": "array"},
                            "sharpe_ratios": {"type": "array", "items": {"type": "number"}},
                            "max_drawdowns": {"type": "array", "items": {"type": "number"}},
                            "long_short_spread": {"type": "number"},
                            "spread_sharpe": {"type": "number"}
                        },
                        "required": ["layers", "returns_by_layer", "long_short_spread"]
                    },
                    "turnover_analysis": {
                        "type": "object",
                        "properties": {
                            "mean_turnover": {"type": "number", "minimum": 0, "maximum": 1},
                            "turnover_std": {"type": "number", "minimum": 0},
                            "turnover_cost_impact": {"type": "number"},
                            "net_returns_after_cost": {"type": "number"},
                            "optimal_rebalance_freq": {"type": "string"}
                        },
                        "required": ["mean_turnover", "turnover_cost_impact"]
                    },
                    "risk_metrics": {
                        "type": "object",
                        "properties": {
                            "volatility": {"type": "number", "minimum": 0},
                            "beta": {"type": "number"},
                            "tracking_error": {"type": "number", "minimum": 0},
                            "value_at_risk": {"type": "number"},
                            "expected_shortfall": {"type": "number"},
                            "factor_exposure": {"type": "object"}
                        },
                        "required": ["volatility", "beta", "tracking_error"]
                    }
                },
                "required": ["summary", "ic_analysis", "layered_returns", "turnover_analysis", "risk_metrics"]
            },
            "metadata": {
                "type": "object",
                "properties": {
                    "validation_period": {"type": "object"},
                    "benchmark": {"type": "string"},
                    "universe": {"type": "string"},
                    "sample_size": {"type": "integer", "minimum": 100},
                    "generation_time": {"type": "string", "format": "date-time"},
                    "computation_duration": {"type": "integer", "minimum": 1}
                },
                "required": ["validation_period", "sample_size", "generation_time"]
            }
        })

        # 执行请求
        response = requests.get(f"{self.report_endpoint}/{validation_id}")

        # 验证合约
        assert response.status_code == 200
        assert_response_time(response, 200)
        assert_json_schema(response.json(), expected_schema)

        # 验证业务逻辑
        data = response.json()
        assert data["validation_id"] == validation_id
        assert data["status"] == "completed"

        # 验证综合评分逻辑
        summary = data["report"]["summary"]
        assert 0 <= summary["overall_score"] <= 100

        # 验证IC分析结果合理性
        ic_analysis = data["report"]["ic_analysis"]
        assert -1 <= ic_analysis["mean_ic"] <= 1
        assert ic_analysis["ic_std"] >= 0
        assert 0 <= ic_analysis["ic_hit_rate"] <= 1

    def test_validation_report_running_status(self):
        """T022-2: 运行中验证的报告合约"""
        validation_id = "val_running123456789abcdef0123456789"

        response = requests.get(f"{self.report_endpoint}/{validation_id}")

        # 运行中的验证应该返回状态但没有完整报告
        assert response.status_code == 200
        assert_response_time(response, 100)  # 运行状态查询应该更快

        data = response.json()
        assert data["status"] == "running"
        assert data["validation_id"] == validation_id

        # 可能包含进度信息
        if "progress" in data:
            assert 0 <= data["progress"]["percentage"] <= 100
            assert "current_step" in data["progress"]

    def test_validation_report_not_found(self):
        """T022-3: 验证ID不存在错误合约"""
        nonexistent_id = "val_nonexistent1234567890abcdef12345"

        response = requests.get(f"{self.report_endpoint}/{nonexistent_id}")

        assert_api_error_response(response, 404)
        assert_response_time(response, 100)

        error_data = response.json()
        assert "validation_id" in error_data["error"]["details"]
        assert nonexistent_id in error_data["error"]["message"]

    def test_validation_report_failed_status(self):
        """T022-4: 失败验证的报告合约"""
        failed_validation_id = "val_failed_abcdef1234567890123456789"

        response = requests.get(f"{self.report_endpoint}/{failed_validation_id}")

        assert response.status_code == 200
        assert_response_time(response, 100)

        data = response.json()
        assert data["status"] == "failed"
        assert data["validation_id"] == failed_validation_id

        # 失败的验证应该包含错误信息
        assert "error" in data
        assert "error_message" in data["error"]
        assert "error_code" in data["error"]
        assert "timestamp" in data["error"]

    def test_validation_report_format_parameter(self):
        """T022-5: 报告格式参数合约"""
        validation_id = "val_a1b2c3d4e5f67890abcdef1234567890"

        # 测试不同输出格式
        formats = ["json", "pdf", "excel"]

        for format_type in formats:
            response = requests.get(
                f"{self.report_endpoint}/{validation_id}",
                params={"format": format_type}
            )

            assert response.status_code == 200
            assert_response_time(response, 500)  # PDF/Excel生成可能更慢

            if format_type == "json":
                assert response.headers["Content-Type"] == "application/json"
            elif format_type == "pdf":
                assert response.headers["Content-Type"] == "application/pdf"
                assert "Content-Disposition" in response.headers
            elif format_type == "excel":
                assert "application/" in response.headers["Content-Type"]
                assert "excel" in response.headers["Content-Type"] or "spreadsheet" in response.headers["Content-Type"]

    def test_validation_report_summary_only(self):
        """T022-6: 仅摘要模式合约"""
        validation_id = "val_a1b2c3d4e5f67890abcdef1234567890"

        response = requests.get(
            f"{self.report_endpoint}/{validation_id}",
            params={"summary_only": "true"}
        )

        assert response.status_code == 200
        assert_response_time(response, 100)  # 摘要应该更快

        data = response.json()
        assert "report" in data
        assert "summary" in data["report"]

        # 摘要模式应该只包含关键指标
        summary = data["report"]["summary"]
        required_fields = ["overall_score", "risk_level", "recommendation"]
        for field in required_fields:
            assert field in summary