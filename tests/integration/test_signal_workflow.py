"""
Signal Workflow Integration Test (T028)
集成测试: 信号生成→风险检查→人工确认→审计
"""

import asyncio
import pytest
import requests
import time
from typing import Dict, Any, List

from tests.integration.test_environment import IntegrationTestBase, integration_test_environment


class TestSignalWorkflowIntegration:
    """信号工作流集成测试"""

    @pytest.mark.asyncio
    async def test_complete_signal_workflow_with_human_confirmation(self):
        """T028: 完整信号生成工作流含人工确认"""
        async with integration_test_environment() as agent_manager:
            test_base = IntegrationTestBase(agent_manager)

            # Step 1: 准备测试策略和因子
            strategy_id = f"test_strategy_{int(time.time())}"
            test_factors = [f"momentum_test_{int(time.time())}", f"reversal_test_{int(time.time())}"]

            # 注册测试因子
            for factor_id in test_factors:
                factor_def = {
                    "factor_id": factor_id,
                    "name": f"信号测试因子 - {factor_id}",
                    "category": "momentum" if "momentum" in factor_id else "reversal",
                    "formula": "CLOSE / REF(CLOSE, 20) - 1" if "momentum" in factor_id else "REF(CLOSE, 5) / CLOSE - 1",
                    "version": "1.0.0"
                }

                register_response = requests.post(
                    test_base.get_api_url("factor", "/api/v1/factors"),
                    json=factor_def
                )
                assert register_response.status_code == 201

            print(f"Test factors registered: {test_factors}")

            # Step 2: 生成交易信号
            signal_request = {
                "strategy_id": strategy_id,
                "factors": test_factors,
                "factor_weights": {test_factors[0]: 0.6, test_factors[1]: 0.4},
                "universe": "custom",
                "stock_universe": ["sh000001", "sh000002", "sz000001", "sz000002", "sh000300"],
                "trade_date": "2023-12-15",
                "position_size": 1000000,  # 100万资金
                "max_positions": 5,
                "sector_neutral": True,
                "risk_budget": 0.15,
                "transaction_cost": 0.003,
                "human_confirmation_required": True,  # 强制人工确认
                "risk_limits": {
                    "max_single_weight": 0.05,  # 单个股票最大权重5%
                    "max_sector_weight": 0.3,   # 单个行业最大权重30%
                    "max_turnover": 0.5,        # 最大换手率50%
                    "min_liquidity": 10000000   # 最小流动性1000万
                }
            }

            signal_response = requests.post(
                self.get_api_url("signal", "/api/v1/signal/generate"),
                json=signal_request
            )
            assert signal_response.status_code == 201

            signal_data = signal_response.json()
            signal_id = signal_data["signal_id"]
            assert signal_data["status"] == "pending_confirmation"
            assert signal_data["confirmation_required"] is True

            print(f"Signal generated: {signal_id}, status: {signal_data['status']}")

            # Step 3: 验证信号内容和风险检查
            assert "signals" in signal_data
            assert "portfolio_metrics" in signal_data
            assert "risk_analysis" in signal_data

            signals = signal_data["signals"]
            portfolio_metrics = signal_data["portfolio_metrics"]
            risk_analysis = signal_data["risk_analysis"]

            # 验证信号数量不超过限制
            assert len(signals) <= signal_request["max_positions"]

            # 验证风险限制
            max_weight = max(signal["target_weight"] for signal in signals)
            assert max_weight <= signal_request["risk_limits"]["max_single_weight"]

            # 验证股票代码格式
            for signal in signals:
                assert signal["stock_code"].startswith(("sh", "sz"))
                assert len(signal["stock_code"]) == 8
                assert signal["action"] in ["buy", "sell", "hold"]
                assert 0 <= signal["confidence"] <= 1

            print(f"Generated {len(signals)} signals with max weight {max_weight:.2%}")

            # 验证组合指标
            assert "total_positions" in portfolio_metrics
            assert "turnover" in portfolio_metrics
            assert "expected_return" in portfolio_metrics
            assert portfolio_metrics["turnover"] <= signal_request["risk_limits"]["max_turnover"]

            # 验证风险分析
            assert "total_risk" in risk_analysis
            assert "factor_risk" in risk_analysis
            assert "specific_risk" in risk_analysis
            assert risk_analysis["total_risk"] >= 0

            # Step 4: 获取信号解释
            explanation_response = requests.get(
                self.get_api_url("signal", f"/api/v1/signal/explanation/{signal_id}")
            )
            assert explanation_response.status_code == 200

            explanation_data = explanation_response.json()
            assert "explanation" in explanation_data
            explanation = explanation_data["explanation"]

            assert "factor_contributions" in explanation
            assert "decision_tree" in explanation
            assert "risk_factors" in explanation

            # 验证因子贡献度
            factor_contributions = explanation["factor_contributions"]
            assert len(factor_contributions) >= len(test_factors)

            total_importance = sum(contrib["importance"] for contrib in factor_contributions)
            assert 0.8 <= total_importance <= 1.2  # 允许一定误差

            print(f"Signal explanation generated with {len(factor_contributions)} factor contributions")

            # Step 5: 模拟人工确认流程
            # 首先进行人工审核调整
            position_adjustments = []

            # 如果某个股票权重过高，调整降低
            for signal in signals:
                if signal["target_weight"] > 0.04:  # 超过4%
                    position_adjustments.append({
                        "stock_code": signal["stock_code"],
                        "target_weight": 0.04,
                        "reason": "风险控制 - 降低集中度"
                    })

            confirmation_request = {
                "signal_id": signal_id,
                "action": "confirm",
                "user_id": "trader_test_001",
                "confirmation_notes": "经过风险审核，适当调整权重后确认执行",
                "position_adjustments": position_adjustments,
                "risk_override": False,
                "execution_timing": "market_open",
                "confirmation_timestamp": time.time()
            }

            confirmation_response = requests.post(
                self.get_api_url("signal", "/api/v1/signal/confirm"),
                json=confirmation_request
            )
            assert confirmation_response.status_code == 200

            confirmation_data = confirmation_response.json()
            assert confirmation_data["signal_id"] == signal_id
            assert confirmation_data["status"] == "confirmed"
            assert "confirmed_at" in confirmation_data
            assert "confirmed_by" in confirmation_data
            assert confirmation_data["confirmed_by"] == confirmation_request["user_id"]

            print(f"Signal confirmed by {confirmation_data['confirmed_by']} with {len(position_adjustments)} adjustments")

            # Step 6: 验证审计记录
            audit_response = requests.get(
                self.get_api_url("signal", f"/api/v1/signal/{signal_id}/audit")
            )
            assert audit_response.status_code == 200

            audit_data = audit_response.json()
            assert "audit_trail" in audit_data

            audit_trail = audit_data["audit_trail"]
            assert len(audit_trail) >= 2  # 至少包含生成和确认两个事件

            # 验证审计事件类型
            event_types = [event["event_type"] for event in audit_trail]
            assert "signal_generated" in event_types
            assert "signal_confirmed" in event_types

            # 验证审计记录完整性
            for event in audit_trail:
                assert "timestamp" in event
                assert "event_type" in event
                assert "user_id" in event or "system" in event
                assert "details" in event

            print(f"Audit trail verified with {len(audit_trail)} events")

            # Step 7: 测试信号拒绝流程
            # 生成另一个信号用于测试拒绝
            rejection_signal_request = signal_request.copy()
            rejection_signal_request["strategy_id"] = f"{strategy_id}_rejection"

            rejection_response = requests.post(
                self.get_api_url("signal", "/api/v1/signal/generate"),
                json=rejection_signal_request
            )
            assert rejection_response.status_code == 201

            rejection_signal_id = rejection_response.json()["signal_id"]

            # 拒绝信号
            rejection_request = {
                "signal_id": rejection_signal_id,
                "action": "reject",
                "user_id": "trader_test_001",
                "rejection_reason": "市场环境不适宜",
                "rejection_notes": "当前市场波动率过高，暂停新策略执行",
                "rejection_timestamp": time.time()
            }

            rejection_confirm_response = requests.post(
                self.get_api_url("signal", "/api/v1/signal/confirm"),
                json=rejection_request
            )
            assert rejection_confirm_response.status_code == 200

            rejection_confirm_data = rejection_confirm_response.json()
            assert rejection_confirm_data["status"] == "rejected"
            assert rejection_confirm_data["rejection_reason"] == rejection_request["rejection_reason"]

            print(f"Signal rejection workflow verified")

            # Step 8: 测试每日信号更新
            daily_update_request = {
                "update_date": "2023-12-16",
                "strategies": [strategy_id],
                "force_update": False,
                "notification_enabled": True,
                "batch_size": 10,
                "performance_monitoring": True
            }

            daily_update_response = requests.post(
                self.get_api_url("signal", "/api/v1/signal/daily-update"),
                json=daily_update_request
            )
            assert daily_update_response.status_code == 202

            update_data = daily_update_response.json()
            assert "update_id" in update_data
            assert update_data["status"] == "running"

            # 验证估计完成时间在15分钟内 (FR-012要求)
            if "estimated_completion" in update_data:
                import datetime
                estimated_time = datetime.datetime.fromisoformat(
                    update_data["estimated_completion"].replace("Z", "+00:00")
                )
                now = datetime.datetime.now(datetime.timezone.utc)
                time_diff = estimated_time - now
                assert time_diff.total_seconds() <= 900, "Daily update should complete within 15 minutes"

            print("Daily signal update workflow initiated")

            # Step 9: 清理测试数据
            for factor_id in test_factors:
                cleanup_response = requests.delete(
                    self.get_api_url("factor", f"/api/v1/factors/{factor_id}")
                )
                assert cleanup_response.status_code == 200

            print("Signal workflow integration test completed successfully")

    @pytest.mark.asyncio
    async def test_signal_risk_limit_enforcement(self):
        """测试信号风险限制强制执行"""
        async with integration_test_environment() as agent_manager:
            self.__init__(agent_manager)

            # 创建会违反风险限制的信号请求
            risky_signal_request = {
                "strategy_id": f"risky_test_{int(time.time())}",
                "factors": ["momentum_test"],
                "universe": "custom",
                "stock_universe": ["sh000001"],  # 只有一只股票，会导致集中度过高
                "trade_date": "2023-12-15",
                "position_size": 1000000,
                "max_positions": 1,
                "risk_limits": {
                    "max_single_weight": 0.02,   # 严格的2%限制
                    "max_sector_weight": 0.1,
                    "max_turnover": 0.1
                },
                "human_confirmation_required": True
            }

            # 应该被风险控制拒绝或生成警告
            risky_response = requests.post(
                self.get_api_url("signal", "/api/v1/signal/generate"),
                json=risky_signal_request
            )

            if risky_response.status_code == 400:
                # 请求被直接拒绝
                error_data = risky_response.json()
                assert "risk" in error_data["error"]["message"].lower()
                print("Risky signal correctly rejected by risk controls")

            elif risky_response.status_code == 201:
                # 信号生成但包含风险警告
                risky_data = risky_response.json()
                assert "risk_warnings" in risky_data
                assert len(risky_data["risk_warnings"]) > 0
                print(f"Risky signal generated with {len(risky_data['risk_warnings'])} risk warnings")

    @pytest.mark.asyncio
    async def test_signal_performance_tracking(self):
        """测试信号性能跟踪"""
        async with integration_test_environment() as agent_manager:
            self.__init__(agent_manager)

            # 生成历史信号用于性能分析
            historical_signal_request = {
                "strategy_id": f"performance_test_{int(time.time())}",
                "factors": ["test_momentum"],
                "universe": "custom",
                "stock_universe": ["sh000001", "sz000001"],
                "trade_date": "2023-11-15",  # 历史日期
                "position_size": 500000,
                "max_positions": 2,
                "human_confirmation_required": False,  # 自动确认用于测试
                "performance_tracking": True
            }

            perf_response = requests.post(
                self.get_api_url("signal", "/api/v1/signal/generate"),
                json=historical_signal_request
            )

            if perf_response.status_code == 201:
                perf_signal_id = perf_response.json()["signal_id"]

                # 查询信号性能
                performance_response = requests.get(
                    self.get_api_url("signal", f"/api/v1/signal/{perf_signal_id}/performance"),
                    params={
                        "start_date": "2023-11-15",
                        "end_date": "2023-12-15",
                        "benchmark": "000300.SH"
                    }
                )

                if performance_response.status_code == 200:
                    performance_data = performance_response.json()
                    assert "performance_metrics" in performance_data

                    metrics = performance_data["performance_metrics"]
                    assert "total_return" in metrics
                    assert "sharpe_ratio" in metrics
                    assert "max_drawdown" in metrics
                    assert "win_rate" in metrics

                    print(f"Signal performance tracked: Return {metrics.get('total_return', 'N/A')}, Sharpe {metrics.get('sharpe_ratio', 'N/A')}")

    @pytest.mark.asyncio
    async def test_signal_workflow_error_handling(self):
        """测试信号工作流错误处理"""
        async with integration_test_environment() as agent_manager:
            self.__init__(agent_manager)

            # 测试无效因子的信号生成
            invalid_signal_request = {
                "strategy_id": "error_test",
                "factors": ["nonexistent_factor_123"],
                "universe": "hs300",
                "trade_date": "2023-12-15",
                "position_size": 1000000
            }

            error_response = requests.post(
                self.get_api_url("signal", "/api/v1/signal/generate"),
                json=invalid_signal_request
            )

            assert error_response.status_code in [400, 404]
            error_data = error_response.json()
            assert "error" in error_data
            print("Invalid factor signal request correctly rejected")

            # 测试无效确认操作
            invalid_confirm_request = {
                "signal_id": "nonexistent_signal_123",
                "action": "confirm",
                "user_id": "test_user"
            }

            confirm_error_response = requests.post(
                self.get_api_url("signal", "/api/v1/signal/confirm"),
                json=invalid_confirm_request
            )

            assert confirm_error_response.status_code == 404
            confirm_error_data = confirm_error_response.json()
            assert "not found" in confirm_error_data["error"]["message"].lower()
            print("Invalid signal confirmation correctly rejected")

    @pytest.mark.asyncio
    async def test_signal_expiration_handling(self):
        """测试信号过期处理"""
        async with integration_test_environment() as agent_manager:
            self.__init__(agent_manager)

            # 生成一个带有较短过期时间的信号
            expiry_signal_request = {
                "strategy_id": f"expiry_test_{int(time.time())}",
                "factors": ["test_factor"],
                "universe": "custom",
                "stock_universe": ["sh000001"],
                "trade_date": "2023-12-15",
                "position_size": 100000,
                "human_confirmation_required": True,
                "expiry_minutes": 5  # 5分钟后过期
            }

            expiry_response = requests.post(
                self.get_api_url("signal", "/api/v1/signal/generate"),
                json=expiry_signal_request
            )

            if expiry_response.status_code == 201:
                expiry_signal_id = expiry_response.json()["signal_id"]
                expires_at = expiry_response.json().get("expires_at")

                if expires_at:
                    print(f"Signal {expiry_signal_id} will expire at {expires_at}")

                    # 等待信号过期后尝试确认
                    await asyncio.sleep(6)  # 等待超过过期时间

                    late_confirm_request = {
                        "signal_id": expiry_signal_id,
                        "action": "confirm",
                        "user_id": "test_user"
                    }

                    late_response = requests.post(
                        self.get_api_url("signal", "/api/v1/signal/confirm"),
                        json=late_confirm_request
                    )

                    # 应该被拒绝因为信号已过期
                    assert late_response.status_code == 410  # Gone
                    late_error_data = late_response.json()
                    assert "expired" in late_error_data["error"]["message"].lower()
                    print("Expired signal confirmation correctly rejected")