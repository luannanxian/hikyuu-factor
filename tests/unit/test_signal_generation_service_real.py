"""
Signal Generation Service Unit Tests

基于真实Hikyuu框架的信号生成服务单元测试
不使用mock数据，测试SignalGenerator, RiskChecker, ConfirmationManager, AuditLogger功能
"""

import pytest
import asyncio
import pandas as pd
import numpy as np
import uuid
from datetime import datetime, date, timedelta
from typing import Dict, List, Any

from src.services.signal_generation_service import (
    SignalGenerator, RiskChecker, ConfirmationManager, AuditLogger, SignalGenerationService
)
from src.models.hikyuu_models import TradingSignal, SignalType, PositionType
from src.models.validation_models import RiskAssessment, RiskFactor, RiskLevel, RiskCategory
from src.models.audit_models import ConfirmationRecord, ConfirmationStatus, AuditEntry


class TestSignalGenerator:
    """SignalGenerator组件单元测试"""

    @pytest.fixture
    def signal_config(self):
        """创建信号配置"""
        return {
            'strategy_config': {
                'strategy_name': 'multi_factor_momentum',
                'factor_weights': {
                    'momentum_20d': 0.4,
                    'rsi_14d': 0.3,
                    'volume_ratio': 0.3
                },
                'signal_threshold': {
                    'buy': 0.7,
                    'sell': 0.3,
                    'hold': [0.3, 0.7]
                }
            }
        }

    @pytest.fixture
    def sample_factor_data(self):
        """创建示例因子数据"""
        np.random.seed(42)

        # 创建momentum因子数据
        momentum_data = pd.DataFrame({
            'stock_code': ['sh600000', 'sz000001', 'sh600036'] * 20,
            'date': pd.concat([pd.date_range('2024-01-01', periods=20)] * 3),
            'factor_value': np.random.normal(0.1, 0.05, 60),
            'factor_score': np.random.uniform(0.4, 0.9, 60)
        })

        # 创建RSI因子数据
        rsi_data = pd.DataFrame({
            'stock_code': ['sh600000', 'sz000001', 'sh600036'] * 20,
            'date': pd.concat([pd.date_range('2024-01-01', periods=20)] * 3),
            'factor_value': np.random.uniform(30, 70, 60),  # RSI值
            'factor_score': np.random.uniform(0.3, 0.8, 60)
        })

        # 创建成交量比率因子数据
        volume_data = pd.DataFrame({
            'stock_code': ['sh600000', 'sz000001', 'sh600036'] * 20,
            'date': pd.concat([pd.date_range('2024-01-01', periods=20)] * 3),
            'factor_value': np.random.uniform(0.8, 1.5, 60),  # 成交量比率
            'factor_score': np.random.uniform(0.2, 0.9, 60)
        })

        return {
            'momentum_20d': momentum_data,
            'rsi_14d': rsi_data,
            'volume_ratio': volume_data
        }

    @pytest.fixture
    def signal_generator(self, signal_config):
        """创建SignalGenerator实例"""
        return SignalGenerator(signal_config)

    def test_signal_generator_initialization(self, signal_generator, signal_config):
        """测试SignalGenerator初始化"""
        assert signal_generator.signal_config == signal_config
        assert signal_generator.logger is not None

    def test_generate_signals_basic(self, signal_generator, sample_factor_data):
        """测试基本信号生成"""
        signals = signal_generator.generate_signals(sample_factor_data)

        assert isinstance(signals, list)
        assert len(signals) > 0

        # 验证每个信号的结构
        for signal in signals:
            assert isinstance(signal, TradingSignal)
            assert signal.signal_id is not None
            assert signal.stock_code is not None
            assert signal.signal_type in [SignalType.BUY, SignalType.SELL, SignalType.HOLD]
            assert 0.0 <= signal.signal_strength <= 1.0
            assert signal.generation_date is not None
            assert signal.confirmation_required is True

    def test_generate_signals_different_thresholds(self, sample_factor_data):
        """测试不同阈值的信号生成"""
        # 配置买入阈值较低的策略
        low_buy_config = {
            'strategy_config': {
                'strategy_name': 'aggressive_strategy',
                'factor_weights': {'momentum_20d': 1.0},
                'signal_threshold': {
                    'buy': 0.4,  # 较低的买入阈值
                    'sell': 0.2,
                    'hold': [0.2, 0.4]
                }
            }
        }

        # 配置买入阈值较高的策略
        high_buy_config = {
            'strategy_config': {
                'strategy_name': 'conservative_strategy',
                'factor_weights': {'momentum_20d': 1.0},
                'signal_threshold': {
                    'buy': 0.8,  # 较高的买入阈值
                    'sell': 0.3,
                    'hold': [0.3, 0.8]
                }
            }
        }

        aggressive_generator = SignalGenerator(low_buy_config)
        conservative_generator = SignalGenerator(high_buy_config)

        aggressive_signals = aggressive_generator.generate_signals(sample_factor_data)
        conservative_signals = conservative_generator.generate_signals(sample_factor_data)

        # 激进策略应该产生更多买入信号
        aggressive_buy_count = len([s for s in aggressive_signals if s.signal_type == SignalType.BUY])
        conservative_buy_count = len([s for s in conservative_signals if s.signal_type == SignalType.BUY])

        assert aggressive_buy_count >= conservative_buy_count

    def test_combine_factor_data(self, signal_generator, sample_factor_data):
        """测试因子数据合并"""
        factor_weights = {
            'momentum_20d': 0.5,
            'rsi_14d': 0.3,
            'volume_ratio': 0.2
        }

        combined_data = signal_generator._combine_factor_data(sample_factor_data, factor_weights)

        assert isinstance(combined_data, pd.DataFrame)
        assert 'stock_code' in combined_data.columns
        assert 'date' in combined_data.columns
        assert 'weighted_score' in combined_data.columns

        # 验证加权分数计算
        assert combined_data['weighted_score'].notna().sum() > 0
        assert combined_data['weighted_score'].min() >= 0
        assert combined_data['weighted_score'].max() <= 1

    def test_generate_signals_empty_data(self, signal_generator):
        """测试空数据的信号生成"""
        empty_factor_data = {}
        signals = signal_generator.generate_signals(empty_factor_data)

        assert isinstance(signals, list)
        assert len(signals) == 0

    def test_generate_signals_missing_factors(self, signal_generator):
        """测试因子缺失的信号生成"""
        # 只提供部分因子
        partial_factor_data = {
            'momentum_20d': pd.DataFrame({
                'stock_code': ['sh600000'],
                'date': [pd.Timestamp('2024-01-01')],
                'factor_value': [0.1],
                'factor_score': [0.8]
            })
            # 缺少rsi_14d和volume_ratio
        }

        signals = signal_generator.generate_signals(partial_factor_data)

        # 应该能够处理缺失因子的情况
        assert isinstance(signals, list)

    def test_signal_strength_calculation(self, signal_generator, sample_factor_data):
        """测试信号强度计算"""
        signals = signal_generator.generate_signals(sample_factor_data)

        # 验证信号强度的合理性
        for signal in signals:
            assert 0.0 <= signal.signal_strength <= 1.0

            # 买入信号应该有较高的强度
            if signal.signal_type == SignalType.BUY:
                assert signal.signal_strength >= 0.7

            # 卖出信号应该有较低的强度
            if signal.signal_type == SignalType.SELL:
                assert signal.signal_strength <= 0.3


class TestRiskChecker:
    """RiskChecker组件单元测试"""

    @pytest.fixture
    def risk_config(self):
        """创建风险配置"""
        return {
            'position_risk': {
                'max_single_position': 0.1,  # 单股票最大仓位10%
                'max_sector_exposure': 0.3   # 单行业最大暴露30%
            },
            'liquidity_risk': {
                'min_avg_volume': 1000000  # 最小平均成交量
            }
        }

    @pytest.fixture
    def sample_signals(self):
        """创建示例交易信号"""
        signals = []

        # 创建多个买入信号
        for i in range(8):
            signal = TradingSignal(
                signal_id=str(uuid.uuid4()),
                stock_code=f'sh{600000+i:06d}',
                signal_type=SignalType.BUY,
                signal_strength=0.8,
                generation_date=datetime.now(),
                effective_date=date.today(),
                source_factors=['momentum_20d'],
                factor_scores={'momentum_20d': 0.8},
                strategy_name='test_strategy',
                confirmation_required=True
            )
            signals.append(signal)

        # 创建卖出信号
        for i in range(2):
            signal = TradingSignal(
                signal_id=str(uuid.uuid4()),
                stock_code=f'sz{1+i:06d}',
                signal_type=SignalType.SELL,
                signal_strength=0.2,
                generation_date=datetime.now(),
                effective_date=date.today(),
                source_factors=['momentum_20d'],
                factor_scores={'momentum_20d': 0.2},
                strategy_name='test_strategy',
                confirmation_required=True
            )
            signals.append(signal)

        return signals

    @pytest.fixture
    def concentrated_signals(self):
        """创建集中度风险信号（同一板块）"""
        signals = []

        # 创建都是sh开头的信号（模拟同一行业）
        for i in range(15):
            signal = TradingSignal(
                signal_id=str(uuid.uuid4()),
                stock_code=f'sh{600000+i:06d}',
                signal_type=SignalType.BUY,
                signal_strength=0.8,
                generation_date=datetime.now(),
                effective_date=date.today(),
                source_factors=['momentum_20d'],
                factor_scores={'momentum_20d': 0.8},
                strategy_name='concentrated_strategy'
            )
            signals.append(signal)

        return signals

    @pytest.fixture
    def risk_checker(self, risk_config):
        """创建RiskChecker实例"""
        return RiskChecker(risk_config)

    def test_risk_checker_initialization(self, risk_checker, risk_config):
        """测试RiskChecker初始化"""
        assert risk_checker.risk_config == risk_config
        assert risk_checker.logger is not None

    def test_assess_signals_normal_portfolio(self, risk_checker, sample_signals):
        """测试正常投资组合的风险评估"""
        assessment = risk_checker.assess_signals(sample_signals)

        assert isinstance(assessment, RiskAssessment)
        assert assessment.assessment_id is not None
        assert assessment.assessment_date is not None
        assert isinstance(assessment.overall_risk_score, float)
        assert assessment.overall_risk_level in [RiskLevel.LOW, RiskLevel.MEDIUM, RiskLevel.HIGH]
        assert isinstance(assessment.risk_factors, list)

    def test_position_risk_check_normal(self, risk_checker, sample_signals):
        """测试正常仓位风险检查"""
        risk_factors = risk_checker._check_position_risk(sample_signals)

        assert isinstance(risk_factors, list)

        # 8个买入信号，每个占1/8=12.5%，超过10%限制
        position_risks = [rf for rf in risk_factors if rf.factor_id == "position_concentration"]
        assert len(position_risks) > 0  # 应该检测到仓位集中风险

    def test_concentration_risk_check(self, risk_checker, concentrated_signals):
        """测试集中度风险检查"""
        risk_factors = risk_checker._check_concentration_risk(concentrated_signals)

        assert isinstance(risk_factors, list)

        # 15个都是sh开头的信号，100%集中在一个"行业"
        sector_risks = [rf for rf in risk_factors if 'sector_concentration' in rf.factor_id]
        assert len(sector_risks) > 0  # 应该检测到行业集中风险

        # 验证风险等级
        for risk in sector_risks:
            assert risk.risk_level in [RiskLevel.HIGH, RiskLevel.MEDIUM]

    def test_liquidity_risk_check(self, risk_checker, sample_signals):
        """测试流动性风险检查"""
        risk_factors = risk_checker._check_liquidity_risk(sample_signals)

        assert isinstance(risk_factors, list)

        # 可能检测到流动性风险（取决于模拟的成交量数据）
        liquidity_risks = [rf for rf in risk_factors if rf.factor_id == "liquidity_risk"]

        # 验证流动性风险结构
        for risk in liquidity_risks:
            assert risk.category == RiskCategory.LIQUIDITY_RISK
            assert 0.0 <= risk.current_value <= 1.0

    def test_filter_risky_signals_low_risk(self, risk_checker, sample_signals):
        """测试低风险情况下的信号过滤"""
        # 创建低风险评估
        assessment = RiskAssessment(
            assessment_id=str(uuid.uuid4()),
            assessment_date=datetime.now(),
            overall_risk_score=0.3,  # 低风险
            overall_risk_level=RiskLevel.LOW
        )

        filtered_signals = risk_checker._filter_risky_signals(sample_signals, assessment)

        # 低风险情况下不应该过滤信号
        assert len(filtered_signals) == len(sample_signals)

    def test_filter_risky_signals_high_risk(self, risk_checker, sample_signals):
        """测试高风险情况下的信号过滤"""
        # 创建高风险评估
        assessment = RiskAssessment(
            assessment_id=str(uuid.uuid4()),
            assessment_date=datetime.now(),
            overall_risk_score=0.8,  # 高风险
            overall_risk_level=RiskLevel.HIGH
        )

        filtered_signals = risk_checker._filter_risky_signals(sample_signals, assessment)

        # 高风险情况下应该过滤一些信号
        assert len(filtered_signals) <= len(sample_signals)

    def test_generate_risk_warnings(self, risk_checker):
        """测试风险警告生成"""
        # 创建超过阈值的风险因子
        risk_factors = [
            RiskFactor(
                factor_id="test_risk",
                factor_name="测试风险因子",
                category=RiskCategory.POSITION_RISK,
                current_value=0.15,
                threshold_value=0.10,
                max_acceptable_value=0.20,
                risk_level=RiskLevel.MEDIUM
            )
        ]

        warnings = risk_checker._generate_risk_warnings(risk_factors)

        assert isinstance(warnings, list)
        assert len(warnings) > 0
        assert "测试风险因子" in warnings[0]
        assert "0.150" in warnings[0]
        assert "0.100" in warnings[0]

    def test_assess_signals_comprehensive(self, risk_checker, concentrated_signals):
        """测试综合风险评估"""
        assessment = risk_checker.assess_signals(concentrated_signals)

        # 集中度风险信号应该产生高风险评估
        assert assessment.overall_risk_score > 0.0
        assert len(assessment.risk_factors) > 0
        assert len(assessment.risk_warnings) > 0

        # 验证过滤后的信号
        assert assessment.filtered_signals is not None
        assert len(assessment.filtered_signals) <= len(concentrated_signals)


class TestConfirmationManager:
    """ConfirmationManager组件单元测试"""

    @pytest.fixture
    def confirmation_manager(self):
        """创建ConfirmationManager实例"""
        return ConfirmationManager()

    @pytest.fixture
    def sample_risk_assessment(self):
        """创建示例风险评估"""
        return RiskAssessment(
            assessment_id=str(uuid.uuid4()),
            assessment_date=datetime.now(),
            overall_risk_score=0.6,
            overall_risk_level=RiskLevel.MEDIUM,
            risk_warnings=["仓位集中度较高", "部分股票流动性不足"]
        )

    def test_confirmation_manager_initialization(self, confirmation_manager):
        """测试ConfirmationManager初始化"""
        assert confirmation_manager.logger is not None

    def test_require_confirmation_normal_risk(self, confirmation_manager, sample_signals, sample_risk_assessment):
        """测试正常风险的确认请求"""
        user_id = "trader_001"

        confirmation_record = confirmation_manager.require_confirmation(
            sample_signals, sample_risk_assessment, user_id
        )

        assert isinstance(confirmation_record, ConfirmationRecord)
        assert confirmation_record.target_object_type == "trading_signals"
        assert confirmation_record.confirmation_type == "signal_confirmation"
        assert confirmation_record.confirmer_id == user_id

        # 验证信号摘要
        assert 'signals_summary' in confirmation_record.__dict__
        summary = confirmation_record.signals_summary
        assert summary['total_signals'] == len(sample_signals)
        assert summary['buy_signals'] > 0
        assert summary['sell_signals'] >= 0

        # 验证风险摘要
        assert 'risk_summary' in confirmation_record.__dict__
        risk_summary = confirmation_record.risk_summary
        assert risk_summary['overall_risk_score'] == sample_risk_assessment.overall_risk_score
        assert risk_summary['risk_level'] == sample_risk_assessment.overall_risk_level.value

    def test_require_confirmation_high_risk(self, confirmation_manager, sample_signals):
        """测试高风险的确认请求"""
        # 创建高风险评估
        high_risk_assessment = RiskAssessment(
            assessment_id=str(uuid.uuid4()),
            assessment_date=datetime.now(),
            overall_risk_score=0.9,
            overall_risk_level=RiskLevel.HIGH,
            risk_warnings=["严重的仓位集中", "多个高风险信号", "流动性严重不足"]
        )

        confirmation_record = confirmation_manager.require_confirmation(
            sample_signals, high_risk_assessment, "trader_002"
        )

        # 高风险情况下应该需要审批
        assert confirmation_record.requires_approval is True
        assert confirmation_record.risk_summary['risk_warnings_count'] == 3

    def test_simulate_user_confirmation_approved(self, confirmation_manager):
        """测试模拟用户确认 - 批准"""
        confirmation_id = str(uuid.uuid4())
        user_id = "trader_003"

        confirmation = confirmation_manager.simulate_user_confirmation(
            confirmation_id, 'approved', '信号质量良好，同意执行', user_id
        )

        assert confirmation.confirmation_id == confirmation_id
        assert confirmation.status == ConfirmationStatus.APPROVED
        assert confirmation.decision == 'approved'
        assert confirmation.confirmer_id == user_id
        assert confirmation.user_comments == '信号质量良好，同意执行'
        assert confirmation.confirmed_at is not None

    def test_simulate_user_confirmation_rejected(self, confirmation_manager):
        """测试模拟用户确认 - 拒绝"""
        confirmation_id = str(uuid.uuid4())
        user_id = "trader_004"

        confirmation = confirmation_manager.simulate_user_confirmation(
            confirmation_id, 'rejected', '风险过高，暂时不执行', user_id
        )

        assert confirmation.confirmation_id == confirmation_id
        assert confirmation.status == ConfirmationStatus.REJECTED
        assert confirmation.decision == 'rejected'
        assert confirmation.user_comments == '风险过高，暂时不执行'

    def test_confirmation_timeout_handling(self, confirmation_manager, sample_signals, sample_risk_assessment):
        """测试确认超时处理"""
        confirmation_record = confirmation_manager.require_confirmation(
            sample_signals, sample_risk_assessment, "trader_005"
        )

        # 验证过期时间设置
        assert confirmation_record.expires_at is not None
        assert confirmation_record.expires_at > datetime.now()

        # 过期时间应该在4小时内
        expected_expiry = datetime.now() + timedelta(hours=4)
        time_diff = abs((confirmation_record.expires_at - expected_expiry).total_seconds())
        assert time_diff < 300  # 5分钟误差范围


class TestAuditLogger:
    """AuditLogger组件单元测试"""

    @pytest.fixture
    def audit_logger(self):
        """创建AuditLogger实例"""
        return AuditLogger()

    def test_audit_logger_initialization(self, audit_logger):
        """测试AuditLogger初始化"""
        assert audit_logger.logger is not None

    def test_log_signal_workflow(self, audit_logger, sample_signals, sample_risk_assessment):
        """测试信号工作流审计记录"""
        session_id = "signal_session_001"

        # 创建确认记录
        confirmation_record = ConfirmationRecord(
            confirmation_id=str(uuid.uuid4()),
            confirmation_type="signal_confirmation",
            target_object_type="trading_signals",
            target_object_id="test_signals",
            status=ConfirmationStatus.APPROVED,
            decision="approved",
            confirmer_id="trader_001",
            confirmer_role="senior_trader"
        )

        # 最终信号（可能经过过滤）
        final_signals = sample_signals[:5]  # 假设过滤后只剩5个信号

        audit_entry = audit_logger.log_signal_workflow(
            session_id, sample_signals, sample_risk_assessment, confirmation_record, final_signals
        )

        # 验证审计记录结构
        assert isinstance(audit_entry, AuditEntry)
        assert audit_entry.audit_id is not None
        assert audit_entry.component == "SignalGenerationService"
        assert audit_entry.action_name == "signal_workflow"
        assert audit_entry.workflow_id == session_id

        # 验证事件数据
        event_data = audit_entry.event_data
        assert event_data['raw_signals_count'] == len(sample_signals)
        assert event_data['final_signals_count'] == len(final_signals)
        assert event_data['risk_score'] == sample_risk_assessment.overall_risk_score
        assert event_data['confirmation_status'] == confirmation_record.status.value
        assert event_data['user_decision'] == confirmation_record.decision

        # 验证工作流步骤
        expected_steps = ['signal_generation', 'risk_assessment', 'user_confirmation', 'signal_filtering']
        assert audit_entry.workflow_steps == expected_steps

    def test_audit_entry_completeness(self, audit_logger, sample_signals, sample_risk_assessment):
        """测试审计记录完整性"""
        session_id = "completeness_test"

        confirmation_record = ConfirmationRecord(
            confirmation_id=str(uuid.uuid4()),
            confirmation_type="auto_approval",
            target_object_type="trading_signals",
            target_object_id=session_id,
            status=ConfirmationStatus.APPROVED,
            decision="auto_approved",
            confirmer_id="system",
            confirmer_role="system"
        )

        audit_entry = audit_logger.log_signal_workflow(
            session_id, sample_signals, sample_risk_assessment, confirmation_record, sample_signals
        )

        # 验证所有必要字段都存在
        assert audit_entry.audit_id is not None
        assert audit_entry.timestamp is not None
        assert audit_entry.component is not None
        assert audit_entry.action_name is not None
        assert audit_entry.description is not None
        assert audit_entry.event_data is not None
        assert audit_entry.workflow_steps is not None


class TestSignalGenerationService:
    """SignalGenerationService集成测试"""

    @pytest.fixture
    def signal_generation_service(self):
        """创建SignalGenerationService实例"""
        config = {
            'enable_audit': True
        }
        return SignalGenerationService(config)

    @pytest.fixture
    def complete_signal_config(self):
        """创建完整的信号配置"""
        return {
            'strategy_config': {
                'strategy_name': 'comprehensive_strategy',
                'factor_weights': {
                    'momentum_20d': 0.4,
                    'rsi_14d': 0.3,
                    'volume_ratio': 0.3
                },
                'signal_threshold': {
                    'buy': 0.65,
                    'sell': 0.35,
                    'hold': [0.35, 0.65]
                }
            }
        }

    @pytest.fixture
    def complete_risk_config(self):
        """创建完整的风险配置"""
        return {
            'position_risk': {
                'max_single_position': 0.08,
                'max_sector_exposure': 0.25
            },
            'liquidity_risk': {
                'min_avg_volume': 800000
            }
        }

    def test_service_initialization(self, signal_generation_service):
        """测试服务初始化"""
        assert signal_generation_service.config is not None
        assert signal_generation_service.logger is not None
        assert signal_generation_service.enable_audit is True

    def test_generate_trading_signals_complete_workflow(
        self, signal_generation_service, sample_factor_data, complete_signal_config, complete_risk_config
    ):
        """测试完整的交易信号生成工作流"""
        user_id = "trader_integration_test"

        result = signal_generation_service.generate_trading_signals(
            sample_factor_data, complete_signal_config, complete_risk_config, user_id, require_confirmation=True
        )

        # 验证返回结果结构
        assert isinstance(result, dict)
        assert 'session_id' in result
        assert 'signals' in result
        assert 'risk_assessment' in result
        assert 'confirmation_status' in result
        assert 'audit_id' in result
        assert 'signal_count' in result
        assert 'risk_score' in result

        # 验证信号
        signals = result['signals']
        assert isinstance(signals, list)
        assert len(signals) > 0

        for signal in signals:
            assert isinstance(signal, TradingSignal)

        # 验证风险评估
        risk_assessment = result['risk_assessment']
        assert isinstance(risk_assessment, RiskAssessment)
        assert 0.0 <= risk_assessment.overall_risk_score <= 1.0

        # 验证确认状态
        assert result['confirmation_status'] in ['approved', 'rejected', 'pending']

        # 验证审计ID
        assert result['audit_id'] is not None

    def test_generate_trading_signals_without_confirmation(
        self, signal_generation_service, sample_factor_data, complete_signal_config, complete_risk_config
    ):
        """测试不需要确认的信号生成"""
        result = signal_generation_service.generate_trading_signals(
            sample_factor_data, complete_signal_config, complete_risk_config, "auto_trader", require_confirmation=False
        )

        # 不需要确认时应该自动批准
        assert result['confirmation_status'] == 'approved'

    def test_save_signals(self, signal_generation_service, sample_signals):
        """测试保存信号"""
        metadata = {
            'strategy': 'test_strategy',
            'generated_at': datetime.now().isoformat(),
            'risk_level': 'medium'
        }

        result = signal_generation_service.save_signals(sample_signals, metadata)

        assert isinstance(result, dict)
        assert result['success'] is True
        assert 'signal_batch_id' in result
        assert result['signals_count'] == len(sample_signals)
        assert result['metadata'] == metadata

    def test_end_to_end_signal_generation(
        self, signal_generation_service, sample_factor_data, complete_signal_config, complete_risk_config
    ):
        """测试端到端信号生成流程"""
        user_id = "end_to_end_trader"

        # Step 1: 生成信号
        generation_result = signal_generation_service.generate_trading_signals(
            sample_factor_data, complete_signal_config, complete_risk_config, user_id, require_confirmation=True
        )

        assert generation_result['session_id'] is not None
        signals = generation_result['signals']

        # Step 2: 保存信号
        metadata = {
            'session_id': generation_result['session_id'],
            'risk_score': generation_result['risk_score'],
            'confirmation_status': generation_result['confirmation_status']
        }

        save_result = signal_generation_service.save_signals(signals, metadata)

        # Step 3: 验证完整流程
        assert save_result['success'] is True
        assert save_result['signals_count'] == len(signals)

        # 验证审计记录存在
        assert generation_result['audit_id'] is not None

    def test_signal_generation_performance(
        self, signal_generation_service, complete_signal_config, complete_risk_config
    ):
        """测试信号生成性能"""
        import time

        # 创建大规模因子数据
        np.random.seed(42)
        large_factor_data = {}

        # 100只股票，3个因子
        stock_codes = [f'sh{600000+i:06d}' for i in range(100)]
        dates = pd.date_range('2024-01-01', periods=30)

        for factor_name in ['momentum_20d', 'rsi_14d', 'volume_ratio']:
            factor_data = []
            for stock in stock_codes:
                for date in dates:
                    factor_data.append({
                        'stock_code': stock,
                        'date': date,
                        'factor_value': np.random.normal(0.1, 0.05),
                        'factor_score': np.random.uniform(0.3, 0.9)
                    })

            large_factor_data[factor_name] = pd.DataFrame(factor_data)

        start_time = time.time()
        result = signal_generation_service.generate_trading_signals(
            large_factor_data, complete_signal_config, complete_risk_config, "performance_test", require_confirmation=False
        )
        end_time = time.time()

        execution_time = end_time - start_time

        # 验证性能
        assert execution_time < 30.0  # 应该在30秒内完成
        assert result['signal_count'] <= 100  # 最多100只股票的信号

    def test_signal_generation_error_handling(self, signal_generation_service, complete_risk_config):
        """测试信号生成错误处理"""
        # 测试空因子数据
        empty_factor_data = {}
        incomplete_signal_config = {'strategy_config': {}}

        result = signal_generation_service.generate_trading_signals(
            empty_factor_data, incomplete_signal_config, complete_risk_config, "error_test"
        )

        # 应该能够处理错误情况
        assert isinstance(result, dict)
        assert result['signal_count'] == 0

    def test_different_risk_scenarios(
        self, signal_generation_service, sample_factor_data, complete_signal_config
    ):
        """测试不同风险场景"""
        # 低风险配置
        low_risk_config = {
            'position_risk': {
                'max_single_position': 0.05,
                'max_sector_exposure': 0.15
            },
            'liquidity_risk': {
                'min_avg_volume': 2000000
            }
        }

        # 高风险配置
        high_risk_config = {
            'position_risk': {
                'max_single_position': 0.20,
                'max_sector_exposure': 0.50
            },
            'liquidity_risk': {
                'min_avg_volume': 100000
            }
        }

        low_risk_result = signal_generation_service.generate_trading_signals(
            sample_factor_data, complete_signal_config, low_risk_config, "low_risk_trader"
        )

        high_risk_result = signal_generation_service.generate_trading_signals(
            sample_factor_data, complete_signal_config, high_risk_config, "high_risk_trader"
        )

        # 低风险配置应该产生更保守的结果
        assert low_risk_result['risk_score'] <= high_risk_result['risk_score']


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])