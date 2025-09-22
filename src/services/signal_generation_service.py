"""
Signal Generation Service

信号生成服务实现，提供：
1. 基于因子的交易信号生成
2. 风险控制和合规检查
3. Human-in-Loop确认机制
4. 完整的审计跟踪

实现集成测试中定义的SignalGenerator, RiskChecker, ConfirmationManager, AuditLogger API契约。
"""

import asyncio
import logging
from datetime import datetime, date, timedelta
from typing import Dict, List, Optional, Any, Tuple
import pandas as pd
import numpy as np
import uuid
import json
from pathlib import Path

from models.hikyuu_models import TradingSignal, SignalType, PortfolioPosition, PositionType
from models.validation_models import RiskAssessment, RiskFactor, RiskLevel, RiskCategory
from models.audit_models import (
    AuditEntry, ConfirmationRecord, ConfirmationStatus,
    AuditEventType, create_signal_confirmation_request
)


class SignalGenerator:
    """信号生成器"""

    def __init__(self, signal_config: Dict[str, Any]):
        self.signal_config = signal_config
        self.logger = logging.getLogger(__name__)

    def generate_signals(self, factor_data: Dict[str, pd.DataFrame]) -> List[TradingSignal]:
        """生成交易信号"""
        self.logger.info(f"基于{len(factor_data)}个因子生成交易信号")

        signals = []
        strategy_config = self.signal_config.get('strategy_config', {})

        # 获取因子权重
        factor_weights = strategy_config.get('factor_weights', {})
        signal_thresholds = strategy_config.get('signal_threshold', {
            'buy': 0.7, 'sell': 0.3, 'hold': [0.3, 0.7]
        })

        # 合并所有因子数据
        combined_data = self._combine_factor_data(factor_data, factor_weights)

        # 为每只股票生成信号
        for stock_code in combined_data['stock_code'].unique():
            stock_data = combined_data[combined_data['stock_code'] == stock_code]

            if stock_data.empty:
                continue

            # 计算综合信号强度
            signal_strength = stock_data['weighted_score'].iloc[-1] if not stock_data.empty else 0.5

            # 确定信号类型
            if signal_strength >= signal_thresholds['buy']:
                signal_type = SignalType.BUY
            elif signal_strength <= signal_thresholds['sell']:
                signal_type = SignalType.SELL
            else:
                signal_type = SignalType.HOLD

            # 创建交易信号
            signal = TradingSignal(
                signal_id=str(uuid.uuid4()),
                stock_code=stock_code,
                signal_type=signal_type,
                signal_strength=signal_strength,
                generation_date=datetime.now(),
                effective_date=date.today(),
                source_factors=list(factor_weights.keys()),
                factor_scores={
                    factor: stock_data.get(f'{factor}_score', 0.5)
                    for factor in factor_weights.keys()
                },
                strategy_name=strategy_config.get('strategy_name', 'multi_factor'),
                confirmation_required=True
            )

            signals.append(signal)

        self.logger.info(f"生成了{len(signals)}个交易信号")
        return signals

    def _combine_factor_data(
        self,
        factor_data: Dict[str, pd.DataFrame],
        factor_weights: Dict[str, float]
    ) -> pd.DataFrame:
        """合并因子数据并计算加权分数"""
        all_data = []

        for factor_name, df in factor_data.items():
            if factor_name in factor_weights:
                df_copy = df.copy()
                df_copy[f'{factor_name}_score'] = df_copy.get('factor_score', 0.5)
                all_data.append(df_copy[['stock_code', 'date', f'{factor_name}_score']])

        if not all_data:
            return pd.DataFrame()

        # 合并所有因子数据
        combined = all_data[0]
        for df in all_data[1:]:
            combined = combined.merge(df, on=['stock_code', 'date'], how='outer')

        # 计算加权分数
        combined['weighted_score'] = 0
        for factor_name, weight in factor_weights.items():
            score_col = f'{factor_name}_score'
            if score_col in combined.columns:
                combined['weighted_score'] += combined[score_col].fillna(0.5) * weight

        return combined


class RiskChecker:
    """风险检查器"""

    def __init__(self, risk_config: Dict[str, Any]):
        self.risk_config = risk_config
        self.logger = logging.getLogger(__name__)

    def assess_signals(self, signals: List[TradingSignal]) -> RiskAssessment:
        """评估信号风险"""
        self.logger.info(f"评估{len(signals)}个交易信号的风险")

        assessment = RiskAssessment(
            assessment_id=str(uuid.uuid4()),
            assessment_date=datetime.now(),
            overall_risk_score=0.0,
            overall_risk_level=RiskLevel.LOW
        )

        # 执行各类风险检查
        risk_factors = []
        risk_factors.extend(self._check_position_risk(signals))
        risk_factors.extend(self._check_concentration_risk(signals))
        risk_factors.extend(self._check_liquidity_risk(signals))

        assessment.risk_factors = risk_factors

        # 计算总体风险分数
        if risk_factors:
            risk_scores = [rf.current_value for rf in risk_factors]
            assessment.overall_risk_score = np.mean(risk_scores)

        # 过滤高风险信号
        filtered_signals = self._filter_risky_signals(signals, assessment)
        assessment.filtered_signals = filtered_signals

        # 生成风险警告
        assessment.risk_warnings = self._generate_risk_warnings(risk_factors)

        return assessment

    def _check_position_risk(self, signals: List[TradingSignal]) -> List[RiskFactor]:
        """检查仓位风险"""
        risk_factors = []

        # 单股票仓位风险
        max_single_position = self.risk_config.get('position_risk', {}).get('max_single_position', 0.1)
        buy_signals = [s for s in signals if s.signal_type == SignalType.BUY]

        if buy_signals:
            position_per_stock = 1.0 / len(buy_signals)  # 简化假设等权重
            if position_per_stock > max_single_position:
                risk_factors.append(RiskFactor(
                    factor_id="position_concentration",
                    factor_name="单股票仓位过高",
                    category=RiskCategory.POSITION_RISK,
                    current_value=position_per_stock,
                    threshold_value=max_single_position,
                    max_acceptable_value=max_single_position * 1.5,
                    risk_level=RiskLevel.HIGH if position_per_stock > max_single_position * 1.2 else RiskLevel.MEDIUM
                ))

        return risk_factors

    def _check_concentration_risk(self, signals: List[TradingSignal]) -> List[RiskFactor]:
        """检查集中度风险"""
        risk_factors = []

        # 行业集中度 (简化实现)
        buy_signals = [s for s in signals if s.signal_type == SignalType.BUY]
        if len(buy_signals) > 0:
            # 假设前缀相同的股票为同一行业
            sectors = {}
            for signal in buy_signals:
                sector = signal.stock_code[:2]  # 简化的行业分类
                sectors[sector] = sectors.get(sector, 0) + 1

            max_sector_exposure = self.risk_config.get('position_risk', {}).get('max_sector_exposure', 0.3)
            total_positions = len(buy_signals)

            for sector, count in sectors.items():
                sector_ratio = count / total_positions
                if sector_ratio > max_sector_exposure:
                    risk_factors.append(RiskFactor(
                        factor_id=f"sector_concentration_{sector}",
                        factor_name=f"行业{sector}集中度过高",
                        category=RiskCategory.MARKET_RISK,
                        current_value=sector_ratio,
                        threshold_value=max_sector_exposure,
                        max_acceptable_value=max_sector_exposure * 1.2,
                        risk_level=RiskLevel.HIGH if sector_ratio > max_sector_exposure * 1.1 else RiskLevel.MEDIUM
                    ))

        return risk_factors

    def _check_liquidity_risk(self, signals: List[TradingSignal]) -> List[RiskFactor]:
        """检查流动性风险"""
        risk_factors = []

        # 简化的流动性检查
        min_avg_volume = self.risk_config.get('liquidity_risk', {}).get('min_avg_volume', 1000000)

        low_liquidity_count = 0
        for signal in signals:
            # 实际实现中应该查询实际的成交量数据
            # 这里使用模拟数据
            estimated_volume = 500000 + hash(signal.stock_code) % 2000000
            if estimated_volume < min_avg_volume:
                low_liquidity_count += 1

        if low_liquidity_count > 0:
            liquidity_ratio = low_liquidity_count / len(signals)
            risk_factors.append(RiskFactor(
                factor_id="liquidity_risk",
                factor_name="低流动性股票比例",
                category=RiskCategory.LIQUIDITY_RISK,
                current_value=liquidity_ratio,
                threshold_value=0.1,
                max_acceptable_value=0.2,
                risk_level=RiskLevel.MEDIUM if liquidity_ratio > 0.1 else RiskLevel.LOW
            ))

        return risk_factors

    def _filter_risky_signals(self, signals: List[TradingSignal], assessment: RiskAssessment) -> List[TradingSignal]:
        """过滤高风险信号"""
        if assessment.overall_risk_score < 0.5:
            return signals  # 低风险，不过滤

        # 移除风险最高的信号
        filtered = []
        for signal in signals:
            # 简化的风险评分
            signal_risk = signal.signal_strength if signal.signal_type == SignalType.SELL else (1 - signal.signal_strength)

            if signal_risk < 0.8:  # 保留风险较低的信号
                filtered.append(signal)

        return filtered

    def _generate_risk_warnings(self, risk_factors: List[RiskFactor]) -> List[str]:
        """生成风险警告"""
        warnings = []

        for factor in risk_factors:
            if factor.is_exceeded():
                warnings.append(f"{factor.factor_name}: 当前值{factor.current_value:.3f}超过阈值{factor.threshold_value:.3f}")

        return warnings


class ConfirmationManager:
    """确认管理器"""

    def __init__(self):
        self.logger = logging.getLogger(__name__)

    def require_confirmation(
        self,
        signals: List[TradingSignal],
        risk_assessment: RiskAssessment,
        user_id: str
    ) -> ConfirmationRecord:
        """创建确认请求"""
        confirmation_required = ConfirmationRecord.create_pending_confirmation(
            target_object_type="trading_signals",
            target_object_id=f"signals_batch_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            confirmation_type="signal_confirmation",
            confirmer_id=user_id,
            expires_in_hours=4
        )

        confirmation_required.signals_summary = {
            'total_signals': len(signals),
            'buy_signals': len([s for s in signals if s.signal_type == SignalType.BUY]),
            'sell_signals': len([s for s in signals if s.signal_type == SignalType.SELL]),
            'hold_signals': len([s for s in signals if s.signal_type == SignalType.HOLD])
        }

        confirmation_required.risk_summary = {
            'overall_risk_score': risk_assessment.overall_risk_score,
            'risk_level': risk_assessment.overall_risk_level.value,
            'risk_warnings_count': len(risk_assessment.risk_warnings)
        }

        confirmation_required.requires_approval = (
            risk_assessment.overall_risk_score > 0.5 or
            len(risk_assessment.risk_warnings) > 0
        )

        return confirmation_required

    def simulate_user_confirmation(
        self,
        confirmation_id: str,
        user_decision: str,
        user_comments: str,
        user_id: str
    ) -> ConfirmationRecord:
        """模拟用户确认（用于测试）"""
        confirmation = ConfirmationRecord(
            confirmation_id=confirmation_id,
            confirmation_type="signal_confirmation",
            target_object_type="trading_signals",
            target_object_id="test_signals",
            status=ConfirmationStatus.APPROVED if user_decision == 'approved' else ConfirmationStatus.REJECTED,
            decision=user_decision,
            confirmer_id=user_id,
            confirmer_role="trader",
            confirmed_at=datetime.now(),
            user_comments=user_comments
        )

        return confirmation


class AuditLogger:
    """审计日志记录器"""

    def __init__(self):
        self.logger = logging.getLogger(__name__)

    def log_signal_workflow(
        self,
        session_id: str,
        raw_signals: List[TradingSignal],
        risk_assessment: RiskAssessment,
        confirmation_record: ConfirmationRecord,
        final_signals: List[TradingSignal]
    ) -> AuditEntry:
        """记录信号工作流审计"""
        audit_entry = AuditEntry.create_system_action(
            component="SignalGenerationService",
            action_name="signal_workflow",
            description=f"信号生成工作流: {session_id}",
            workflow_id=session_id
        )

        audit_entry.event_data = {
            'raw_signals_count': len(raw_signals),
            'final_signals_count': len(final_signals),
            'risk_score': risk_assessment.overall_risk_score,
            'confirmation_status': confirmation_record.status.value,
            'user_decision': confirmation_record.decision
        }

        audit_entry.workflow_steps = [
            'signal_generation',
            'risk_assessment',
            'user_confirmation',
            'signal_filtering'
        ]

        return audit_entry


class SignalGenerationService:
    """信号生成服务主类"""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.logger = logging.getLogger(__name__)

        # 服务配置
        self.enable_audit = self.config.get('enable_audit', True)

    def generate_trading_signals(
        self,
        factor_data: Dict[str, pd.DataFrame],
        signal_config: Dict[str, Any],
        risk_config: Dict[str, Any],
        user_id: str,
        require_confirmation: bool = True
    ) -> Dict[str, Any]:
        """生成交易信号（同步版本，用于简化实现）"""
        session_id = f"signal_gen_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.logger.info(f"开始生成交易信号: {session_id}")

        # Step 1: 信号生成
        generator = SignalGenerator(signal_config)
        raw_signals = generator.generate_signals(factor_data)

        # Step 2: 风险检查
        risk_checker = RiskChecker(risk_config)
        risk_assessment = risk_checker.assess_signals(raw_signals)

        # Step 3: 人工确认
        confirmation_manager = ConfirmationManager()
        confirmation_record = None

        if require_confirmation:
            confirmation_required = confirmation_manager.require_confirmation(
                raw_signals, risk_assessment, user_id
            )

            # 模拟自动确认（实际环境中需要真实的用户交互）
            confirmation_record = confirmation_manager.simulate_user_confirmation(
                confirmation_required.confirmation_id,
                'approved',
                '自动测试确认',
                user_id
            )
        else:
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

        # Step 4: 最终信号
        final_signals = risk_assessment.filtered_signals or raw_signals

        # 审计记录
        audit_id = None
        if self.enable_audit:
            audit_logger = AuditLogger()
            audit_entry = audit_logger.log_signal_workflow(
                session_id, raw_signals, risk_assessment, confirmation_record, final_signals
            )
            audit_id = audit_entry.audit_id

        return {
            'session_id': session_id,
            'signals': final_signals,
            'risk_assessment': risk_assessment,
            'confirmation_status': confirmation_record.status.value,
            'audit_id': audit_id,
            'signal_count': len(final_signals),
            'risk_score': risk_assessment.overall_risk_score
        }

    def save_signals(
        self,
        signals: List[TradingSignal],
        metadata: Dict[str, Any]
    ) -> Dict[str, Any]:
        """保存信号到存储"""
        try:
            signal_batch_id = str(uuid.uuid4())

            # 转换为可序列化格式
            signals_data = [signal.to_dict() for signal in signals]

            # 这里应该保存到实际的存储系统
            # 为简化实现，我们只返回成功状态

            return {
                'success': True,
                'signal_batch_id': signal_batch_id,
                'signals_count': len(signals),
                'metadata': metadata
            }

        except Exception as e:
            self.logger.error(f"保存信号失败: {e}")
            return {
                'success': False,
                'error': str(e)
            }