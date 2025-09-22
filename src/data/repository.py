"""
Data Repository Layer
数据仓库服务层 - 提供统一的数据访问接口
"""

from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime, date
from decimal import Decimal
import pandas as pd
from sqlalchemy.orm import Session
from sqlalchemy import and_, or_, desc, func

from lib.database import db_manager
from lib.environment import env_manager, warn_mock_data
from models.database_models import (
    Stock, MarketData, FinancialData,
    FactorDefinition, FactorValue, FactorCalculationTask,
    StockStatus, FactorCategory, FactorStatus, TaskStatus
)
from data.hikyuu_interface import hikyuu_interface


class StockRepository:
    """股票数据仓库"""

    def __init__(self):
        self.hikyuu = hikyuu_interface

    def get_all_stocks(self) -> List[Dict[str, Any]]:
        """获取所有股票信息"""
        if not env_manager.is_mock_data_allowed():
            # 生产环境使用数据库
            with db_manager.session_scope() as session:
                stocks = session.query(Stock).filter(Stock.status == StockStatus.ACTIVE).all()
                return [self._stock_to_dict(stock) for stock in stocks]
        else:
            # 开发环境可以使用Hikyuu直接获取
            return self.hikyuu.get_all_stocks()

    def get_stock_by_code(self, stock_code: str) -> Optional[Dict[str, Any]]:
        """根据股票代码获取股票信息"""
        if not env_manager.is_mock_data_allowed():
            with db_manager.session_scope() as session:
                stock = session.query(Stock).filter(Stock.stock_code == stock_code).first()
                return self._stock_to_dict(stock) if stock else None
        else:
            stocks = self.hikyuu.get_all_stocks()
            return next((s for s in stocks if s['stock_code'] == stock_code), None)

    def get_stocks_by_market(self, market: str) -> List[Dict[str, Any]]:
        """根据市场获取股票列表"""
        if not env_manager.is_mock_data_allowed():
            with db_manager.session_scope() as session:
                stocks = session.query(Stock).filter(
                    and_(Stock.market == market, Stock.status == StockStatus.ACTIVE)
                ).all()
                return [self._stock_to_dict(stock) for stock in stocks]
        else:
            stocks = self.hikyuu.get_all_stocks()
            return [s for s in stocks if s['market'] == market]

    def get_stocks_by_sector(self, sector: str) -> List[Dict[str, Any]]:
        """根据行业板块获取股票列表"""
        if not env_manager.is_mock_data_allowed():
            with db_manager.session_scope() as session:
                stocks = session.query(Stock).filter(
                    and_(Stock.sector == sector, Stock.status == StockStatus.ACTIVE)
                ).all()
                return [self._stock_to_dict(stock) for stock in stocks]
        else:
            stocks = self.hikyuu.get_all_stocks()
            return [s for s in stocks if s.get('sector') == sector]

    def save_stock(self, stock_data: Dict[str, Any]) -> bool:
        """保存股票信息"""
        if env_manager.is_mock_data_allowed():
            warn_mock_data("Stock save operation in development mode")
            return True

        try:
            with db_manager.session_scope() as session:
                stock = session.query(Stock).filter(Stock.stock_code == stock_data['stock_code']).first()

                if stock:
                    # 更新现有股票
                    for key, value in stock_data.items():
                        if hasattr(stock, key):
                            setattr(stock, key, value)
                else:
                    # 创建新股票
                    stock = Stock(**stock_data)
                    session.add(stock)

                return True
        except Exception as e:
            print(f"Failed to save stock: {e}")
            return False

    def _stock_to_dict(self, stock: Stock) -> Dict[str, Any]:
        """转换Stock对象为字典"""
        return {
            'stock_code': stock.stock_code,
            'stock_name': stock.stock_name,
            'market': stock.market,
            'sector': stock.sector,
            'list_date': stock.list_date,
            'delist_date': stock.delist_date,
            'status': stock.status.value if isinstance(stock.status, StockStatus) else stock.status,
            'created_at': stock.created_at,
            'updated_at': stock.updated_at
        }


class MarketDataRepository:
    """市场数据仓库"""

    def __init__(self):
        self.hikyuu = hikyuu_interface

    def get_market_data(self, stock_code: str, start_date: date, end_date: date) -> pd.DataFrame:
        """获取市场数据"""
        if not env_manager.is_mock_data_allowed():
            # 生产环境优先使用数据库
            with db_manager.session_scope() as session:
                data = session.query(MarketData).filter(
                    and_(
                        MarketData.stock_code == stock_code,
                        MarketData.trade_date >= start_date,
                        MarketData.trade_date <= end_date
                    )
                ).order_by(MarketData.trade_date).all()

                if data:
                    return pd.DataFrame([self._market_data_to_dict(d) for d in data])
                else:
                    # 数据库中没有数据，尝试从Hikyuu获取
                    return self.hikyuu.get_market_data(stock_code, start_date, end_date)
        else:
            # 开发环境直接使用Hikyuu
            return self.hikyuu.get_market_data(stock_code, start_date, end_date)

    def get_latest_market_data(self, stock_code: str, limit: int = 1) -> pd.DataFrame:
        """获取最新的市场数据"""
        if not env_manager.is_mock_data_allowed():
            with db_manager.session_scope() as session:
                data = session.query(MarketData).filter(
                    MarketData.stock_code == stock_code
                ).order_by(desc(MarketData.trade_date)).limit(limit).all()

                return pd.DataFrame([self._market_data_to_dict(d) for d in data])
        else:
            # 开发环境使用模拟数据
            from datetime import timedelta
            end_date = date.today()
            start_date = end_date - timedelta(days=limit)
            return self.hikyuu.get_market_data(stock_code, start_date, end_date).tail(limit)

    def save_market_data_batch(self, data_list: List[Dict[str, Any]]) -> bool:
        """批量保存市场数据"""
        if env_manager.is_mock_data_allowed():
            warn_mock_data("Market data batch save in development mode")
            return True

        try:
            with db_manager.session_scope() as session:
                for data in data_list:
                    market_data = MarketData(**data)
                    session.merge(market_data)  # 使用merge处理重复数据
                return True
        except Exception as e:
            print(f"Failed to save market data batch: {e}")
            return False

    def _market_data_to_dict(self, data: MarketData) -> Dict[str, Any]:
        """转换MarketData对象为字典"""
        return {
            'stock_code': data.stock_code,
            'trade_date': data.trade_date,
            'open_price': data.open_price,
            'high_price': data.high_price,
            'low_price': data.low_price,
            'close_price': data.close_price,
            'volume': data.volume,
            'amount': data.amount,
            'adj_factor': data.adj_factor,
            'turnover_rate': data.turnover_rate
        }


class FactorRepository:
    """因子数据仓库"""

    def get_all_factors(self, category: Optional[str] = None, status: Optional[str] = None) -> List[Dict[str, Any]]:
        """获取所有因子定义"""
        if not env_manager.is_mock_data_allowed():
            with db_manager.session_scope() as session:
                query = session.query(FactorDefinition)

                if category:
                    query = query.filter(FactorDefinition.category == FactorCategory(category))
                if status:
                    query = query.filter(FactorDefinition.status == FactorStatus(status))

                factors = query.all()
                return [self._factor_definition_to_dict(f) for f in factors]
        else:
            # 开发环境返回模拟因子
            return self._get_mock_factors(category, status)

    def get_factor_by_id(self, factor_id: str) -> Optional[Dict[str, Any]]:
        """根据ID获取因子定义"""
        if not env_manager.is_mock_data_allowed():
            with db_manager.session_scope() as session:
                factor = session.query(FactorDefinition).filter(
                    FactorDefinition.factor_id == factor_id
                ).first()
                return self._factor_definition_to_dict(factor) if factor else None
        else:
            factors = self._get_mock_factors()
            return next((f for f in factors if f['factor_id'] == factor_id), None)

    def get_factor_values(self, factor_id: str, stock_codes: List[str],
                         start_date: date, end_date: date) -> pd.DataFrame:
        """获取因子值"""
        if not env_manager.is_mock_data_allowed():
            with db_manager.session_scope() as session:
                values = session.query(FactorValue).filter(
                    and_(
                        FactorValue.factor_id == factor_id,
                        FactorValue.stock_code.in_(stock_codes),
                        FactorValue.trade_date >= start_date,
                        FactorValue.trade_date <= end_date
                    )
                ).order_by(FactorValue.trade_date, FactorValue.stock_code).all()

                return pd.DataFrame([self._factor_value_to_dict(v) for v in values])
        else:
            # 开发环境使用Hikyuu计算或返回模拟数据
            return self._get_mock_factor_values(factor_id, stock_codes, start_date, end_date)

    def save_factor_definition(self, factor_data: Dict[str, Any]) -> bool:
        """保存因子定义"""
        if env_manager.is_mock_data_allowed():
            warn_mock_data("Factor definition save in development mode")
            return True

        try:
            with db_manager.session_scope() as session:
                factor = FactorDefinition(**factor_data)
                session.merge(factor)
                return True
        except Exception as e:
            print(f"Failed to save factor definition: {e}")
            return False

    def save_factor_values_batch(self, values_list: List[Dict[str, Any]]) -> bool:
        """批量保存因子值"""
        if env_manager.is_mock_data_allowed():
            warn_mock_data("Factor values batch save in development mode")
            return True

        try:
            with db_manager.session_scope() as session:
                for value_data in values_list:
                    factor_value = FactorValue(**value_data)
                    session.merge(factor_value)
                return True
        except Exception as e:
            print(f"Failed to save factor values batch: {e}")
            return False

    def _factor_definition_to_dict(self, factor: FactorDefinition) -> Dict[str, Any]:
        """转换FactorDefinition对象为字典"""
        return {
            'factor_id': factor.factor_id,
            'factor_name': factor.factor_name,
            'category': factor.category.value if isinstance(factor.category, FactorCategory) else factor.category,
            'formula': factor.formula,
            'description': factor.description,
            'economic_logic': factor.economic_logic,
            'status': factor.status.value if isinstance(factor.status, FactorStatus) else factor.status,
            'calculation_params': factor.calculation_params,
            'coverage_ratio': factor.coverage_ratio,
            'avg_calculation_time_ms': factor.avg_calculation_time_ms,
            'created_by': factor.created_by,
            'created_at': factor.created_at,
            'updated_at': factor.updated_at
        }

    def _factor_value_to_dict(self, value: FactorValue) -> Dict[str, Any]:
        """转换FactorValue对象为字典"""
        return {
            'factor_id': value.factor_id,
            'stock_code': value.stock_code,
            'trade_date': value.trade_date,
            'factor_value': value.factor_value,
            'factor_score': value.factor_score,
            'percentile_rank': value.percentile_rank,
            'calculation_id': value.calculation_id,
            'data_version': value.data_version,
            'created_at': value.created_at
        }

    def _get_mock_factors(self, category: Optional[str] = None, status: Optional[str] = None) -> List[Dict[str, Any]]:
        """获取模拟因子数据"""
        warn_mock_data("Using mock factor definitions")

        factors = [
            {
                'factor_id': 'momentum_20d',
                'factor_name': '20日动量因子',
                'category': 'momentum',
                'formula': 'CLOSE() / MA(CLOSE(), 20) - 1',
                'description': '基于20日价格动量的因子',
                'economic_logic': '捕捉短期价格趋势',
                'status': 'active',
                'calculation_params': {'period': 20},
                'coverage_ratio': Decimal('0.95'),
                'avg_calculation_time_ms': 166,
                'created_by': 'system',
                'created_at': datetime(2024, 1, 1),
                'updated_at': datetime(2024, 1, 1)
            },
            {
                'factor_id': 'rsi_14d',
                'factor_name': '14日RSI因子',
                'category': 'momentum',
                'formula': 'RSI(CLOSE(), 14)',
                'description': '相对强弱指标',
                'economic_logic': '衡量股票超买超卖状态',
                'status': 'active',
                'calculation_params': {'period': 14},
                'coverage_ratio': Decimal('0.92'),
                'avg_calculation_time_ms': 145,
                'created_by': 'system',
                'created_at': datetime(2024, 1, 1),
                'updated_at': datetime(2024, 1, 1)
            },
            {
                'factor_id': 'pe_ratio',
                'factor_name': '市盈率因子',
                'category': 'value',
                'formula': 'CLOSE() / EPS',
                'description': '股票估值因子',
                'economic_logic': '价值投资基础指标',
                'status': 'active',
                'calculation_params': {},
                'coverage_ratio': Decimal('0.88'),
                'avg_calculation_time_ms': 89,
                'created_by': 'system',
                'created_at': datetime(2024, 1, 1),
                'updated_at': datetime(2024, 1, 1)
            }
        ]

        # 应用筛选
        if category:
            factors = [f for f in factors if f['category'] == category]
        if status:
            factors = [f for f in factors if f['status'] == status]

        return factors

    def _get_mock_factor_values(self, factor_id: str, stock_codes: List[str],
                               start_date: date, end_date: date) -> pd.DataFrame:
        """获取模拟因子值"""
        warn_mock_data(f"Using mock factor values for {factor_id}")

        import numpy as np

        data = []
        dates = pd.date_range(start=start_date, end=end_date, freq='D')

        for stock_code in stock_codes:
            for date_val in dates:
                data.append({
                    'factor_id': factor_id,
                    'stock_code': stock_code,
                    'trade_date': date_val.date(),
                    'factor_value': Decimal(str(round(np.random.normal(0, 0.1), 6))),
                    'factor_score': Decimal(str(round(np.random.uniform(0, 1), 6))),
                    'percentile_rank': Decimal(str(round(np.random.uniform(0, 1), 4))),
                    'calculation_id': 'mock_calc_001',
                    'data_version': 'mock_v1.0',
                    'created_at': datetime.now()
                })

        return pd.DataFrame(data)


# 创建仓库实例
stock_repository = StockRepository()
market_data_repository = MarketDataRepository()
factor_repository = FactorRepository()