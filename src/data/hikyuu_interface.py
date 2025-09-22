"""
Hikyuu Framework Interface
Hikyuu量化框架接口层
"""

import os
import logging
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime, date
from decimal import Decimal
import pandas as pd

try:
    import hikyuu as hku
    from hikyuu import (
        Stock, KData, Query, KQuery, Datetime,
        CLOSE, OPEN, HIGH, LOW, VOL, AMO,
        MA, EMA, RSI, MACD, KDJ
    )
    HIKYUU_AVAILABLE = True
except ImportError as e:
    logging.warning(f"Hikyuu not available: {e}")
    HIKYUU_AVAILABLE = False
    # 定义空的占位类
    class Stock: pass
    class KData: pass
    class Query: pass
    class KQuery: pass
    class Datetime: pass

from lib.environment import env_manager, warn_mock_data


class HikyuuDataInterface:
    """Hikyuu数据接口"""

    def __init__(self, config_path: Optional[str] = None):
        self.config_path = config_path or self._get_default_config_path()
        self._initialized = False
        self._stock_manager = None

    def _get_default_config_path(self) -> str:
        """获取默认配置路径"""
        if env_manager.is_development_only():
            return os.path.join(os.path.expanduser("~"), ".hikyuu", "hikyuu.ini")
        else:
            config_path = os.getenv('HIKYUU_CONFIG_PATH')
            if not config_path:
                raise ValueError(
                    "HIKYUU_CONFIG_PATH environment variable is required for production"
                )
            return config_path

    def initialize(self) -> bool:
        """初始化Hikyuu"""
        if not HIKYUU_AVAILABLE:
            if not env_manager.is_mock_data_allowed():
                raise RuntimeError(
                    "Hikyuu is not available and mock data is prohibited in this environment"
                )
            warn_mock_data("Hikyuu framework not available, using mock mode")
            self._initialized = True
            return True

        try:
            # 初始化Hikyuu
            hku.set_global_config({
                'log_level': 'INFO' if env_manager.is_development_only() else 'ERROR',
                'data_dir': os.path.join(os.path.dirname(self.config_path), 'data'),
                'tmp_dir': '/tmp/hikyuu'
            })

            # 加载配置并初始化
            if os.path.exists(self.config_path):
                self._stock_manager = hku.StockManager.instance()
                self._stock_manager.init(self.config_path)
                logging.info(f"Hikyuu initialized with config: {self.config_path}")
            else:
                logging.warning(f"Hikyuu config not found: {self.config_path}")
                if not env_manager.is_mock_data_allowed():
                    raise FileNotFoundError(f"Hikyuu config file not found: {self.config_path}")

            self._initialized = True
            return True

        except Exception as e:
            logging.error(f"Failed to initialize Hikyuu: {e}")
            if not env_manager.is_mock_data_allowed():
                raise
            warn_mock_data("Hikyuu initialization failed, using mock mode")
            self._initialized = True
            return False

    def get_all_stocks(self) -> List[Dict[str, Any]]:
        """获取所有股票信息"""
        if not self._initialized:
            self.initialize()

        if not HIKYUU_AVAILABLE or not self._stock_manager:
            return self._get_mock_stocks()

        try:
            stocks = []
            stock_list = self._stock_manager.get_stock_type_info(hku.STOCKTYPE.A)

            for stock_info in stock_list:
                stock = self._stock_manager[stock_info.market + stock_info.code]
                if stock.valid:
                    stocks.append({
                        'stock_code': f"{stock_info.market}{stock_info.code}",
                        'stock_name': stock.name,
                        'market': stock_info.market,
                        'sector': getattr(stock, 'sector', None),
                        'list_date': self._convert_datetime_to_date(stock.start_datetime),
                        'delist_date': self._convert_datetime_to_date(stock.last_datetime) if stock.last_datetime != hku.Null_Datetime else None,
                        'status': 'active'
                    })

            return stocks

        except Exception as e:
            logging.error(f"Failed to get stocks from Hikyuu: {e}")
            if env_manager.is_mock_data_allowed():
                return self._get_mock_stocks()
            raise

    def get_market_data(self, stock_code: str, start_date: date, end_date: date) -> pd.DataFrame:
        """获取市场数据"""
        if not self._initialized:
            self.initialize()

        if not HIKYUU_AVAILABLE or not self._stock_manager:
            return self._get_mock_market_data(stock_code, start_date, end_date)

        try:
            stock = self._stock_manager[stock_code]
            if not stock.valid:
                raise ValueError(f"Invalid stock code: {stock_code}")

            # 构建查询
            query = KQuery(
                start=hku.Datetime(start_date),
                end=hku.Datetime(end_date),
                ktype=hku.KType.DAY
            )

            # 获取K线数据
            kdata = stock.get_kdata(query)
            if kdata.empty():
                return pd.DataFrame()

            # 转换为DataFrame
            data = []
            for i in range(len(kdata)):
                record = kdata[i]
                data.append({
                    'stock_code': stock_code,
                    'trade_date': record.datetime.date(),
                    'open_price': Decimal(str(record.open_price)),
                    'high_price': Decimal(str(record.high_price)),
                    'low_price': Decimal(str(record.low_price)),
                    'close_price': Decimal(str(record.close_price)),
                    'volume': int(record.volume),
                    'amount': Decimal(str(record.amount)),
                    'adj_factor': Decimal('1.0'),  # 需要实际计算复权因子
                    'turnover_rate': None  # 需要额外计算
                })

            return pd.DataFrame(data)

        except Exception as e:
            logging.error(f"Failed to get market data from Hikyuu: {e}")
            if env_manager.is_mock_data_allowed():
                return self._get_mock_market_data(stock_code, start_date, end_date)
            raise

    def calculate_factor(self, factor_formula: str, stock_code: str, start_date: date, end_date: date) -> pd.DataFrame:
        """计算因子值"""
        if not self._initialized:
            self.initialize()

        if not HIKYUU_AVAILABLE or not self._stock_manager:
            return self._get_mock_factor_data(factor_formula, stock_code, start_date, end_date)

        try:
            stock = self._stock_manager[stock_code]
            if not stock.valid:
                raise ValueError(f"Invalid stock code: {stock_code}")

            # 解析并执行因子公式
            factor_values = self._execute_factor_formula(factor_formula, stock, start_date, end_date)

            return factor_values

        except Exception as e:
            logging.error(f"Failed to calculate factor: {e}")
            if env_manager.is_mock_data_allowed():
                return self._get_mock_factor_data(factor_formula, stock_code, start_date, end_date)
            raise

    def _execute_factor_formula(self, formula: str, stock: Stock, start_date: date, end_date: date) -> pd.DataFrame:
        """执行因子计算公式"""
        # 创建查询
        query = KQuery(
            start=hku.Datetime(start_date),
            end=hku.Datetime(end_date),
            ktype=hku.KType.DAY
        )

        # 获取基础数据
        kdata = stock.get_kdata(query)
        if kdata.empty():
            return pd.DataFrame()

        # 解析公式并计算
        if "MA(" in formula:
            # 示例：移动平均相关因子
            return self._calculate_ma_factor(formula, stock, query)
        elif "RSI(" in formula:
            # 示例：RSI因子
            return self._calculate_rsi_factor(formula, stock, query)
        elif "CLOSE()" in formula:
            # 示例：价格动量因子
            return self._calculate_momentum_factor(formula, stock, query)
        else:
            # 通用计算逻辑
            return self._calculate_generic_factor(formula, stock, query)

    def _calculate_ma_factor(self, formula: str, stock: Stock, query: KQuery) -> pd.DataFrame:
        """计算移动平均相关因子"""
        # 示例实现
        close_data = CLOSE(stock, query)
        ma20_data = MA(close_data, 20)

        data = []
        for i in range(len(close_data)):
            if i >= 19:  # 确保有足够的数据计算MA20
                factor_value = float(close_data[i] / ma20_data[i] - 1) if ma20_data[i] != 0 else 0
                data.append({
                    'stock_code': stock.market_code + stock.code,
                    'trade_date': close_data.get_datetime(i).date(),
                    'factor_value': Decimal(str(factor_value)),
                    'factor_score': None,
                    'percentile_rank': None
                })

        return pd.DataFrame(data)

    def _calculate_rsi_factor(self, formula: str, stock: Stock, query: KQuery) -> pd.DataFrame:
        """计算RSI因子"""
        close_data = CLOSE(stock, query)
        rsi_data = RSI(close_data, 14)

        data = []
        for i in range(len(rsi_data)):
            if i >= 13:  # 确保有足够的数据计算RSI14
                data.append({
                    'stock_code': stock.market_code + stock.code,
                    'trade_date': close_data.get_datetime(i).date(),
                    'factor_value': Decimal(str(float(rsi_data[i]))),
                    'factor_score': None,
                    'percentile_rank': None
                })

        return pd.DataFrame(data)

    def _calculate_momentum_factor(self, formula: str, stock: Stock, query: KQuery) -> pd.DataFrame:
        """计算动量因子"""
        close_data = CLOSE(stock, query)

        data = []
        for i in range(20, len(close_data)):  # 20日动量
            current_price = float(close_data[i])
            past_price = float(close_data[i-20])
            momentum = (current_price / past_price - 1) if past_price != 0 else 0

            data.append({
                'stock_code': stock.market_code + stock.code,
                'trade_date': close_data.get_datetime(i).date(),
                'factor_value': Decimal(str(momentum)),
                'factor_score': None,
                'percentile_rank': None
            })

        return pd.DataFrame(data)

    def _calculate_generic_factor(self, formula: str, stock: Stock, query: KQuery) -> pd.DataFrame:
        """通用因子计算"""
        # 这里需要更复杂的公式解析和执行逻辑
        # 暂时返回空DataFrame
        return pd.DataFrame()

    def _convert_datetime_to_date(self, dt) -> Optional[date]:
        """转换Hikyuu Datetime到Python date"""
        if not dt or dt == hku.Null_Datetime:
            return None
        return dt.date()

    def _get_mock_stocks(self) -> List[Dict[str, Any]]:
        """获取模拟股票数据"""
        warn_mock_data("Using mock stock data")

        return [
            {
                'stock_code': 'sh000001',
                'stock_name': '上证指数',
                'market': 'sh',
                'sector': '指数',
                'list_date': date(1990, 12, 19),
                'delist_date': None,
                'status': 'active'
            },
            {
                'stock_code': 'sz000001',
                'stock_name': '平安银行',
                'market': 'sz',
                'sector': '银行',
                'list_date': date(1991, 4, 3),
                'delist_date': None,
                'status': 'active'
            },
            {
                'stock_code': 'sh600036',
                'stock_name': '招商银行',
                'market': 'sh',
                'sector': '银行',
                'list_date': date(2002, 4, 9),
                'delist_date': None,
                'status': 'active'
            }
        ]

    def _get_mock_market_data(self, stock_code: str, start_date: date, end_date: date) -> pd.DataFrame:
        """获取模拟市场数据"""
        warn_mock_data(f"Using mock market data for {stock_code}")

        import numpy as np
        from datetime import timedelta

        # 生成日期范围
        dates = pd.date_range(start=start_date, end=end_date, freq='D')
        dates = [d.date() for d in dates]

        data = []
        base_price = 10.0

        for i, trade_date in enumerate(dates):
            # 模拟价格走势
            price_change = np.random.normal(0, 0.02)
            base_price *= (1 + price_change)

            data.append({
                'stock_code': stock_code,
                'trade_date': trade_date,
                'open_price': Decimal(str(round(base_price * 0.999, 3))),
                'high_price': Decimal(str(round(base_price * 1.01, 3))),
                'low_price': Decimal(str(round(base_price * 0.99, 3))),
                'close_price': Decimal(str(round(base_price, 3))),
                'volume': int(np.random.uniform(1000000, 10000000)),
                'amount': Decimal(str(round(base_price * np.random.uniform(1000000, 10000000), 2))),
                'adj_factor': Decimal('1.0'),
                'turnover_rate': Decimal(str(round(np.random.uniform(0.5, 5.0), 4)))
            })

        return pd.DataFrame(data)

    def _get_mock_factor_data(self, formula: str, stock_code: str, start_date: date, end_date: date) -> pd.DataFrame:
        """获取模拟因子数据"""
        warn_mock_data(f"Using mock factor data for {formula}")

        import numpy as np

        # 生成日期范围
        dates = pd.date_range(start=start_date, end=end_date, freq='D')
        dates = [d.date() for d in dates]

        data = []
        for trade_date in dates:
            data.append({
                'stock_code': stock_code,
                'trade_date': trade_date,
                'factor_value': Decimal(str(round(np.random.normal(0, 0.1), 6))),
                'factor_score': Decimal(str(round(np.random.uniform(0, 1), 6))),
                'percentile_rank': Decimal(str(round(np.random.uniform(0, 1), 4)))
            })

        return pd.DataFrame(data)


# 全局Hikyuu接口实例
hikyuu_interface = HikyuuDataInterface()