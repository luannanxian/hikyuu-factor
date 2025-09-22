"""
Hikyuu Framework Interface
Hikyuu量化框架接口层
"""

import os
import logging
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime, date, timedelta
from decimal import Decimal
import pandas as pd
import numpy as np

try:
    import hikyuu as hku
    from hikyuu import (
        Stock, KData, Query, KQuery, Datetime,
        CLOSE, OPEN, HIGH, LOW, VOL, AMO,
        MA, EMA, RSI, MACD, KDJ, BOLL,
        FINANCE, MF,  # 财务数据和多因子功能
        StockManager, Timeline
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
    class StockManager: pass
    class Timeline: pass
    FINANCE = None
    MF = None

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

    def get_financial_data(self, stock_code: str, start_date: date, end_date: date) -> pd.DataFrame:
        """获取财务数据（使用Hikyuu FINANCE功能）"""
        if not self._initialized:
            self.initialize()

        if not HIKYUU_AVAILABLE or not self._stock_manager or not FINANCE:
            return self._get_mock_financial_data(stock_code, start_date, end_date)

        try:
            stock = self._stock_manager[stock_code]
            if not stock.valid:
                raise ValueError(f"Invalid stock code: {stock_code}")

            # 使用Hikyuu FINANCE功能获取财务数据
            financial_data = []

            # 获取常用财务指标
            financial_indicators = {
                'eps': 'EPS',  # 每股收益
                'bvps': 'BVPS',  # 每股净资产
                'roe': 'ROE',  # 净资产收益率
                'roa': 'ROA',  # 总资产收益率
                'pe': 'PE',  # 市盈率
                'pb': 'PB',  # 市净率
                'total_revenue': 'REVENUE',  # 营业收入
                'net_profit': 'NETPROFIT',  # 净利润
                'total_assets': 'TOTALASSETS',  # 总资产
                'total_equity': 'TOTALEQUITY'  # 股东权益
            }

            # 获取日期范围内的财务数据
            current_date = start_date
            while current_date <= end_date:
                date_data = {'date': current_date, 'stock_code': stock_code}

                for key, finance_key in financial_indicators.items():
                    try:
                        # 使用FINANCE获取指定日期的财务数据
                        value = FINANCE(stock, hku.Datetime(current_date))[finance_key]
                        date_data[key] = float(value) if value is not None else None
                    except Exception as e:
                        logging.debug(f"Failed to get {finance_key} for {stock_code} on {current_date}: {e}")
                        date_data[key] = None

                financial_data.append(date_data)
                current_date += timedelta(days=1)

            return pd.DataFrame(financial_data)

        except Exception as e:
            logging.error(f"Failed to get financial data from Hikyuu: {e}")
            if env_manager.is_mock_data_allowed():
                return self._get_mock_financial_data(stock_code, start_date, end_date)
            raise

    def calculate_multi_factors(self, stock_codes: List[str], start_date: date, end_date: date,
                               factor_list: List[str]) -> Dict[str, pd.DataFrame]:
        """使用Hikyuu MF功能批量计算多个因子"""
        if not self._initialized:
            self.initialize()

        if not HIKYUU_AVAILABLE or not self._stock_manager or not MF:
            return self._get_mock_multi_factors(stock_codes, start_date, end_date, factor_list)

        try:
            # 使用Hikyuu MF(多因子)功能
            results = {}

            # 构建查询
            query = KQuery(
                start=hku.Datetime(start_date),
                end=hku.Datetime(end_date),
                ktype=hku.KType.DAY
            )

            for factor_name in factor_list:
                factor_data = []

                for stock_code in stock_codes:
                    try:
                        stock = self._stock_manager[stock_code]
                        if not stock.valid:
                            continue

                        # 根据因子名称选择相应的计算方法
                        factor_values = self._calculate_factor_with_mf(
                            stock, query, factor_name
                        )

                        # 转换为DataFrame格式
                        for i, value in enumerate(factor_values):
                            if not pd.isna(value):
                                factor_data.append({
                                    'stock_code': stock_code,
                                    'trade_date': start_date + timedelta(days=i),
                                    'factor_value': float(value),
                                    'factor_score': None,
                                    'percentile_rank': None
                                })

                    except Exception as e:
                        logging.warning(f"Failed to calculate {factor_name} for {stock_code}: {e}")
                        continue

                results[factor_name] = pd.DataFrame(factor_data)

            return results

        except Exception as e:
            logging.error(f"Failed to calculate multi-factors with Hikyuu MF: {e}")
            if env_manager.is_mock_data_allowed():
                return self._get_mock_multi_factors(stock_codes, start_date, end_date, factor_list)
            raise

    def _calculate_factor_with_mf(self, stock: Stock, query: KQuery, factor_name: str):
        """使用Hikyuu MF计算具体因子"""
        try:
            # 获取基本数据
            close_data = CLOSE(stock, query)
            open_data = OPEN(stock, query)
            high_data = HIGH(stock, query)
            low_data = LOW(stock, query)
            vol_data = VOL(stock, query)

            # 根据因子名称计算
            if factor_name == 'momentum_20d':
                # 20日动量因子
                ma20 = MA(close_data, 20)
                return (close_data / ma20 - 1).to_np()

            elif factor_name == 'rsi_14d':
                # RSI因子
                rsi_values = RSI(close_data, 14)
                return rsi_values.to_np()

            elif factor_name == 'volatility_20d':
                # 20日波动率
                returns = close_data.pct_change()
                volatility = returns.rolling(20).std() * np.sqrt(252)
                return volatility.to_np() if hasattr(volatility, 'to_np') else np.array(volatility)

            elif factor_name == 'macd_signal':
                # MACD信号
                macd_result = MACD(close_data)
                return macd_result.to_np() if hasattr(macd_result, 'to_np') else np.array(macd_result)

            elif factor_name == 'bollinger_position':
                # 布林带位置
                boll_result = BOLL(close_data, 20, 2)
                # 返回价格在布林带中的位置
                upper = boll_result.upper
                lower = boll_result.lower
                position = (close_data - lower) / (upper - lower)
                return position.to_np() if hasattr(position, 'to_np') else np.array(position)

            elif factor_name == 'volume_ratio':
                # 成交量比率
                vol_ma = MA(vol_data, 20)
                vol_ratio = vol_data / vol_ma
                return vol_ratio.to_np() if hasattr(vol_ratio, 'to_np') else np.array(vol_ratio)

            else:
                # 默认返回空数组
                return np.full(len(close_data), np.nan)

        except Exception as e:
            logging.error(f"MF factor calculation failed for {factor_name}: {e}")
            return np.array([])

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

    def _get_mock_financial_data(self, stock_code: str, start_date: date, end_date: date) -> pd.DataFrame:
        """获取模拟财务数据"""
        warn_mock_data(f"Using mock financial data for {stock_code}")

        dates = pd.date_range(start=start_date, end=end_date, freq='D')
        data = []

        np.random.seed(hash(stock_code) % 2**32)
        for trade_date in dates:
            data.append({
                'date': trade_date.date(),
                'stock_code': stock_code,
                'eps': round(np.random.uniform(0.5, 2.0), 4),
                'bvps': round(np.random.uniform(3.0, 8.0), 4),
                'roe': round(np.random.uniform(0.05, 0.25), 4),
                'roa': round(np.random.uniform(0.02, 0.15), 4),
                'pe': round(np.random.uniform(8.0, 50.0), 2),
                'pb': round(np.random.uniform(0.8, 5.0), 2),
                'total_revenue': round(np.random.uniform(1e8, 1e10), 0),
                'net_profit': round(np.random.uniform(1e7, 1e9), 0),
                'total_assets': round(np.random.uniform(1e9, 1e11), 0),
                'total_equity': round(np.random.uniform(1e8, 1e10), 0)
            })

        return pd.DataFrame(data)

    def _get_mock_multi_factors(self, stock_codes: List[str], start_date: date, end_date: date,
                               factor_list: List[str]) -> Dict[str, pd.DataFrame]:
        """获取模拟多因子数据"""
        warn_mock_data(f"Using mock multi-factor data for {len(stock_codes)} stocks, {len(factor_list)} factors")

        results = {}
        dates = pd.date_range(start=start_date, end=end_date, freq='D')

        for factor_name in factor_list:
            factor_data = []
            for stock_code in stock_codes:
                np.random.seed(hash(stock_code + factor_name) % 2**32)
                for trade_date in dates:
                    factor_data.append({
                        'stock_code': stock_code,
                        'trade_date': trade_date.date(),
                        'factor_value': round(np.random.normal(0, 0.1), 6),
                        'factor_score': round(np.random.uniform(0, 1), 6),
                        'percentile_rank': round(np.random.uniform(0, 1), 4)
                    })
            results[factor_name] = pd.DataFrame(factor_data)

        return results


# 全局Hikyuu接口实例
hikyuu_interface = HikyuuDataInterface()