"""
Point-in-Time Data Access Service
Point-in-Time数据访问服务，确保数据查询时间点的严格一致性，避免前视偏差
"""

import logging
from datetime import date, datetime, timedelta
from typing import Dict, List, Optional, Any, Union, Tuple
from dataclasses import dataclass, field
import pandas as pd
import numpy as np
from pathlib import Path
import threading
from contextlib import contextmanager

from ..models.database import get_db_session
from ..lib.exceptions import InsufficientDataException, ValidationException
from ..lib.hikyuu_wrapper import get_hikyuu_wrapper

logger = logging.getLogger(__name__)


@dataclass
class PitDataRequest:
    """Point-in-Time数据请求"""
    stock_codes: List[str]
    as_of_date: date
    data_types: List[str]  # ["price", "volume", "financial", "factor"]
    lookback_days: int = 252  # 回望天数
    include_delisted: bool = False
    exclude_st: bool = True

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            "stock_codes": self.stock_codes,
            "as_of_date": self.as_of_date.isoformat(),
            "data_types": self.data_types,
            "lookback_days": self.lookback_days,
            "include_delisted": self.include_delisted,
            "exclude_st": self.exclude_st
        }


@dataclass
class PitDataResponse:
    """Point-in-Time数据响应"""
    request: PitDataRequest
    data: Dict[str, pd.DataFrame]
    metadata: Dict[str, Any] = field(default_factory=dict)
    warnings: List[str] = field(default_factory=list)
    execution_time_ms: Optional[int] = None

    def get_stock_data(self, stock_code: str, data_type: str) -> Optional[pd.DataFrame]:
        """获取特定股票和数据类型的数据"""
        key = f"{stock_code}_{data_type}"
        return self.data.get(key)

    def get_cross_sectional_data(self, data_type: str, field: str) -> pd.Series:
        """获取横截面数据"""
        result = {}
        for stock_code in self.request.stock_codes:
            stock_data = self.get_stock_data(stock_code, data_type)
            if stock_data is not None and not stock_data.empty:
                # 获取as_of_date或之前最近的数据
                valid_data = stock_data[stock_data.index <= self.request.as_of_date]
                if not valid_data.empty and field in valid_data.columns:
                    result[stock_code] = valid_data[field].iloc[-1]

        return pd.Series(result)


class PitDataValidator:
    """Point-in-Time数据验证器"""

    @staticmethod
    def validate_request(request: PitDataRequest) -> List[str]:
        """验证数据请求"""
        errors = []

        if not request.stock_codes:
            errors.append("Stock codes cannot be empty")

        if not request.data_types:
            errors.append("Data types cannot be empty")

        if request.lookback_days < 1:
            errors.append("Lookback days must be at least 1")

        # 验证as_of_date不能是未来时间
        if request.as_of_date > date.today():
            errors.append("As-of date cannot be in the future")

        # 验证数据类型
        valid_data_types = {"price", "volume", "financial", "factor", "index"}
        invalid_types = set(request.data_types) - valid_data_types
        if invalid_types:
            errors.append(f"Invalid data types: {invalid_types}")

        return errors

    @staticmethod
    def validate_temporal_consistency(data: pd.DataFrame, as_of_date: date) -> List[str]:
        """验证时间一致性"""
        warnings = []

        if data.empty:
            return warnings

        # 检查是否有未来数据
        if hasattr(data.index, 'date'):
            future_data = data[data.index.date > as_of_date]
        else:
            future_data = data[data.index > as_of_date]

        if not future_data.empty:
            warnings.append(f"Found {len(future_data)} future data points after {as_of_date}")

        return warnings


class PitDataCache:
    """Point-in-Time数据缓存"""

    def __init__(self, max_size: int = 1000):
        self.cache: Dict[str, Tuple[datetime, pd.DataFrame]] = {}
        self.max_size = max_size
        self.lock = threading.RLock()

    def _get_cache_key(self, stock_code: str, data_type: str, as_of_date: date) -> str:
        """生成缓存键"""
        return f"{stock_code}_{data_type}_{as_of_date.isoformat()}"

    def get(self, stock_code: str, data_type: str, as_of_date: date) -> Optional[pd.DataFrame]:
        """获取缓存数据"""
        key = self._get_cache_key(stock_code, data_type, as_of_date)
        with self.lock:
            if key in self.cache:
                timestamp, data = self.cache[key]
                # 检查缓存是否过期（1小时）
                if datetime.now() - timestamp < timedelta(hours=1):
                    return data.copy()
                else:
                    del self.cache[key]
        return None

    def put(self, stock_code: str, data_type: str, as_of_date: date, data: pd.DataFrame) -> None:
        """存储缓存数据"""
        key = self._get_cache_key(stock_code, data_type, as_of_date)
        with self.lock:
            # 检查缓存大小
            if len(self.cache) >= self.max_size:
                # 删除最旧的缓存项
                oldest_key = min(self.cache.keys(), key=lambda k: self.cache[k][0])
                del self.cache[oldest_key]

            self.cache[key] = (datetime.now(), data.copy())

    def clear(self) -> None:
        """清空缓存"""
        with self.lock:
            self.cache.clear()


class PitDataService:
    """Point-in-Time数据访问服务"""

    def __init__(self):
        self.cache = PitDataCache()
        self.hikyuu_wrapper = get_hikyuu_wrapper()
        self.validator = PitDataValidator()

    async def get_data(self, request: PitDataRequest) -> PitDataResponse:
        """获取Point-in-Time数据"""
        start_time = datetime.now()

        # 验证请求
        validation_errors = self.validator.validate_request(request)
        if validation_errors:
            raise ValidationException(f"Invalid PIT data request: {'; '.join(validation_errors)}")

        response = PitDataResponse(
            request=request,
            data={}
        )

        try:
            # 并行获取不同类型的数据
            for data_type in request.data_types:
                for stock_code in request.stock_codes:
                    try:
                        data = await self._get_stock_data(stock_code, data_type, request)
                        if data is not None:
                            key = f"{stock_code}_{data_type}"
                            response.data[key] = data

                            # 验证时间一致性
                            warnings = self.validator.validate_temporal_consistency(data, request.as_of_date)
                            response.warnings.extend(warnings)

                    except Exception as e:
                        logger.warning(f"Failed to get {data_type} data for {stock_code}: {e}")
                        response.warnings.append(f"Failed to get {data_type} data for {stock_code}: {e}")

            # 计算执行时间
            end_time = datetime.now()
            response.execution_time_ms = int((end_time - start_time).total_seconds() * 1000)

            # 添加元数据
            response.metadata = {
                "total_stocks_requested": len(request.stock_codes),
                "total_data_types": len(request.data_types),
                "successful_queries": len(response.data),
                "cache_hits": 0,  # TODO: 实现缓存命中统计
                "as_of_date": request.as_of_date.isoformat()
            }

            logger.info(f"PIT data query completed: {len(response.data)} datasets in {response.execution_time_ms}ms")
            return response

        except Exception as e:
            logger.error(f"PIT data query failed: {e}")
            raise

    async def _get_stock_data(self, stock_code: str, data_type: str, request: PitDataRequest) -> Optional[pd.DataFrame]:
        """获取单个股票的数据"""
        # 检查缓存
        cached_data = self.cache.get(stock_code, data_type, request.as_of_date)
        if cached_data is not None:
            return cached_data

        # 根据数据类型获取数据
        data = None
        if data_type == "price":
            data = await self._get_price_data(stock_code, request)
        elif data_type == "volume":
            data = await self._get_volume_data(stock_code, request)
        elif data_type == "financial":
            data = await self._get_financial_data(stock_code, request)
        elif data_type == "factor":
            data = await self._get_factor_data(stock_code, request)
        elif data_type == "index":
            data = await self._get_index_data(stock_code, request)

        # 缓存数据
        if data is not None:
            self.cache.put(stock_code, data_type, request.as_of_date, data)

        return data

    async def _get_price_data(self, stock_code: str, request: PitDataRequest) -> Optional[pd.DataFrame]:
        """获取价格数据"""
        try:
            if not self.hikyuu_wrapper.is_ready():
                return self._get_price_data_from_db(stock_code, request)

            # 从Hikyuu获取数据
            start_date = request.as_of_date - timedelta(days=request.lookback_days)
            kdata = self.hikyuu_wrapper.get_kdata(
                stock_code,
                query=None  # 这里应该设置具体的查询条件
            )

            if not kdata or len(kdata) == 0:
                return None

            # 转换为DataFrame
            dates = []
            opens = []
            highs = []
            lows = []
            closes = []
            volumes = []

            for i in range(len(kdata)):
                record = kdata[i]
                trade_date = record.datetime.date()

                # 只包含as_of_date及之前的数据
                if trade_date <= request.as_of_date:
                    dates.append(trade_date)
                    opens.append(float(record.openPrice))
                    highs.append(float(record.highPrice))
                    lows.append(float(record.lowPrice))
                    closes.append(float(record.closePrice))
                    volumes.append(float(record.volume))

            if not dates:
                return None

            df = pd.DataFrame({
                "open": opens,
                "high": highs,
                "low": lows,
                "close": closes,
                "volume": volumes
            }, index=pd.to_datetime(dates))

            return df

        except Exception as e:
            logger.warning(f"Failed to get price data from Hikyuu for {stock_code}: {e}")
            return self._get_price_data_from_db(stock_code, request)

    def _get_price_data_from_db(self, stock_code: str, request: PitDataRequest) -> Optional[pd.DataFrame]:
        """从数据库获取价格数据"""
        try:
            start_date = request.as_of_date - timedelta(days=request.lookback_days)

            with get_db_session() as session:
                result = session.execute("""
                    SELECT trade_date, open_price, high_price, low_price, close_price, volume
                    FROM stock_daily_data
                    WHERE stock_code = :stock_code
                      AND trade_date BETWEEN :start_date AND :as_of_date
                    ORDER BY trade_date
                """, {
                    "stock_code": stock_code,
                    "start_date": start_date,
                    "as_of_date": request.as_of_date
                }).fetchall()

                if not result:
                    return None

                df = pd.DataFrame(result, columns=["date", "open", "high", "low", "close", "volume"])
                df["date"] = pd.to_datetime(df["date"])
                df.set_index("date", inplace=True)

                return df

        except Exception as e:
            logger.error(f"Failed to get price data from database for {stock_code}: {e}")
            return None

    async def _get_volume_data(self, stock_code: str, request: PitDataRequest) -> Optional[pd.DataFrame]:
        """获取成交量数据"""
        # 成交量数据通常包含在价格数据中
        price_data = await self._get_price_data(stock_code, request)
        if price_data is not None and "volume" in price_data.columns:
            return price_data[["volume"]].copy()
        return None

    async def _get_financial_data(self, stock_code: str, request: PitDataRequest) -> Optional[pd.DataFrame]:
        """获取财务数据"""
        try:
            # 财务数据通常按季度发布，需要特别处理Point-in-Time逻辑
            with get_db_session() as session:
                result = session.execute("""
                    SELECT report_date, publish_date, total_revenue, net_profit, total_assets, total_equity
                    FROM financial_data
                    WHERE stock_code = :stock_code
                      AND publish_date <= :as_of_date
                      AND report_date >= :start_date
                    ORDER BY report_date
                """, {
                    "stock_code": stock_code,
                    "as_of_date": request.as_of_date,
                    "start_date": request.as_of_date - timedelta(days=request.lookback_days)
                }).fetchall()

                if not result:
                    return None

                df = pd.DataFrame(result, columns=[
                    "report_date", "publish_date", "total_revenue", "net_profit", "total_assets", "total_equity"
                ])
                df["report_date"] = pd.to_datetime(df["report_date"])
                df["publish_date"] = pd.to_datetime(df["publish_date"])
                df.set_index("report_date", inplace=True)

                return df

        except Exception as e:
            logger.error(f"Failed to get financial data for {stock_code}: {e}")
            return None

    async def _get_factor_data(self, stock_code: str, request: PitDataRequest) -> Optional[pd.DataFrame]:
        """获取因子数据"""
        try:
            start_date = request.as_of_date - timedelta(days=request.lookback_days)

            with get_db_session() as session:
                result = session.execute("""
                    SELECT trade_date, factor_id, factor_value
                    FROM factor_data
                    WHERE stock_code = :stock_code
                      AND trade_date BETWEEN :start_date AND :as_of_date
                    ORDER BY trade_date, factor_id
                """, {
                    "stock_code": stock_code,
                    "start_date": start_date,
                    "as_of_date": request.as_of_date
                }).fetchall()

                if not result:
                    return None

                df = pd.DataFrame(result, columns=["date", "factor_id", "value"])
                df["date"] = pd.to_datetime(df["date"])

                # 透视表，将因子ID作为列
                pivot_df = df.pivot_table(index="date", columns="factor_id", values="value")

                return pivot_df

        except Exception as e:
            logger.error(f"Failed to get factor data for {stock_code}: {e}")
            return None

    async def _get_index_data(self, stock_code: str, request: PitDataRequest) -> Optional[pd.DataFrame]:
        """获取指数数据"""
        # 指数数据处理逻辑类似价格数据，但可能来源不同
        return await self._get_price_data(stock_code, request)

    def get_cross_sectional_snapshot(self, stock_codes: List[str], as_of_date: date,
                                     fields: List[str]) -> pd.DataFrame:
        """获取横截面快照数据"""
        try:
            # 构建查询
            field_selects = ", ".join(fields)
            placeholders = ", ".join([f":stock_{i}" for i in range(len(stock_codes))])

            with get_db_session() as session:
                query = f"""
                    SELECT stock_code, {field_selects}
                    FROM (
                        SELECT stock_code, {field_selects},
                               ROW_NUMBER() OVER (PARTITION BY stock_code ORDER BY trade_date DESC) as rn
                        FROM stock_daily_data
                        WHERE stock_code IN ({placeholders})
                          AND trade_date <= :as_of_date
                    ) ranked
                    WHERE rn = 1
                """

                params = {"as_of_date": as_of_date}
                for i, code in enumerate(stock_codes):
                    params[f"stock_{i}"] = code

                result = session.execute(query, params).fetchall()

                if not result:
                    return pd.DataFrame()

                columns = ["stock_code"] + fields
                df = pd.DataFrame(result, columns=columns)
                df.set_index("stock_code", inplace=True)

                return df

        except Exception as e:
            logger.error(f"Failed to get cross-sectional snapshot: {e}")
            return pd.DataFrame()

    @contextmanager
    def batch_mode(self):
        """批量模式上下文管理器，优化批量查询性能"""
        # 在批量模式下可以进行一些优化，比如预加载、批量查询等
        original_cache_size = self.cache.max_size
        self.cache.max_size = original_cache_size * 2  # 增加缓存大小

        try:
            yield
        finally:
            self.cache.max_size = original_cache_size

    def clear_cache(self) -> None:
        """清空缓存"""
        self.cache.clear()
        logger.info("PIT data cache cleared")


# 全局Point-in-Time数据服务实例
pit_data_service = PitDataService()


def get_pit_data_service() -> PitDataService:
    """获取Point-in-Time数据服务实例"""
    return pit_data_service