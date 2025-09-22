"""
Stock Pool Management Models
股票池管理相关的数据模型，集成Hikyuu框架的股票管理功能
"""

import enum
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta
from typing import Dict, List, Optional, Any, Set, Union, Callable
import json
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


class StockStatus(enum.Enum):
    """股票状态"""
    NORMAL = "normal"  # 正常交易
    ST = "st"  # ST股票
    STAR_ST = "star_st"  # *ST股票
    SUSPENDED = "suspended"  # 停牌
    DELISTED = "delisted"  # 退市
    NEW_LISTED = "new_listed"  # 新上市


class MarketType(enum.Enum):
    """市场类型"""
    SH_A = "sh_a"  # 上海A股
    SZ_A = "sz_a"  # 深圳A股
    SH_B = "sh_b"  # 上海B股
    SZ_B = "sz_b"  # 深圳B股
    STAR = "star"  # 科创板
    GEM = "gem"  # 创业板
    NEEQ = "neeq"  # 新三板


class IndustryType(enum.Enum):
    """行业分类类型"""
    SW_L1 = "sw_l1"  # 申万一级行业
    SW_L2 = "sw_l2"  # 申万二级行业
    SW_L3 = "sw_l3"  # 申万三级行业
    CITIC = "citic"  # 中信行业
    WIND = "wind"  # Wind行业
    CUSTOM = "custom"  # 自定义行业


@dataclass
class StockInfo:
    """股票基础信息"""
    stock_code: str
    stock_name: str
    market_type: MarketType
    status: StockStatus = StockStatus.NORMAL
    list_date: Optional[date] = None
    delist_date: Optional[date] = None
    industry_code: Optional[str] = None
    industry_name: Optional[str] = None
    industry_type: Optional[IndustryType] = None
    market_cap: Optional[float] = None  # 市值（万元）
    total_shares: Optional[float] = None  # 总股本（万股）
    float_shares: Optional[float] = None  # 流通股本（万股）
    last_update: datetime = field(default_factory=datetime.now)

    @property
    def is_tradable(self) -> bool:
        """是否可交易"""
        return self.status in [StockStatus.NORMAL, StockStatus.NEW_LISTED]

    @property
    def is_st_stock(self) -> bool:
        """是否为ST股票"""
        return self.status in [StockStatus.ST, StockStatus.STAR_ST]

    @property
    def trading_days_since_list(self) -> Optional[int]:
        """上市以来交易天数"""
        if not self.list_date:
            return None
        return (date.today() - self.list_date).days

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            "stock_code": self.stock_code,
            "stock_name": self.stock_name,
            "market_type": self.market_type.value,
            "status": self.status.value,
            "list_date": self.list_date.isoformat() if self.list_date else None,
            "delist_date": self.delist_date.isoformat() if self.delist_date else None,
            "industry_code": self.industry_code,
            "industry_name": self.industry_name,
            "industry_type": self.industry_type.value if self.industry_type else None,
            "market_cap": self.market_cap,
            "total_shares": self.total_shares,
            "float_shares": self.float_shares,
            "last_update": self.last_update.isoformat()
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'StockInfo':
        """从字典创建"""
        return cls(
            stock_code=data["stock_code"],
            stock_name=data["stock_name"],
            market_type=MarketType(data["market_type"]),
            status=StockStatus(data.get("status", "normal")),
            list_date=date.fromisoformat(data["list_date"]) if data.get("list_date") else None,
            delist_date=date.fromisoformat(data["delist_date"]) if data.get("delist_date") else None,
            industry_code=data.get("industry_code"),
            industry_name=data.get("industry_name"),
            industry_type=IndustryType(data["industry_type"]) if data.get("industry_type") else None,
            market_cap=data.get("market_cap"),
            total_shares=data.get("total_shares"),
            float_shares=data.get("float_shares"),
            last_update=datetime.fromisoformat(data.get("last_update", datetime.now().isoformat()))
        )


@dataclass
class StockPoolFilter:
    """股票池过滤条件"""
    exclude_st: bool = True  # 排除ST股票
    exclude_suspended: bool = True  # 排除停牌股票
    exclude_new_listed: bool = False  # 排除新上市股票
    min_list_days: Optional[int] = None  # 最少上市天数
    max_list_days: Optional[int] = None  # 最多上市天数
    min_market_cap: Optional[float] = None  # 最小市值（万元）
    max_market_cap: Optional[float] = None  # 最大市值（万元）
    min_price: Optional[float] = None  # 最小价格
    max_price: Optional[float] = None  # 最大价格
    allowed_markets: Optional[List[MarketType]] = None  # 允许的市场
    allowed_industries: Optional[List[str]] = None  # 允许的行业
    excluded_industries: Optional[List[str]] = None  # 排除的行业
    custom_filters: List[Callable[[StockInfo], bool]] = field(default_factory=list)

    def apply(self, stock: StockInfo) -> bool:
        """应用过滤条件"""
        # ST股票过滤
        if self.exclude_st and stock.is_st_stock:
            return False

        # 停牌股票过滤
        if self.exclude_suspended and stock.status == StockStatus.SUSPENDED:
            return False

        # 新上市股票过滤
        if self.exclude_new_listed and stock.status == StockStatus.NEW_LISTED:
            return False

        # 上市天数过滤
        if stock.list_date:
            days_since_list = stock.trading_days_since_list
            if self.min_list_days and days_since_list and days_since_list < self.min_list_days:
                return False
            if self.max_list_days and days_since_list and days_since_list > self.max_list_days:
                return False

        # 市值过滤
        if stock.market_cap:
            if self.min_market_cap and stock.market_cap < self.min_market_cap:
                return False
            if self.max_market_cap and stock.market_cap > self.max_market_cap:
                return False

        # 市场过滤
        if self.allowed_markets and stock.market_type not in self.allowed_markets:
            return False

        # 行业过滤
        if self.allowed_industries and stock.industry_code:
            if stock.industry_code not in self.allowed_industries:
                return False

        if self.excluded_industries and stock.industry_code:
            if stock.industry_code in self.excluded_industries:
                return False

        # 自定义过滤器
        for custom_filter in self.custom_filters:
            if not custom_filter(stock):
                return False

        return True

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            "exclude_st": self.exclude_st,
            "exclude_suspended": self.exclude_suspended,
            "exclude_new_listed": self.exclude_new_listed,
            "min_list_days": self.min_list_days,
            "max_list_days": self.max_list_days,
            "min_market_cap": self.min_market_cap,
            "max_market_cap": self.max_market_cap,
            "min_price": self.min_price,
            "max_price": self.max_price,
            "allowed_markets": [m.value for m in self.allowed_markets] if self.allowed_markets else None,
            "allowed_industries": self.allowed_industries,
            "excluded_industries": self.excluded_industries
        }


@dataclass
class StockPool:
    """股票池定义"""
    pool_id: str
    pool_name: str
    description: str
    stocks: Dict[str, StockInfo] = field(default_factory=dict)
    filter_config: Optional[StockPoolFilter] = None
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)
    is_dynamic: bool = False  # 是否动态股票池（根据过滤条件自动更新）
    update_frequency: str = "daily"  # 更新频率：daily, weekly, monthly
    metadata: Dict[str, Any] = field(default_factory=dict)

    def add_stock(self, stock: StockInfo) -> bool:
        """添加股票"""
        if stock.stock_code in self.stocks:
            return False

        self.stocks[stock.stock_code] = stock
        self.updated_at = datetime.now()
        return True

    def remove_stock(self, stock_code: str) -> bool:
        """移除股票"""
        if stock_code not in self.stocks:
            return False

        del self.stocks[stock_code]
        self.updated_at = datetime.now()
        return True

    def update_stock(self, stock: StockInfo) -> bool:
        """更新股票信息"""
        if stock.stock_code not in self.stocks:
            return False

        self.stocks[stock.stock_code] = stock
        self.updated_at = datetime.now()
        return True

    def get_stock(self, stock_code: str) -> Optional[StockInfo]:
        """获取股票信息"""
        return self.stocks.get(stock_code)

    def get_stock_codes(self) -> List[str]:
        """获取所有股票代码"""
        return list(self.stocks.keys())

    def get_stocks_by_market(self, market_type: MarketType) -> List[StockInfo]:
        """根据市场类型获取股票"""
        return [stock for stock in self.stocks.values() if stock.market_type == market_type]

    def get_stocks_by_industry(self, industry_code: str) -> List[StockInfo]:
        """根据行业获取股票"""
        return [stock for stock in self.stocks.values() if stock.industry_code == industry_code]

    def get_stocks_by_status(self, status: StockStatus) -> List[StockInfo]:
        """根据状态获取股票"""
        return [stock for stock in self.stocks.values() if stock.status == status]

    def get_tradable_stocks(self) -> List[StockInfo]:
        """获取可交易股票"""
        return [stock for stock in self.stocks.values() if stock.is_tradable]

    def apply_filter(self, filter_config: StockPoolFilter) -> List[StockInfo]:
        """应用过滤条件，返回符合条件的股票"""
        return [stock for stock in self.stocks.values() if filter_config.apply(stock)]

    def update_with_filter(self, all_stocks: List[StockInfo]) -> int:
        """使用过滤条件更新股票池"""
        if not self.filter_config or not self.is_dynamic:
            return 0

        # 清空当前股票
        old_count = len(self.stocks)
        self.stocks.clear()

        # 应用过滤条件
        for stock in all_stocks:
            if self.filter_config.apply(stock):
                self.stocks[stock.stock_code] = stock

        self.updated_at = datetime.now()
        new_count = len(self.stocks)

        logger.info(f"Pool {self.pool_id} updated: {old_count} -> {new_count} stocks")
        return new_count

    def get_pool_statistics(self) -> Dict[str, Any]:
        """获取股票池统计信息"""
        if not self.stocks:
            return {"total_stocks": 0}

        # 市场分布
        market_distribution = {}
        for market in MarketType:
            market_distribution[market.value] = len(self.get_stocks_by_market(market))

        # 状态分布
        status_distribution = {}
        for status in StockStatus:
            status_distribution[status.value] = len(self.get_stocks_by_status(status))

        # 行业分布（Top 10）
        industry_counts = {}
        for stock in self.stocks.values():
            if stock.industry_name:
                industry_counts[stock.industry_name] = industry_counts.get(stock.industry_name, 0) + 1

        top_industries = sorted(industry_counts.items(), key=lambda x: x[1], reverse=True)[:10]

        # 市值统计
        market_caps = [stock.market_cap for stock in self.stocks.values() if stock.market_cap]
        market_cap_stats = {}
        if market_caps:
            market_cap_stats = {
                "min": min(market_caps),
                "max": max(market_caps),
                "avg": sum(market_caps) / len(market_caps),
                "median": sorted(market_caps)[len(market_caps) // 2]
            }

        return {
            "total_stocks": len(self.stocks),
            "tradable_stocks": len(self.get_tradable_stocks()),
            "market_distribution": market_distribution,
            "status_distribution": status_distribution,
            "top_industries": top_industries,
            "market_cap_statistics": market_cap_stats,
            "last_updated": self.updated_at.isoformat()
        }

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            "pool_id": self.pool_id,
            "pool_name": self.pool_name,
            "description": self.description,
            "stocks": {code: stock.to_dict() for code, stock in self.stocks.items()},
            "filter_config": self.filter_config.to_dict() if self.filter_config else None,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
            "is_dynamic": self.is_dynamic,
            "update_frequency": self.update_frequency,
            "metadata": self.metadata
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'StockPool':
        """从字典创建"""
        stocks = {}
        for code, stock_data in data.get("stocks", {}).items():
            stocks[code] = StockInfo.from_dict(stock_data)

        filter_config = None
        if data.get("filter_config"):
            filter_data = data["filter_config"].copy()
            # 处理市场类型
            if filter_data.get("allowed_markets"):
                filter_data["allowed_markets"] = [MarketType(m) for m in filter_data["allowed_markets"]]
            filter_config = StockPoolFilter(**filter_data)

        return cls(
            pool_id=data["pool_id"],
            pool_name=data["pool_name"],
            description=data["description"],
            stocks=stocks,
            filter_config=filter_config,
            created_at=datetime.fromisoformat(data.get("created_at", datetime.now().isoformat())),
            updated_at=datetime.fromisoformat(data.get("updated_at", datetime.now().isoformat())),
            is_dynamic=data.get("is_dynamic", False),
            update_frequency=data.get("update_frequency", "daily"),
            metadata=data.get("metadata", {})
        )

    def save(self, file_path: Union[str, Path]) -> None:
        """保存到文件"""
        path = Path(file_path)
        path.parent.mkdir(parents=True, exist_ok=True)

        with open(path, 'w', encoding='utf-8') as f:
            json.dump(self.to_dict(), f, indent=2, ensure_ascii=False)

    @classmethod
    def load(cls, file_path: Union[str, Path]) -> 'StockPool':
        """从文件加载"""
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return cls.from_dict(data)


class StockPoolManager:
    """股票池管理器"""

    def __init__(self):
        self.pools: Dict[str, StockPool] = {}
        self.hikyuu_stocks: Dict[str, StockInfo] = {}

    def initialize_hikyuu_integration(self) -> bool:
        """初始化Hikyuu集成"""
        try:
            # 这里应该集成Hikyuu的StockManager
            # 暂时使用模拟数据
            logger.info("Hikyuu integration initialized (mock)")
            return True
        except Exception as e:
            logger.error(f"Failed to initialize Hikyuu integration: {e}")
            return False

    def sync_from_hikyuu(self) -> int:
        """从Hikyuu同步股票数据"""
        try:
            # 这里应该从Hikyuu获取股票数据
            # 暂时生成一些模拟数据
            mock_stocks = self._generate_mock_stocks()
            self.hikyuu_stocks.update(mock_stocks)
            logger.info(f"Synced {len(mock_stocks)} stocks from Hikyuu")
            return len(mock_stocks)
        except Exception as e:
            logger.error(f"Failed to sync from Hikyuu: {e}")
            return 0

    def _generate_mock_stocks(self) -> Dict[str, StockInfo]:
        """生成模拟股票数据"""
        mock_stocks = {}

        # 生成一些上海A股
        for i in range(1, 101):
            stock_code = f"600{i:03d}.SH"
            stock = StockInfo(
                stock_code=stock_code,
                stock_name=f"测试股票{i:03d}",
                market_type=MarketType.SH_A,
                status=StockStatus.NORMAL,
                list_date=date(2010, 1, 1) + timedelta(days=i*10),
                industry_code=f"IND{(i-1)//10:02d}",
                industry_name=f"测试行业{(i-1)//10:02d}",
                industry_type=IndustryType.SW_L1,
                market_cap=1000000 + i * 10000,
                total_shares=100000 + i * 1000,
                float_shares=80000 + i * 800
            )
            mock_stocks[stock_code] = stock

        # 生成一些深圳A股
        for i in range(1, 51):
            stock_code = f"000{i:03d}.SZ"
            stock = StockInfo(
                stock_code=stock_code,
                stock_name=f"深圳股票{i:03d}",
                market_type=MarketType.SZ_A,
                status=StockStatus.NORMAL,
                list_date=date(2012, 1, 1) + timedelta(days=i*15),
                industry_code=f"IND{(i-1)//5:02d}",
                industry_name=f"深圳行业{(i-1)//5:02d}",
                industry_type=IndustryType.SW_L1,
                market_cap=500000 + i * 5000,
                total_shares=50000 + i * 500,
                float_shares=40000 + i * 400
            )
            mock_stocks[stock_code] = stock

        return mock_stocks

    def create_pool(self, pool_id: str, pool_name: str, description: str,
                    filter_config: Optional[StockPoolFilter] = None,
                    is_dynamic: bool = False) -> StockPool:
        """创建股票池"""
        pool = StockPool(
            pool_id=pool_id,
            pool_name=pool_name,
            description=description,
            filter_config=filter_config,
            is_dynamic=is_dynamic
        )

        self.pools[pool_id] = pool
        logger.info(f"Created stock pool: {pool_id}")
        return pool

    def get_pool(self, pool_id: str) -> Optional[StockPool]:
        """获取股票池"""
        return self.pools.get(pool_id)

    def list_pools(self) -> List[StockPool]:
        """列出所有股票池"""
        return list(self.pools.values())

    def update_dynamic_pools(self) -> Dict[str, int]:
        """更新所有动态股票池"""
        results = {}
        all_stocks = list(self.hikyuu_stocks.values())

        for pool in self.pools.values():
            if pool.is_dynamic:
                count = pool.update_with_filter(all_stocks)
                results[pool.pool_id] = count

        return results

    def create_preset_pools(self) -> Dict[str, StockPool]:
        """创建预设股票池"""
        pools = {}

        # 全市场股票池
        all_market_filter = StockPoolFilter(
            exclude_st=False,
            exclude_suspended=False,
            exclude_new_listed=False
        )
        pools["all_market"] = self.create_pool(
            "all_market",
            "全市场",
            "包含所有上市股票",
            all_market_filter,
            is_dynamic=True
        )

        # 主板股票池
        main_board_filter = StockPoolFilter(
            exclude_st=True,
            exclude_suspended=True,
            exclude_new_listed=False,
            allowed_markets=[MarketType.SH_A, MarketType.SZ_A],
            min_list_days=365
        )
        pools["main_board"] = self.create_pool(
            "main_board",
            "主板股票",
            "主板正常交易股票，排除ST和停牌",
            main_board_filter,
            is_dynamic=True
        )

        # 大盘股池
        large_cap_filter = StockPoolFilter(
            exclude_st=True,
            exclude_suspended=True,
            min_market_cap=1000000,  # 100亿市值
            allowed_markets=[MarketType.SH_A, MarketType.SZ_A]
        )
        pools["large_cap"] = self.create_pool(
            "large_cap",
            "大盘股",
            "市值超过100亿的大盘股",
            large_cap_filter,
            is_dynamic=True
        )

        # 更新动态股票池
        self.update_dynamic_pools()

        return pools

    def remove_pool(self, pool_id: str) -> bool:
        """删除股票池"""
        if pool_id in self.pools:
            del self.pools[pool_id]
            logger.info(f"Removed stock pool: {pool_id}")
            return True
        return False

    def export_pools(self, file_path: Union[str, Path]) -> None:
        """导出所有股票池"""
        data = {
            "pools": {pool_id: pool.to_dict() for pool_id, pool in self.pools.items()},
            "exported_at": datetime.now().isoformat()
        }

        path = Path(file_path)
        path.parent.mkdir(parents=True, exist_ok=True)

        with open(path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

    def import_pools(self, file_path: Union[str, Path]) -> int:
        """导入股票池"""
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        count = 0
        for pool_id, pool_data in data.get("pools", {}).items():
            pool = StockPool.from_dict(pool_data)
            self.pools[pool_id] = pool
            count += 1

        logger.info(f"Imported {count} stock pools")
        return count