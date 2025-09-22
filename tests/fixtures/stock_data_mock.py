"""
股票数据Mock生成器 - Stock Data Mock Generators

专为A股量化因子系统提供股票数据Mock生成，遵循Hikyuu数据结构
"""
import random
from datetime import datetime, timedelta, date
from typing import Dict, List, Optional, Any, Union, Tuple
import pandas as pd
import numpy as np


class MockStockDataGenerator:
    """
    股票数据Mock生成器

    生成符合Hikyuu框架格式的股票数据，用于测试
    """

    def __init__(self, seed: Optional[int] = None):
        """
        初始化股票数据生成器

        Args:
            seed: 随机种子，确保可重现性
        """
        self.seed = seed
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)

        # A股市场常用股票代码池
        self.stock_pool = [
            # 上证主板
            "sh600000", "sh600001", "sh600004", "sh600005", "sh600006",
            "sh600009", "sh600010", "sh600011", "sh600015", "sh600016",
            "sh600018", "sh600019", "sh600025", "sh600028", "sh600029",
            "sh600030", "sh600031", "sh600036", "sh600038", "sh600048",

            # 深证主板
            "sz000001", "sz000002", "sz000063", "sz000069", "sz000100",
            "sz000157", "sz000166", "sz000333", "sz000338", "sz000402",
            "sz000425", "sz000503", "sz000538", "sz000540", "sz000547",
            "sz000568", "sz000576", "sz000581", "sz000596", "sz000623",

            # 创业板
            "sz300001", "sz300002", "sz300003", "sz300009", "sz300012",
            "sz300014", "sz300015", "sz300017", "sz300024", "sz300027",
            "sz300033", "sz300037", "sz300045", "sz300049", "sz300059",
            "sz300070", "sz300072", "sz300104", "sz300122", "sz300124"
        ]

    def generate_stock_code(self, market: str = "random") -> str:
        """
        生成股票代码

        Args:
            market: 市场类型 ("sh", "sz", "random")

        Returns:
            股票代码
        """
        if market == "random":
            return random.choice(self.stock_pool)
        elif market == "sh":
            return random.choice([code for code in self.stock_pool if code.startswith("sh")])
        elif market == "sz":
            return random.choice([code for code in self.stock_pool if code.startswith("sz")])
        else:
            raise ValueError(f"Unknown market: {market}")

    def generate_stock_list(self, count: int = 10, market: str = "random") -> List[str]:
        """
        生成股票代码列表

        Args:
            count: 股票数量
            market: 市场类型

        Returns:
            股票代码列表
        """
        if market == "random":
            pool = self.stock_pool
        elif market == "sh":
            pool = [code for code in self.stock_pool if code.startswith("sh")]
        elif market == "sz":
            pool = [code for code in self.stock_pool if code.startswith("sz")]
        else:
            raise ValueError(f"Unknown market: {market}")

        return random.sample(pool, min(count, len(pool)))

    def generate_kdata(self,
                      stock_code: str,
                      start_date: Union[str, datetime] = None,
                      end_date: Union[str, datetime] = None,
                      days: int = 100,
                      initial_price: float = None,
                      volatility: float = 0.02,
                      trend: float = 0.0) -> pd.DataFrame:
        """
        生成K线数据

        Args:
            stock_code: 股票代码
            start_date: 开始日期
            end_date: 结束日期
            days: 天数（当start_date和end_date为None时使用）
            initial_price: 初始价格
            volatility: 波动率
            trend: 趋势

        Returns:
            K线数据DataFrame
        """
        # 处理日期
        if start_date is None and end_date is None:
            end_date = datetime.now().date()
            start_date = end_date - timedelta(days=days)
        elif isinstance(start_date, str):
            start_date = datetime.strptime(start_date, "%Y-%m-%d").date()
        elif isinstance(end_date, str):
            end_date = datetime.strptime(end_date, "%Y-%m-%d").date()

        # 生成交易日期（排除周末）
        dates = []
        current_date = start_date
        while current_date <= end_date:
            if current_date.weekday() < 5:  # 周一到周五
                dates.append(current_date)
            current_date += timedelta(days=1)

        if not dates:
            raise ValueError("No trading days found in date range")

        # 设置初始价格
        if initial_price is None:
            initial_price = random.uniform(5.0, 100.0)

        # 生成价格序列
        prices = [initial_price]
        current_price = initial_price

        for i in range(1, len(dates)):
            # 生成日收益率
            daily_return = np.random.normal(trend / 252, volatility)
            current_price *= (1 + daily_return)
            current_price = max(0.01, current_price)  # 避免负价格
            prices.append(current_price)

        # 生成OHLC数据
        kdata = []
        for i, (date, close) in enumerate(zip(dates, prices)):
            # 生成开盘价（基于前一日收盘价波动）
            if i == 0:
                open_price = close * random.uniform(0.98, 1.02)
            else:
                open_price = prices[i-1] * random.uniform(0.98, 1.02)

            # 生成最高价和最低价
            high = max(open_price, close) * random.uniform(1.0, 1.03)
            low = min(open_price, close) * random.uniform(0.97, 1.0)

            # 生成成交量
            volume = random.randint(100000, 10000000)

            # 计算成交额
            amount = volume * close

            kdata.append({
                "datetime": date,
                "stock_code": stock_code,
                "open": round(open_price, 2),
                "high": round(high, 2),
                "low": round(low, 2),
                "close": round(close, 2),
                "volume": volume,
                "amount": round(amount, 2)
            })

        return pd.DataFrame(kdata)

    def generate_minute_data(self,
                           stock_code: str,
                           date: Union[str, datetime],
                           start_time: str = "09:30",
                           end_time: str = "15:00",
                           base_price: float = None) -> pd.DataFrame:
        """
        生成分钟级数据

        Args:
            stock_code: 股票代码
            date: 交易日期
            start_time: 开始时间
            end_time: 结束时间
            base_price: 基准价格

        Returns:
            分钟级数据DataFrame
        """
        if isinstance(date, str):
            date = datetime.strptime(date, "%Y-%m-%d").date()

        if base_price is None:
            base_price = random.uniform(10.0, 50.0)

        # 生成交易时间点
        times = []
        current_time = datetime.strptime(start_time, "%H:%M").time()
        end_time_obj = datetime.strptime(end_time, "%H:%M").time()

        while current_time <= end_time_obj:
            # 跳过中午休市时间 11:30-13:00
            if not (datetime.strptime("11:30", "%H:%M").time() <= current_time <= datetime.strptime("13:00", "%H:%M").time()):
                times.append(current_time)

            # 加一分钟
            dt = datetime.combine(date, current_time) + timedelta(minutes=1)
            current_time = dt.time()

        # 生成价格序列
        prices = [base_price]
        current_price = base_price

        for _ in range(1, len(times)):
            # 分钟级波动较小
            minute_return = np.random.normal(0, 0.001)
            current_price *= (1 + minute_return)
            current_price = max(0.01, current_price)
            prices.append(current_price)

        # 生成分钟级OHLC数据
        minute_data = []
        for i, (time, close) in enumerate(zip(times, prices)):
            if i == 0:
                open_price = close
            else:
                open_price = prices[i-1]

            high = max(open_price, close) * random.uniform(1.0, 1.005)
            low = min(open_price, close) * random.uniform(0.995, 1.0)
            volume = random.randint(1000, 100000)
            amount = volume * close

            minute_data.append({
                "datetime": datetime.combine(date, time),
                "stock_code": stock_code,
                "open": round(open_price, 2),
                "high": round(high, 2),
                "low": round(low, 2),
                "close": round(close, 2),
                "volume": volume,
                "amount": round(amount, 2)
            })

        return pd.DataFrame(minute_data)

    def generate_stock_info(self, stock_code: str) -> Dict[str, Any]:
        """
        生成股票基本信息

        Args:
            stock_code: 股票代码

        Returns:
            股票信息字典
        """
        # 模拟股票名称
        name_suffixes = ["科技", "实业", "集团", "股份", "发展", "控股", "投资", "建设", "工业", "贸易"]
        name_prefixes = ["华", "中", "上", "大", "新", "金", "银", "宝", "德", "正"]

        stock_name = random.choice(name_prefixes) + random.choice(name_suffixes)

        # 行业分类
        industries = [
            "银行", "房地产", "保险", "证券", "医药生物", "食品饮料",
            "家用电器", "电子", "计算机", "通信", "汽车", "化工",
            "钢铁", "有色金属", "煤炭", "石油石化", "电力", "公用事业"
        ]

        return {
            "stock_code": stock_code,
            "stock_name": stock_name,
            "market": stock_code[:2],
            "industry": random.choice(industries),
            "listing_date": (datetime.now() - timedelta(days=random.randint(365, 3650))).date(),
            "total_shares": random.randint(100000000, 10000000000),  # 总股本
            "float_shares": random.randint(50000000, 5000000000),    # 流通股本
            "market_cap": random.uniform(1000000000, 100000000000)   # 市值
        }

    def generate_financial_data(self, stock_code: str, years: int = 3) -> pd.DataFrame:
        """
        生成财务数据

        Args:
            stock_code: 股票代码
            years: 年数

        Returns:
            财务数据DataFrame
        """
        financial_data = []

        for year in range(years):
            report_date = datetime.now().date() - timedelta(days=365 * year)

            # 基础财务指标
            revenue = random.uniform(1000000000, 50000000000)  # 营业收入
            profit = revenue * random.uniform(0.05, 0.25)     # 净利润
            assets = revenue * random.uniform(1.5, 5.0)       # 总资产
            equity = assets * random.uniform(0.3, 0.7)        # 净资产

            financial_data.append({
                "stock_code": stock_code,
                "report_date": report_date,
                "revenue": revenue,
                "net_profit": profit,
                "total_assets": assets,
                "total_equity": equity,
                "roe": profit / equity if equity > 0 else 0,  # ROE
                "roa": profit / assets if assets > 0 else 0,  # ROA
                "debt_ratio": (assets - equity) / assets if assets > 0 else 0,  # 资产负债率
                "eps": profit / random.uniform(100000000, 1000000000),  # 每股收益
                "bps": equity / random.uniform(100000000, 1000000000)   # 每股净资产
            })

        return pd.DataFrame(financial_data)


def create_sample_stock_pool(size: str = "small") -> List[str]:
    """
    创建样本股票池

    Args:
        size: 股票池大小 ("small", "medium", "large")

    Returns:
        股票代码列表
    """
    generator = MockStockDataGenerator(seed=42)

    size_mapping = {
        "small": 10,
        "medium": 50,
        "large": 200
    }

    count = size_mapping.get(size, 10)
    return generator.generate_stock_list(count)


def create_test_kdata_dataset(stock_codes: List[str],
                            start_date: str = "2024-01-01",
                            end_date: str = "2024-12-31") -> Dict[str, pd.DataFrame]:
    """
    创建测试用K线数据集

    Args:
        stock_codes: 股票代码列表
        start_date: 开始日期
        end_date: 结束日期

    Returns:
        股票代码到K线数据的映射
    """
    generator = MockStockDataGenerator(seed=42)
    dataset = {}

    for stock_code in stock_codes:
        dataset[stock_code] = generator.generate_kdata(
            stock_code=stock_code,
            start_date=start_date,
            end_date=end_date
        )

    return dataset


def create_market_scenario_data(scenario: str = "normal") -> Dict[str, Any]:
    """
    创建市场场景数据

    Args:
        scenario: 市场场景 ("normal", "bull", "bear", "volatile")

    Returns:
        市场场景数据
    """
    scenario_configs = {
        "normal": {"volatility": 0.015, "trend": 0.08, "correlation": 0.3},
        "bull": {"volatility": 0.012, "trend": 0.25, "correlation": 0.7},
        "bear": {"volatility": 0.025, "trend": -0.15, "correlation": 0.8},
        "volatile": {"volatility": 0.045, "trend": 0.0, "correlation": 0.2}
    }

    config = scenario_configs.get(scenario, scenario_configs["normal"])
    generator = MockStockDataGenerator(seed=42)

    # 生成股票池
    stock_codes = generator.generate_stock_list(20)

    # 生成K线数据
    kdata_dict = {}
    for stock_code in stock_codes:
        kdata_dict[stock_code] = generator.generate_kdata(
            stock_code=stock_code,
            volatility=config["volatility"],
            trend=config["trend"]
        )

    return {
        "scenario": scenario,
        "config": config,
        "stock_codes": stock_codes,
        "kdata": kdata_dict,
        "description": f"市场场景: {scenario}"
    }