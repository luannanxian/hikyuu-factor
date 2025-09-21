"""
外部API Mock - External API Mock

模拟外部API调用，避免测试依赖真实的外部服务
"""
import json
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Union
from unittest.mock import Mock, MagicMock
import random


class MockExternalAPI:
    """
    外部API模拟器

    模拟各种外部API调用，包括行情数据、新闻、基本面数据等
    """

    def __init__(self, seed: Optional[int] = None):
        """
        初始化外部API模拟器

        Args:
            seed: 随机种子，确保可重现的响应
        """
        self.seed = seed
        if seed is not None:
            random.seed(seed)

        self.call_count = {}
        self.response_delay = 0.1  # 模拟网络延迟

    def get_market_status(self) -> Dict[str, Any]:
        """
        获取市场状态

        Returns:
            市场状态信息
        """
        self._track_call("get_market_status")
        time.sleep(self.response_delay)

        return {
            "status": "success",
            "data": {
                "market_status": "open",
                "trading_session": "continuous",
                "last_update": datetime.now().isoformat(),
                "next_close": "15:00:00"
            },
            "timestamp": datetime.now().isoformat()
        }

    def get_real_time_quote(self, symbol: str) -> Dict[str, Any]:
        """
        获取实时报价

        Args:
            symbol: 股票代码

        Returns:
            实时报价数据
        """
        self._track_call("get_real_time_quote")
        time.sleep(self.response_delay)

        # 模拟报价数据
        base_price = 10.0 + hash(symbol) % 50
        change_pct = random.uniform(-0.05, 0.05)

        return {
            "status": "success",
            "data": {
                "symbol": symbol,
                "price": round(base_price * (1 + change_pct), 2),
                "change": round(base_price * change_pct, 2),
                "change_percent": round(change_pct * 100, 2),
                "volume": random.randint(1000000, 10000000),
                "timestamp": datetime.now().isoformat()
            }
        }

    def get_company_info(self, symbol: str) -> Dict[str, Any]:
        """
        获取公司基本信息

        Args:
            symbol: 股票代码

        Returns:
            公司信息
        """
        self._track_call("get_company_info")
        time.sleep(self.response_delay)

        return {
            "status": "success",
            "data": {
                "symbol": symbol,
                "name": f"公司{symbol[-4:]}",
                "industry": random.choice(["科技", "金融", "制造", "消费", "医药"]),
                "market_cap": random.uniform(50, 5000),
                "employees": random.randint(100, 50000),
                "founded": random.randint(1990, 2020)
            }
        }

    def _track_call(self, method_name: str):
        """跟踪API调用次数"""
        self.call_count[method_name] = self.call_count.get(method_name, 0) + 1


def mock_market_data_api(responses: Dict[str, Any] = None,
                        delay: float = 0.1,
                        failure_rate: float = 0.0) -> MockExternalAPI:
    """
    创建市场数据API mock

    Args:
        responses: 预定义的响应数据
        delay: 模拟的网络延迟
        failure_rate: 失败率 (0.0-1.0)

    Returns:
        Mock API实例
    """
    api = MockExternalAPI()
    api.response_delay = delay

    if responses:
        # 如果提供了预定义响应，覆盖默认行为
        for method_name, response_data in responses.items():
            setattr(api, method_name, lambda: response_data)

    # 模拟API失败
    if failure_rate > 0:
        original_methods = {}
        for method_name in ["get_market_status", "get_real_time_quote", "get_company_info"]:
            original_method = getattr(api, method_name)
            original_methods[method_name] = original_method

            def create_failing_method(orig_method, method_name):
                def failing_method(*args, **kwargs):
                    if random.random() < failure_rate:
                        raise ConnectionError(f"Mock API failure for {method_name}")
                    return orig_method(*args, **kwargs)
                return failing_method

            setattr(api, method_name, create_failing_method(original_method, method_name))

    return api


def mock_news_api(news_count: int = 10,
                 languages: List[str] = None,
                 sentiment_bias: float = 0.0) -> Mock:
    """
    创建新闻API mock

    Args:
        news_count: 返回的新闻数量
        languages: 支持的语言列表
        sentiment_bias: 情感倾向 (-1.0 到 1.0)

    Returns:
        Mock新闻API
    """
    if languages is None:
        languages = ["zh-CN", "en-US"]

    def get_news(symbol: str = None, category: str = None) -> Dict[str, Any]:
        """模拟获取新闻"""
        news_items = []

        for i in range(news_count):
            sentiment_score = random.normalvariate(sentiment_bias, 0.3)
            sentiment_score = max(-1.0, min(1.0, sentiment_score))

            news_items.append({
                "id": f"news_{i}_{hash(symbol or 'market')}",
                "title": f"市场新闻 {i+1}: {symbol or '全市场'}",
                "content": f"这是关于{symbol or '市场'}的新闻内容...",
                "sentiment_score": round(sentiment_score, 2),
                "publish_time": (datetime.now() - timedelta(hours=i)).isoformat(),
                "source": random.choice(["财经网", "证券时报", "新浪财经"]),
                "category": category or random.choice(["公司", "行业", "政策", "市场"])
            })

        return {
            "status": "success",
            "data": {
                "news": news_items,
                "total": news_count,
                "symbol": symbol
            }
        }

    news_api = Mock()
    news_api.get_news = get_news
    news_api.supported_languages = languages

    return news_api


def mock_fundamental_api(update_frequency: str = "quarterly") -> Mock:
    """
    创建基本面数据API mock

    Args:
        update_frequency: 更新频率 ("quarterly", "annual", "monthly")

    Returns:
        Mock基本面API
    """
    def get_financial_statements(symbol: str, statement_type: str = "income") -> Dict[str, Any]:
        """模拟获取财务报表"""
        # 生成模拟财务数据
        base_revenue = random.uniform(1000, 50000)  # 百万元

        if statement_type == "income":
            data = {
                "revenue": base_revenue,
                "gross_profit": base_revenue * random.uniform(0.2, 0.6),
                "operating_profit": base_revenue * random.uniform(0.1, 0.3),
                "net_profit": base_revenue * random.uniform(0.05, 0.2),
                "eps": random.uniform(0.1, 5.0)
            }
        elif statement_type == "balance":
            data = {
                "total_assets": base_revenue * random.uniform(2, 8),
                "total_liabilities": base_revenue * random.uniform(0.5, 3),
                "shareholders_equity": base_revenue * random.uniform(1, 4),
                "cash": base_revenue * random.uniform(0.1, 1),
                "debt": base_revenue * random.uniform(0.2, 2)
            }
        elif statement_type == "cash_flow":
            data = {
                "operating_cash_flow": base_revenue * random.uniform(0.1, 0.4),
                "investing_cash_flow": base_revenue * random.uniform(-0.3, 0.1),
                "financing_cash_flow": base_revenue * random.uniform(-0.2, 0.2),
                "free_cash_flow": base_revenue * random.uniform(0.05, 0.25)
            }
        else:
            data = {}

        return {
            "status": "success",
            "data": {
                "symbol": symbol,
                "statement_type": statement_type,
                "period": "2023Q4",
                "currency": "CNY",
                "unit": "million",
                **data
            }
        }

    def get_financial_ratios(symbol: str) -> Dict[str, Any]:
        """模拟获取财务比率"""
        return {
            "status": "success",
            "data": {
                "symbol": symbol,
                "pe_ratio": random.uniform(10, 50),
                "pb_ratio": random.uniform(1, 5),
                "roe": random.uniform(0.05, 0.25),
                "roa": random.uniform(0.02, 0.15),
                "debt_to_equity": random.uniform(0.2, 2.0),
                "current_ratio": random.uniform(1.0, 3.0),
                "quick_ratio": random.uniform(0.5, 2.0)
            }
        }

    fundamental_api = Mock()
    fundamental_api.get_financial_statements = get_financial_statements
    fundamental_api.get_financial_ratios = get_financial_ratios
    fundamental_api.update_frequency = update_frequency

    return fundamental_api


class APIErrorSimulator:
    """
    API错误模拟器

    模拟各种API错误情况，用于测试错误处理逻辑
    """

    @staticmethod
    def create_timeout_error():
        """创建超时错误"""
        import requests
        raise requests.exceptions.Timeout("Mock API timeout")

    @staticmethod
    def create_connection_error():
        """创建连接错误"""
        import requests
        raise requests.exceptions.ConnectionError("Mock connection failed")

    @staticmethod
    def create_http_error(status_code: int = 500):
        """创建HTTP错误"""
        import requests
        response = Mock()
        response.status_code = status_code
        response.text = f"Mock HTTP {status_code} error"
        raise requests.exceptions.HTTPError(response=response)

    @staticmethod
    def create_rate_limit_error():
        """创建限流错误"""
        return {
            "status": "error",
            "error": {
                "code": "RATE_LIMIT_EXCEEDED",
                "message": "API rate limit exceeded",
                "retry_after": 60
            }
        }

    @staticmethod
    def create_invalid_symbol_error(symbol: str):
        """创建无效股票代码错误"""
        return {
            "status": "error",
            "error": {
                "code": "INVALID_SYMBOL",
                "message": f"Invalid symbol: {symbol}",
                "symbol": symbol
            }
        }