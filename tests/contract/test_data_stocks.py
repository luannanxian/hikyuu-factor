"""
数据股票池API契约测试 - Data Stocks API Contract Tests

测试 GET /api/v1/data/stocks 端点
根据 data_manager_api.yaml 合约规范，验证股票池获取功能
"""
import pytest
import requests
from datetime import datetime, date
from tests.utils import assert_json_schema, assert_response_time


class TestDataStocksContract:
    """
    数据股票池API契约测试

    这些测试必须在实现前编写并失败，遵循TDD原则
    """

    def setup_method(self):
        """测试setup - 配置测试环境"""
        self.base_url = "http://localhost:8001"  # Data Manager Agent端口
        self.stocks_endpoint = f"{self.base_url}/api/v1/data/stocks"

    def test_stocks_endpoint_exists(self):
        """
        测试: GET /api/v1/data/stocks 端点存在

        期望: 端点应该存在且返回股票列表
        当前状态: 应该失败 (端点尚未实现)
        """
        response = requests.get(self.stocks_endpoint)

        # 这个测试现在应该失败，因为端点尚未实现
        assert response.status_code == 200, f"Stocks endpoint should exist, got {response.status_code}"

    def test_stocks_response_schema(self):
        """
        测试: 股票列表响应schema验证

        期望返回格式:
        {
          "status": "success",
          "data": {
            "stocks": [
              {
                "code": "sh000001",
                "name": "上证指数",
                "market": "sh",
                "industry": "指数",
                "listing_date": "1991-07-15",
                "total_shares": 1000000000,
                "float_shares": 800000000,
                "market_cap": 50000000000.0,
                "is_active": true,
                "last_updated": "2025-01-15T10:30:00Z"
              }
            ],
            "total_count": 5000,
            "filtered_count": 4500,
            "page": 1,
            "page_size": 100
          }
        }
        """
        response = requests.get(self.stocks_endpoint)
        assert response.status_code == 200

        data = response.json()

        # 验证顶级响应结构
        assert "status" in data
        assert "data" in data
        assert data["status"] == "success"

        stocks_data = data["data"]

        # 验证数据结构
        required_fields = ["stocks", "total_count", "filtered_count", "page", "page_size"]
        for field in required_fields:
            assert field in stocks_data, f"Missing required field: {field}"

        # 验证股票列表
        stocks = stocks_data["stocks"]
        assert isinstance(stocks, list)
        assert len(stocks) > 0, "Should return at least some stocks"

        # 验证单个股票对象结构
        if stocks:
            stock = stocks[0]
            stock_required_fields = [
                "code", "name", "market", "industry", "listing_date",
                "total_shares", "float_shares", "market_cap", "is_active", "last_updated"
            ]
            for field in stock_required_fields:
                assert field in stock, f"Missing stock field: {field}"

            # 验证字段类型
            assert isinstance(stock["code"], str)
            assert isinstance(stock["name"], str)
            assert stock["market"] in ["sh", "sz"]
            assert isinstance(stock["total_shares"], int)
            assert isinstance(stock["float_shares"], int)
            assert isinstance(stock["market_cap"], (int, float))
            assert isinstance(stock["is_active"], bool)

    def test_stocks_code_format_validation(self):
        """
        测试: 股票代码格式验证

        A股股票代码应该遵循sh/sz前缀格式
        """
        response = requests.get(self.stocks_endpoint)
        assert response.status_code == 200

        data = response.json()
        stocks = data["data"]["stocks"]

        for stock in stocks:
            code = stock["code"]

            # 验证代码格式
            assert len(code) == 8, f"Stock code should be 8 characters: {code}"
            assert code.startswith(("sh", "sz")), f"Stock code should start with sh/sz: {code}"

            # 验证数字部分
            number_part = code[2:]
            assert number_part.isdigit(), f"Stock code number part should be digits: {code}"

            # 验证市场一致性
            market = stock["market"]
            expected_prefix = market.lower()
            assert code.startswith(expected_prefix), \
                f"Stock code {code} should match market {market}"

    def test_st_stocks_exclusion(self):
        """
        测试: ST/*ST股票排除 (FR-001)

        默认情况下应该排除ST和*ST股票
        """
        response = requests.get(self.stocks_endpoint)
        assert response.status_code == 200

        data = response.json()
        stocks = data["data"]["stocks"]

        # 验证没有ST股票
        for stock in stocks:
            name = stock["name"]
            assert "ST" not in name, f"ST stock should be excluded by default: {name}"
            assert "*ST" not in name, f"*ST stock should be excluded by default: {name}"

    def test_stocks_listing_date_filter(self):
        """
        测试: 上市时间过滤

        应该能够根据上市时间过滤股票 (上市不足60日的排除)
        """
        # 测试日期过滤参数
        today = date.today()
        min_listing_date = today.replace(year=today.year - 1)  # 1年前

        params = {
            "min_listing_date": min_listing_date.isoformat()
        }

        response = requests.get(self.stocks_endpoint, params=params)
        assert response.status_code == 200

        data = response.json()
        stocks = data["data"]["stocks"]

        # 验证所有返回的股票都满足上市时间要求
        for stock in stocks:
            listing_date = datetime.fromisoformat(stock["listing_date"]).date()
            assert listing_date <= min_listing_date, \
                f"Stock {stock['code']} listing date {listing_date} should be before {min_listing_date}"

    def test_stocks_market_filter(self):
        """
        测试: 市场过滤参数

        应该能够按市场(sh/sz)过滤股票
        """
        # 测试上证市场过滤
        response = requests.get(self.stocks_endpoint, params={"market": "sh"})
        assert response.status_code == 200

        data = response.json()
        stocks = data["data"]["stocks"]

        # 验证只返回上证股票
        for stock in stocks:
            assert stock["market"] == "sh", f"Should only return SH stocks, got {stock['market']}"
            assert stock["code"].startswith("sh"), f"SH stock code should start with 'sh': {stock['code']}"

        # 测试深证市场过滤
        response = requests.get(self.stocks_endpoint, params={"market": "sz"})
        assert response.status_code == 200

        data = response.json()
        stocks = data["data"]["stocks"]

        # 验证只返回深证股票
        for stock in stocks:
            assert stock["market"] == "sz", f"Should only return SZ stocks, got {stock['market']}"
            assert stock["code"].startswith("sz"), f"SZ stock code should start with 'sz': {stock['code']}"

    def test_stocks_industry_filter(self):
        """
        测试: 行业过滤参数
        """
        # 测试行业过滤
        response = requests.get(self.stocks_endpoint, params={"industry": "银行"})
        assert response.status_code == 200

        data = response.json()
        stocks = data["data"]["stocks"]

        # 如果有结果，验证行业匹配
        if stocks:
            for stock in stocks:
                assert stock["industry"] == "银行", \
                    f"Industry filter failed: expected '银行', got '{stock['industry']}'"

    def test_stocks_pagination(self):
        """
        测试: 分页功能

        验证page和page_size参数
        """
        # 测试第一页
        response = requests.get(self.stocks_endpoint, params={"page": 1, "page_size": 50})
        assert response.status_code == 200

        data = response.json()
        stocks_data = data["data"]

        assert stocks_data["page"] == 1
        assert stocks_data["page_size"] == 50
        assert len(stocks_data["stocks"]) <= 50

        # 测试第二页
        response = requests.get(self.stocks_endpoint, params={"page": 2, "page_size": 50})
        assert response.status_code == 200

        data2 = response.json()
        stocks_data2 = data2["data"]

        assert stocks_data2["page"] == 2
        assert stocks_data2["page_size"] == 50

        # 验证不同页的股票不重复
        page1_codes = {stock["code"] for stock in stocks_data["stocks"]}
        page2_codes = {stock["code"] for stock in stocks_data2["stocks"]}

        overlap = page1_codes.intersection(page2_codes)
        assert len(overlap) == 0, f"Pages should not have overlapping stocks: {overlap}"

    def test_stocks_sorting(self):
        """
        测试: 排序功能

        支持按不同字段排序
        """
        # 测试按市值排序
        response = requests.get(self.stocks_endpoint, params={
            "sort_by": "market_cap",
            "sort_order": "desc",
            "page_size": 10
        })
        assert response.status_code == 200

        data = response.json()
        stocks = data["data"]["stocks"]

        # 验证按市值降序排列
        if len(stocks) > 1:
            for i in range(1, len(stocks)):
                assert stocks[i-1]["market_cap"] >= stocks[i]["market_cap"], \
                    "Stocks should be sorted by market_cap in descending order"

    def test_stocks_search_functionality(self):
        """
        测试: 搜索功能

        支持按股票代码或名称搜索
        """
        # 按代码搜索
        response = requests.get(self.stocks_endpoint, params={"search": "000001"})
        assert response.status_code == 200

        data = response.json()
        stocks = data["data"]["stocks"]

        # 验证搜索结果包含查询关键词
        if stocks:
            found_match = any("000001" in stock["code"] or "000001" in stock["name"]
                            for stock in stocks)
            assert found_match, "Search results should contain the search term"

    def test_stocks_active_status_filter(self):
        """
        测试: 活跃状态过滤

        默认只返回活跃股票，可选择包含非活跃股票
        """
        # 默认只返回活跃股票
        response = requests.get(self.stocks_endpoint)
        assert response.status_code == 200

        data = response.json()
        stocks = data["data"]["stocks"]

        # 验证所有股票都是活跃的
        for stock in stocks:
            assert stock["is_active"] is True, f"Should only return active stocks by default"

        # 测试包含非活跃股票
        response = requests.get(self.stocks_endpoint, params={"include_inactive": "true"})
        assert response.status_code == 200

        data = response.json()
        all_stocks = data["data"]["stocks"]

        # 现在应该包含非活跃股票
        active_count = sum(1 for stock in all_stocks if stock["is_active"])
        inactive_count = sum(1 for stock in all_stocks if not stock["is_active"])

        # 应该有一些非活跃股票 (除非数据库中没有)
        assert len(all_stocks) >= len(stocks), "Should include more stocks when including inactive"

    def test_invalid_parameters_handling(self):
        """
        测试: 无效参数处理

        验证各种无效参数的错误处理
        """
        invalid_params_tests = [
            # 无效页码
            ({"page": 0}, 400),
            ({"page": -1}, 400),
            # 无效页面大小
            ({"page_size": 0}, 400),
            ({"page_size": 1001}, 400),  # 假设最大1000
            # 无效市场
            ({"market": "invalid"}, 400),
            # 无效排序字段
            ({"sort_by": "invalid_field"}, 400),
            # 无效排序顺序
            ({"sort_order": "invalid"}, 400),
            # 无效日期格式
            ({"min_listing_date": "invalid-date"}, 400)
        ]

        for params, expected_status in invalid_params_tests:
            response = requests.get(self.stocks_endpoint, params=params)
            assert response.status_code == expected_status, \
                f"Invalid params {params} should return {expected_status}, got {response.status_code}"

    def test_response_time_performance(self):
        """
        测试: API响应时间应该 < 200ms (性能要求)
        """
        response = requests.get(self.stocks_endpoint, params={"page_size": 100})
        assert response.status_code == 200

        # 检查响应时间
        assert_response_time(response, max_time_ms=200)

    def test_data_consistency_validation(self):
        """
        测试: 数据一致性验证

        验证股票数据的业务逻辑一致性
        """
        response = requests.get(self.stocks_endpoint)
        assert response.status_code == 200

        data = response.json()
        stocks = data["data"]["stocks"]

        for stock in stocks:
            # 流通股本不应该超过总股本
            assert stock["float_shares"] <= stock["total_shares"], \
                f"Float shares should not exceed total shares for {stock['code']}"

            # 市值应该为正数
            assert stock["market_cap"] > 0, \
                f"Market cap should be positive for {stock['code']}"

            # 股本数量应该为正数
            assert stock["total_shares"] > 0, \
                f"Total shares should be positive for {stock['code']}"
            assert stock["float_shares"] >= 0, \
                f"Float shares should be non-negative for {stock['code']}"

    def test_content_type_and_headers(self):
        """
        测试: Content-Type和响应头
        """
        response = requests.get(self.stocks_endpoint)
        assert response.status_code == 200
        assert response.headers.get("Content-Type") == "application/json"


# 这个测试文件应该运行失败，因为端点尚未实现
# 这是TDD的正确流程：先写测试，测试失败，然后实现功能使测试通过