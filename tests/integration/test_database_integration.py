"""
数据库集成测试

测试与MySQL数据库的集成，包括：
1. 数据库连接和事务管理
2. 数据CRUD操作
3. 数据完整性和约束验证
4. 并发访问和锁机制
5. 数据迁移和备份恢复
6. 性能和优化
7. 连接池管理
"""

import pytest
import asyncio
import pandas as pd
import numpy as np
from datetime import datetime, timedelta, date
from typing import Dict, Any, List, Optional, Tuple
from sqlalchemy import create_engine, text, inspect
from sqlalchemy.orm import sessionmaker, Session
from sqlalchemy.exc import IntegrityError, OperationalError
import pymysql
import json
import time

# 导入数据库模型和服务
try:
    from src.models.database_models import (
        Base, Stock, MarketData, FinancialData,
        FactorDefinition, FactorValue, FactorCalculationTask
    )
    from src.lib.database import DatabaseManager
    from src.services.data_manager_service import DataManagerService
except ImportError as e:
    pytest.skip(f"Required database modules not available: {e}", allow_module_level=True)


@pytest.mark.integration
@pytest.mark.database
@pytest.mark.requires_mysql
class TestDatabaseIntegration:
    """数据库集成测试"""

    def setup_method(self):
        """设置测试环境"""
        self.db_config = {
            "host": "192.168.3.46",
            "port": 3306,
            "user": "remote",
            "password": "remote123456",
            "database": "hikyuu_factor_test",
            "charset": "utf8mb4",
            "pool_size": 10,
            "max_overflow": 20,
            "pool_pre_ping": True
        }

        # 创建数据库连接
        self.engine = None
        self.Session = None
        self.db_manager = None

        # 测试数据
        self.test_stocks = [
            {"stock_code": "sh600000", "stock_name": "浦发银行", "market": "SH"},
            {"stock_code": "sz000001", "stock_name": "平安银行", "market": "SZ"},
            {"stock_code": "sh000001", "stock_name": "上证指数", "market": "SH"}
        ]

    @pytest.fixture(autouse=True)
    async def setup_database(self):
        """设置数据库连接"""
        try:
            # 创建数据库引擎
            connection_url = f"mysql+pymysql://{self.db_config['user']}:{self.db_config['password']}@{self.db_config['host']}:{self.db_config['port']}/{self.db_config['database']}?charset={self.db_config['charset']}"

            self.engine = create_engine(
                connection_url,
                pool_size=self.db_config["pool_size"],
                max_overflow=self.db_config["max_overflow"],
                pool_pre_ping=self.db_config["pool_pre_ping"],
                echo=False  # 测试时不显示SQL
            )

            # 创建Session工厂
            self.Session = sessionmaker(bind=self.engine)

            # 初始化数据库管理器
            self.db_manager = DatabaseManager(self.db_config)
            await self.db_manager.initialize()

            # 确保表存在
            Base.metadata.create_all(self.engine)

            yield

        finally:
            # 清理
            if self.db_manager:
                await self.db_manager.close()
            if self.engine:
                self.engine.dispose()

    @pytest.mark.integration
    def test_database_connection_and_basic_operations(self):
        """测试数据库连接和基本操作"""
        # 测试连接
        with self.engine.connect() as conn:
            result = conn.execute(text("SELECT 1 as test"))
            assert result.fetchone()[0] == 1, "数据库连接测试失败"

        # 测试会话
        session = self.Session()
        try:
            # 测试表是否存在
            inspector = inspect(self.engine)
            tables = inspector.get_table_names()
            expected_tables = ['stocks', 'market_data', 'financial_data', 'factor_definitions', 'factor_values', 'factor_calculation_tasks']

            for table in expected_tables:
                assert table in tables, f"表 {table} 不存在"

        finally:
            session.close()

    @pytest.mark.integration
    def test_stock_data_crud_operations(self):
        """测试股票数据的CRUD操作"""
        session = self.Session()

        try:
            # === Create 操作 ===
            # 插入测试股票
            test_stocks = []
            for stock_data in self.test_stocks:
                stock = Stock(
                    stock_code=stock_data["stock_code"],
                    stock_name=stock_data["stock_name"],
                    market=stock_data["market"],
                    sector="测试板块",
                    list_date=date(2000, 1, 1),
                    status="ACTIVE"
                )
                test_stocks.append(stock)
                session.add(stock)

            session.commit()

            # === Read 操作 ===
            # 查询所有测试股票
            queried_stocks = session.query(Stock).filter(
                Stock.stock_code.in_([s["stock_code"] for s in self.test_stocks])
            ).all()

            assert len(queried_stocks) == len(self.test_stocks), "查询到的股票数量不匹配"

            # 按条件查询
            sh_stocks = session.query(Stock).filter(Stock.market == "SH").all()
            assert len(sh_stocks) >= 2, "上海股票数量不正确"

            # === Update 操作 ===
            # 更新股票信息
            stock_to_update = session.query(Stock).filter(Stock.stock_code == "sh600000").first()
            assert stock_to_update is not None, "未找到要更新的股票"

            original_sector = stock_to_update.sector
            stock_to_update.sector = "更新后的板块"
            session.commit()

            # 验证更新
            updated_stock = session.query(Stock).filter(Stock.stock_code == "sh600000").first()
            assert updated_stock.sector == "更新后的板块", "股票信息更新失败"

            # === Delete 操作 ===
            # 删除测试数据（清理）
            session.query(Stock).filter(
                Stock.stock_code.in_([s["stock_code"] for s in self.test_stocks])
            ).delete(synchronize_session=False)
            session.commit()

            # 验证删除
            remaining_stocks = session.query(Stock).filter(
                Stock.stock_code.in_([s["stock_code"] for s in self.test_stocks])
            ).all()
            assert len(remaining_stocks) == 0, "股票数据删除失败"

        finally:
            session.rollback()
            session.close()

    @pytest.mark.integration
    def test_market_data_operations(self):
        """测试市场数据操作"""
        session = self.Session()

        try:
            # 首先插入股票数据
            test_stock = Stock(
                stock_code="test001",
                stock_name="测试股票",
                market="SH",
                status="ACTIVE"
            )
            session.add(test_stock)
            session.commit()

            # 生成测试市场数据
            market_data_list = []
            base_price = 10.0

            for i in range(10):  # 10天数据
                trade_date = date(2024, 1, 1) + timedelta(days=i)
                price_change = np.random.randn() * 0.02

                market_data = MarketData(
                    stock_code="test001",
                    trade_date=trade_date,
                    open_price=base_price * (1 + price_change),
                    high_price=base_price * (1 + abs(price_change) + 0.01),
                    low_price=base_price * (1 - abs(price_change) - 0.01),
                    close_price=base_price * (1 + price_change * 0.5),
                    volume=1000000 + int(np.random.randn() * 100000),
                    amount=0  # 将计算得出
                )
                market_data.amount = market_data.close_price * market_data.volume
                market_data_list.append(market_data)

                base_price = market_data.close_price

            # 批量插入
            session.add_all(market_data_list)
            session.commit()

            # 查询验证
            inserted_data = session.query(MarketData).filter(
                MarketData.stock_code == "test001"
            ).order_by(MarketData.trade_date).all()

            assert len(inserted_data) == 10, "市场数据插入数量不正确"

            # 验证数据完整性
            for data in inserted_data:
                assert data.high_price >= data.low_price, "最高价应该大于等于最低价"
                assert data.high_price >= data.open_price, "最高价应该大于等于开盘价"
                assert data.high_price >= data.close_price, "最高价应该大于等于收盘价"
                assert data.low_price <= data.open_price, "最低价应该小于等于开盘价"
                assert data.low_price <= data.close_price, "最低价应该小于等于收盘价"
                assert data.volume > 0, "成交量应该大于0"
                assert data.amount > 0, "成交额应该大于0"

            # 测试时间范围查询
            date_range_data = session.query(MarketData).filter(
                MarketData.stock_code == "test001",
                MarketData.trade_date >= date(2024, 1, 3),
                MarketData.trade_date <= date(2024, 1, 7)
            ).all()

            assert len(date_range_data) == 5, "日期范围查询结果不正确"

        finally:
            # 清理测试数据
            session.query(MarketData).filter(MarketData.stock_code == "test001").delete()
            session.query(Stock).filter(Stock.stock_code == "test001").delete()
            session.commit()
            session.close()

    @pytest.mark.integration
    def test_factor_data_operations(self):
        """测试因子数据操作"""
        session = self.Session()

        try:
            # 创建测试股票和因子定义
            test_stock = Stock(stock_code="test_factor", stock_name="因子测试股票", market="SH", status="ACTIVE")
            session.add(test_stock)

            test_factor = FactorDefinition(
                factor_id="test_momentum",
                factor_name="测试动量因子",
                category="MOMENTUM",
                formula="price_change_20d / volatility_20d",
                description="测试用动量因子",
                status="ACTIVE"
            )
            session.add(test_factor)
            session.commit()

            # 创建因子值数据
            factor_values = []
            for i in range(20):  # 20天因子值
                trade_date = date(2024, 1, 1) + timedelta(days=i)

                factor_value = FactorValue(
                    factor_id="test_momentum",
                    stock_code="test_factor",
                    trade_date=trade_date,
                    factor_value=np.random.randn() * 0.1,  # 随机因子值
                    factor_score=np.random.uniform(0, 1),   # 0-1分数
                    percentile_rank=np.random.uniform(0, 1)  # 百分位排名
                )
                factor_values.append(factor_value)

            session.add_all(factor_values)
            session.commit()

            # 验证因子值插入
            inserted_values = session.query(FactorValue).filter(
                FactorValue.factor_id == "test_momentum",
                FactorValue.stock_code == "test_factor"
            ).count()

            assert inserted_values == 20, "因子值插入数量不正确"

            # 测试因子值查询和统计
            avg_score = session.query(
                session.query(FactorValue.factor_score).filter(
                    FactorValue.factor_id == "test_momentum"
                ).subquery().c.factor_score
            ).scalar()

            # 测试因子定义查询
            factor_def = session.query(FactorDefinition).filter(
                FactorDefinition.factor_id == "test_momentum"
            ).first()

            assert factor_def is not None, "因子定义查询失败"
            assert factor_def.category == "MOMENTUM", "因子类别不正确"

        finally:
            # 清理测试数据
            session.query(FactorValue).filter(FactorValue.factor_id == "test_momentum").delete()
            session.query(FactorDefinition).filter(FactorDefinition.factor_id == "test_momentum").delete()
            session.query(Stock).filter(Stock.stock_code == "test_factor").delete()
            session.commit()
            session.close()

    @pytest.mark.integration
    def test_data_integrity_constraints(self):
        """测试数据完整性约束"""
        session = self.Session()

        try:
            # 测试主键约束
            stock1 = Stock(stock_code="duplicate_test", stock_name="重复测试1", market="SH")
            stock2 = Stock(stock_code="duplicate_test", stock_name="重复测试2", market="SZ")

            session.add(stock1)
            session.commit()

            # 尝试插入重复主键
            session.add(stock2)
            with pytest.raises(IntegrityError):
                session.commit()
            session.rollback()

            # 测试外键约束
            # 尝试插入不存在股票的市场数据
            invalid_market_data = MarketData(
                stock_code="nonexistent_stock",
                trade_date=date(2024, 1, 1),
                open_price=10.0,
                high_price=10.5,
                low_price=9.5,
                close_price=10.2,
                volume=1000000,
                amount=10200000
            )

            session.add(invalid_market_data)
            with pytest.raises(IntegrityError):
                session.commit()
            session.rollback()

            # 测试唯一约束
            # 尝试插入重复的股票-日期组合
            test_stock = Stock(stock_code="unique_test", stock_name="唯一性测试", market="SH")
            session.add(test_stock)
            session.commit()

            market_data1 = MarketData(
                stock_code="unique_test",
                trade_date=date(2024, 1, 1),
                open_price=10.0,
                high_price=10.5,
                low_price=9.5,
                close_price=10.2,
                volume=1000000,
                amount=10200000
            )

            market_data2 = MarketData(
                stock_code="unique_test",
                trade_date=date(2024, 1, 1),  # 相同日期
                open_price=11.0,
                high_price=11.5,
                low_price=10.5,
                close_price=11.2,
                volume=1100000,
                amount=12320000
            )

            session.add(market_data1)
            session.commit()

            session.add(market_data2)
            with pytest.raises(IntegrityError):
                session.commit()

        finally:
            session.rollback()
            # 清理可能成功插入的数据
            session.query(MarketData).filter(MarketData.stock_code.in_(["duplicate_test", "unique_test"])).delete(synchronize_session=False)
            session.query(Stock).filter(Stock.stock_code.in_(["duplicate_test", "unique_test"])).delete(synchronize_session=False)
            session.commit()
            session.close()

    @pytest.mark.integration
    def test_transaction_management(self):
        """测试事务管理"""
        session = self.Session()

        try:
            # 测试事务回滚
            test_stock = Stock(stock_code="transaction_test", stock_name="事务测试", market="SH")
            session.add(test_stock)

            # 制造一个会失败的操作
            invalid_data = MarketData(
                stock_code="transaction_test",
                trade_date=date(2024, 1, 1),
                open_price=None,  # 违反NOT NULL约束
                high_price=10.5,
                low_price=9.5,
                close_price=10.2,
                volume=1000000,
                amount=10200000
            )
            session.add(invalid_data)

            # 这应该导致整个事务回滚
            with pytest.raises(Exception):
                session.commit()

            session.rollback()

            # 验证回滚成功 - 股票也不应该被插入
            remaining_stock = session.query(Stock).filter(Stock.stock_code == "transaction_test").first()
            assert remaining_stock is None, "事务回滚失败，股票仍然存在"

            # 测试正确的事务提交
            valid_stock = Stock(stock_code="transaction_test", stock_name="事务测试", market="SH")
            valid_data = MarketData(
                stock_code="transaction_test",
                trade_date=date(2024, 1, 1),
                open_price=10.0,
                high_price=10.5,
                low_price=9.5,
                close_price=10.2,
                volume=1000000,
                amount=10200000
            )

            session.add(valid_stock)
            session.add(valid_data)
            session.commit()

            # 验证提交成功
            committed_stock = session.query(Stock).filter(Stock.stock_code == "transaction_test").first()
            committed_data = session.query(MarketData).filter(MarketData.stock_code == "transaction_test").first()

            assert committed_stock is not None, "正确的股票事务提交失败"
            assert committed_data is not None, "正确的市场数据事务提交失败"

        finally:
            # 清理
            session.query(MarketData).filter(MarketData.stock_code == "transaction_test").delete()
            session.query(Stock).filter(Stock.stock_code == "transaction_test").delete()
            session.commit()
            session.close()

    @pytest.mark.integration
    def test_concurrent_database_access(self):
        """测试并发数据库访问"""
        import threading
        import queue

        # 结果队列
        results = queue.Queue()
        errors = queue.Queue()

        def worker_function(worker_id: int):
            """工作线程函数"""
            session = self.Session()
            try:
                # 每个工作线程插入不同的股票
                test_stock = Stock(
                    stock_code=f"concurrent_{worker_id:03d}",
                    stock_name=f"并发测试股票{worker_id}",
                    market="SH" if worker_id % 2 == 0 else "SZ"
                )
                session.add(test_stock)
                session.commit()

                # 查询验证
                queried_stock = session.query(Stock).filter(
                    Stock.stock_code == f"concurrent_{worker_id:03d}"
                ).first()

                if queried_stock:
                    results.put(f"worker_{worker_id}_success")
                else:
                    errors.put(f"worker_{worker_id}_query_failed")

            except Exception as e:
                errors.put(f"worker_{worker_id}_error: {str(e)}")
            finally:
                session.close()

        # 启动多个并发线程
        threads = []
        num_workers = 10

        for i in range(num_workers):
            thread = threading.Thread(target=worker_function, args=(i,))
            threads.append(thread)
            thread.start()

        # 等待所有线程完成
        for thread in threads:
            thread.join()

        # 检查结果
        success_count = results.qsize()
        error_count = errors.qsize()

        print(f"并发测试结果: {success_count} 成功, {error_count} 失败")

        # 打印错误信息
        while not errors.empty():
            print(f"错误: {errors.get()}")

        # 至少90%的操作应该成功
        success_rate = success_count / num_workers
        assert success_rate >= 0.9, f"并发成功率过低: {success_rate:.2%}"

        # 清理并发测试数据
        cleanup_session = self.Session()
        try:
            cleanup_session.query(Stock).filter(
                Stock.stock_code.like("concurrent_%")
            ).delete(synchronize_session=False)
            cleanup_session.commit()
        finally:
            cleanup_session.close()

    @pytest.mark.integration
    def test_connection_pool_management(self):
        """测试连接池管理"""
        # 测试连接池大小限制
        sessions = []

        try:
            # 创建多个会话（超过池大小）
            for i in range(15):  # 超过配置的池大小10
                session = self.Session()
                sessions.append(session)

                # 执行简单查询确保连接被使用
                result = session.execute(text("SELECT 1"))
                assert result.fetchone()[0] == 1

            # 应该能够成功创建所有会话（通过overflow）
            assert len(sessions) == 15, "连接池应该支持overflow"

        finally:
            # 关闭所有会话
            for session in sessions:
                session.close()

        # 测试连接回收
        # 创建并关闭会话，然后重新创建
        for i in range(5):
            session = self.Session()
            session.execute(text("SELECT 1"))
            session.close()

        # 应该能够成功重用连接
        final_session = self.Session()
        try:
            result = final_session.execute(text("SELECT 1"))
            assert result.fetchone()[0] == 1, "连接回收测试失败"
        finally:
            final_session.close()

    @pytest.mark.integration
    @pytest.mark.slow
    def test_database_performance(self):
        """测试数据库性能"""
        session = self.Session()

        try:
            # 性能测试：批量插入
            test_stock = Stock(stock_code="perf_test", stock_name="性能测试", market="SH")
            session.add(test_stock)
            session.commit()

            # 生成大量测试数据
            num_records = 1000
            market_data_list = []

            start_time = time.time()

            for i in range(num_records):
                trade_date = date(2024, 1, 1) + timedelta(days=i % 365)
                market_data = MarketData(
                    stock_code="perf_test",
                    trade_date=trade_date,
                    open_price=10.0 + np.random.randn() * 0.5,
                    high_price=10.5 + np.random.randn() * 0.3,
                    low_price=9.5 + np.random.randn() * 0.3,
                    close_price=10.0 + np.random.randn() * 0.5,
                    volume=1000000 + int(np.random.randn() * 100000),
                    amount=(10.0 + np.random.randn() * 0.5) * (1000000 + int(np.random.randn() * 100000))
                )
                market_data_list.append(market_data)

            # 批量插入
            session.add_all(market_data_list)
            session.commit()

            insert_time = time.time() - start_time
            print(f"批量插入 {num_records} 条记录耗时: {insert_time:.3f} 秒")

            # 性能要求：插入时间应该在合理范围内
            assert insert_time < 10.0, f"批量插入性能过低: {insert_time:.3f} 秒"

            # 性能测试：查询
            start_time = time.time()

            queried_data = session.query(MarketData).filter(
                MarketData.stock_code == "perf_test"
            ).limit(100).all()

            query_time = time.time() - start_time
            print(f"查询 100 条记录耗时: {query_time:.3f} 秒")

            assert len(queried_data) == 100, "查询结果数量不正确"
            assert query_time < 1.0, f"查询性能过低: {query_time:.3f} 秒"

            # 性能测试：聚合查询
            start_time = time.time()

            avg_price = session.query(
                session.query(MarketData.close_price).filter(
                    MarketData.stock_code == "perf_test"
                ).subquery().c.close_price
            ).scalar()

            aggregation_time = time.time() - start_time
            print(f"聚合查询耗时: {aggregation_time:.3f} 秒")

            assert aggregation_time < 2.0, f"聚合查询性能过低: {aggregation_time:.3f} 秒"

        finally:
            # 清理性能测试数据
            session.query(MarketData).filter(MarketData.stock_code == "perf_test").delete()
            session.query(Stock).filter(Stock.stock_code == "perf_test").delete()
            session.commit()
            session.close()

    def teardown_method(self):
        """清理测试环境"""
        if self.Session:
            # 清理可能残留的测试数据
            session = self.Session()
            try:
                # 清理所有测试相关的数据
                test_patterns = ["test%", "concurrent_%", "duplicate_%", "unique_%", "transaction_%", "perf_%"]

                for pattern in test_patterns:
                    session.query(MarketData).filter(MarketData.stock_code.like(pattern)).delete(synchronize_session=False)
                    session.query(FactorValue).filter(FactorValue.stock_code.like(pattern)).delete(synchronize_session=False)
                    session.query(Stock).filter(Stock.stock_code.like(pattern)).delete(synchronize_session=False)
                    session.query(FactorDefinition).filter(FactorDefinition.factor_id.like(pattern)).delete(synchronize_session=False)

                session.commit()
            except Exception as e:
                print(f"清理测试数据时出错: {e}")
                session.rollback()
            finally:
                session.close()