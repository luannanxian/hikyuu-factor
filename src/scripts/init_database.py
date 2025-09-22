#!/usr/bin/env python3
"""
数据库初始化脚本
创建hikyuu_factor_test数据库中的所有表结构
"""

import sys
import os
from pathlib import Path

# 添加src目录到Python路径
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root / "src"))

from sqlalchemy import create_engine, text
from sqlalchemy.exc import SQLAlchemyError
from models.database_models import Base, Stock, MarketData, FinancialData, FactorDefinition, FactorValue, FactorCalculationTask
from datetime import date, datetime
import logging

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def create_database_connection():
    """创建数据库连接"""
    # 数据库连接配置
    DB_CONFIG = {
        'host': '192.168.3.46',
        'port': 3306,
        'user': 'remote',
        'password': 'remote123456',
        'database': 'hikyuu_factor_test',
        'charset': 'utf8mb4'
    }

    # 创建连接URL
    connection_url = f"mysql+pymysql://{DB_CONFIG['user']}:{DB_CONFIG['password']}@{DB_CONFIG['host']}:{DB_CONFIG['port']}/{DB_CONFIG['database']}?charset={DB_CONFIG['charset']}"

    try:
        engine = create_engine(
            connection_url,
            echo=True,  # 显示SQL语句
            pool_pre_ping=True,  # 连接池预检查
            pool_recycle=3600    # 连接回收时间
        )
        return engine
    except Exception as e:
        logger.error(f"创建数据库连接失败: {e}")
        raise

def create_tables(engine):
    """创建所有表"""
    try:
        logger.info("开始创建数据库表...")

        # 使用Base.metadata.create_all()创建所有表
        Base.metadata.create_all(engine)

        logger.info("数据库表创建成功！")
        return True

    except SQLAlchemyError as e:
        logger.error(f"创建数据库表失败: {e}")
        return False

def verify_tables(engine):
    """验证表是否创建成功"""
    try:
        with engine.connect() as conn:
            # 查询所有表
            result = conn.execute(text("SHOW TABLES"))
            tables = [row[0] for row in result]

            expected_tables = [
                'stocks', 'market_data', 'financial_data',
                'factor_definitions', 'factor_values', 'factor_calculation_tasks'
            ]

            logger.info("数据库中的表:")
            for table in tables:
                logger.info(f"  - {table}")

            # 检查是否所有期望的表都存在
            missing_tables = set(expected_tables) - set(tables)
            if missing_tables:
                logger.warning(f"缺少的表: {missing_tables}")
                return False
            else:
                logger.info("✅ 所有期望的表都已创建")
                return True

    except SQLAlchemyError as e:
        logger.error(f"验证表失败: {e}")
        return False

def insert_sample_data(engine):
    """插入示例数据"""
    try:
        from sqlalchemy.orm import sessionmaker
        Session = sessionmaker(bind=engine)
        session = Session()

        logger.info("插入示例数据...")

        # 插入股票数据
        sample_stocks = [
            Stock(
                stock_code="sh600000",
                stock_name="浦发银行",
                market="SH",
                sector="金融业",
                industry="银行",
                list_date=date(1999, 11, 10),
                is_active=True
            ),
            Stock(
                stock_code="sz000001",
                stock_name="平安银行",
                market="SZ",
                sector="金融业",
                industry="银行",
                list_date=date(1991, 4, 3),
                is_active=True
            ),
            Stock(
                stock_code="sh000001",
                stock_name="上证指数",
                market="SH",
                sector="指数",
                industry="指数",
                list_date=date(1990, 12, 19),
                is_active=True
            )
        ]

        for stock in sample_stocks:
            session.merge(stock)  # 使用merge避免重复插入

        # 插入因子定义
        sample_factors = [
            FactorDefinition(
                factor_name="momentum_20d",
                factor_type="MOMENTUM",
                description="20日动量因子",
                calculation_method="price_change_20d / volatility_20d",
                parameters={"period": 20, "method": "simple"},
                is_active=True,
                created_by="system"
            ),
            FactorDefinition(
                factor_name="pe_ratio",
                factor_type="VALUATION",
                description="市盈率因子",
                calculation_method="market_cap / net_profit",
                parameters={"ttm": True},
                is_active=True,
                created_by="system"
            )
        ]

        for factor in sample_factors:
            session.merge(factor)

        session.commit()
        session.close()

        logger.info("✅ 示例数据插入成功")
        return True

    except Exception as e:
        logger.error(f"插入示例数据失败: {e}")
        if 'session' in locals():
            session.rollback()
            session.close()
        return False

def main():
    """主函数"""
    try:
        logger.info("=== Hikyuu Factor 数据库初始化开始 ===")

        # 创建数据库连接
        engine = create_database_connection()
        logger.info("✅ 数据库连接创建成功")

        # 创建表
        if create_tables(engine):
            logger.info("✅ 数据库表创建成功")
        else:
            logger.error("❌ 数据库表创建失败")
            return False

        # 验证表
        if verify_tables(engine):
            logger.info("✅ 表验证成功")
        else:
            logger.error("❌ 表验证失败")
            return False

        # 插入示例数据
        if insert_sample_data(engine):
            logger.info("✅ 示例数据插入成功")
        else:
            logger.warning("⚠️ 示例数据插入失败，但表结构已创建")

        logger.info("=== 数据库初始化完成 ===")
        return True

    except Exception as e:
        logger.error(f"数据库初始化失败: {e}")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)