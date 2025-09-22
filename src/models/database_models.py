"""
Database Models using SQLAlchemy
数据库模型定义
"""

from datetime import datetime, date
from typing import Optional, Dict, Any
from decimal import Decimal
from enum import Enum

from sqlalchemy import (
    Column, String, DateTime, Date, BigInteger, Integer,
    Numeric, Text, JSON, ForeignKey,
    UniqueConstraint, Index, Enum as SQLEnum
)
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func

Base = declarative_base()


class StockStatus(Enum):
    """股票状态枚举"""
    ACTIVE = "active"
    SUSPENDED = "suspended"
    DELISTED = "delisted"


class FactorCategory(Enum):
    """因子类别枚举"""
    MOMENTUM = "momentum"
    VALUE = "value"
    QUALITY = "quality"
    VOLATILITY = "volatility"
    GROWTH = "growth"
    TECHNICAL = "technical"


class FactorStatus(Enum):
    """因子状态枚举"""
    ACTIVE = "active"
    INACTIVE = "inactive"
    TESTING = "testing"


class TaskStatus(Enum):
    """任务状态枚举"""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class Stock(Base):
    """股票基础信息表"""
    __tablename__ = 'stocks'

    stock_code = Column(String(20), primary_key=True, comment='股票代码 (如: sh000001)')
    stock_name = Column(String(100), nullable=False, comment='股票名称')
    market = Column(String(10), nullable=False, comment='市场代码 (sh/sz/bj)')
    sector = Column(String(50), comment='行业板块')
    list_date = Column(Date, comment='上市日期')
    delist_date = Column(Date, comment='退市日期')
    status = Column(SQLEnum(StockStatus), default=StockStatus.ACTIVE, comment='状态')

    created_at = Column(DateTime, default=func.current_timestamp())
    updated_at = Column(DateTime, default=func.current_timestamp(), onupdate=func.current_timestamp())

    # 关系
    market_data = relationship("MarketData", back_populates="stock", cascade="all, delete-orphan")
    financial_data = relationship("FinancialData", back_populates="stock", cascade="all, delete-orphan")
    factor_values = relationship("FactorValue", back_populates="stock", cascade="all, delete-orphan")

    # 索引
    __table_args__ = (
        Index('idx_market', 'market'),
        Index('idx_sector', 'sector'),
        Index('idx_status', 'status'),
        Index('idx_list_date', 'list_date'),
        {'comment': '股票基础信息表'}
    )

    def __repr__(self):
        return f"<Stock(code={self.stock_code}, name={self.stock_name})>"


class MarketData(Base):
    """市场数据表 (日K线数据)"""
    __tablename__ = 'market_data'

    id = Column(BigInteger, primary_key=True, autoincrement=True)
    stock_code = Column(String(20), ForeignKey('stocks.stock_code', ondelete='CASCADE'),
                       nullable=False, comment='股票代码')
    trade_date = Column(Date, nullable=False, comment='交易日期')

    open_price = Column(Numeric(10, 3), nullable=False, comment='开盘价')
    high_price = Column(Numeric(10, 3), nullable=False, comment='最高价')
    low_price = Column(Numeric(10, 3), nullable=False, comment='最低价')
    close_price = Column(Numeric(10, 3), nullable=False, comment='收盘价')
    volume = Column(BigInteger, nullable=False, comment='成交量')
    amount = Column(Numeric(18, 2), nullable=False, comment='成交金额')

    adj_factor = Column(Numeric(10, 6), default=1.0, comment='复权因子')
    turnover_rate = Column(Numeric(8, 4), comment='换手率')

    created_at = Column(DateTime, default=func.current_timestamp())
    updated_at = Column(DateTime, default=func.current_timestamp(), onupdate=func.current_timestamp())

    # 关系
    stock = relationship("Stock", back_populates="market_data")

    # 约束和索引
    __table_args__ = (
        UniqueConstraint('stock_code', 'trade_date', name='uk_stock_date'),
        Index('idx_trade_date', 'trade_date'),
        Index('idx_stock_code', 'stock_code'),
        {'comment': '市场数据表'}
    )

    def __repr__(self):
        return f"<MarketData(stock={self.stock_code}, date={self.trade_date}, close={self.close_price})>"


class FinancialData(Base):
    """财务数据表"""
    __tablename__ = 'financial_data'

    id = Column(BigInteger, primary_key=True, autoincrement=True)
    stock_code = Column(String(20), ForeignKey('stocks.stock_code', ondelete='CASCADE'),
                       nullable=False, comment='股票代码')
    report_date = Column(Date, nullable=False, comment='报告期')

    # 基本财务指标
    pe_ratio = Column(Numeric(10, 3), comment='市盈率')
    pb_ratio = Column(Numeric(10, 3), comment='市净率')
    roe = Column(Numeric(8, 4), comment='净资产收益率')
    roa = Column(Numeric(8, 4), comment='总资产收益率')

    # 财务数据
    total_revenue = Column(Numeric(18, 2), comment='营业收入')
    net_profit = Column(Numeric(18, 2), comment='净利润')
    total_assets = Column(Numeric(18, 2), comment='总资产')
    total_equity = Column(Numeric(18, 2), comment='股东权益')

    created_at = Column(DateTime, default=func.current_timestamp())
    updated_at = Column(DateTime, default=func.current_timestamp(), onupdate=func.current_timestamp())

    # 关系
    stock = relationship("Stock", back_populates="financial_data")

    # 约束和索引
    __table_args__ = (
        UniqueConstraint('stock_code', 'report_date', name='uk_stock_report'),
        Index('idx_report_date', 'report_date'),
        {'comment': '财务数据表'}
    )

    def __repr__(self):
        return f"<FinancialData(stock={self.stock_code}, date={self.report_date}, pe={self.pe_ratio})>"


class FactorDefinition(Base):
    """因子定义表"""
    __tablename__ = 'factor_definitions'

    factor_id = Column(String(50), primary_key=True, comment='因子ID')
    factor_name = Column(String(100), nullable=False, comment='因子名称')
    category = Column(SQLEnum(FactorCategory), nullable=False, comment='因子类别')

    formula = Column(Text, comment='Hikyuu计算公式')
    description = Column(Text, comment='因子描述')
    economic_logic = Column(Text, comment='经济学逻辑')

    status = Column(SQLEnum(FactorStatus), default=FactorStatus.ACTIVE, comment='状态')

    # 计算参数
    calculation_params = Column(JSON, comment='计算参数 JSON格式')

    # 统计信息
    coverage_ratio = Column(Numeric(5, 4), default=0, comment='覆盖率')
    avg_calculation_time_ms = Column(Integer, default=0, comment='平均计算时间(毫秒)')

    created_by = Column(String(50), comment='创建者')
    created_at = Column(DateTime, default=func.current_timestamp())
    updated_at = Column(DateTime, default=func.current_timestamp(), onupdate=func.current_timestamp())

    # 关系
    factor_values = relationship("FactorValue", back_populates="factor_definition", cascade="all, delete-orphan")

    # 索引
    __table_args__ = (
        Index('idx_category', 'category'),
        Index('idx_status', 'status'),
        Index('idx_created_at', 'created_at'),
        {'comment': '因子定义表'}
    )

    def __repr__(self):
        return f"<FactorDefinition(id={self.factor_id}, name={self.factor_name}, category={self.category})>"


class FactorValue(Base):
    """因子值表 (核心表，数据量大)"""
    __tablename__ = 'factor_values'

    id = Column(BigInteger, primary_key=True, autoincrement=True)
    factor_id = Column(String(50), ForeignKey('factor_definitions.factor_id', ondelete='CASCADE'),
                      nullable=False, comment='因子ID')
    stock_code = Column(String(20), ForeignKey('stocks.stock_code', ondelete='CASCADE'),
                       nullable=False, comment='股票代码')
    trade_date = Column(Date, nullable=False, comment='交易日期')

    factor_value = Column(Numeric(15, 6), nullable=False, comment='因子原始值')
    factor_score = Column(Numeric(8, 6), comment='因子标准化分数 (0-1)')
    percentile_rank = Column(Numeric(5, 4), comment='百分位排名')

    # 计算元信息
    calculation_id = Column(String(50), comment='计算批次ID')
    data_version = Column(String(20), comment='数据版本')

    created_at = Column(DateTime, default=func.current_timestamp())

    # 关系
    factor_definition = relationship("FactorDefinition", back_populates="factor_values")
    stock = relationship("Stock", back_populates="factor_values")

    # 约束和索引
    __table_args__ = (
        UniqueConstraint('factor_id', 'stock_code', 'trade_date', name='uk_factor_stock_date'),
        Index('idx_factor_date', 'factor_id', 'trade_date'),
        Index('idx_stock_date', 'stock_code', 'trade_date'),
        Index('idx_calculation_id', 'calculation_id'),
        {'comment': '因子值表'}
    )

    def __repr__(self):
        return f"<FactorValue(factor={self.factor_id}, stock={self.stock_code}, date={self.trade_date}, value={self.factor_value})>"


class FactorCalculationTask(Base):
    """因子计算任务表"""
    __tablename__ = 'factor_calculation_tasks'

    task_id = Column(String(50), primary_key=True, comment='任务ID')
    factor_ids = Column(JSON, nullable=False, comment='因子ID列表')
    stock_list = Column(JSON, comment='股票列表 (null表示全市场)')

    date_range = Column(JSON, nullable=False, comment='日期范围 {start_date, end_date}')

    status = Column(SQLEnum(TaskStatus), default=TaskStatus.PENDING, comment='状态')
    progress = Column(Numeric(5, 2), default=0, comment='进度百分比')

    # 执行信息
    started_at = Column(DateTime, comment='开始时间')
    completed_at = Column(DateTime, comment='完成时间')
    error_message = Column(Text, comment='错误信息')

    # 统计信息
    total_calculations = Column(Integer, default=0, comment='总计算次数')
    successful_calculations = Column(Integer, default=0, comment='成功计算次数')

    created_by = Column(String(50), comment='创建者')
    created_at = Column(DateTime, default=func.current_timestamp())
    updated_at = Column(DateTime, default=func.current_timestamp(), onupdate=func.current_timestamp())

    # 索引
    __table_args__ = (
        Index('idx_status', 'status'),
        Index('idx_created_at', 'created_at'),
        {'comment': '因子计算任务表'}
    )

    def __repr__(self):
        return f"<FactorCalculationTask(id={self.task_id}, status={self.status}, progress={self.progress})>"


class DatabaseManager:
    """数据库管理器"""

    @staticmethod
    def get_all_models():
        """获取所有模型类"""
        return [
            Stock,
            MarketData,
            FinancialData,
            FactorDefinition,
            FactorValue,
            FactorCalculationTask
        ]

    @staticmethod
    def get_table_names():
        """获取所有表名"""
        return [model.__tablename__ for model in DatabaseManager.get_all_models()]