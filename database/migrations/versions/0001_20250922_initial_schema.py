"""Initial schema creation

Revision ID: 0001
Revises:
Create Date: 2025-09-22 10:00:00.000000

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import mysql

# revision identifiers, used by Alembic.
revision: str = '0001'
down_revision: Union[str, None] = None
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade database schema"""

    # 设置字符集和时区
    op.execute("SET NAMES utf8mb4")
    op.execute("SET TIME_ZONE = '+08:00'")

    # 创建股票基础信息表
    op.create_table(
        'stocks',
        sa.Column('stock_code', sa.VARCHAR(20), primary_key=True, comment='股票代码 (如: sh000001)'),
        sa.Column('stock_name', sa.VARCHAR(100), nullable=False, comment='股票名称'),
        sa.Column('market', sa.VARCHAR(10), nullable=False, comment='市场代码 (sh/sz/bj)'),
        sa.Column('sector', sa.VARCHAR(50), comment='行业板块'),
        sa.Column('list_date', sa.DATE, comment='上市日期'),
        sa.Column('delist_date', sa.DATE, comment='退市日期'),
        sa.Column('status', sa.Enum('active', 'suspended', 'delisted', name='stock_status'),
                 server_default='active', comment='状态'),
        sa.Column('created_at', sa.TIMESTAMP, server_default=sa.text('CURRENT_TIMESTAMP')),
        sa.Column('updated_at', sa.TIMESTAMP,
                 server_default=sa.text('CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP')),

        mysql_engine='InnoDB',
        mysql_charset='utf8mb4',
        comment='股票基础信息表'
    )

    # 创建索引
    op.create_index('idx_market', 'stocks', ['market'])
    op.create_index('idx_sector', 'stocks', ['sector'])
    op.create_index('idx_status', 'stocks', ['status'])
    op.create_index('idx_list_date', 'stocks', ['list_date'])

    # 创建市场数据表
    op.create_table(
        'market_data',
        sa.Column('id', sa.BIGINT, primary_key=True, autoincrement=True),
        sa.Column('stock_code', sa.VARCHAR(20), nullable=False, comment='股票代码'),
        sa.Column('trade_date', sa.DATE, nullable=False, comment='交易日期'),

        sa.Column('open_price', sa.DECIMAL(10, 3), nullable=False, comment='开盘价'),
        sa.Column('high_price', sa.DECIMAL(10, 3), nullable=False, comment='最高价'),
        sa.Column('low_price', sa.DECIMAL(10, 3), nullable=False, comment='最低价'),
        sa.Column('close_price', sa.DECIMAL(10, 3), nullable=False, comment='收盘价'),
        sa.Column('volume', sa.BIGINT, nullable=False, comment='成交量'),
        sa.Column('amount', sa.DECIMAL(18, 2), nullable=False, comment='成交金额'),

        sa.Column('adj_factor', sa.DECIMAL(10, 6), server_default='1.0', comment='复权因子'),
        sa.Column('turnover_rate', sa.DECIMAL(8, 4), comment='换手率'),

        sa.Column('created_at', sa.TIMESTAMP, server_default=sa.text('CURRENT_TIMESTAMP')),
        sa.Column('updated_at', sa.TIMESTAMP,
                 server_default=sa.text('CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP')),

        mysql_engine='InnoDB',
        mysql_charset='utf8mb4',
        comment='市场数据表'
    )

    # 创建唯一约束和索引
    op.create_unique_constraint('uk_stock_date', 'market_data', ['stock_code', 'trade_date'])
    op.create_index('idx_trade_date', 'market_data', ['trade_date'])
    op.create_index('idx_stock_code', 'market_data', ['stock_code'])

    # 创建外键约束
    op.create_foreign_key(
        'fk_market_data_stock',
        'market_data', 'stocks',
        ['stock_code'], ['stock_code'],
        ondelete='CASCADE'
    )

    # 创建财务数据表
    op.create_table(
        'financial_data',
        sa.Column('id', sa.BIGINT, primary_key=True, autoincrement=True),
        sa.Column('stock_code', sa.VARCHAR(20), nullable=False, comment='股票代码'),
        sa.Column('report_date', sa.DATE, nullable=False, comment='报告期'),

        # 基本财务指标
        sa.Column('pe_ratio', sa.DECIMAL(10, 3), comment='市盈率'),
        sa.Column('pb_ratio', sa.DECIMAL(10, 3), comment='市净率'),
        sa.Column('roe', sa.DECIMAL(8, 4), comment='净资产收益率'),
        sa.Column('roa', sa.DECIMAL(8, 4), comment='总资产收益率'),

        # 财务数据
        sa.Column('total_revenue', sa.DECIMAL(18, 2), comment='营业收入'),
        sa.Column('net_profit', sa.DECIMAL(18, 2), comment='净利润'),
        sa.Column('total_assets', sa.DECIMAL(18, 2), comment='总资产'),
        sa.Column('total_equity', sa.DECIMAL(18, 2), comment='股东权益'),

        sa.Column('created_at', sa.TIMESTAMP, server_default=sa.text('CURRENT_TIMESTAMP')),
        sa.Column('updated_at', sa.TIMESTAMP,
                 server_default=sa.text('CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP')),

        mysql_engine='InnoDB',
        mysql_charset='utf8mb4',
        comment='财务数据表'
    )

    # 创建唯一约束和索引
    op.create_unique_constraint('uk_stock_report', 'financial_data', ['stock_code', 'report_date'])
    op.create_index('idx_report_date', 'financial_data', ['report_date'])

    # 创建外键约束
    op.create_foreign_key(
        'fk_financial_data_stock',
        'financial_data', 'stocks',
        ['stock_code'], ['stock_code'],
        ondelete='CASCADE'
    )

    # 创建因子定义表
    op.create_table(
        'factor_definitions',
        sa.Column('factor_id', sa.VARCHAR(50), primary_key=True, comment='因子ID'),
        sa.Column('factor_name', sa.VARCHAR(100), nullable=False, comment='因子名称'),
        sa.Column('category', sa.Enum('momentum', 'value', 'quality', 'volatility', 'growth', 'technical',
                                    name='factor_category'), nullable=False, comment='因子类别'),

        sa.Column('formula', sa.TEXT, comment='Hikyuu计算公式'),
        sa.Column('description', sa.TEXT, comment='因子描述'),
        sa.Column('economic_logic', sa.TEXT, comment='经济学逻辑'),

        sa.Column('status', sa.Enum('active', 'inactive', 'testing', name='factor_status'),
                 server_default='active', comment='状态'),

        # 计算参数
        sa.Column('calculation_params', sa.JSON, comment='计算参数 JSON格式'),

        # 统计信息
        sa.Column('coverage_ratio', sa.DECIMAL(5, 4), server_default='0', comment='覆盖率'),
        sa.Column('avg_calculation_time_ms', sa.INT, server_default='0', comment='平均计算时间(毫秒)'),

        sa.Column('created_by', sa.VARCHAR(50), comment='创建者'),
        sa.Column('created_at', sa.TIMESTAMP, server_default=sa.text('CURRENT_TIMESTAMP')),
        sa.Column('updated_at', sa.TIMESTAMP,
                 server_default=sa.text('CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP')),

        mysql_engine='InnoDB',
        mysql_charset='utf8mb4',
        comment='因子定义表'
    )

    # 创建索引
    op.create_index('idx_category', 'factor_definitions', ['category'])
    op.create_index('idx_status', 'factor_definitions', ['status'])
    op.create_index('idx_created_at', 'factor_definitions', ['created_at'])

    # 创建因子值表
    op.create_table(
        'factor_values',
        sa.Column('id', sa.BIGINT, primary_key=True, autoincrement=True),
        sa.Column('factor_id', sa.VARCHAR(50), nullable=False, comment='因子ID'),
        sa.Column('stock_code', sa.VARCHAR(20), nullable=False, comment='股票代码'),
        sa.Column('trade_date', sa.DATE, nullable=False, comment='交易日期'),

        sa.Column('factor_value', sa.DECIMAL(15, 6), nullable=False, comment='因子原始值'),
        sa.Column('factor_score', sa.DECIMAL(8, 6), comment='因子标准化分数 (0-1)'),
        sa.Column('percentile_rank', sa.DECIMAL(5, 4), comment='百分位排名'),

        # 计算元信息
        sa.Column('calculation_id', sa.VARCHAR(50), comment='计算批次ID'),
        sa.Column('data_version', sa.VARCHAR(20), comment='数据版本'),

        sa.Column('created_at', sa.TIMESTAMP, server_default=sa.text('CURRENT_TIMESTAMP')),

        mysql_engine='InnoDB',
        mysql_charset='utf8mb4',
        comment='因子值表'
    )

    # 创建唯一约束和索引
    op.create_unique_constraint('uk_factor_stock_date', 'factor_values',
                               ['factor_id', 'stock_code', 'trade_date'])
    op.create_index('idx_factor_date', 'factor_values', ['factor_id', 'trade_date'])
    op.create_index('idx_stock_date', 'factor_values', ['stock_code', 'trade_date'])
    op.create_index('idx_calculation_id', 'factor_values', ['calculation_id'])

    # 创建外键约束
    op.create_foreign_key(
        'fk_factor_values_factor',
        'factor_values', 'factor_definitions',
        ['factor_id'], ['factor_id'],
        ondelete='CASCADE'
    )
    op.create_foreign_key(
        'fk_factor_values_stock',
        'factor_values', 'stocks',
        ['stock_code'], ['stock_code'],
        ondelete='CASCADE'
    )

    # 创建因子计算任务表
    op.create_table(
        'factor_calculation_tasks',
        sa.Column('task_id', sa.VARCHAR(50), primary_key=True, comment='任务ID'),
        sa.Column('factor_ids', sa.JSON, nullable=False, comment='因子ID列表'),
        sa.Column('stock_list', sa.JSON, comment='股票列表 (null表示全市场)'),

        sa.Column('date_range', sa.JSON, nullable=False, comment='日期范围 {start_date, end_date}'),

        sa.Column('status', sa.Enum('pending', 'running', 'completed', 'failed', 'cancelled',
                                  name='task_status'), server_default='pending', comment='状态'),
        sa.Column('progress', sa.DECIMAL(5, 2), server_default='0', comment='进度百分比'),

        # 执行信息
        sa.Column('started_at', sa.TIMESTAMP, comment='开始时间'),
        sa.Column('completed_at', sa.TIMESTAMP, comment='完成时间'),
        sa.Column('error_message', sa.TEXT, comment='错误信息'),

        # 统计信息
        sa.Column('total_calculations', sa.INT, server_default='0', comment='总计算次数'),
        sa.Column('successful_calculations', sa.INT, server_default='0', comment='成功计算次数'),

        sa.Column('created_by', sa.VARCHAR(50), comment='创建者'),
        sa.Column('created_at', sa.TIMESTAMP, server_default=sa.text('CURRENT_TIMESTAMP')),
        sa.Column('updated_at', sa.TIMESTAMP,
                 server_default=sa.text('CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP')),

        mysql_engine='InnoDB',
        mysql_charset='utf8mb4',
        comment='因子计算任务表'
    )

    # 创建索引
    op.create_index('idx_status', 'factor_calculation_tasks', ['status'])
    op.create_index('idx_created_at', 'factor_calculation_tasks', ['created_at'])


def downgrade() -> None:
    """Downgrade database schema"""

    # 删除表（逆序）
    op.drop_table('factor_calculation_tasks')
    op.drop_table('factor_values')
    op.drop_table('factor_definitions')
    op.drop_table('financial_data')
    op.drop_table('market_data')
    op.drop_table('stocks')

    # 删除枚举类型
    op.execute("DROP TYPE IF EXISTS task_status")
    op.execute("DROP TYPE IF EXISTS factor_status")
    op.execute("DROP TYPE IF EXISTS factor_category")
    op.execute("DROP TYPE IF EXISTS stock_status")