"""Add table partitioning for performance

Revision ID: 0002
Revises: 0001
Create Date: 2025-09-22 10:05:00.000000

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa

# revision identifiers, used by Alembic.
revision: str = '0002'
down_revision: Union[str, None] = '0001'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Add partitioning to large tables"""

    # 为market_data表添加分区
    # 注意：MySQL的Alembic不直接支持分区，需要使用原生SQL
    op.execute("""
        ALTER TABLE market_data
        PARTITION BY RANGE COLUMNS(trade_date) (
            PARTITION p2020 VALUES LESS THAN ('2021-01-01'),
            PARTITION p2021 VALUES LESS THAN ('2022-01-01'),
            PARTITION p2022 VALUES LESS THAN ('2023-01-01'),
            PARTITION p2023 VALUES LESS THAN ('2024-01-01'),
            PARTITION p2024 VALUES LESS THAN ('2025-01-01'),
            PARTITION p_future VALUES LESS THAN MAXVALUE
        )
    """)

    # 为factor_values表添加分区
    op.execute("""
        ALTER TABLE factor_values
        PARTITION BY RANGE COLUMNS(trade_date) (
            PARTITION p2020 VALUES LESS THAN ('2021-01-01'),
            PARTITION p2021 VALUES LESS THAN ('2022-01-01'),
            PARTITION p2022 VALUES LESS THAN ('2023-01-01'),
            PARTITION p2023 VALUES LESS THAN ('2024-01-01'),
            PARTITION p2024 VALUES LESS THAN ('2025-01-01'),
            PARTITION p_future VALUES LESS THAN MAXVALUE
        )
    """)


def downgrade() -> None:
    """Remove partitioning from tables"""

    # 移除market_data表的分区
    op.execute("ALTER TABLE market_data REMOVE PARTITIONING")

    # 移除factor_values表的分区
    op.execute("ALTER TABLE factor_values REMOVE PARTITIONING")