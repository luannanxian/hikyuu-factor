"""
Alembic Environment Configuration
数据库迁移环境配置
"""
import sys
import os
from logging.config import fileConfig
from sqlalchemy import engine_from_config
from sqlalchemy import pool
from alembic import context

# 添加项目根目录到Python路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

# 导入环境管理器
from lib.environment import env_manager

# this is the Alembic Config object, which provides
# access to the values within the .ini file in use.
config = context.config

# Interpret the config file for Python logging.
# This line sets up loggers basically.
if config.config_file_name is not None:
    fileConfig(config.config_file_name)

# add your model's MetaData object here
# for 'autogenerate' support
# from myapp import mymodel
# target_metadata = mymodel.Base.metadata
target_metadata = None

# other values from the config, defined by the needs of env.py,
# can be acquired:
# my_important_option = config.get_main_option("my_important_option")
# ... etc.


def get_database_url():
    """获取数据库连接URL"""
    # 检查环境
    if env_manager.is_development_only():
        # 开发环境使用配置文件中的默认值
        database_url = config.get_main_option("sqlalchemy.url")
        if database_url.startswith("mysql+pymysql://user:password@localhost"):
            # 替换为实际的开发环境数据库配置
            return "mysql+pymysql://dev_user:dev_password@localhost/hikyuu_factor_dev?charset=utf8mb4"
        return database_url
    else:
        # 非开发环境从环境变量获取
        database_url = os.getenv('DATABASE_URL')
        if not database_url:
            raise ValueError(
                "DATABASE_URL environment variable is required for non-development environments"
            )
        return database_url


def run_migrations_offline() -> None:
    """Run migrations in 'offline' mode.

    This configures the context with just a URL
    and not an Engine, though an Engine is acceptable
    here as well.  By skipping the Engine creation
    we don't even need a DBAPI to be available.

    Calls to context.execute() here emit the given string to the
    script output.

    """
    url = get_database_url()
    context.configure(
        url=url,
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
        compare_type=True,
        compare_server_default=True,
    )

    with context.begin_transaction():
        context.run_migrations()


def run_migrations_online() -> None:
    """Run migrations in 'online' mode.

    In this scenario we need to create an Engine
    and associate a connection with the context.

    """
    # 获取数据库URL并更新配置
    database_url = get_database_url()
    config.set_main_option("sqlalchemy.url", database_url)

    connectable = engine_from_config(
        config.get_section(config.config_ini_section),
        prefix="sqlalchemy.",
        poolclass=pool.NullPool,
        connect_args={
            "charset": "utf8mb4",
            "autocommit": False,
        }
    )

    with connectable.connect() as connection:
        context.configure(
            connection=connection,
            target_metadata=target_metadata,
            compare_type=True,
            compare_server_default=True,
        )

        with context.begin_transaction():
            context.run_migrations()


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()