"""
Database Connection and Session Management
数据库连接和会话管理
"""

import os
from typing import Optional, Generator
from contextlib import contextmanager

from sqlalchemy import create_engine, Engine
from sqlalchemy.orm import sessionmaker, Session
from sqlalchemy.pool import QueuePool

from lib.environment import env_manager
from models.database_models import Base


class DatabaseConfig:
    """数据库配置类"""

    def __init__(self):
        self.database_url = self._get_database_url()
        self.echo = env_manager.is_development_only()
        self.pool_size = 10
        self.max_overflow = 20
        self.pool_timeout = 30
        self.pool_recycle = 3600

    def _get_database_url(self) -> str:
        """获取数据库连接URL"""
        if env_manager.is_development_only():
            # 开发环境使用默认配置
            return os.getenv(
                'DATABASE_URL',
                'mysql+pymysql://dev_user:dev_password@localhost/hikyuu_factor_dev?charset=utf8mb4'
            )
        else:
            # 生产环境必须提供DATABASE_URL
            database_url = os.getenv('DATABASE_URL')
            if not database_url:
                raise ValueError(
                    "DATABASE_URL environment variable is required for production environment"
                )
            return database_url


class DatabaseManager:
    """数据库管理器"""

    def __init__(self, config: Optional[DatabaseConfig] = None):
        self.config = config or DatabaseConfig()
        self._engine: Optional[Engine] = None
        self._session_factory: Optional[sessionmaker] = None

    @property
    def engine(self) -> Engine:
        """获取数据库引擎"""
        if self._engine is None:
            self._engine = create_engine(
                self.config.database_url,
                echo=self.config.echo,
                poolclass=QueuePool,
                pool_size=self.config.pool_size,
                max_overflow=self.config.max_overflow,
                pool_timeout=self.config.pool_timeout,
                pool_recycle=self.config.pool_recycle,
                connect_args={
                    "charset": "utf8mb4",
                    "autocommit": False,
                }
            )
        return self._engine

    @property
    def session_factory(self) -> sessionmaker:
        """获取会话工厂"""
        if self._session_factory is None:
            self._session_factory = sessionmaker(
                bind=self.engine,
                autocommit=False,
                autoflush=False
            )
        return self._session_factory

    def get_session(self) -> Session:
        """获取数据库会话"""
        return self.session_factory()

    @contextmanager
    def session_scope(self) -> Generator[Session, None, None]:
        """数据库会话上下文管理器"""
        session = self.get_session()
        try:
            yield session
            session.commit()
        except Exception:
            session.rollback()
            raise
        finally:
            session.close()

    def create_all_tables(self):
        """创建所有表"""
        Base.metadata.create_all(self.engine)

    def drop_all_tables(self):
        """删除所有表"""
        Base.metadata.drop_all(self.engine)

    def test_connection(self) -> bool:
        """测试数据库连接"""
        try:
            with self.engine.connect() as conn:
                conn.execute("SELECT 1")
            return True
        except Exception as e:
            print(f"Database connection test failed: {e}")
            return False

    def close(self):
        """关闭数据库连接"""
        if self._engine:
            self._engine.dispose()
            self._engine = None
        self._session_factory = None


# 全局数据库管理器实例
db_manager = DatabaseManager()


def get_database_session() -> Session:
    """获取数据库会话 (FastAPI依赖项)"""
    return db_manager.get_session()


def init_database():
    """初始化数据库"""
    if not env_manager.is_mock_data_allowed():
        # 生产环境需要真实的数据库初始化
        print("Initializing production database...")
        db_manager.create_all_tables()
        print("Database initialization completed.")
    else:
        # 开发环境警告
        print("WARNING: Development environment detected.")
        print("Database initialization requires real database connection in production.")
        print("Use Alembic migrations for production database setup.")


def cleanup_database():
    """清理数据库连接"""
    db_manager.close()