"""
Database Models and Session Management
数据库连接、会话管理和基础数据访问模型
"""

import asyncio
import logging
from contextlib import asynccontextmanager, contextmanager
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, AsyncGenerator, Generator, Union
import pymysql
from sqlalchemy import create_engine, text, Engine, MetaData, Table
from sqlalchemy.orm import sessionmaker, Session, declarative_base
from sqlalchemy.pool import StaticPool, QueuePool
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession, async_sessionmaker
import json
from datetime import datetime, timedelta
import threading
import time


# 声明基类
Base = declarative_base()

logger = logging.getLogger(__name__)


@dataclass
class DatabaseConfig:
    """数据库配置"""
    host: str = "localhost"
    port: int = 3306
    user: str = "root"
    password: str = ""
    database: str = "hikyuu_factor"
    charset: str = "utf8mb4"

    # 连接池配置
    pool_size: int = 10
    max_overflow: int = 20
    pool_timeout: int = 30
    pool_recycle: int = 3600  # 1小时

    # 连接超时配置
    connect_timeout: int = 10
    read_timeout: int = 30
    write_timeout: int = 30

    # 其他配置
    autocommit: bool = False
    echo: bool = False
    echo_pool: bool = False

    def get_sync_url(self) -> str:
        """获取同步数据库连接URL"""
        return (f"mysql+pymysql://{self.user}:{self.password}@"
                f"{self.host}:{self.port}/{self.database}"
                f"?charset={self.charset}")

    def get_async_url(self) -> str:
        """获取异步数据库连接URL"""
        return (f"mysql+aiomysql://{self.user}:{self.password}@"
                f"{self.host}:{self.port}/{self.database}"
                f"?charset={self.charset}")

    def get_connection_params(self) -> Dict[str, Any]:
        """获取连接参数"""
        return {
            "host": self.host,
            "port": self.port,
            "user": self.user,
            "password": self.password,
            "database": self.database,
            "charset": self.charset,
            "connect_timeout": self.connect_timeout,
            "read_timeout": self.read_timeout,
            "write_timeout": self.write_timeout,
            "autocommit": self.autocommit
        }

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            "host": self.host,
            "port": self.port,
            "user": self.user,
            "password": "***",  # 隐藏密码
            "database": self.database,
            "charset": self.charset,
            "pool_size": self.pool_size,
            "max_overflow": self.max_overflow,
            "pool_timeout": self.pool_timeout,
            "pool_recycle": self.pool_recycle,
            "connect_timeout": self.connect_timeout,
            "read_timeout": self.read_timeout,
            "write_timeout": self.write_timeout,
            "autocommit": self.autocommit,
            "echo": self.echo
        }


@dataclass
class ConnectionHealth:
    """连接健康状态"""
    is_healthy: bool
    latency_ms: float
    error_message: Optional[str] = None
    last_check: datetime = field(default_factory=datetime.now)
    connection_count: int = 0
    active_connections: int = 0

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            "is_healthy": self.is_healthy,
            "latency_ms": self.latency_ms,
            "error_message": self.error_message,
            "last_check": self.last_check.isoformat(),
            "connection_count": self.connection_count,
            "active_connections": self.active_connections
        }


class DatabaseManager:
    """数据库管理器"""

    def __init__(self, config: DatabaseConfig):
        self.config = config
        self.sync_engine: Optional[Engine] = None
        self.async_engine = None
        self.sync_session_factory: Optional[sessionmaker] = None
        self.async_session_factory = None
        self._health_status = ConnectionHealth(is_healthy=False, latency_ms=0.0)
        self._lock = threading.Lock()

    def initialize_sync(self) -> bool:
        """初始化同步数据库连接"""
        try:
            # 创建同步引擎
            self.sync_engine = create_engine(
                self.config.get_sync_url(),
                pool_size=self.config.pool_size,
                max_overflow=self.config.max_overflow,
                pool_timeout=self.config.pool_timeout,
                pool_recycle=self.config.pool_recycle,
                echo=self.config.echo,
                echo_pool=self.config.echo_pool,
                connect_args={
                    "connect_timeout": self.config.connect_timeout,
                    "read_timeout": self.config.read_timeout,
                    "write_timeout": self.config.write_timeout
                }
            )

            # 创建会话工厂
            self.sync_session_factory = sessionmaker(
                bind=self.sync_engine,
                autocommit=self.config.autocommit,
                autoflush=True
            )

            # 测试连接
            with self.sync_engine.connect() as conn:
                conn.execute(text("SELECT 1"))

            logger.info("Sync database connection initialized successfully")
            return True

        except Exception as e:
            logger.error(f"Failed to initialize sync database connection: {e}")
            return False

    async def initialize_async(self) -> bool:
        """初始化异步数据库连接"""
        try:
            # 创建异步引擎
            self.async_engine = create_async_engine(
                self.config.get_async_url(),
                pool_size=self.config.pool_size,
                max_overflow=self.config.max_overflow,
                pool_timeout=self.config.pool_timeout,
                pool_recycle=self.config.pool_recycle,
                echo=self.config.echo,
                echo_pool=self.config.echo_pool,
                connect_args={
                    "connect_timeout": self.config.connect_timeout,
                    "read_timeout": self.config.read_timeout,
                    "write_timeout": self.config.write_timeout
                }
            )

            # 创建异步会话工厂
            self.async_session_factory = async_sessionmaker(
                bind=self.async_engine,
                autocommit=self.config.autocommit,
                autoflush=True,
                class_=AsyncSession
            )

            # 测试连接
            async with self.async_engine.connect() as conn:
                await conn.execute(text("SELECT 1"))

            logger.info("Async database connection initialized successfully")
            return True

        except Exception as e:
            logger.error(f"Failed to initialize async database connection: {e}")
            return False

    @contextmanager
    def get_sync_session(self) -> Generator[Session, None, None]:
        """获取同步数据库会话"""
        if not self.sync_session_factory:
            raise RuntimeError("Sync database not initialized")

        session = self.sync_session_factory()
        try:
            yield session
            session.commit()
        except Exception:
            session.rollback()
            raise
        finally:
            session.close()

    @asynccontextmanager
    async def get_async_session(self) -> AsyncGenerator[AsyncSession, None]:
        """获取异步数据库会话"""
        if not self.async_session_factory:
            raise RuntimeError("Async database not initialized")

        session = self.async_session_factory()
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise
        finally:
            await session.close()

    def execute_sync(self, query: str, params: Optional[Dict[str, Any]] = None) -> Any:
        """执行同步查询"""
        with self.get_sync_session() as session:
            result = session.execute(text(query), params or {})
            return result.fetchall()

    async def execute_async(self, query: str, params: Optional[Dict[str, Any]] = None) -> Any:
        """执行异步查询"""
        async with self.get_async_session() as session:
            result = await session.execute(text(query), params or {})
            return result.fetchall()

    def check_health(self) -> ConnectionHealth:
        """检查数据库健康状态"""
        try:
            start_time = time.time()

            # 测试同步连接
            if self.sync_engine:
                with self.sync_engine.connect() as conn:
                    conn.execute(text("SELECT 1"))

            latency_ms = (time.time() - start_time) * 1000

            # 获取连接池信息
            pool_info = {}
            if self.sync_engine and hasattr(self.sync_engine.pool, 'size'):
                pool_info = {
                    "connection_count": self.sync_engine.pool.size(),
                    "active_connections": self.sync_engine.pool.checkedout()
                }

            self._health_status = ConnectionHealth(
                is_healthy=True,
                latency_ms=latency_ms,
                last_check=datetime.now(),
                **pool_info
            )

        except Exception as e:
            self._health_status = ConnectionHealth(
                is_healthy=False,
                latency_ms=0.0,
                error_message=str(e),
                last_check=datetime.now()
            )

        return self._health_status

    async def check_health_async(self) -> ConnectionHealth:
        """异步检查数据库健康状态"""
        try:
            start_time = time.time()

            # 测试异步连接
            if self.async_engine:
                async with self.async_engine.connect() as conn:
                    await conn.execute(text("SELECT 1"))

            latency_ms = (time.time() - start_time) * 1000

            self._health_status = ConnectionHealth(
                is_healthy=True,
                latency_ms=latency_ms,
                last_check=datetime.now()
            )

        except Exception as e:
            self._health_status = ConnectionHealth(
                is_healthy=False,
                latency_ms=0.0,
                error_message=str(e),
                last_check=datetime.now()
            )

        return self._health_status

    def get_database_info(self) -> Dict[str, Any]:
        """获取数据库信息"""
        try:
            with self.get_sync_session() as session:
                # 获取数据库版本
                version_result = session.execute(text("SELECT VERSION()"))
                version = version_result.scalar()

                # 获取表列表
                tables_result = session.execute(text("SHOW TABLES"))
                tables = [row[0] for row in tables_result.fetchall()]

                # 获取数据库大小
                size_result = session.execute(text("""
                    SELECT
                        ROUND(SUM(data_length + index_length) / 1024 / 1024, 1) AS size_mb
                    FROM information_schema.tables
                    WHERE table_schema = :database
                """), {"database": self.config.database})
                size_mb = size_result.scalar() or 0

                return {
                    "version": version,
                    "database": self.config.database,
                    "tables": tables,
                    "table_count": len(tables),
                    "size_mb": size_mb,
                    "config": self.config.to_dict()
                }

        except Exception as e:
            logger.error(f"Failed to get database info: {e}")
            return {"error": str(e)}

    def close(self) -> None:
        """关闭数据库连接"""
        if self.sync_engine:
            self.sync_engine.dispose()
            logger.info("Sync database connection closed")

        if self.async_engine:
            # 异步引擎需要在异步上下文中关闭
            loop = asyncio.get_event_loop()
            if loop.is_running():
                asyncio.create_task(self.async_engine.dispose())
            else:
                asyncio.run(self.async_engine.dispose())
            logger.info("Async database connection closed")


class DatabaseConnectionPool:
    """数据库连接池管理器"""

    def __init__(self):
        self._pools: Dict[str, DatabaseManager] = {}
        self._default_pool: Optional[str] = None

    def add_pool(self, name: str, config: DatabaseConfig, set_as_default: bool = False) -> bool:
        """添加连接池"""
        try:
            manager = DatabaseManager(config)
            if manager.initialize_sync():
                self._pools[name] = manager
                if set_as_default or not self._default_pool:
                    self._default_pool = name
                logger.info(f"Database pool '{name}' added successfully")
                return True
        except Exception as e:
            logger.error(f"Failed to add database pool '{name}': {e}")

        return False

    def get_pool(self, name: Optional[str] = None) -> Optional[DatabaseManager]:
        """获取连接池"""
        pool_name = name or self._default_pool
        return self._pools.get(pool_name) if pool_name else None

    def remove_pool(self, name: str) -> bool:
        """移除连接池"""
        if name in self._pools:
            self._pools[name].close()
            del self._pools[name]

            if self._default_pool == name:
                self._default_pool = next(iter(self._pools.keys())) if self._pools else None

            logger.info(f"Database pool '{name}' removed")
            return True

        return False

    def get_health_status(self) -> Dict[str, ConnectionHealth]:
        """获取所有连接池的健康状态"""
        status = {}
        for name, pool in self._pools.items():
            status[name] = pool.check_health()
        return status

    def close_all(self) -> None:
        """关闭所有连接池"""
        for name, pool in self._pools.items():
            pool.close()
            logger.info(f"Database pool '{name}' closed")
        self._pools.clear()
        self._default_pool = None


# 全局连接池实例
db_pool = DatabaseConnectionPool()


def init_database(config: DatabaseConfig, pool_name: str = "default") -> bool:
    """初始化数据库连接"""
    return db_pool.add_pool(pool_name, config, set_as_default=True)


def get_db_manager(pool_name: Optional[str] = None) -> Optional[DatabaseManager]:
    """获取数据库管理器"""
    return db_pool.get_pool(pool_name)


@contextmanager
def get_db_session(pool_name: Optional[str] = None) -> Generator[Session, None, None]:
    """获取数据库会话上下文管理器"""
    manager = get_db_manager(pool_name)
    if not manager:
        raise RuntimeError("Database not initialized")

    with manager.get_sync_session() as session:
        yield session


@asynccontextmanager
async def get_async_db_session(pool_name: Optional[str] = None) -> AsyncGenerator[AsyncSession, None]:
    """获取异步数据库会话上下文管理器"""
    manager = get_db_manager(pool_name)
    if not manager:
        raise RuntimeError("Database not initialized")

    async with manager.get_async_session() as session:
        yield session


def execute_query(query: str, params: Optional[Dict[str, Any]] = None,
                  pool_name: Optional[str] = None) -> Any:
    """执行数据库查询"""
    manager = get_db_manager(pool_name)
    if not manager:
        raise RuntimeError("Database not initialized")

    return manager.execute_sync(query, params)


async def execute_query_async(query: str, params: Optional[Dict[str, Any]] = None,
                               pool_name: Optional[str] = None) -> Any:
    """异步执行数据库查询"""
    manager = get_db_manager(pool_name)
    if not manager:
        raise RuntimeError("Database not initialized")

    return await manager.execute_async(query, params)


def check_database_health(pool_name: Optional[str] = None) -> ConnectionHealth:
    """检查数据库健康状态"""
    manager = get_db_manager(pool_name)
    if not manager:
        return ConnectionHealth(
            is_healthy=False,
            latency_ms=0.0,
            error_message="Database not initialized"
        )

    return manager.check_health()


def get_database_info(pool_name: Optional[str] = None) -> Dict[str, Any]:
    """获取数据库信息"""
    manager = get_db_manager(pool_name)
    if not manager:
        return {"error": "Database not initialized"}

    return manager.get_database_info()


def close_database_connections() -> None:
    """关闭所有数据库连接"""
    db_pool.close_all()