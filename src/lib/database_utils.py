"""
Database Utilities and Migrations
数据库工具类和迁移管理，支持数据库结构管理和数据迁移
"""

import os
import json
import hashlib
from pathlib import Path
from typing import Dict, List, Optional, Any, Union, Callable
from dataclasses import dataclass, field
from datetime import datetime, date
import logging
from sqlalchemy import (
    create_engine, MetaData, Table, Column, Integer, String, Text, DateTime,
    Boolean, Float, Date, Index, ForeignKey, text, inspect
)
from sqlalchemy.engine import Engine
from sqlalchemy.orm import Session
from contextlib import contextmanager

from ..models.database import DatabaseManager, DatabaseConfig, get_db_session
from ..lib.exceptions import DatabaseConnectionException, DatabaseQueryException

logger = logging.getLogger(__name__)


@dataclass
class MigrationInfo:
    """迁移信息"""
    version: str
    name: str
    description: str
    created_at: datetime
    applied_at: Optional[datetime] = None
    checksum: Optional[str] = None
    execution_time_ms: Optional[int] = None

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            "version": self.version,
            "name": self.name,
            "description": self.description,
            "created_at": self.created_at.isoformat(),
            "applied_at": self.applied_at.isoformat() if self.applied_at else None,
            "checksum": self.checksum,
            "execution_time_ms": self.execution_time_ms
        }


class Migration:
    """数据库迁移基类"""

    def __init__(self, version: str, name: str, description: str = ""):
        self.version = version
        self.name = name
        self.description = description

    def up(self, session: Session) -> None:
        """执行迁移（向上）"""
        raise NotImplementedError("Migration must implement up() method")

    def down(self, session: Session) -> None:
        """回滚迁移（向下）"""
        raise NotImplementedError("Migration must implement down() method")

    def get_checksum(self) -> str:
        """获取迁移校验和"""
        content = f"{self.version}_{self.name}_{self.description}"
        # 包含up和down方法的源代码
        import inspect
        try:
            up_source = inspect.getsource(self.up)
            down_source = inspect.getsource(self.down)
            content += up_source + down_source
        except OSError:
            # 如果无法获取源代码，使用类名
            content += self.__class__.__name__

        return hashlib.sha256(content.encode()).hexdigest()


class DatabaseSchema:
    """数据库Schema管理"""

    def __init__(self, db_manager: DatabaseManager):
        self.db_manager = db_manager
        self.metadata = MetaData()

    def create_schema_tables(self) -> None:
        """创建Schema管理相关表"""
        # 迁移历史表
        self.migration_history = Table(
            'schema_migration_history',
            self.metadata,
            Column('id', Integer, primary_key=True, autoincrement=True),
            Column('version', String(50), nullable=False, unique=True),
            Column('name', String(200), nullable=False),
            Column('description', Text),
            Column('checksum', String(64), nullable=False),
            Column('applied_at', DateTime, nullable=False),
            Column('execution_time_ms', Integer),
            Index('idx_migration_version', 'version'),
            Index('idx_migration_applied_at', 'applied_at')
        )

        # 数据库版本表
        self.database_version = Table(
            'schema_database_version',
            self.metadata,
            Column('id', Integer, primary_key=True),
            Column('current_version', String(50), nullable=False),
            Column('updated_at', DateTime, nullable=False)
        )

        # 创建表
        with self.db_manager.get_sync_session() as session:
            self.metadata.create_all(session.bind)

    def get_current_version(self) -> Optional[str]:
        """获取当前数据库版本"""
        try:
            with self.db_manager.get_sync_session() as session:
                result = session.execute(
                    text("SELECT current_version FROM schema_database_version WHERE id = 1")
                ).fetchone()
                return result[0] if result else None
        except Exception:
            return None

    def set_current_version(self, version: str) -> None:
        """设置当前数据库版本"""
        with self.db_manager.get_sync_session() as session:
            # 尝试更新
            result = session.execute(
                text("UPDATE schema_database_version SET current_version = :version, updated_at = :now WHERE id = 1"),
                {"version": version, "now": datetime.now()}
            )

            # 如果更新失败，插入新记录
            if result.rowcount == 0:
                session.execute(
                    text("INSERT INTO schema_database_version (id, current_version, updated_at) VALUES (1, :version, :now)"),
                    {"version": version, "now": datetime.now()}
                )

    def get_applied_migrations(self) -> List[MigrationInfo]:
        """获取已应用的迁移"""
        migrations = []
        try:
            with self.db_manager.get_sync_session() as session:
                result = session.execute(
                    text("""
                        SELECT version, name, description, checksum, applied_at, execution_time_ms
                        FROM schema_migration_history
                        ORDER BY applied_at
                    """)
                ).fetchall()

                for row in result:
                    migrations.append(MigrationInfo(
                        version=row[0],
                        name=row[1],
                        description=row[2] or "",
                        created_at=datetime.now(),  # 无法从历史获取
                        applied_at=row[4],
                        checksum=row[3],
                        execution_time_ms=row[5]
                    ))

        except Exception as e:
            logger.warning(f"Failed to get applied migrations: {e}")

        return migrations

    def record_migration(self, migration: Migration, execution_time_ms: int) -> None:
        """记录迁移执行"""
        with self.db_manager.get_sync_session() as session:
            session.execute(
                text("""
                    INSERT INTO schema_migration_history
                    (version, name, description, checksum, applied_at, execution_time_ms)
                    VALUES (:version, :name, :description, :checksum, :applied_at, :execution_time_ms)
                """),
                {
                    "version": migration.version,
                    "name": migration.name,
                    "description": migration.description,
                    "checksum": migration.get_checksum(),
                    "applied_at": datetime.now(),
                    "execution_time_ms": execution_time_ms
                }
            )

    def remove_migration_record(self, version: str) -> None:
        """移除迁移记录"""
        with self.db_manager.get_sync_session() as session:
            session.execute(
                text("DELETE FROM schema_migration_history WHERE version = :version"),
                {"version": version}
            )


class MigrationManager:
    """迁移管理器"""

    def __init__(self, db_manager: DatabaseManager, migrations_dir: Union[str, Path] = "migrations"):
        self.db_manager = db_manager
        self.migrations_dir = Path(migrations_dir)
        self.schema = DatabaseSchema(db_manager)
        self.migrations: Dict[str, Migration] = {}

    def initialize(self) -> None:
        """初始化迁移管理器"""
        # 创建迁移目录
        self.migrations_dir.mkdir(exist_ok=True)

        # 创建Schema表
        self.schema.create_schema_tables()

        # 加载迁移
        self.load_migrations()

        logger.info("Migration manager initialized")

    def load_migrations(self) -> None:
        """加载迁移文件"""
        self.migrations.clear()

        # 扫描迁移目录中的Python文件
        for migration_file in self.migrations_dir.glob("*.py"):
            if migration_file.name.startswith("__"):
                continue

            try:
                self._load_migration_file(migration_file)
            except Exception as e:
                logger.error(f"Failed to load migration {migration_file}: {e}")

        logger.info(f"Loaded {len(self.migrations)} migrations")

    def _load_migration_file(self, file_path: Path) -> None:
        """加载单个迁移文件"""
        # 动态导入迁移模块
        import importlib.util
        spec = importlib.util.spec_from_file_location("migration", file_path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)

        # 查找Migration类
        for attr_name in dir(module):
            attr = getattr(module, attr_name)
            if (isinstance(attr, type) and
                issubclass(attr, Migration) and
                attr != Migration):

                migration_instance = attr()
                self.migrations[migration_instance.version] = migration_instance
                break

    def register_migration(self, migration: Migration) -> None:
        """注册迁移"""
        self.migrations[migration.version] = migration

    def get_pending_migrations(self) -> List[Migration]:
        """获取待执行的迁移"""
        applied_migrations = {m.version for m in self.schema.get_applied_migrations()}
        pending = []

        # 按版本排序
        sorted_versions = sorted(self.migrations.keys())
        for version in sorted_versions:
            if version not in applied_migrations:
                pending.append(self.migrations[version])

        return pending

    def migrate(self, target_version: Optional[str] = None) -> List[MigrationInfo]:
        """执行迁移"""
        pending_migrations = self.get_pending_migrations()

        if target_version:
            # 只执行到指定版本
            pending_migrations = [
                m for m in pending_migrations
                if m.version <= target_version
            ]

        executed_migrations = []

        for migration in pending_migrations:
            logger.info(f"Applying migration: {migration.version} - {migration.name}")

            start_time = datetime.now()
            try:
                with self.db_manager.get_sync_session() as session:
                    migration.up(session)

                end_time = datetime.now()
                execution_time_ms = int((end_time - start_time).total_seconds() * 1000)

                # 记录迁移
                self.schema.record_migration(migration, execution_time_ms)

                # 更新数据库版本
                self.schema.set_current_version(migration.version)

                migration_info = MigrationInfo(
                    version=migration.version,
                    name=migration.name,
                    description=migration.description,
                    created_at=start_time,
                    applied_at=end_time,
                    checksum=migration.get_checksum(),
                    execution_time_ms=execution_time_ms
                )

                executed_migrations.append(migration_info)
                logger.info(f"Migration applied successfully: {migration.version} ({execution_time_ms}ms)")

            except Exception as e:
                logger.error(f"Migration failed: {migration.version} - {e}")
                raise DatabaseQueryException(f"Migration {migration.version} failed: {e}")

        return executed_migrations

    def rollback(self, target_version: str) -> List[MigrationInfo]:
        """回滚到指定版本"""
        applied_migrations = self.schema.get_applied_migrations()

        # 找到需要回滚的迁移（按应用时间倒序）
        rollback_migrations = []
        for migration_info in reversed(applied_migrations):
            if migration_info.version > target_version:
                if migration_info.version in self.migrations:
                    rollback_migrations.append(self.migrations[migration_info.version])
                else:
                    logger.warning(f"Migration not found for rollback: {migration_info.version}")

        rolled_back = []

        for migration in rollback_migrations:
            logger.info(f"Rolling back migration: {migration.version} - {migration.name}")

            start_time = datetime.now()
            try:
                with self.db_manager.get_sync_session() as session:
                    migration.down(session)

                end_time = datetime.now()
                execution_time_ms = int((end_time - start_time).total_seconds() * 1000)

                # 移除迁移记录
                self.schema.remove_migration_record(migration.version)

                migration_info = MigrationInfo(
                    version=migration.version,
                    name=migration.name,
                    description=migration.description,
                    created_at=start_time,
                    applied_at=end_time,
                    execution_time_ms=execution_time_ms
                )

                rolled_back.append(migration_info)
                logger.info(f"Migration rolled back: {migration.version} ({execution_time_ms}ms)")

            except Exception as e:
                logger.error(f"Rollback failed: {migration.version} - {e}")
                raise DatabaseQueryException(f"Rollback {migration.version} failed: {e}")

        # 更新数据库版本
        if rolled_back:
            self.schema.set_current_version(target_version)

        return rolled_back

    def get_migration_status(self) -> Dict[str, Any]:
        """获取迁移状态"""
        applied_migrations = self.schema.get_applied_migrations()
        pending_migrations = self.get_pending_migrations()

        return {
            "current_version": self.schema.get_current_version(),
            "total_migrations": len(self.migrations),
            "applied_count": len(applied_migrations),
            "pending_count": len(pending_migrations),
            "applied_migrations": [m.to_dict() for m in applied_migrations],
            "pending_migrations": [
                {
                    "version": m.version,
                    "name": m.name,
                    "description": m.description,
                    "checksum": m.get_checksum()
                }
                for m in pending_migrations
            ]
        }


class DatabaseUtils:
    """数据库工具类"""

    def __init__(self, db_manager: DatabaseManager):
        self.db_manager = db_manager

    def backup_database(self, backup_path: Union[str, Path]) -> bool:
        """备份数据库"""
        try:
            # 使用mysqldump进行备份
            import subprocess

            config = self.db_manager.config
            cmd = [
                "mysqldump",
                f"--host={config.host}",
                f"--port={config.port}",
                f"--user={config.user}",
                f"--password={config.password}",
                "--single-transaction",
                "--routines",
                "--triggers",
                config.database
            ]

            backup_path = Path(backup_path)
            backup_path.parent.mkdir(parents=True, exist_ok=True)

            with open(backup_path, 'w') as f:
                result = subprocess.run(cmd, stdout=f, stderr=subprocess.PIPE, text=True)

            if result.returncode == 0:
                logger.info(f"Database backup created: {backup_path}")
                return True
            else:
                logger.error(f"Database backup failed: {result.stderr}")
                return False

        except Exception as e:
            logger.error(f"Database backup error: {e}")
            return False

    def restore_database(self, backup_path: Union[str, Path]) -> bool:
        """恢复数据库"""
        try:
            import subprocess

            config = self.db_manager.config
            cmd = [
                "mysql",
                f"--host={config.host}",
                f"--port={config.port}",
                f"--user={config.user}",
                f"--password={config.password}",
                config.database
            ]

            with open(backup_path, 'r') as f:
                result = subprocess.run(cmd, stdin=f, stderr=subprocess.PIPE, text=True)

            if result.returncode == 0:
                logger.info(f"Database restored from: {backup_path}")
                return True
            else:
                logger.error(f"Database restore failed: {result.stderr}")
                return False

        except Exception as e:
            logger.error(f"Database restore error: {e}")
            return False

    def get_table_info(self, table_name: str) -> Dict[str, Any]:
        """获取表信息"""
        with self.db_manager.get_sync_session() as session:
            # 获取表结构
            result = session.execute(text(f"DESCRIBE {table_name}")).fetchall()
            columns = []
            for row in result:
                columns.append({
                    "field": row[0],
                    "type": row[1],
                    "null": row[2],
                    "key": row[3],
                    "default": row[4],
                    "extra": row[5]
                })

            # 获取表状态
            result = session.execute(text(f"SHOW TABLE STATUS LIKE '{table_name}'")).fetchone()
            status = {
                "rows": result[4] if result else 0,
                "data_length": result[6] if result else 0,
                "index_length": result[8] if result else 0,
                "engine": result[1] if result else "Unknown"
            } if result else {}

            return {
                "table_name": table_name,
                "columns": columns,
                "status": status
            }

    def optimize_table(self, table_name: str) -> bool:
        """优化表"""
        try:
            with self.db_manager.get_sync_session() as session:
                session.execute(text(f"OPTIMIZE TABLE {table_name}"))
                logger.info(f"Table optimized: {table_name}")
                return True
        except Exception as e:
            logger.error(f"Table optimization failed for {table_name}: {e}")
            return False

    def analyze_table(self, table_name: str) -> bool:
        """分析表"""
        try:
            with self.db_manager.get_sync_session() as session:
                session.execute(text(f"ANALYZE TABLE {table_name}"))
                logger.info(f"Table analyzed: {table_name}")
                return True
        except Exception as e:
            logger.error(f"Table analysis failed for {table_name}: {e}")
            return False

    def get_database_size(self) -> Dict[str, Any]:
        """获取数据库大小信息"""
        with self.db_manager.get_sync_session() as session:
            result = session.execute(text("""
                SELECT
                    table_name,
                    ROUND(((data_length + index_length) / 1024 / 1024), 2) as size_mb,
                    table_rows
                FROM information_schema.tables
                WHERE table_schema = :database
                ORDER BY (data_length + index_length) DESC
            """), {"database": self.db_manager.config.database}).fetchall()

            tables = []
            total_size = 0
            total_rows = 0

            for row in result:
                table_info = {
                    "table_name": row[0],
                    "size_mb": float(row[1]),
                    "rows": int(row[2]) if row[2] else 0
                }
                tables.append(table_info)
                total_size += table_info["size_mb"]
                total_rows += table_info["rows"]

            return {
                "database": self.db_manager.config.database,
                "total_size_mb": round(total_size, 2),
                "total_rows": total_rows,
                "table_count": len(tables),
                "tables": tables
            }


# 预定义迁移示例
class InitialSchemaMigration(Migration):
    """初始Schema迁移"""

    def __init__(self):
        super().__init__("001", "initial_schema", "Create initial database schema")

    def up(self, session: Session) -> None:
        """创建初始表结构"""
        # 因子数据表
        session.execute(text("""
            CREATE TABLE IF NOT EXISTS factor_data (
                id BIGINT AUTO_INCREMENT PRIMARY KEY,
                factor_id VARCHAR(50) NOT NULL,
                stock_code VARCHAR(20) NOT NULL,
                trade_date DATE NOT NULL,
                factor_value DOUBLE,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                INDEX idx_factor_stock_date (factor_id, stock_code, trade_date),
                INDEX idx_trade_date (trade_date)
            ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4
        """))

        # 交易信号表
        session.execute(text("""
            CREATE TABLE IF NOT EXISTS trading_signals (
                id BIGINT AUTO_INCREMENT PRIMARY KEY,
                signal_id VARCHAR(50) NOT NULL UNIQUE,
                stock_code VARCHAR(20) NOT NULL,
                signal_type ENUM('buy', 'sell', 'hold') NOT NULL,
                target_price DECIMAL(10, 3) NOT NULL,
                target_quantity INT NOT NULL,
                status ENUM('pending', 'confirmed', 'rejected', 'executed', 'expired') NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                valid_until TIMESTAMP NOT NULL,
                INDEX idx_signal_status (status),
                INDEX idx_stock_code (stock_code),
                INDEX idx_created_at (created_at)
            ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4
        """))

        # 审计日志表
        session.execute(text("""
            CREATE TABLE IF NOT EXISTS audit_logs (
                id BIGINT AUTO_INCREMENT PRIMARY KEY,
                log_id VARCHAR(50) NOT NULL UNIQUE,
                event_type VARCHAR(50) NOT NULL,
                timestamp TIMESTAMP NOT NULL,
                severity ENUM('info', 'warning', 'error', 'critical') NOT NULL,
                message TEXT NOT NULL,
                details JSON,
                user_id VARCHAR(50),
                previous_hash VARCHAR(64),
                current_hash VARCHAR(64) NOT NULL,
                sequence_number INT NOT NULL,
                INDEX idx_event_type (event_type),
                INDEX idx_timestamp (timestamp),
                INDEX idx_user_id (user_id),
                INDEX idx_sequence (sequence_number)
            ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4
        """))

    def down(self, session: Session) -> None:
        """删除表结构"""
        session.execute(text("DROP TABLE IF EXISTS audit_logs"))
        session.execute(text("DROP TABLE IF EXISTS trading_signals"))
        session.execute(text("DROP TABLE IF EXISTS factor_data"))