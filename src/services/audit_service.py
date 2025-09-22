"""
Audit Logging Service with Hash Chains
审计日志服务，实现基于哈希链的不可变审计记录
"""

import asyncio
import logging
from datetime import datetime
from typing import Dict, List, Optional, Any, Union, Callable
from dataclasses import dataclass
import threading
import json
from pathlib import Path

from ..models.audit_log import AuditLog, AuditLogChain, AuditLogger, AuditEventType, AuditSeverity, AuditContext
from ..models.database import get_db_session
from ..lib.exceptions import DatabaseQueryException

logger = logging.getLogger(__name__)


@dataclass
class AuditServiceConfig:
    """审计服务配置"""
    enable_database_storage: bool = True
    enable_file_storage: bool = True
    file_storage_path: str = "logs/audit"
    auto_flush_interval: int = 60  # 秒
    max_memory_buffer: int = 1000  # 最大内存缓冲区大小
    enable_chain_verification: bool = True
    verification_interval: int = 300  # 链验证间隔（秒）
    backup_enabled: bool = True
    backup_interval: int = 3600  # 备份间隔（秒）


class AuditLogStorage:
    """审计日志存储接口"""

    async def store_log(self, log: AuditLog) -> bool:
        """存储审计日志"""
        raise NotImplementedError

    async def get_logs(self, start_time: Optional[datetime] = None,
                       end_time: Optional[datetime] = None,
                       event_types: Optional[List[AuditEventType]] = None,
                       user_id: Optional[str] = None,
                       limit: int = 1000) -> List[AuditLog]:
        """获取审计日志"""
        raise NotImplementedError

    async def verify_chain(self) -> bool:
        """验证审计链完整性"""
        raise NotImplementedError


class DatabaseAuditStorage(AuditLogStorage):
    """数据库审计日志存储"""

    async def store_log(self, log: AuditLog) -> bool:
        """存储审计日志到数据库"""
        try:
            with get_db_session() as session:
                session.execute("""
                    INSERT INTO audit_logs (
                        log_id, event_type, timestamp, severity, message, details,
                        user_id, previous_hash, current_hash, sequence_number
                    ) VALUES (
                        :log_id, :event_type, :timestamp, :severity, :message, :details,
                        :user_id, :previous_hash, :current_hash, :sequence_number
                    )
                """, {
                    "log_id": log.log_id,
                    "event_type": log.event_type.value,
                    "timestamp": log.timestamp,
                    "severity": log.severity.value,
                    "message": log.message,
                    "details": json.dumps(log.details) if log.details else None,
                    "user_id": log.context.user_id if log.context else None,
                    "previous_hash": log.previous_hash,
                    "current_hash": log.current_hash,
                    "sequence_number": log.sequence_number
                })
                return True

        except Exception as e:
            logger.error(f"Failed to store audit log to database: {e}")
            return False

    async def get_logs(self, start_time: Optional[datetime] = None,
                       end_time: Optional[datetime] = None,
                       event_types: Optional[List[AuditEventType]] = None,
                       user_id: Optional[str] = None,
                       limit: int = 1000) -> List[AuditLog]:
        """从数据库获取审计日志"""
        try:
            conditions = []
            params = {"limit": limit}

            if start_time:
                conditions.append("timestamp >= :start_time")
                params["start_time"] = start_time

            if end_time:
                conditions.append("timestamp <= :end_time")
                params["end_time"] = end_time

            if event_types:
                event_type_values = [et.value for et in event_types]
                conditions.append(f"event_type IN ({','.join([f':event_type_{i}' for i in range(len(event_type_values))])})")
                for i, event_type in enumerate(event_type_values):
                    params[f"event_type_{i}"] = event_type

            if user_id:
                conditions.append("user_id = :user_id")
                params["user_id"] = user_id

            where_clause = " AND ".join(conditions) if conditions else "1=1"

            with get_db_session() as session:
                result = session.execute(f"""
                    SELECT log_id, event_type, timestamp, severity, message, details,
                           user_id, previous_hash, current_hash, sequence_number
                    FROM audit_logs
                    WHERE {where_clause}
                    ORDER BY sequence_number
                    LIMIT :limit
                """, params).fetchall()

                logs = []
                for row in result:
                    context = AuditContext(user_id=row[6]) if row[6] else None
                    details = json.loads(row[5]) if row[5] else {}

                    log = AuditLog(
                        log_id=row[0],
                        event_type=AuditEventType(row[1]),
                        timestamp=row[2],
                        severity=AuditSeverity(row[3]),
                        message=row[4],
                        details=details,
                        context=context,
                        previous_hash=row[7],
                        current_hash=row[8],
                        sequence_number=row[9]
                    )
                    logs.append(log)

                return logs

        except Exception as e:
            logger.error(f"Failed to get audit logs from database: {e}")
            return []

    async def verify_chain(self) -> bool:
        """验证数据库中的审计链完整性"""
        try:
            with get_db_session() as session:
                result = session.execute("""
                    SELECT log_id, previous_hash, current_hash, sequence_number
                    FROM audit_logs
                    ORDER BY sequence_number
                """).fetchall()

                if not result:
                    return True  # 空链是有效的

                previous_hash = None
                for row in result:
                    expected_previous_hash = previous_hash
                    actual_previous_hash = row[1]

                    if expected_previous_hash != actual_previous_hash:
                        logger.error(f"Chain integrity violation at sequence {row[3]}: "
                                   f"expected previous hash {expected_previous_hash}, "
                                   f"got {actual_previous_hash}")
                        return False

                    previous_hash = row[2]

                return True

        except Exception as e:
            logger.error(f"Failed to verify audit chain in database: {e}")
            return False


class FileAuditStorage(AuditLogStorage):
    """文件审计日志存储"""

    def __init__(self, storage_path: str):
        self.storage_path = Path(storage_path)
        self.storage_path.mkdir(parents=True, exist_ok=True)

    async def store_log(self, log: AuditLog) -> bool:
        """存储审计日志到文件"""
        try:
            # 按日期创建文件
            date_str = log.timestamp.strftime("%Y-%m-%d")
            file_path = self.storage_path / f"audit_{date_str}.jsonl"

            # 追加写入JSON Lines格式
            with open(file_path, 'a', encoding='utf-8') as f:
                f.write(log.to_json() + '\n')

            return True

        except Exception as e:
            logger.error(f"Failed to store audit log to file: {e}")
            return False

    async def get_logs(self, start_time: Optional[datetime] = None,
                       end_time: Optional[datetime] = None,
                       event_types: Optional[List[AuditEventType]] = None,
                       user_id: Optional[str] = None,
                       limit: int = 1000) -> List[AuditLog]:
        """从文件获取审计日志"""
        logs = []
        try:
            # 确定需要读取的文件
            files_to_read = []
            if start_time and end_time:
                current_date = start_time.date()
                while current_date <= end_time.date():
                    file_path = self.storage_path / f"audit_{current_date.strftime('%Y-%m-%d')}.jsonl"
                    if file_path.exists():
                        files_to_read.append(file_path)
                    current_date += timedelta(days=1)
            else:
                # 读取所有文件
                files_to_read = list(self.storage_path.glob("audit_*.jsonl"))

            # 读取日志
            for file_path in sorted(files_to_read):
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        for line in f:
                            if len(logs) >= limit:
                                break

                            try:
                                log = AuditLog.from_json(line.strip())

                                # 应用过滤条件
                                if start_time and log.timestamp < start_time:
                                    continue
                                if end_time and log.timestamp > end_time:
                                    continue
                                if event_types and log.event_type not in event_types:
                                    continue
                                if user_id and (not log.context or log.context.user_id != user_id):
                                    continue

                                logs.append(log)

                            except Exception as e:
                                logger.warning(f"Failed to parse audit log line: {e}")

                except Exception as e:
                    logger.warning(f"Failed to read audit file {file_path}: {e}")

                if len(logs) >= limit:
                    break

            return logs

        except Exception as e:
            logger.error(f"Failed to get audit logs from files: {e}")
            return []

    async def verify_chain(self) -> bool:
        """验证文件中的审计链完整性"""
        try:
            all_logs = await self.get_logs(limit=10000)  # 获取所有日志
            if not all_logs:
                return True

            # 按序列号排序
            all_logs.sort(key=lambda x: x.sequence_number)

            previous_hash = None
            for log in all_logs:
                if log.previous_hash != previous_hash:
                    logger.error(f"File chain integrity violation at sequence {log.sequence_number}")
                    return False
                previous_hash = log.current_hash

            return True

        except Exception as e:
            logger.error(f"Failed to verify audit chain in files: {e}")
            return False


class AuditService:
    """审计服务主类"""

    def __init__(self, config: AuditServiceConfig):
        self.config = config
        self.chain = AuditLogChain()
        self.logger = AuditLogger(self.chain)
        self.storages: List[AuditLogStorage] = []
        self.memory_buffer: List[AuditLog] = []
        self.buffer_lock = threading.Lock()
        self._running = False
        self._flush_task = None
        self._verification_task = None
        self._backup_task = None

        # 初始化存储
        self._initialize_storages()

    def _initialize_storages(self) -> None:
        """初始化存储后端"""
        if self.config.enable_database_storage:
            self.storages.append(DatabaseAuditStorage())

        if self.config.enable_file_storage:
            self.storages.append(FileAuditStorage(self.config.file_storage_path))

    async def start(self) -> None:
        """启动审计服务"""
        if self._running:
            return

        self._running = True

        # 启动后台任务
        self._flush_task = asyncio.create_task(self._auto_flush_loop())

        if self.config.enable_chain_verification:
            self._verification_task = asyncio.create_task(self._verification_loop())

        if self.config.backup_enabled:
            self._backup_task = asyncio.create_task(self._backup_loop())

        logger.info("Audit service started")

    async def stop(self) -> None:
        """停止审计服务"""
        if not self._running:
            return

        self._running = False

        # 取消后台任务
        if self._flush_task:
            self._flush_task.cancel()
        if self._verification_task:
            self._verification_task.cancel()
        if self._backup_task:
            self._backup_task.cancel()

        # 刷新剩余的日志
        await self.flush_logs()

        logger.info("Audit service stopped")

    async def log_event(self, event_type: AuditEventType, message: str,
                        severity: AuditSeverity = AuditSeverity.INFO,
                        details: Optional[Dict[str, Any]] = None,
                        context: Optional[AuditContext] = None) -> AuditLog:
        """记录审计事件"""
        log = self.logger.chain.create_log(event_type, message, severity, details, context)

        # 添加到内存缓冲区
        with self.buffer_lock:
            self.memory_buffer.append(log)

            # 如果缓冲区满了，立即刷新
            if len(self.memory_buffer) >= self.config.max_memory_buffer:
                asyncio.create_task(self.flush_logs())

        return log

    async def flush_logs(self) -> None:
        """刷新日志到存储"""
        with self.buffer_lock:
            if not self.memory_buffer:
                return

            logs_to_flush = self.memory_buffer.copy()
            self.memory_buffer.clear()

        # 并行写入到所有存储
        tasks = []
        for storage in self.storages:
            for log in logs_to_flush:
                tasks.append(storage.store_log(log))

        if tasks:
            results = await asyncio.gather(*tasks, return_exceptions=True)
            errors = [r for r in results if isinstance(r, Exception)]
            if errors:
                logger.error(f"Failed to store some audit logs: {len(errors)} errors")

        logger.debug(f"Flushed {len(logs_to_flush)} audit logs")

    async def get_logs(self, start_time: Optional[datetime] = None,
                       end_time: Optional[datetime] = None,
                       event_types: Optional[List[AuditEventType]] = None,
                       user_id: Optional[str] = None,
                       limit: int = 1000) -> List[AuditLog]:
        """获取审计日志"""
        # 优先从数据库获取
        for storage in self.storages:
            if isinstance(storage, DatabaseAuditStorage):
                return await storage.get_logs(start_time, end_time, event_types, user_id, limit)

        # 如果没有数据库存储，从文件获取
        for storage in self.storages:
            if isinstance(storage, FileAuditStorage):
                return await storage.get_logs(start_time, end_time, event_types, user_id, limit)

        return []

    async def verify_integrity(self) -> Dict[str, bool]:
        """验证审计链完整性"""
        results = {}

        # 验证内存链
        results["memory_chain"] = self.chain.verify_chain()

        # 验证存储链
        for i, storage in enumerate(self.storages):
            storage_name = f"storage_{i}_{type(storage).__name__}"
            results[storage_name] = await storage.verify_chain()

        return results

    async def search_logs(self, query: str, limit: int = 100) -> List[AuditLog]:
        """搜索审计日志"""
        all_logs = await self.get_logs(limit=limit * 2)  # 获取更多日志进行搜索
        return self.chain.search_logs(query)[:limit]

    async def get_statistics(self) -> Dict[str, Any]:
        """获取审计统计信息"""
        try:
            recent_logs = await self.get_logs(limit=10000)

            # 统计事件类型分布
            event_type_counts = {}
            severity_counts = {}
            user_activity = {}

            for log in recent_logs:
                # 事件类型统计
                event_type = log.event_type.value
                event_type_counts[event_type] = event_type_counts.get(event_type, 0) + 1

                # 严重性统计
                severity = log.severity.value
                severity_counts[severity] = severity_counts.get(severity, 0) + 1

                # 用户活动统计
                if log.context and log.context.user_id:
                    user_id = log.context.user_id
                    user_activity[user_id] = user_activity.get(user_id, 0) + 1

            # 链完整性验证
            integrity_results = await self.verify_integrity()

            return {
                "total_logs": len(recent_logs),
                "event_type_distribution": event_type_counts,
                "severity_distribution": severity_counts,
                "top_users": sorted(user_activity.items(), key=lambda x: x[1], reverse=True)[:10],
                "chain_integrity": integrity_results,
                "memory_buffer_size": len(self.memory_buffer),
                "storage_backends": len(self.storages)
            }

        except Exception as e:
            logger.error(f"Failed to get audit statistics: {e}")
            return {"error": str(e)}

    async def _auto_flush_loop(self) -> None:
        """自动刷新循环"""
        while self._running:
            try:
                await asyncio.sleep(self.config.auto_flush_interval)
                await self.flush_logs()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Auto flush error: {e}")

    async def _verification_loop(self) -> None:
        """定期验证循环"""
        while self._running:
            try:
                await asyncio.sleep(self.config.verification_interval)
                integrity_results = await self.verify_integrity()

                # 记录验证结果
                all_valid = all(integrity_results.values())
                if not all_valid:
                    await self.log_event(
                        AuditEventType.SECURITY_ALERT,
                        "Audit chain integrity verification failed",
                        AuditSeverity.CRITICAL,
                        {"integrity_results": integrity_results}
                    )

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Verification loop error: {e}")

    async def _backup_loop(self) -> None:
        """定期备份循环"""
        while self._running:
            try:
                await asyncio.sleep(self.config.backup_interval)
                await self._create_backup()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Backup loop error: {e}")

    async def _create_backup(self) -> None:
        """创建审计日志备份"""
        try:
            backup_path = Path(self.config.file_storage_path) / "backups"
            backup_path.mkdir(parents=True, exist_ok=True)

            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            backup_file = backup_path / f"audit_backup_{timestamp}.json"

            # 导出审计链
            self.chain.export_chain(str(backup_file))

            logger.info(f"Audit backup created: {backup_file}")

        except Exception as e:
            logger.error(f"Failed to create audit backup: {e}")


# 全局审计服务实例
audit_service: Optional[AuditService] = None


def get_audit_service() -> Optional[AuditService]:
    """获取全局审计服务实例"""
    return audit_service


def initialize_audit_service(config: Optional[AuditServiceConfig] = None) -> AuditService:
    """初始化全局审计服务"""
    global audit_service

    if config is None:
        config = AuditServiceConfig()

    audit_service = AuditService(config)
    return audit_service


# 便捷函数
async def audit_log(event_type: AuditEventType, message: str,
                   severity: AuditSeverity = AuditSeverity.INFO,
                   details: Optional[Dict[str, Any]] = None,
                   context: Optional[AuditContext] = None) -> Optional[AuditLog]:
    """记录审计日志的便捷函数"""
    service = get_audit_service()
    if service:
        return await service.log_event(event_type, message, severity, details, context)
    return None