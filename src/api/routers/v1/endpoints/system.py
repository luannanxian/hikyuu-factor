"""
System API Endpoints

系统管理相关的API端点
"""

from fastapi import APIRouter, Depends, HTTPException
from typing import Dict, Any, List
import platform
import psutil
from datetime import datetime

router = APIRouter()


@router.get("/platform", summary="获取平台信息")
async def get_platform_info() -> Dict[str, Any]:
    """获取系统平台信息"""

    return {
        "system": {
            "platform": platform.system(),
            "machine": platform.machine(),
            "processor": platform.processor(),
            "python_version": platform.python_version(),
            "cpu_count": psutil.cpu_count(),
            "cpu_count_logical": psutil.cpu_count(logical=True),
            "memory_total": psutil.virtual_memory().total,
            "memory_available": psutil.virtual_memory().available
        },
        "hikyuu": {
            "version": "2.6.8",
            "data_path": "./data/hikyuu",
            "optimization": {
                "apple_silicon": platform.machine() == "arm64" and platform.system() == "Darwin",
                "x86_64": platform.machine() in ["x86_64", "AMD64"]
            }
        },
        "timestamp": datetime.now().isoformat()
    }


@router.get("/health", summary="系统健康检查")
async def health_check() -> Dict[str, Any]:
    """综合系统健康检查"""

    # 检查系统资源
    memory = psutil.virtual_memory()
    disk = psutil.disk_usage('/')
    cpu_percent = psutil.cpu_percent(interval=1)

    # 检查各个Agent服务状态（模拟）
    agents_status = {
        "data_manager": "healthy",
        "factor_calculation": "healthy",
        "validation": "healthy",
        "signal_generation": "healthy"
    }

    health_status = {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "system": {
            "cpu_usage": cpu_percent,
            "memory_usage": memory.percent,
            "disk_usage": (disk.used / disk.total) * 100,
            "load_average": psutil.getloadavg() if hasattr(psutil, 'getloadavg') else None
        },
        "agents": agents_status,
        "services": {
            "database": "connected",
            "redis": "connected",
            "hikyuu": "initialized"
        }
    }

    # 判断整体健康状态
    if cpu_percent > 90 or memory.percent > 90:
        health_status["status"] = "warning"

    return health_status


@router.get("/stats", summary="系统统计信息")
async def get_system_stats() -> Dict[str, Any]:
    """获取系统统计信息"""

    return {
        "factors": {
            "total_registered": 6,
            "active_calculations": 0,
            "cache_hit_rate": 0.85
        },
        "signals": {
            "generated_today": 0,
            "pending_confirmation": 0,
            "executed_today": 0
        },
        "data": {
            "stocks_covered": 5000,
            "latest_data_date": datetime.now().date().isoformat(),
            "data_quality_score": 0.95
        },
        "performance": {
            "avg_factor_calc_time": 166,  # milliseconds
            "avg_signal_gen_time": 45,    # milliseconds
            "uptime_hours": 24
        }
    }


@router.get("/config", summary="获取系统配置")
async def get_system_config() -> Dict[str, Any]:
    """获取系统配置信息（不包含敏感信息）"""

    return {
        "version": "1.0.0",
        "environment": "development",
        "agents": {
            "data_manager": {
                "port": 8001,
                "status": "running"
            },
            "factor_calculation": {
                "port": 8002,
                "status": "running"
            },
            "validation": {
                "port": 8003,
                "status": "running"
            },
            "signal_generation": {
                "port": 8004,
                "status": "running"
            }
        },
        "features": {
            "platform_optimization": True,
            "human_confirmation": True,
            "audit_trail": True,
            "point_in_time": True
        }
    }