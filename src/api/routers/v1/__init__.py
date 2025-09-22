"""
API v1 Router

API版本1路由定义
"""

from fastapi import APIRouter
from .endpoints import system, factors, signals, data, validation

router = APIRouter()

# 包含各个模块的路由
router.include_router(system.router, prefix="/system", tags=["system"])
router.include_router(factors.router, prefix="/factors", tags=["factors"])
router.include_router(signals.router, prefix="/signals", tags=["signals"])
router.include_router(data.router, prefix="/data", tags=["data"])
router.include_router(validation.router, prefix="/validation", tags=["validation"])