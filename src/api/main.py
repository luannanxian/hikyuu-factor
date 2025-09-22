"""
API Main Application

FastAPI主应用程序，提供统一的REST API入口
"""

from fastapi import FastAPI, Request, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import JSONResponse
import time
import logging
from typing import Any

from .routers import v1
from .middleware import LoggingMiddleware, AuthMiddleware
from .exceptions import APIException, api_exception_handler


def create_app() -> FastAPI:
    """创建FastAPI应用程序"""

    app = FastAPI(
        title="Hikyuu Factor API",
        description="A股全市场量化因子挖掘与决策支持系统 REST API",
        version="1.0.0",
        docs_url="/docs",
        redoc_url="/redoc",
        openapi_url="/openapi.json"
    )

    # 配置CORS
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],  # 生产环境应该限制域名
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # 配置压缩
    app.add_middleware(GZipMiddleware, minimum_size=1000)

    # 配置自定义中间件
    app.add_middleware(LoggingMiddleware)
    app.add_middleware(AuthMiddleware)

    # 配置异常处理器
    app.add_exception_handler(APIException, api_exception_handler)

    # 注册路由
    app.include_router(v1.router, prefix="/api/v1")

    # 根路径
    @app.get("/")
    async def root():
        """API根路径"""
        return {
            "name": "Hikyuu Factor API",
            "version": "1.0.0",
            "description": "A股全市场量化因子挖掘与决策支持系统",
            "docs_url": "/docs",
            "redoc_url": "/redoc"
        }

    # 健康检查
    @app.get("/health")
    async def health_check():
        """健康检查接口"""
        return {
            "status": "healthy",
            "timestamp": time.time(),
            "version": "1.0.0"
        }

    return app


# 创建应用实例
app = create_app()


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)