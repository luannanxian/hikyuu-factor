"""
API Middleware

API中间件定义
"""

from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware
import time
import logging
import uuid


class LoggingMiddleware(BaseHTTPMiddleware):
    """请求日志中间件"""

    async def dispatch(self, request: Request, call_next):
        # 生成请求ID
        request_id = str(uuid.uuid4())
        request.state.request_id = request_id

        # 记录请求开始
        start_time = time.time()
        logger = logging.getLogger("api.requests")

        logger.info(
            f"Request started: {request.method} {request.url}",
            extra={
                "request_id": request_id,
                "method": request.method,
                "url": str(request.url),
                "user_agent": request.headers.get("user-agent"),
                "client_ip": request.client.host if request.client else None
            }
        )

        # 处理请求
        response = await call_next(request)

        # 记录请求完成
        process_time = time.time() - start_time
        logger.info(
            f"Request completed: {request.method} {request.url} - {response.status_code}",
            extra={
                "request_id": request_id,
                "method": request.method,
                "url": str(request.url),
                "status_code": response.status_code,
                "process_time": process_time
            }
        )

        # 添加响应头
        response.headers["X-Request-ID"] = request_id
        response.headers["X-Process-Time"] = str(process_time)

        return response


class AuthMiddleware(BaseHTTPMiddleware):
    """认证中间件"""

    async def dispatch(self, request: Request, call_next):
        # 跳过不需要认证的路径
        public_paths = ["/", "/health", "/docs", "/redoc", "/openapi.json"]

        if request.url.path in public_paths:
            return await call_next(request)

        # 这里可以添加实际的认证逻辑
        # 当前为演示模式，跳过认证
        return await call_next(request)