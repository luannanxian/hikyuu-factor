"""
RESTful API接口模块

FastAPI实现的Agent API接口，支持系统管理、数据管理、因子计算、验证和信号生成。

API版本: v1
文档地址: /docs
健康检查: /health

主要端点:
- /api/v1/system/* - 系统管理
- /api/v1/factors/* - 因子管理
- /api/v1/signals/* - 交易信号
- /api/v1/data/* - 数据管理
- /api/v1/validation/* - 验证管理
"""

from .main import app

__all__ = ['app']