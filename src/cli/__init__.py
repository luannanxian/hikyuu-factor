"""
命令行界面模块

提供因子计算、信号生成、数据管理等功能的CLI工具。

Available Commands:
- factor: 因子计算相关命令
- signal: 交易信号相关命令
- data: 数据管理相关命令
- db: 数据库管理相关命令

Usage:
    python -m src.cli.main --help
"""

from .main import cli

__all__ = ['cli']