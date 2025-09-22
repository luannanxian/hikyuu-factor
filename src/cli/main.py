"""
Main CLI Entry Point

hikyuu-factor命令行工具主入口
"""

import click
from .factor_calculate import factor
from .signal_generate import signal
from .data_manager import data
from .db_migrate import db


@click.group()
@click.version_option(version='1.0.0')
def cli():
    """
    hikyuu-factor - A股全市场量化因子挖掘与决策支持系统

    基于Hikyuu框架的Agent架构设计
    """
    pass


# 添加子命令组
cli.add_command(factor)
cli.add_command(signal)
cli.add_command(data)
cli.add_command(db)


# 添加快捷命令别名
@cli.command()
@click.option('--factor-id', required=True, help='因子ID')
@click.option('--stocks', required=True, help='股票列表')
@click.pass_context
def calculate(ctx, factor_id: str, stocks: str):
    """快捷因子计算命令"""
    ctx.invoke(factor.commands['calculate'], factor_id=factor_id, stocks=stocks)


@cli.command()
@click.option('--strategy', required=True, help='策略名称')
@click.option('--confirm', is_flag=True, help='人工确认')
@click.pass_context
def generate_signal(ctx, strategy: str, confirm: bool):
    """快捷信号生成命令"""
    ctx.invoke(signal.commands['generate'], strategy=strategy, confirm=confirm)


@cli.command()
@click.option('--market', default='sh,sz', help='市场代码')
@click.pass_context
def data_update(ctx, market: str):
    """快捷数据更新命令"""
    ctx.invoke(data.commands['update'], market=market)


if __name__ == '__main__':
    cli()