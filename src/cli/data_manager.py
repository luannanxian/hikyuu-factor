"""
Data Management CLI

数据管理命令行工具
支持数据更新、质量检查等功能
"""

import click
import asyncio
from datetime import datetime
from typing import List, Optional
import sys
import os

# 添加项目根目录到路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from agents.data_manager_agent import DataManagerAgent
from services.data_manager_service import DataManagerService


@click.command()
@click.option('--market', default='sh,sz', help='市场代码 (sh,sz,bj)')
@click.option('--start-date', default=None, help='开始日期 (YYYY-MM-DD)')
@click.option('--end-date', default=None, help='结束日期 (YYYY-MM-DD)')
@click.option('--force', is_flag=True, help='强制更新已存在的数据')
@click.option('--verbose', '-v', is_flag=True, help='详细输出')
def update(market: str, start_date: Optional[str], end_date: Optional[str],
           force: bool, verbose: bool):
    """更新股票数据"""

    if verbose:
        click.echo(f"开始更新股票数据: {market}")

    markets = [m.strip() for m in market.split(',')]

    try:
        # 初始化数据管理服务
        service = DataManagerService()

        if verbose:
            click.echo(f"目标市场: {markets}")
            click.echo(f"日期范围: {start_date} 到 {end_date}")
            click.echo(f"强制更新: {'是' if force else '否'}")

        # 执行数据更新
        config = {
            'markets': markets,
            'start_date': start_date,
            'end_date': end_date,
            'force_update': force
        }

        result = asyncio.run(service.update_market_data(config))

        if result.get('success'):
            click.echo(f"数据更新完成")
            if verbose:
                click.echo(f"更新记录数: {result.get('updated_count', 0)}")
                click.echo(f"处理时间: {result.get('processing_time', 0):.2f}秒")
        else:
            click.echo(f"Error: 数据更新失败: {result.get('error')}", err=True)

    except Exception as e:
        click.echo(f"Error: 数据更新失败: {e}", err=True)
        sys.exit(1)


@click.command()
@click.option('--market', default='sh,sz', help='市场代码 (sh,sz,bj)')
@click.option('--check-type', default='all',
              type=click.Choice(['all', 'missing', 'duplicate', 'outlier']))
@click.option('--output', '-o', default=None, help='输出报告文件路径')
@click.option('--verbose', '-v', is_flag=True, help='详细输出')
def quality_check(market: str, check_type: str, output: Optional[str], verbose: bool):
    """执行数据质量检查"""

    if verbose:
        click.echo(f"开始数据质量检查: {market}")

    markets = [m.strip() for m in market.split(',')]

    try:
        # 初始化数据管理服务
        service = DataManagerService()

        if verbose:
            click.echo(f"检查市场: {markets}")
            click.echo(f"检查类型: {check_type}")

        # 执行质量检查
        config = {
            'markets': markets,
            'check_types': [check_type] if check_type != 'all' else ['missing', 'duplicate', 'outlier']
        }

        result = asyncio.run(service.check_data_quality(config))

        if result.get('success'):
            click.echo("数据质量检查完成")

            # 显示检查结果
            for check in result.get('checks', []):
                click.echo(f"检查项: {check.get('check_type')}")
                click.echo(f"状态: {check.get('status')}")
                if check.get('issues'):
                    click.echo(f"问题数量: {len(check.get('issues'))}")
                click.echo("-" * 40)

            # 保存报告
            if output:
                import json
                with open(output, 'w', encoding='utf-8') as f:
                    json.dump(result, f, ensure_ascii=False, indent=2, default=str)
                click.echo(f"报告已保存到: {output}")

        else:
            click.echo(f"Error: 数据质量检查失败: {result.get('error')}", err=True)

    except Exception as e:
        click.echo(f"Error: 数据质量检查失败: {e}", err=True)
        sys.exit(1)


@click.command()
@click.option('--market', default='sh,sz', help='市场代码 (sh,sz,bj)')
@click.option('--detailed', is_flag=True, help='显示详细状态')
def status(market: str, detailed: bool):
    """查看数据状态"""

    markets = [m.strip() for m in market.split(',')]

    try:
        # 初始化数据管理服务
        service = DataManagerService()

        # 获取数据状态
        result = service.get_data_status({'markets': markets})

        click.echo("数据状态:")
        click.echo("=" * 50)

        for market_info in result.get('market_status', []):
            click.echo(f"市场: {market_info.get('market')}")
            click.echo(f"最新数据日期: {market_info.get('latest_date')}")
            click.echo(f"股票数量: {market_info.get('stock_count')}")

            if detailed:
                click.echo(f"数据完整性: {market_info.get('completeness', 0):.1%}")
                click.echo(f"最后更新: {market_info.get('last_update')}")

            click.echo("-" * 30)

    except Exception as e:
        click.echo(f"Error: 获取数据状态失败: {e}", err=True)
        sys.exit(1)


@click.group()
def data():
    """数据管理相关命令"""
    pass


data.add_command(update)
data.add_command(quality_check, name='check')
data.add_command(status)


if __name__ == '__main__':
    data()