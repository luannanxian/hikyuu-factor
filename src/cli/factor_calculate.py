"""
Factor Calculate CLI

因子计算命令行工具
支持计算指定因子在指定股票列表上的值
"""

import click
import asyncio
import pandas as pd
from datetime import datetime, date
from typing import List, Optional
import sys
import os

# 添加项目根目录到路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from agents.factor_calculation_agent import FactorCalculationAgent
from services.factor_calculation_service import FactorCalculationService


@click.command()
@click.option('--factor-id', required=True, help='因子ID (如: momentum_20d)')
@click.option('--stocks', required=True, help='股票代码列表，用逗号分隔 (如: sh000001,sz000002)')
@click.option('--start-date', default=None, help='开始日期 (YYYY-MM-DD)')
@click.option('--end-date', default=None, help='结束日期 (YYYY-MM-DD)')
@click.option('--output', '-o', default=None, help='输出文件路径 (可选)')
@click.option('--format', 'output_format', default='csv', type=click.Choice(['csv', 'json', 'parquet']))
@click.option('--verbose', '-v', is_flag=True, help='详细输出')
def calculate(factor_id: str, stocks: str, start_date: Optional[str],
              end_date: Optional[str], output: Optional[str],
              output_format: str, verbose: bool):
    """计算指定因子的值"""

    if verbose:
        click.echo(f"开始计算因子: {factor_id}")

    # 解析股票列表
    stock_list = [s.strip() for s in stocks.split(',')]

    # 解析日期范围
    date_range = {}
    if start_date:
        date_range['start_date'] = start_date
    if end_date:
        date_range['end_date'] = end_date
    else:
        date_range['end_date'] = datetime.now().strftime('%Y-%m-%d')

    try:
        # 初始化因子计算服务
        service = FactorCalculationService()

        if verbose:
            click.echo(f"股票列表: {stock_list}")
            click.echo(f"日期范围: {date_range}")

        # 计算因子
        results = asyncio.run(service.calculate_factor_batch(
            factor_ids=[factor_id],
            stock_list=stock_list,
            date_range=date_range
        ))

        if factor_id not in results:
            click.echo(f"Error: 因子 {factor_id} 计算失败", err=True)
            return

        factor_data = results[factor_id]

        if verbose:
            click.echo(f"计算完成，共 {len(factor_data)} 条记录")

        # 输出结果
        if output:
            # 保存到文件
            if output_format == 'csv':
                factor_data.to_csv(output, index=False)
            elif output_format == 'json':
                factor_data.to_json(output, orient='records', date_format='iso')
            elif output_format == 'parquet':
                factor_data.to_parquet(output, index=False)

            click.echo(f"结果已保存到: {output}")
        else:
            # 打印到控制台
            click.echo("计算结果:")
            click.echo(factor_data.to_string(index=False))

    except Exception as e:
        click.echo(f"Error: 计算失败: {e}", err=True)
        sys.exit(1)


@click.command()
@click.option('--list-all', is_flag=True, help='列出所有可用因子')
@click.option('--category', default=None, help='按类别筛选因子 (momentum, value, quality, volatility)')
@click.option('--search', default=None, help='搜索因子名称')
def list_factors(list_all: bool, category: Optional[str], search: Optional[str]):
    """列出可用的因子"""

    try:
        service = FactorCalculationService()
        factors = service.list_factors()

        # 应用筛选
        if category:
            factors = [f for f in factors if f.get('category') == category]

        if search:
            factors = [f for f in factors if search.lower() in f.get('name', '').lower()]

        if not factors:
            click.echo("没有找到匹配的因子")
            return

        click.echo(f"可用因子 ({len(factors)}个):")
        click.echo("=" * 80)

        for factor in factors:
            click.echo(f"ID: {factor.get('factor_id')}")
            click.echo(f"名称: {factor.get('name')}")
            click.echo(f"类别: {factor.get('category')}")
            click.echo(f"描述: {factor.get('description', 'N/A')}")
            click.echo("-" * 40)

    except Exception as e:
        click.echo(f"Error: 获取因子列表失败: {e}", err=True)
        sys.exit(1)


@click.group()
def factor():
    """因子计算相关命令"""
    pass


factor.add_command(calculate)
factor.add_command(list_factors, name='list')


if __name__ == '__main__':
    factor()