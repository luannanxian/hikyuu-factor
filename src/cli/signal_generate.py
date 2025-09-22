"""
Signal Generate CLI

信号生成命令行工具
支持生成交易信号并进行人工确认
"""

import click
import asyncio
import json
from datetime import datetime
from typing import List, Optional, Dict, Any
import sys
import os

# 添加项目根目录到路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from agents.signal_generation_agent import SignalGenerationAgent
from services.signal_generation_service import SignalGenerationService


@click.command()
@click.option('--strategy', required=True, help='策略名称 (如: momentum_v1)')
@click.option('--stocks', default=None, help='股票代码列表，用逗号分隔 (默认: 全市场)')
@click.option('--confirm', is_flag=True, help='启用人工确认机制')
@click.option('--output', '-o', default=None, help='输出文件路径 (可选)')
@click.option('--format', 'output_format', default='json', type=click.Choice(['json', 'csv']))
@click.option('--verbose', '-v', is_flag=True, help='详细输出')
def generate(strategy: str, stocks: Optional[str], confirm: bool,
             output: Optional[str], output_format: str, verbose: bool):
    """生成交易信号"""

    if verbose:
        click.echo(f"开始生成交易信号: {strategy}")

    # 解析股票列表
    stock_list = None
    if stocks:
        stock_list = [s.strip() for s in stocks.split(',')]

    try:
        # 初始化信号生成服务
        service = SignalGenerationService()

        # 获取因子数据 (从因子计算服务)
        factor_data = _get_factor_data_from_service(stock_list or ["sh000001", "sz000002"])

        # 配置参数
        signal_config = {
            'strategy_config': {
                'strategy_name': strategy,
                'factor_weights': {
                    'momentum_20d': 0.4,
                    'rsi_14d': 0.3,
                    'pe_ratio': 0.3
                },
                'signal_threshold': {
                    'buy': 0.7,
                    'sell': 0.3,
                    'hold': [0.3, 0.7]
                }
            }
        }

        risk_config = {
            'position_risk': {
                'max_single_position': 0.1,
                'max_sector_exposure': 0.3
            },
            'liquidity_risk': {
                'min_avg_volume': 1000000
            }
        }

        if verbose:
            click.echo(f"股票范围: {stock_list or '全市场'}")
            click.echo(f"人工确认: {'启用' if confirm else '禁用'}")

        # 生成信号
        results = service.generate_trading_signals(
            factor_data=factor_data,
            signal_config=signal_config,
            risk_config=risk_config,
            user_id="cli_user",
            require_confirmation=confirm
        )

        if verbose:
            click.echo(f"信号生成完成")
            click.echo(f"生成信号数量: {results['signal_count']}")
            click.echo(f"风险评分: {results['risk_score']:.3f}")
            click.echo(f"确认状态: {results['confirmation_status']}")

        # 处理确认
        if confirm and results['confirmation_status'] == 'approved':
            confirmation = click.confirm("确认执行这些交易信号吗?")
            if not confirmation:
                click.echo("信号生成已取消")
                return

        # 输出结果
        if output:
            # 保存到文件
            if output_format == 'json':
                with open(output, 'w', encoding='utf-8') as f:
                    json.dump(results, f, ensure_ascii=False, indent=2, default=str)
            elif output_format == 'csv':
                # 将信号转换为DataFrame并保存
                signals_data = []
                for signal in results['signals']:
                    signals_data.append({
                        'signal_id': signal.signal_id,
                        'stock_code': signal.stock_code,
                        'signal_type': signal.signal_type.value,
                        'signal_strength': signal.signal_strength,
                        'generation_date': signal.generation_date,
                        'strategy_name': signal.strategy_name
                    })

                import pandas as pd
                df = pd.DataFrame(signals_data)
                df.to_csv(output, index=False)

            click.echo(f"结果已保存到: {output}")
        else:
            # 打印到控制台
            click.echo("生成的交易信号:")
            click.echo("=" * 60)
            for signal in results['signals']:
                click.echo(f"股票: {signal.stock_code}")
                click.echo(f"信号: {signal.signal_type.value}")
                click.echo(f"强度: {signal.signal_strength:.3f}")
                click.echo(f"策略: {signal.strategy_name}")
                click.echo("-" * 30)

    except Exception as e:
        click.echo(f"Error: 信号生成失败: {e}", err=True)
        sys.exit(1)


def _get_factor_data_from_service(stock_list: List[str]) -> Dict[str, Any]:
    """从因子计算服务获取真实因子数据"""
    import sys
    import os

    # 添加项目根目录到路径
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

    from lib.environment import env_manager
    from services.factor_calculation_service import FactorCalculationService

    # 环境检查
    if not env_manager.is_mock_data_allowed():
        # 非开发环境，必须使用真实数据
        service = FactorCalculationService()
        factors = ['momentum_20d', 'rsi_14d', 'pe_ratio']

        # 调用真实的因子计算服务
        # 注意: 这里需要实现真实的因子数据获取逻辑
        raise NotImplementedError(
            env_manager.get_real_data_requirement_message() +
            " Factor calculation service integration required."
        )
    else:
        # 开发环境，允许使用模拟数据（但需要明确标注）
        return _development_only_mock_factor_data(stock_list)


def _development_only_mock_factor_data(stock_list: List[str]) -> Dict[str, Any]:
    """开发环境专用的模拟因子数据 - 仅用于开发调试"""
    import pandas as pd
    import numpy as np
    from lib.environment import warn_mock_data

    # MOCK DATA WARNING
    warn_mock_data("Signal generation CLI using mock factor data")

    mock_data = {}  # MOCK DATA
    factors = ['momentum_20d', 'rsi_14d', 'pe_ratio']

    for factor in factors:
        data = []
        for stock in stock_list:
            data.append({
                'stock_code': stock,
                'date': datetime.now().date(),
                'factor_value': np.random.random(),  # MOCK DATA
                'factor_score': np.random.random()   # MOCK DATA
            })
        mock_data[factor] = pd.DataFrame(data)  # MOCK DATA

    return mock_data  # MOCK DATA


@click.command()
@click.option('--strategy', default=None, help='策略名称筛选')
@click.option('--status', default=None, help='状态筛选 (pending, approved, rejected)')
def list_signals(strategy: Optional[str], status: Optional[str]):
    """列出历史交易信号"""

    click.echo("历史交易信号:")
    click.echo("=" * 60)
    click.echo("注意: 此功能需要连接数据库实现")
    click.echo("当前为演示模式")


@click.group()
def signal():
    """交易信号相关命令"""
    pass


signal.add_command(generate)
signal.add_command(list_signals, name='list')


if __name__ == '__main__':
    signal()