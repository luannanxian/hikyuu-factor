"""
Factor Calculation Performance Test
因子计算性能测试脚本
"""

import asyncio
import time
import sys
import os
from datetime import date, timedelta
from typing import List

# 添加项目根目录到路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from services.factor_calculation_service import (
    FactorRegistry, FactorCalculator, PlatformOptimizer,
    FactorCalculationService
)
from models.hikyuu_models import FactorCalculationRequest, FactorType
from lib.performance import calculation_optimizer, platform_optimizer
from data.repository import stock_repository


async def test_performance_targets():
    """测试性能目标是否达成"""
    print("=" * 80)
    print("因子计算性能测试")
    print("=" * 80)

    # 输出平台信息
    platform_info = platform_optimizer.platform_info
    print(f"平台信息:")
    print(f"  系统: {platform_info['system']} {platform_info['machine']}")
    print(f"  CPU核数: {platform_info['cpu_count']}")
    print(f"  内存: {platform_info['memory_gb']}GB")
    print(f"  Apple Silicon: {platform_info['is_apple_silicon']}")
    print(f"  x86_64: {platform_info['is_x86_64']}")
    print()

    # 模拟全市场股票数量
    full_market_stocks = 5000  # A股全市场约5000只股票
    factors_count = 20  # 20个因子

    # 性能估算
    estimate = calculation_optimizer.estimate_calculation_time(full_market_stocks, factors_count)
    strategy = calculation_optimizer.optimize_calculation_strategy(full_market_stocks, factors_count)

    print(f"全市场性能估算:")
    print(f"  股票数量: {full_market_stocks:,}")
    print(f"  因子数量: {factors_count}")
    print(f"  预计计算时间: {estimate['estimated_minutes']:.1f}分钟 ({estimate['estimated_hours']:.2f}小时)")
    print(f"  目标时间: {estimate['target_minutes']}分钟")
    print(f"  满足性能目标: {'是' if estimate['meets_target'] else '否'}")
    print(f"  性能比率: {estimate['performance_ratio']:.2f}x")
    print()

    print(f"优化策略:")
    print(f"  并行计算: {strategy['use_parallel']}")
    print(f"  工作进程数: {strategy['worker_count']}")
    print(f"  批次大小: {strategy['chunk_size']}")
    print(f"  内存优化: {strategy['memory_optimization']}")
    print(f"  缓存中间结果: {strategy['cache_intermediate_results']}")
    print()

    # 小规模性能测试
    print("正在进行小规模性能测试...")

    try:
        # 初始化服务
        registry = FactorRegistry()
        calculator = FactorCalculator(registry)
        service = FactorCalculationService()

        # 测试股票列表（小规模）
        test_stocks = ["sh000001", "sz000001", "sh600036", "sz000002", "sh600519"]

        # 创建测试请求
        request = FactorCalculationRequest(
            request_id="perf_test_001",
            factor_name="momentum_20d",
            factor_type=FactorType.MOMENTUM,
            stock_codes=test_stocks,
            start_date=date.today() - timedelta(days=60),
            end_date=date.today(),
            data_source="hikyuu",
            chunk_size=50
        )

        # 执行计算
        start_time = time.perf_counter()
        result = await calculator.calculate_factor(request)
        end_time = time.perf_counter()

        actual_time = end_time - start_time
        stocks_per_second = len(test_stocks) / actual_time

        print(f"小规模测试结果:")
        print(f"  测试股票: {len(test_stocks)}只")
        print(f"  实际耗时: {actual_time:.2f}秒")
        print(f"  成功计算: {result.successful_calculations}")
        print(f"  失败计算: {result.failed_calculations}")
        print(f"  处理速率: {stocks_per_second:.1f}股票/秒")
        print()

        # 根据小规模测试结果推算全市场性能
        if stocks_per_second > 0:
            estimated_full_market_time = (full_market_stocks * factors_count) / stocks_per_second / 60  # 分钟
            print(f"基于实际测试的全市场性能推算:")
            print(f"  预计全市场计算时间: {estimated_full_market_time:.1f}分钟 ({estimated_full_market_time/60:.2f}小时)")
            print(f"  目标达成情况: {'达成' if estimated_full_market_time <= 30 else '未达成'}")

            if estimated_full_market_time > 30:
                needed_improvement = estimated_full_market_time / 30
                print(f"  需要性能提升: {needed_improvement:.1f}x")
                print(f"  建议措施:")
                print(f"    - 增加并行度至 {min(platform_info['cpu_count'], int(strategy['worker_count'] * needed_improvement))}个进程")
                print(f"    - 优化数据缓存和预计算")
                print(f"    - 使用更高性能的硬件平台")
        print()

    except Exception as e:
        print(f"性能测试失败: {e}")
        print("请检查Hikyuu环境和数据源配置")

    print("=" * 80)
    print("性能测试完成")
    print("=" * 80)


if __name__ == "__main__":
    asyncio.run(test_performance_targets())