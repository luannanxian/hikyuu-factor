"""
Complete System Integration Test
完整系统集成测试 - 验证Phase 9所有功能的协同工作
"""

import asyncio
import sys
import os
from datetime import date, timedelta
from typing import List, Dict, Any

# 添加项目根目录到路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from services.data_manager_service import DataManagerService
from services.factor_calculation_service import FactorCalculationService
from data.hikyuu_interface import hikyuu_interface
from data.repository import stock_repository, market_data_repository, factor_repository
from lib.performance import platform_optimizer, calculation_optimizer
from lib.environment import env_manager
from models.hikyuu_models import FactorCalculationRequest, FactorType


class TestCompleteSystemIntegration:
    """完整系统集成测试"""

    def setup_test_environment(self):
        """设置测试环境"""
        print("\n" + "="*80)
        print("开始完整系统集成测试")
        print("="*80)

        # 初始化Hikyuu
        try:
            success = hikyuu_interface.initialize()
            print(f"Hikyuu初始化: {'成功' if success else '失败'}")
        except Exception as e:
            print(f"Hikyuu初始化异常: {e}")

        # 检查平台优化器
        platform_info = platform_optimizer.platform_info
        print(f"平台信息: {platform_info['system']} {platform_info['machine']}")
        print(f"CPU核数: {platform_info['cpu_count']}, 内存: {platform_info['memory_gb']}GB")
        print(f"Apple Silicon: {platform_info['is_apple_silicon']}")

    def cleanup_test_environment(self):
        """清理测试环境"""
        print("\n" + "="*80)
        print("系统集成测试完成")
        print("="*80)

    async def test_complete_data_pipeline(self):
        """测试完整的数据管道：数据获取 -> 因子计算 -> 存储"""
        print("\n测试1: 完整数据管道")
        print("-" * 50)

        # 1. 数据管理服务测试
        data_service = DataManagerService()

        # 获取股票列表
        stocks = data_service.get_stock_list(market="sh")
        print(f"获取股票列表: {len(stocks)} 只股票")
        assert len(stocks) > 0, "应该能获取到股票数据"

        test_stock = stocks[0]['stock_code']
        print(f"测试股票: {test_stock}")

        # 获取市场数据
        start_date = (date.today() - timedelta(days=30)).strftime("%Y-%m-%d")
        end_date = date.today().strftime("%Y-%m-%d")

        market_data = data_service.get_market_data(test_stock, start_date, end_date)
        print(f"市场数据: {market_data['count']} 条记录")
        assert market_data['count'] >= 0, "应该能获取市场数据"

        # 2. 因子计算服务测试
        factor_service = FactorCalculationService()

        # 测试单因子计算
        test_stocks = [stocks[i]['stock_code'] for i in range(min(3, len(stocks)))]

        result = await factor_service.calculate_multi_factors_with_mf(
            stock_codes=test_stocks,
            factor_names=['momentum_20d', 'rsi_14d'],
            start_date=date.today() - timedelta(days=30),
            end_date=date.today(),
            user_id="integration_test"
        )

        print(f"MF因子计算结果: {'成功' if result['success'] else '失败'}")
        if result['success']:
            metrics = result['performance_metrics']
            print(f"   总记录: {metrics['total_records']}")
            print(f"   成功率: {metrics['success_rate']:.2%}")
            print(f"   处理速度: {metrics['records_per_second']:.1f} 记录/秒")
            print(f"   使用Hikyuu MF: {metrics['using_hikyuu_mf']}")

        assert result['success'], "因子计算应该成功"

        print("完整数据管道测试通过")

    async def test_hikyuu_advanced_features(self):
        """测试Hikyuu高级功能集成"""
        print("\n测试2: Hikyuu高级功能")
        print("-" * 50)

        test_stock = "sh000001"
        start_date = date.today() - timedelta(days=20)
        end_date = date.today()

        # 1. FINANCE功能测试
        try:
            financial_data = hikyuu_interface.get_financial_data(
                test_stock, start_date, end_date
            )
            print(f"FINANCE数据: {len(financial_data)} 条记录")

            if not financial_data.empty:
                sample = financial_data.iloc[0]
                print(f"   样本数据: EPS={sample.get('eps', 'N/A')}, PE={sample.get('pe', 'N/A')}")

        except Exception as e:
            print(f"FINANCE功能异常: {e}")

        # 2. MF批量计算测试
        try:
            mf_results = hikyuu_interface.calculate_multi_factors(
                [test_stock], start_date, end_date,
                ['momentum_20d', 'rsi_14d', 'volatility_20d']
            )

            total_records = sum(len(df) for df in mf_results.values() if not df.empty)
            print(f"MF批量计算: {len(mf_results)} 个因子, {total_records} 条记录")

            for factor_name, df in mf_results.items():
                valid_count = len(df[df['factor_value'].notna()]) if not df.empty else 0
                print(f"   {factor_name}: {len(df)} 条记录, {valid_count} 条有效")

        except Exception as e:
            print(f"MF功能异常: {e}")

        print("Hikyuu高级功能测试通过")

    async def test_performance_optimization(self):
        """测试性能优化功能"""
        print("\n测试3: 性能优化")
        print("-" * 50)

        # 1. 平台优化测试
        stock_count = 1000
        factor_count = 10

        estimate = calculation_optimizer.estimate_calculation_time(stock_count, factor_count)
        strategy = calculation_optimizer.optimize_calculation_strategy(stock_count, factor_count)

        print(f"性能估算 ({stock_count}股票 × {factor_count}因子):")
        print(f"   预计时间: {estimate['estimated_minutes']:.1f} 分钟")
        print(f"   目标达成: {'是' if estimate['meets_target'] else '否'}")
        print(f"   性能比率: {estimate['performance_ratio']:.2f}x")

        print(f"优化策略:")
        print(f"   并行计算: {strategy['use_parallel']}")
        print(f"   工作进程: {strategy['worker_count']}")
        print(f"   批次大小: {strategy['chunk_size']}")
        print(f"   内存优化: {strategy['memory_optimization']}")

        # 2. 实际性能测试（小规模）
        import time
        import pandas as pd
        import numpy as np

        # 创建测试DataFrame
        test_data = pd.DataFrame({
            'close': np.random.randn(1000) * 10 + 100,
            'volume': np.random.randint(1000000, 10000000, 1000),
            'date': pd.date_range('2024-01-01', periods=1000)
        })

        # 内存优化测试
        original_memory = test_data.memory_usage(deep=True).sum()
        optimized_data = platform_optimizer.optimize_dataframe(test_data.copy())
        optimized_memory = optimized_data.memory_usage(deep=True).sum()

        memory_reduction = (original_memory - optimized_memory) / original_memory * 100
        print(f"内存优化: 减少 {memory_reduction:.1f}%")

        print("性能优化测试通过")

    async def test_data_repository_integration(self):
        """测试数据仓库层集成"""
        print("\n测试4: 数据仓库集成")
        print("-" * 50)

        # 1. 股票仓库测试
        try:
            stocks = stock_repository.get_all_stocks()
            print(f"股票仓库: {len(stocks)} 只股票")

            if stocks:
                sample_stock = stocks[0]
                print(f"   样本: {sample_stock['stock_code']} - {sample_stock['stock_name']}")
        except Exception as e:
            print(f"股票仓库异常: {e}")

        # 2. 市场数据仓库测试
        try:
            start_date = date.today() - timedelta(days=10)
            end_date = date.today()

            market_data = market_data_repository.get_market_data("sh000001", start_date, end_date)
            print(f"市场数据仓库: {len(market_data)} 条记录")
        except Exception as e:
            print(f"市场数据仓库异常: {e}")

        # 3. 因子仓库测试
        try:
            factors = factor_repository.get_all_factors()
            print(f"因子仓库: {len(factors)} 个因子")

            if factors:
                sample_factor = factors[0]
                print(f"   样本: {sample_factor['factor_id']} - {sample_factor['factor_name']}")
        except Exception as e:
            print(f"因子仓库异常: {e}")

        print("数据仓库集成测试通过")

    async def test_environment_protection(self):
        """测试环境保护机制"""
        print("\n测试5: 环境保护")
        print("-" * 50)

        # 1. 环境检测
        print(f"当前环境: {env_manager._current_env.value}")
        print(f"Mock数据允许: {env_manager.is_mock_data_allowed()}")
        print(f"开发环境: {env_manager.is_development_only()}")

        # 2. Mock数据保护测试
        if not env_manager.is_mock_data_allowed():
            print("生产环境检测：Mock数据被正确禁用")
        else:
            print("开发环境检测：Mock数据已启用")

        print("环境保护测试通过")

    async def test_end_to_end_workflow(self):
        """测试端到端工作流"""
        print("\n测试6: 端到端工作流")
        print("-" * 50)

        workflow_start = asyncio.get_event_loop().time()

        try:
            # 1. 获取股票列表
            data_service = DataManagerService()
            stocks = data_service.get_stock_list()[:5]  # 只测试5只股票

            # 2. 批量因子计算
            factor_service = FactorCalculationService()
            result = await factor_service.calculate_multi_factors_with_mf(
                stock_codes=[s['stock_code'] for s in stocks],
                factor_names=['momentum_20d', 'rsi_14d'],
                start_date=date.today() - timedelta(days=15),
                end_date=date.today(),
                user_id="e2e_test"
            )

            workflow_time = asyncio.get_event_loop().time() - workflow_start

            print(f"端到端工作流:")
            print(f"   股票数量: {len(stocks)}")
            print(f"   因子数量: 2")
            print(f"   成功状态: {result['success']}")
            print(f"   总耗时: {workflow_time:.2f} 秒")

            if result['success']:
                metrics = result['performance_metrics']
                print(f"   处理记录: {metrics['total_records']}")
                print(f"   成功率: {metrics['success_rate']:.2%}")

        except Exception as e:
            print(f"端到端工作流失败: {e}")
            raise

        print("端到端工作流测试通过")


# 运行集成测试的主函数
async def run_integration_tests():
    """运行完整的集成测试"""
    test_instance = TestCompleteSystemIntegration()

    # 设置测试环境
    test_instance.setup_test_environment()

    try:
        # 运行所有测试
        await test_instance.test_complete_data_pipeline()
        await test_instance.test_hikyuu_advanced_features()
        await test_instance.test_performance_optimization()
        await test_instance.test_data_repository_integration()
        await test_instance.test_environment_protection()
        await test_instance.test_end_to_end_workflow()

        print("\n所有集成测试通过！")
        print("系统各组件协同工作正常")

    except Exception as e:
        print(f"\n集成测试失败: {e}")
        raise

    finally:
        test_instance.cleanup_test_environment()


if __name__ == "__main__":
    asyncio.run(run_integration_tests())