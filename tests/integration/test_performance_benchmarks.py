"""
性能基准测试

建立系统的性能基准线，包括：
1. 因子计算性能基准
2. 数据处理吞吐量基准
3. Agent响应时间基准
4. 内存使用基准
5. 并发处理能力基准
6. 数据库查询性能基准
7. 端到端工作流性能基准
"""

import pytest
import asyncio
import time
import psutil
import numpy as np
import pandas as pd
from datetime import datetime, timedelta, date
from typing import Dict, Any, List, Tuple, Optional
from unittest.mock import Mock
import threading
import multiprocessing
import json
import gc
try:
    import memory_profiler
    HAS_MEMORY_PROFILER = True
except ImportError:
    HAS_MEMORY_PROFILER = False

# 导入系统模块
try:
    from src.agents.data_manager_agent import DataManagerAgent
    from src.agents.factor_calculation_agent import FactorCalculationAgent
    from src.agents.validation_agent import ValidationAgent
    from src.agents.signal_generation_agent import SignalGenerationAgent
    from src.services.factor_calculation_service import FactorCalculationService
    from src.services.data_manager_service import DataManagerService
    from src.models.hikyuu_models import FactorCalculationRequest, FactorType
    from src.models.agent_models import TaskRequest, AgentType, Priority
except ImportError as e:
    pytest.skip(f"Required modules not available: {e}", allow_module_level=True)


class PerformanceProfiler:
    """性能分析器"""

    def __init__(self):
        self.measurements = {}
        self.baselines = {}

    def start_measurement(self, name: str):
        """开始性能测量"""
        self.measurements[name] = {
            'start_time': time.time(),
            'start_memory': psutil.Process().memory_info().rss / 1024 / 1024,  # MB
            'start_cpu': psutil.Process().cpu_percent()
        }

    def end_measurement(self, name: str) -> Dict[str, float]:
        """结束性能测量并返回结果"""
        if name not in self.measurements:
            raise ValueError(f"测量 {name} 未开始")

        start_data = self.measurements[name]
        end_time = time.time()
        end_memory = psutil.Process().memory_info().rss / 1024 / 1024  # MB
        end_cpu = psutil.Process().cpu_percent()

        result = {
            'duration': end_time - start_data['start_time'],
            'memory_delta': end_memory - start_data['start_memory'],
            'peak_memory': end_memory,
            'avg_cpu': (start_data['start_cpu'] + end_cpu) / 2
        }

        del self.measurements[name]
        return result

    def set_baseline(self, name: str, metrics: Dict[str, float]):
        """设置性能基准"""
        self.baselines[name] = metrics

    def compare_to_baseline(self, name: str, current_metrics: Dict[str, float]) -> Dict[str, float]:
        """与基准比较"""
        if name not in self.baselines:
            return {}

        baseline = self.baselines[name]
        comparison = {}

        for metric, current_value in current_metrics.items():
            if metric in baseline:
                baseline_value = baseline[metric]
                if baseline_value != 0:
                    comparison[f"{metric}_ratio"] = current_value / baseline_value
                    comparison[f"{metric}_improvement"] = (baseline_value - current_value) / baseline_value
                else:
                    comparison[f"{metric}_ratio"] = float('inf') if current_value > 0 else 1.0

        return comparison


@pytest.mark.integration
@pytest.mark.performance
@pytest.mark.slow
@pytest.mark.requires_hikyuu
class TestPerformanceBenchmarks:
    """性能基准测试"""

    def setup_method(self):
        """设置测试环境"""
        self.profiler = PerformanceProfiler()
        self.test_config = {
            "database": {
                "host": "192.168.3.46",
                "port": 3306,
                "user": "remote",
                "password": "remote123456",
                "database": "hikyuu_factor_test",
                "charset": "utf8mb4"
            },
            "performance": {
                "target_throughput": 1000,  # 每秒处理量
                "max_response_time": 5.0,   # 最大响应时间(秒)
                "max_memory_usage": 1000,   # 最大内存使用(MB)
                "min_cpu_efficiency": 0.7   # 最小CPU效率
            }
        }

        # 生成测试数据
        self.test_stocks = [f"sh{600000 + i:06d}" for i in range(100)]
        self.large_test_stocks = [f"sh{600000 + i:06d}" for i in range(1000)]
        self.test_market_data = self._generate_market_data()

        # 性能基准线（理想值）
        self.performance_baselines = {
            "factor_calculation": {
                "duration": 5.0,      # 5秒内完成
                "memory_delta": 100,  # 增加100MB内存
                "avg_cpu": 50.0       # 平均CPU使用50%
            },
            "data_processing": {
                "duration": 2.0,
                "memory_delta": 50,
                "avg_cpu": 30.0
            },
            "agent_response": {
                "duration": 1.0,
                "memory_delta": 10,
                "avg_cpu": 20.0
            }
        }

        for name, baseline in self.performance_baselines.items():
            self.profiler.set_baseline(name, baseline)

    def _generate_market_data(self) -> pd.DataFrame:
        """生成测试市场数据"""
        np.random.seed(42)
        data = []

        for stock in self.test_stocks:
            base_price = 10.0 + np.random.randn() * 2.0
            for i in range(252):  # 一年的交易数据
                trade_date = date(2024, 1, 1) + timedelta(days=i)
                price_change = np.random.randn() * 0.02

                data.append({
                    'stock_code': stock,
                    'trade_date': trade_date,
                    'open_price': base_price * (1 + price_change),
                    'high_price': base_price * (1 + abs(price_change) + 0.01),
                    'low_price': base_price * (1 - abs(price_change) - 0.01),
                    'close_price': base_price * (1 + price_change * 0.5),
                    'volume': 1000000 + int(np.random.randn() * 100000),
                    'turnover_rate': np.random.uniform(0.1, 5.0)
                })

                base_price = data[-1]['close_price']

        df = pd.DataFrame(data)
        df['amount'] = df['close_price'] * df['volume']
        return df

    @pytest.mark.performance
    def test_factor_calculation_performance(self):
        """测试因子计算性能"""
        # 初始化因子计算服务
        factor_service = FactorCalculationService()

        # 创建因子计算请求
        calculation_request = FactorCalculationRequest(
            factor_types=[FactorType.MOMENTUM, FactorType.TECHNICAL, FactorType.VALUATION],
            stock_codes=self.test_stocks,
            start_date=date(2024, 1, 1),
            end_date=date(2024, 12, 31),
            calculation_params={
                "momentum_periods": [5, 10, 20],
                "technical_indicators": ["RSI", "MACD", "BOLL"],
                "valuation_metrics": ["PE", "PB"]
            }
        )

        # 开始性能测量
        self.profiler.start_measurement("factor_calculation")

        # 执行因子计算
        try:
            # 由于这是集成测试，实际实现可能需要调整
            result = factor_service.calculate_factors(calculation_request)

            # 结束测量
            metrics = self.profiler.end_measurement("factor_calculation")

            # 验证结果
            assert result is not None, "因子计算应该返回结果"

            # 性能验证
            assert metrics['duration'] < self.test_config['performance']['max_response_time'], \
                f"因子计算时间过长: {metrics['duration']:.3f}秒"

            assert metrics['memory_delta'] < self.test_config['performance']['max_memory_usage'], \
                f"内存使用过多: {metrics['memory_delta']:.1f}MB"

            # 与基准比较
            comparison = self.profiler.compare_to_baseline("factor_calculation", metrics)
            if comparison:
                print(f"因子计算性能比较: {comparison}")

            # 计算吞吐量
            total_calculations = len(self.test_stocks) * len(calculation_request.factor_types)
            throughput = total_calculations / metrics['duration']

            print(f"因子计算性能指标:")
            print(f"  - 计算时间: {metrics['duration']:.3f}秒")
            print(f"  - 内存增量: {metrics['memory_delta']:.1f}MB")
            print(f"  - 平均CPU: {metrics['avg_cpu']:.1f}%")
            print(f"  - 吞吐量: {throughput:.1f} 计算/秒")

            assert throughput > 10, f"因子计算吞吐量过低: {throughput:.1f}"

        except Exception as e:
            self.profiler.end_measurement("factor_calculation")
            raise e

    @pytest.mark.performance
    @pytest.mark.asyncio
    async def test_agent_response_performance(self):
        """测试Agent响应性能"""
        # 初始化测试Agent
        data_agent = DataManagerAgent(config=self.test_config)
        await data_agent.initialize()

        try:
            response_times = []
            memory_usages = []

            # 执行多次请求测试
            for i in range(20):
                self.profiler.start_measurement(f"agent_response_{i}")

                # 创建测试任务
                task = TaskRequest(
                    task_type="health_check",
                    parameters={"check_id": i},
                    priority=Priority.MEDIUM
                )

                message = task.to_agent_message(
                    sender_agent=AgentType.TEST,
                    receiver_agent=AgentType.DATA_MANAGER
                )

                # 发送请求
                response = await data_agent.process_message(message)

                # 测量结果
                metrics = self.profiler.end_measurement(f"agent_response_{i}")

                response_times.append(metrics['duration'])
                memory_usages.append(metrics['memory_delta'])

                # 验证响应
                assert response is not None, f"第{i}次请求无响应"

            # 计算统计指标
            avg_response_time = np.mean(response_times)
            p95_response_time = np.percentile(response_times, 95)
            max_response_time = np.max(response_times)
            avg_memory_delta = np.mean(memory_usages)

            print(f"Agent响应性能指标:")
            print(f"  - 平均响应时间: {avg_response_time:.3f}秒")
            print(f"  - P95响应时间: {p95_response_time:.3f}秒")
            print(f"  - 最大响应时间: {max_response_time:.3f}秒")
            print(f"  - 平均内存增量: {avg_memory_delta:.1f}MB")

            # 性能要求验证
            assert avg_response_time < 1.0, f"平均响应时间过长: {avg_response_time:.3f}秒"
            assert p95_response_time < 2.0, f"P95响应时间过长: {p95_response_time:.3f}秒"
            assert max_response_time < 5.0, f"最大响应时间过长: {max_response_time:.3f}秒"

        finally:
            await data_agent.shutdown()

    @pytest.mark.performance
    def test_data_processing_throughput(self):
        """测试数据处理吞吐量"""
        data_service = DataManagerService()

        # 生成大量测试数据
        large_dataset = self._generate_large_dataset(10000)  # 10K记录

        self.profiler.start_measurement("data_processing")

        try:
            # 执行数据处理
            processed_data = data_service.process_market_data(large_dataset)

            metrics = self.profiler.end_measurement("data_processing")

            # 验证处理结果
            assert processed_data is not None, "数据处理应该返回结果"
            assert len(processed_data) > 0, "处理后的数据不应为空"

            # 计算吞吐量
            records_per_second = len(large_dataset) / metrics['duration']

            print(f"数据处理性能指标:")
            print(f"  - 处理时间: {metrics['duration']:.3f}秒")
            print(f"  - 处理记录数: {len(large_dataset)}")
            print(f"  - 吞吐量: {records_per_second:.1f} 记录/秒")
            print(f"  - 内存增量: {metrics['memory_delta']:.1f}MB")

            # 性能要求
            assert records_per_second > self.test_config['performance']['target_throughput'], \
                f"数据处理吞吐量不达标: {records_per_second:.1f}"

        except Exception as e:
            self.profiler.end_measurement("data_processing")
            raise e

    def _generate_large_dataset(self, size: int) -> pd.DataFrame:
        """生成大型测试数据集"""
        np.random.seed(42)
        data = []

        for i in range(size):
            stock_id = i % 1000
            data.append({
                'stock_code': f"test{stock_id:06d}",
                'trade_date': date(2024, 1, 1) + timedelta(days=i % 365),
                'open_price': 10.0 + np.random.randn() * 2.0,
                'close_price': 10.0 + np.random.randn() * 2.0,
                'volume': 1000000 + int(np.random.randn() * 500000),
                'amount': (10.0 + np.random.randn() * 2.0) * (1000000 + int(np.random.randn() * 500000))
            })

        return pd.DataFrame(data)

    @pytest.mark.performance
    def test_concurrent_processing_performance(self):
        """测试并发处理性能"""
        num_workers = multiprocessing.cpu_count()
        tasks_per_worker = 50
        total_tasks = num_workers * tasks_per_worker

        def worker_function(worker_id: int, results_queue):
            """工作进程函数"""
            worker_start_time = time.time()
            completed_tasks = 0

            try:
                for task_id in range(tasks_per_worker):
                    # 模拟计算密集型任务
                    data = np.random.randn(1000, 100)
                    result = np.mean(data @ data.T)

                    completed_tasks += 1

                worker_end_time = time.time()
                results_queue.put({
                    'worker_id': worker_id,
                    'completed_tasks': completed_tasks,
                    'duration': worker_end_time - worker_start_time,
                    'success': True
                })

            except Exception as e:
                results_queue.put({
                    'worker_id': worker_id,
                    'completed_tasks': completed_tasks,
                    'error': str(e),
                    'success': False
                })

        # 启动并发测试
        import multiprocessing as mp
        results_queue = mp.Queue()
        processes = []

        start_time = time.time()

        # 创建并启动工作进程
        for worker_id in range(num_workers):
            process = mp.Process(
                target=worker_function,
                args=(worker_id, results_queue)
            )
            processes.append(process)
            process.start()

        # 等待所有进程完成
        for process in processes:
            process.join()

        end_time = time.time()
        total_duration = end_time - start_time

        # 收集结果
        results = []
        while not results_queue.empty():
            results.append(results_queue.get())

        # 分析结果
        successful_workers = [r for r in results if r['success']]
        total_completed_tasks = sum(r['completed_tasks'] for r in successful_workers)
        avg_worker_duration = np.mean([r['duration'] for r in successful_workers])

        # 计算性能指标
        overall_throughput = total_completed_tasks / total_duration
        parallel_efficiency = avg_worker_duration / total_duration
        cpu_utilization = len(successful_workers) / num_workers

        print(f"并发处理性能指标:")
        print(f"  - 工作进程数: {num_workers}")
        print(f"  - 总任务数: {total_tasks}")
        print(f"  - 完成任务数: {total_completed_tasks}")
        print(f"  - 总耗时: {total_duration:.3f}秒")
        print(f"  - 整体吞吐量: {overall_throughput:.1f} 任务/秒")
        print(f"  - 并行效率: {parallel_efficiency:.2%}")
        print(f"  - CPU利用率: {cpu_utilization:.2%}")

        # 性能验证
        assert len(successful_workers) >= num_workers * 0.9, "至少90%的工作进程应该成功"
        assert total_completed_tasks >= total_tasks * 0.9, "至少90%的任务应该完成"
        assert parallel_efficiency >= 0.7, f"并行效率过低: {parallel_efficiency:.2%}"

    @pytest.mark.performance
    @pytest.mark.asyncio
    async def test_end_to_end_workflow_performance(self):
        """测试端到端工作流性能"""
        # 初始化所有Agents
        agents = {}
        try:
            agents["data_manager"] = DataManagerAgent(config=self.test_config)
            agents["factor_calculator"] = FactorCalculationAgent(config=self.test_config)
            agents["validator"] = ValidationAgent(config=self.test_config)

            for agent in agents.values():
                await agent.initialize()

            # 端到端工作流测试
            self.profiler.start_measurement("end_to_end_workflow")

            # Step 1: 数据准备
            data_task = TaskRequest(
                task_type="data_preparation",
                parameters={
                    "stock_codes": self.test_stocks[:10],  # 减少规模以确保测试完成
                    "date_range": ["2024-01-01", "2024-01-31"]
                }
            )

            data_message = data_task.to_agent_message(
                sender_agent=AgentType.TEST,
                receiver_agent=AgentType.DATA_MANAGER
            )

            data_response = await agents["data_manager"].process_message(data_message)
            assert data_response.success, "数据准备失败"

            # Step 2: 因子计算
            factor_task = TaskRequest(
                task_type="factor_calculation",
                parameters={
                    "factor_types": ["momentum"],
                    "stock_codes": self.test_stocks[:10]
                }
            )

            factor_message = factor_task.to_agent_message(
                sender_agent=AgentType.TEST,
                receiver_agent=AgentType.FACTOR_CALCULATOR
            )

            factor_response = await agents["factor_calculator"].process_message(factor_message)
            assert factor_response.success, "因子计算失败"

            # Step 3: 验证
            validation_task = TaskRequest(
                task_type="factor_validation",
                parameters={
                    "factors": factor_response.data.get("factors", [])
                }
            )

            validation_message = validation_task.to_agent_message(
                sender_agent=AgentType.TEST,
                receiver_agent=AgentType.VALIDATOR
            )

            validation_response = await agents["validator"].process_message(validation_message)
            assert validation_response.success, "因子验证失败"

            # 结束测量
            metrics = self.profiler.end_measurement("end_to_end_workflow")

            print(f"端到端工作流性能指标:")
            print(f"  - 总耗时: {metrics['duration']:.3f}秒")
            print(f"  - 内存增量: {metrics['memory_delta']:.1f}MB")
            print(f"  - 平均CPU: {metrics['avg_cpu']:.1f}%")

            # 性能要求
            assert metrics['duration'] < 30.0, f"端到端工作流时间过长: {metrics['duration']:.3f}秒"
            assert metrics['memory_delta'] < 500, f"工作流内存使用过多: {metrics['memory_delta']:.1f}MB"

        finally:
            # 清理资源
            for agent in agents.values():
                try:
                    await agent.shutdown()
                except:
                    pass

    @pytest.mark.performance
    def test_memory_usage_profiling(self):
        """测试内存使用分析"""

        @memory_profiler.profile
        def memory_intensive_operation():
            """内存密集型操作"""
            # 创建大型数据结构
            large_data = []
            for i in range(10000):
                data_chunk = {
                    'id': i,
                    'values': np.random.randn(100).tolist(),
                    'metadata': {'created': datetime.now().isoformat()}
                }
                large_data.append(data_chunk)

            # 处理数据
            processed_data = []
            for item in large_data:
                processed_item = {
                    'id': item['id'],
                    'mean': np.mean(item['values']),
                    'std': np.std(item['values']),
                    'processed_at': datetime.now().isoformat()
                }
                processed_data.append(processed_item)

            return processed_data

        # 测量内存使用
        start_memory = psutil.Process().memory_info().rss / 1024 / 1024

        # 执行内存密集型操作
        result = memory_intensive_operation()

        # 强制垃圾回收
        gc.collect()

        end_memory = psutil.Process().memory_info().rss / 1024 / 1024
        memory_delta = end_memory - start_memory

        print(f"内存使用分析:")
        print(f"  - 开始内存: {start_memory:.1f}MB")
        print(f"  - 结束内存: {end_memory:.1f}MB")
        print(f"  - 内存增量: {memory_delta:.1f}MB")
        print(f"  - 处理记录数: {len(result)}")

        # 验证内存使用合理
        assert memory_delta < 200, f"内存使用过多: {memory_delta:.1f}MB"
        assert len(result) == 10000, "处理结果数量不正确"

    @pytest.mark.performance
    def test_cpu_utilization_benchmark(self):
        """测试CPU利用率基准"""

        def cpu_intensive_task(duration: float = 5.0):
            """CPU密集型任务"""
            start_time = time.time()
            operations = 0

            while time.time() - start_time < duration:
                # 执行计算密集型操作
                result = sum(i * i for i in range(1000))
                operations += 1

            return operations

        # 监控CPU使用率
        initial_cpu = psutil.cpu_percent(interval=1)

        start_time = time.time()
        operations = cpu_intensive_task(5.0)
        duration = time.time() - start_time

        final_cpu = psutil.cpu_percent(interval=1)

        # 计算操作效率
        operations_per_second = operations / duration
        cpu_efficiency = operations_per_second / (final_cpu if final_cpu > 0 else 1)

        print(f"CPU利用率基准:")
        print(f"  - 测试时长: {duration:.1f}秒")
        print(f"  - 总操作数: {operations}")
        print(f"  - 操作/秒: {operations_per_second:.1f}")
        print(f"  - 初始CPU: {initial_cpu:.1f}%")
        print(f"  - 最终CPU: {final_cpu:.1f}%")
        print(f"  - CPU效率: {cpu_efficiency:.1f}")

        # 验证CPU利用率
        assert operations_per_second > 1000, f"CPU处理效率过低: {operations_per_second:.1f}"
        assert cpu_efficiency >= self.test_config['performance']['min_cpu_efficiency'], \
            f"CPU效率不达标: {cpu_efficiency:.2f}"

    def teardown_method(self):
        """清理测试环境"""
        # 强制垃圾回收
        gc.collect()

        # 清理测试数据
        del self.test_market_data
        del self.test_stocks
        del self.large_test_stocks