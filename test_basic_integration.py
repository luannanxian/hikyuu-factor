#!/usr/bin/env python3
"""
简单的集成测试脚本
用于验证单个Agent是否正常运行
"""

import requests
import time
import json

def test_data_agent_basic():
    """测试DataManagerAgent基本功能"""
    base_url = "http://127.0.0.1:8081"

    print("=== DataManagerAgent 基本功能测试 ===")

    # 1. 测试健康检查
    print("1. 测试健康检查...")
    try:
        response = requests.get(f"{base_url}/health", timeout=5)
        print(f"   状态码: {response.status_code}")
        if response.status_code == 200:
            data = response.json()
            print(f"   Agent ID: {data.get('agent_id')}")
            print(f"   状态: {data.get('status')}")
            print(f"   运行时间: {data.get('uptime_seconds')}秒")
        else:
            print(f"   错误: {response.text}")
    except Exception as e:
        print(f"   连接失败: {e}")
        return False

    # 2. 测试API路由
    print("\n2. 测试API路由发现...")
    try:
        response = requests.get(f"{base_url}/docs", timeout=5)
        if response.status_code == 200:
            print(f"   OpenAPI文档可访问: {response.status_code}")
        else:
            print(f"   OpenAPI文档状态: {response.status_code}")
    except Exception as e:
        print(f"   文档访问失败: {e}")

    # 3. 测试一个简单的数据API (如果实现了)
    print("\n3. 测试数据API...")
    try:
        # 尝试股票列表API
        response = requests.get(f"{base_url}/api/v1/data/stocks",
                               params={"limit": 5}, timeout=10)
        print(f"   股票列表API状态码: {response.status_code}")
        if response.status_code == 200:
            data = response.json()
            print(f"   返回数据类型: {type(data)}")
            if isinstance(data, dict) and "stocks" in data:
                print(f"   股票数量: {len(data['stocks'])}")
            elif isinstance(data, list):
                print(f"   股票数量: {len(data)}")
        else:
            print(f"   响应内容: {response.text[:200]}")
    except Exception as e:
        print(f"   数据API测试失败: {e}")

    print("\n=== 测试完成 ===")
    return True

def test_response_time():
    """测试响应时间"""
    base_url = "http://127.0.0.1:8081"
    print("\n=== 响应时间测试 ===")

    total_time = 0
    success_count = 0

    for i in range(5):
        try:
            start_time = time.time()
            response = requests.get(f"{base_url}/health", timeout=5)
            end_time = time.time()

            response_time_ms = (end_time - start_time) * 1000
            total_time += response_time_ms
            success_count += 1

            print(f"   请求 {i+1}: {response_time_ms:.1f}ms")

        except Exception as e:
            print(f"   请求 {i+1} 失败: {e}")

    if success_count > 0:
        avg_time = total_time / success_count
        print(f"\n   平均响应时间: {avg_time:.1f}ms")
        if avg_time < 200:
            print("   ✅ 响应时间符合要求 (<200ms)")
        else:
            print("   ⚠️  响应时间较慢 (>200ms)")

    return success_count > 0

if __name__ == "__main__":
    print("开始DataManagerAgent集成测试...")

    # 等待Agent完全启动
    print("等待Agent启动...")
    time.sleep(2)

    # 运行测试
    basic_test = test_data_agent_basic()
    time_test = test_response_time()

    if basic_test and time_test:
        print("\n🎉 所有基础测试通过！Agent运行正常。")
    else:
        print("\n❌ 某些测试失败，请检查Agent状态。")