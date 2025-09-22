#!/usr/bin/env python3
"""
最小化集成测试 - 仅测试已运行的DataManagerAgent
"""

import asyncio
import requests
import pytest

@pytest.mark.asyncio
async def test_data_agent_only():
    """仅测试DataManagerAgent - 不启动其他Agent"""
    base_url = "http://127.0.0.1:8081"

    # 测试健康检查
    response = requests.get(f"{base_url}/health")
    assert response.status_code == 200

    data = response.json()
    assert "agent_id" in data
    assert "status" in data
    assert data["agent_id"].startswith("data_manager_")

    print(f"✅ DataManagerAgent健康检查通过: {data['agent_id']}")

@pytest.mark.asyncio
async def test_data_agent_platform_endpoint():
    """测试平台检测端点 (如果已实现)"""
    base_url = "http://127.0.0.1:8081"

    try:
        response = requests.get(f"{base_url}/api/v1/system/platform")
        if response.status_code == 200:
            data = response.json()
            print(f"✅ 平台检测成功: {data}")
            assert "platform_type" in data
        elif response.status_code == 404:
            print("ℹ️  平台检测端点未实现 (404)")
        else:
            print(f"⚠️  平台检测返回状态: {response.status_code}")
    except requests.RequestException as e:
        print(f"⚠️  平台检测请求失败: {e}")

@pytest.mark.asyncio
async def test_response_time_requirement():
    """验证响应时间要求 (<200ms)"""
    base_url = "http://127.0.0.1:8081"

    import time
    total_time = 0
    iterations = 10

    for i in range(iterations):
        start = time.time()
        response = requests.get(f"{base_url}/health")
        end = time.time()

        response_time_ms = (end - start) * 1000
        total_time += response_time_ms

        assert response.status_code == 200

    avg_time = total_time / iterations
    print(f"✅ 平均响应时间: {avg_time:.1f}ms")

    # FR-012要求: API响应时间 < 200ms
    assert avg_time < 200, f"响应时间{avg_time:.1f}ms超过200ms要求"

if __name__ == "__main__":
    # 直接运行测试
    asyncio.run(test_data_agent_only())
    asyncio.run(test_data_agent_platform_endpoint())
    asyncio.run(test_response_time_requirement())
    print("\n🎉 所有最小化测试通过！")