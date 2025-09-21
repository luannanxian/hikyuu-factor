"""
网络服务Mock - Network Service Mock

模拟网络服务调用，包括HTTP客户端、WebSocket连接等
"""
import asyncio
import json
import time
from datetime import datetime
from typing import Dict, List, Optional, Any, Callable
from unittest.mock import Mock, MagicMock, AsyncMock
import random


class MockNetworkService:
    """
    网络服务模拟器

    模拟各种网络服务调用，包括HTTP、WebSocket等
    """

    def __init__(self,
                 latency_range: tuple = (0.01, 0.1),
                 failure_rate: float = 0.0,
                 timeout_rate: float = 0.0):
        """
        初始化网络服务模拟器

        Args:
            latency_range: 延迟范围 (最小, 最大) 秒
            failure_rate: 失败率 (0.0-1.0)
            timeout_rate: 超时率 (0.0-1.0)
        """
        self.latency_range = latency_range
        self.failure_rate = failure_rate
        self.timeout_rate = timeout_rate
        self.request_count = 0
        self.error_count = 0

    def simulate_request(self,
                        url: str,
                        method: str = "GET",
                        data: Optional[Dict] = None,
                        headers: Optional[Dict] = None) -> Dict[str, Any]:
        """
        模拟HTTP请求

        Args:
            url: 请求URL
            method: HTTP方法
            data: 请求数据
            headers: 请求头

        Returns:
            模拟响应

        Raises:
            TimeoutError: 模拟超时
            ConnectionError: 模拟连接错误
        """
        self.request_count += 1

        # 模拟网络延迟
        delay = random.uniform(*self.latency_range)
        time.sleep(delay)

        # 模拟超时
        if random.random() < self.timeout_rate:
            self.error_count += 1
            raise TimeoutError(f"Mock timeout for {url}")

        # 模拟连接失败
        if random.random() < self.failure_rate:
            self.error_count += 1
            raise ConnectionError(f"Mock connection failed for {url}")

        # 模拟成功响应
        return {
            "status_code": 200,
            "headers": {"Content-Type": "application/json"},
            "data": {
                "url": url,
                "method": method,
                "timestamp": datetime.now().isoformat(),
                "mock_response": True
            },
            "request_id": f"req_{self.request_count}",
            "latency_ms": int(delay * 1000)
        }

    def get_stats(self) -> Dict[str, Any]:
        """获取请求统计"""
        return {
            "total_requests": self.request_count,
            "total_errors": self.error_count,
            "success_rate": (self.request_count - self.error_count) / max(1, self.request_count),
            "error_rate": self.error_count / max(1, self.request_count)
        }


def mock_http_client(response_data: Optional[Dict] = None,
                    status_code: int = 200,
                    delay: float = 0.1) -> Mock:
    """
    创建HTTP客户端mock

    Args:
        response_data: 预设的响应数据
        status_code: HTTP状态码
        delay: 响应延迟

    Returns:
        Mock HTTP客户端
    """
    def mock_get(url: str, **kwargs) -> Mock:
        time.sleep(delay)
        response = Mock()
        response.status_code = status_code
        response.headers = {"Content-Type": "application/json"}

        if response_data:
            response.json.return_value = response_data
            response.text = json.dumps(response_data)
        else:
            default_data = {
                "url": url,
                "timestamp": datetime.now().isoformat(),
                "mock": True
            }
            response.json.return_value = default_data
            response.text = json.dumps(default_data)

        return response

    def mock_post(url: str, data=None, json=None, **kwargs) -> Mock:
        time.sleep(delay)
        response = Mock()
        response.status_code = status_code
        response.headers = {"Content-Type": "application/json"}

        response_data_post = {
            "url": url,
            "received_data": data or json,
            "timestamp": datetime.now().isoformat(),
            "mock": True
        }
        response.json.return_value = response_data_post
        response.text = json.dumps(response_data_post)

        return response

    client = Mock()
    client.get = mock_get
    client.post = mock_post
    client.put = mock_post  # 简化处理
    client.delete = mock_get  # 简化处理

    return client


class MockWebSocketClient:
    """
    WebSocket客户端模拟器
    """

    def __init__(self, url: str):
        """
        初始化WebSocket客户端

        Args:
            url: WebSocket URL
        """
        self.url = url
        self.connected = False
        self.messages = []
        self.subscribers = []
        self.connection_count = 0

    async def connect(self):
        """连接到WebSocket服务器"""
        await asyncio.sleep(0.1)  # 模拟连接延迟
        self.connected = True
        self.connection_count += 1

    async def disconnect(self):
        """断开WebSocket连接"""
        self.connected = False

    async def send(self, message: str):
        """发送消息"""
        if not self.connected:
            raise ConnectionError("WebSocket not connected")

        self.messages.append({
            "type": "sent",
            "message": message,
            "timestamp": datetime.now().isoformat()
        })

    async def receive(self) -> str:
        """接收消息"""
        if not self.connected:
            raise ConnectionError("WebSocket not connected")

        # 模拟接收消息
        await asyncio.sleep(0.05)

        mock_message = {
            "type": "market_data",
            "symbol": "sh000001",
            "price": round(3000 + random.uniform(-50, 50), 2),
            "timestamp": datetime.now().isoformat()
        }

        return json.dumps(mock_message)

    def subscribe(self, callback: Callable):
        """订阅消息"""
        self.subscribers.append(callback)

    async def start_message_loop(self, duration: float = 1.0):
        """启动消息循环"""
        start_time = time.time()

        while time.time() - start_time < duration and self.connected:
            try:
                message = await self.receive()
                for callback in self.subscribers:
                    callback(message)
                await asyncio.sleep(0.1)
            except Exception as e:
                break


def mock_websocket_client(url: str = "ws://localhost:8080") -> MockWebSocketClient:
    """
    创建WebSocket客户端mock

    Args:
        url: WebSocket URL

    Returns:
        Mock WebSocket客户端
    """
    return MockWebSocketClient(url)


def simulate_network_delay(min_delay: float = 0.01,
                          max_delay: float = 0.5,
                          jitter: float = 0.1):
    """
    模拟网络延迟装饰器

    Args:
        min_delay: 最小延迟
        max_delay: 最大延迟
        jitter: 抖动范围

    Returns:
        装饰器函数
    """
    def decorator(func):
        def wrapper(*args, **kwargs):
            # 计算延迟
            base_delay = random.uniform(min_delay, max_delay)
            jitter_adjustment = random.uniform(-jitter, jitter)
            total_delay = max(0, base_delay + jitter_adjustment)

            # 模拟延迟
            time.sleep(total_delay)

            # 执行原函数
            return func(*args, **kwargs)

        return wrapper
    return decorator


class NetworkConditionSimulator:
    """
    网络状况模拟器

    模拟不同的网络状况，如慢网络、不稳定连接等
    """

    def __init__(self):
        self.conditions = {
            "excellent": {"latency": (0.01, 0.05), "loss_rate": 0.001, "jitter": 0.01},
            "good": {"latency": (0.05, 0.15), "loss_rate": 0.01, "jitter": 0.02},
            "fair": {"latency": (0.15, 0.5), "loss_rate": 0.05, "jitter": 0.1},
            "poor": {"latency": (0.5, 2.0), "loss_rate": 0.15, "jitter": 0.5},
            "terrible": {"latency": (2.0, 10.0), "loss_rate": 0.3, "jitter": 2.0}
        }
        self.current_condition = "good"

    def set_condition(self, condition: str):
        """设置网络状况"""
        if condition in self.conditions:
            self.current_condition = condition
        else:
            raise ValueError(f"Unknown network condition: {condition}")

    def apply_condition(self, func):
        """应用网络状况到函数"""
        condition = self.conditions[self.current_condition]

        def wrapper(*args, **kwargs):
            # 模拟丢包
            if random.random() < condition["loss_rate"]:
                raise ConnectionError("Simulated packet loss")

            # 模拟延迟和抖动
            latency = random.uniform(*condition["latency"])
            jitter = random.uniform(-condition["jitter"], condition["jitter"])
            total_delay = max(0, latency + jitter)

            time.sleep(total_delay)

            return func(*args, **kwargs)

        return wrapper

    def get_current_condition(self) -> Dict[str, Any]:
        """获取当前网络状况"""
        return {
            "condition": self.current_condition,
            **self.conditions[self.current_condition]
        }


# 预定义的网络状况实例
network_simulator = NetworkConditionSimulator()


def create_retry_mock(max_retries: int = 3,
                     backoff_factor: float = 1.0,
                     success_on_retry: int = 2) -> Mock:
    """
    创建重试机制mock

    Args:
        max_retries: 最大重试次数
        backoff_factor: 退避因子
        success_on_retry: 第几次重试成功

    Returns:
        带重试逻辑的Mock函数
    """
    call_count = [0]  # 使用列表来修改闭包变量

    def mock_function(*args, **kwargs):
        call_count[0] += 1

        if call_count[0] < success_on_retry:
            # 前几次调用失败
            raise ConnectionError(f"Mock failure on attempt {call_count[0]}")

        # 最后成功
        return {
            "status": "success",
            "attempt": call_count[0],
            "data": "Mock successful response"
        }

    mock_func = Mock(side_effect=mock_function)
    mock_func.call_count_tracker = call_count

    return mock_func