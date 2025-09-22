"""
Integration Test Environment Setup
提供集成测试环境的启动和管理功能
"""

import asyncio
import subprocess
import time
import requests
import logging
from pathlib import Path
from typing import Dict, List, Optional, Any
from contextlib import asynccontextmanager
import pytest

logger = logging.getLogger(__name__)


class AgentTestManager:
    """Agent测试环境管理器"""

    def __init__(self):
        self.agents: Dict[str, subprocess.Popen] = {}
        self.base_ports = {
            'data': 8081,
            'factor': 8082,
            'validation': 8083,
            'signal': 8084
        }
        self.startup_timeout = 30  # 30秒启动超时

    async def start_agent(self, agent_type: str, config_override: Optional[Dict[str, Any]] = None) -> bool:
        """启动指定Agent"""
        if agent_type in self.agents:
            logger.warning(f"Agent {agent_type} already running")
            return True

        port = self.base_ports.get(agent_type)
        if not port:
            raise ValueError(f"Unknown agent type: {agent_type}")

        # 构建启动命令
        cmd = [
            "python", "start_agents.py", agent_type,
            "--port", str(port)
        ]

        if config_override:
            import tempfile
            import json
            # 创建临时配置文件
            config_file = tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False)
            json.dump(config_override, config_file)
            config_file.flush()
            cmd.extend(["--config", config_file.name])

        logger.info(f"Starting {agent_type} agent on port {port}: {' '.join(cmd)}")

        try:
            # 启动Agent进程
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                cwd=Path.cwd()
            )

            self.agents[agent_type] = process

            # 等待Agent启动
            return await self._wait_for_agent_ready(agent_type, port)

        except Exception as e:
            logger.error(f"Failed to start {agent_type} agent: {e}")
            return False

    async def _wait_for_agent_ready(self, agent_type: str, port: int) -> bool:
        """等待Agent就绪"""
        health_url = f"http://127.0.0.1:{port}/health"
        start_time = time.time()

        while time.time() - start_time < self.startup_timeout:
            try:
                response = requests.get(health_url, timeout=2)
                if response.status_code == 200:
                    logger.info(f"Agent {agent_type} is ready on port {port}")
                    return True
            except requests.RequestException:
                pass

            await asyncio.sleep(1)

        logger.error(f"Agent {agent_type} failed to start within {self.startup_timeout}s")
        return False

    async def start_all_agents(self) -> Dict[str, bool]:
        """启动所有Agent"""
        results = {}

        # 并行启动所有Agent
        tasks = []
        for agent_type in self.base_ports.keys():
            task = asyncio.create_task(self.start_agent(agent_type))
            tasks.append((agent_type, task))

        # 等待所有启动完成
        for agent_type, task in tasks:
            results[agent_type] = await task

        return results

    def stop_agent(self, agent_type: str) -> bool:
        """停止指定Agent"""
        if agent_type not in self.agents:
            logger.warning(f"Agent {agent_type} not running")
            return True

        process = self.agents[agent_type]

        try:
            # 优雅关闭
            process.terminate()

            # 等待进程结束
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                # 强制关闭
                process.kill()
                process.wait()

            del self.agents[agent_type]
            logger.info(f"Agent {agent_type} stopped")
            return True

        except Exception as e:
            logger.error(f"Failed to stop {agent_type} agent: {e}")
            return False

    def stop_all_agents(self) -> Dict[str, bool]:
        """停止所有Agent"""
        results = {}

        for agent_type in list(self.agents.keys()):
            results[agent_type] = self.stop_agent(agent_type)

        return results

    def get_agent_status(self, agent_type: str) -> Dict[str, Any]:
        """获取Agent状态"""
        if agent_type not in self.agents:
            return {"status": "stopped"}

        process = self.agents[agent_type]
        port = self.base_ports[agent_type]

        # 检查进程状态
        if process.poll() is not None:
            return {"status": "crashed", "exit_code": process.returncode}

        # 检查健康状态
        try:
            response = requests.get(f"http://127.0.0.1:{port}/health", timeout=2)
            if response.status_code == 200:
                health_data = response.json()
                return {"status": "healthy", "details": health_data}
            else:
                return {"status": "unhealthy", "http_status": response.status_code}
        except requests.RequestException as e:
            return {"status": "unreachable", "error": str(e)}

    async def reset_all_agents(self) -> Dict[str, bool]:
        """重置所有Agent（重启）"""
        logger.info("Resetting all agents...")

        # 停止所有Agent
        stop_results = self.stop_all_agents()

        # 等待一下让端口释放
        await asyncio.sleep(2)

        # 重新启动所有Agent
        start_results = await self.start_all_agents()

        return {agent: stop_results.get(agent, False) and start_results.get(agent, False)
                for agent in self.base_ports.keys()}


# 全局测试环境管理器
test_manager = AgentTestManager()


@asynccontextmanager
async def integration_test_environment():
    """集成测试环境上下文管理器"""
    try:
        logger.info("Setting up integration test environment...")

        # 启动所有Agent
        start_results = await test_manager.start_all_agents()

        # 检查启动结果
        failed_agents = [agent for agent, success in start_results.items() if not success]
        if failed_agents:
            logger.error(f"Failed to start agents: {failed_agents}")
            raise RuntimeError(f"Failed to start agents: {failed_agents}")

        logger.info("Integration test environment ready")
        yield test_manager

    finally:
        logger.info("Tearing down integration test environment...")
        stop_results = test_manager.stop_all_agents()

        failed_stops = [agent for agent, success in stop_results.items() if not success]
        if failed_stops:
            logger.warning(f"Failed to stop agents: {failed_stops}")


@pytest.fixture(scope="session")
async def agent_manager():
    """pytest fixture for agent manager"""
    async with integration_test_environment() as manager:
        yield manager


@pytest.fixture(scope="function")
async def fresh_agents(agent_manager):
    """为每个测试提供全新的Agent环境"""
    # 在每个测试前重置所有Agent
    await agent_manager.reset_all_agents()
    yield agent_manager


class IntegrationTestBase:
    """集成测试基类"""

    def __init__(self, agent_manager: AgentTestManager):
        self.agent_manager = agent_manager
        self.base_urls = {
            agent_type: f"http://127.0.0.1:{port}"
            for agent_type, port in agent_manager.base_ports.items()
        }

    async def wait_for_task_completion(self, agent_type: str, task_id: str, timeout: int = 300) -> Dict[str, Any]:
        """等待异步任务完成"""
        start_time = time.time()
        status_url = f"{self.base_urls[agent_type]}/api/v1/tasks/{task_id}/status"

        while time.time() - start_time < timeout:
            try:
                response = requests.get(status_url)
                if response.status_code == 200:
                    data = response.json()
                    if data["status"] in ["completed", "failed"]:
                        return data

                await asyncio.sleep(2)  # 2秒检查一次

            except requests.RequestException as e:
                logger.warning(f"Error checking task status: {e}")
                await asyncio.sleep(5)

        raise TimeoutError(f"Task {task_id} did not complete within {timeout}s")

    def assert_agent_healthy(self, agent_type: str):
        """断言Agent健康状态"""
        status = self.agent_manager.get_agent_status(agent_type)
        assert status["status"] == "healthy", f"Agent {agent_type} is not healthy: {status}"

    def get_api_url(self, agent_type: str, endpoint: str) -> str:
        """构建API URL"""
        base_url = self.base_urls[agent_type]
        if not endpoint.startswith("/"):
            endpoint = f"/{endpoint}"
        return f"{base_url}{endpoint}"