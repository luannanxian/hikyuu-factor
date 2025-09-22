#!/usr/bin/env python3
"""
Agent启动脚本
启动指定的Agent服务用于集成测试
"""

import sys
import argparse
import asyncio
import logging
from pathlib import Path

# 添加src路径
sys.path.insert(0, str(Path(__file__).parent / "src"))

from agents.data_manager_agent import DataManagerAgent
from agents.factor_calculation_agent import FactorCalculationAgent
from agents.validation_agent import ValidationAgent
from agents.signal_generation_agent import SignalGenerationAgent
from models.agent_models import AgentType

# 集成测试端口配置
INTEGRATION_TEST_PORTS = {
    'data': 8081,
    'factor': 8082,
    'validation': 8083,
    'signal': 8084
}

def create_agent(agent_type: str, port: int = None):
    """创建指定类型的Agent"""
    if port is None:
        port = INTEGRATION_TEST_PORTS.get(agent_type, 8000)

    config = {
        'host': '127.0.0.1',
        'port': port,
        'debug': True,
        'enable_cors': True
    }

    if agent_type == 'data':
        return DataManagerAgent(config=config)
    elif agent_type == 'factor':
        return FactorCalculationAgent(config=config)
    elif agent_type == 'validation':
        return ValidationAgent(config=config)
    elif agent_type == 'signal':
        return SignalGenerationAgent(config=config)
    else:
        raise ValueError(f"Unknown agent type: {agent_type}")

def start_agent_sync(agent_type: str, port: int = None):
    """启动指定Agent (同步版本)"""
    try:
        agent = create_agent(agent_type, port)
        print(f"Starting {agent_type} agent on port {agent.port}...")

        # 初始化Agent (需要异步)
        import asyncio
        initialized = asyncio.run(agent.initialize())
        if not initialized:
            print(f"Failed to initialize {agent_type} agent")
            return False

        print(f"{agent_type} agent initialized successfully")

        # 使用run方法启动FastAPI服务器
        print(f"{agent_type} agent starting web server...")
        agent.run()  # 这会阻塞运行uvicorn
        return True

    except Exception as e:
        print(f"Error starting {agent_type} agent: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    parser = argparse.ArgumentParser(description="Start Agent for integration testing")
    parser.add_argument("agent_type",
                       choices=['data', 'factor', 'validation', 'signal', 'all'],
                       help="Type of agent to start")
    parser.add_argument("--port", type=int,
                       help="Port to run the agent on (overrides default)")
    parser.add_argument("--verbose", "-v", action="store_true",
                       help="Enable verbose logging")

    args = parser.parse_args()

    # 配置日志
    level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    if args.agent_type == 'all':
        print("Starting all agents for integration testing...")
        print("This will start agents on ports 8081-8084")
        print("Press Ctrl+C to stop all agents")

        # TODO: Implement concurrent agent startup
        print("Multi-agent startup not yet implemented")
        print("Please start agents individually:")
        for agent_type, port in INTEGRATION_TEST_PORTS.items():
            print(f"  python start_agents.py {agent_type} --port {port}")
        return

    # 启动单个Agent
    try:
        start_agent_sync(args.agent_type, args.port)
    except KeyboardInterrupt:
        print(f"\n{args.agent_type} agent stopped by user")
    except Exception as e:
        print(f"Failed to start {args.agent_type} agent: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()