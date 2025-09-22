"""
Mock数据生成器模块 - A股量化因子系统

专注于外部依赖的Mock，遵循"真实优先"策略：
- 使用真实的股票数据库数据
- 使用真实的Hikyuu框架
- 仅对真正的外部依赖进行Mock
"""

# 导入外部依赖Mock模块
from .external_api_mock import *
from .network_service_mock import *
from .file_system_mock import *
from .datetime_mock import *
from .random_data_mock import *

__all__ = [
    # External API mocks
    'MockExternalAPI',
    'mock_market_data_api',
    'mock_news_api',
    'mock_fundamental_api',

    # Network service mocks
    'MockNetworkService',
    'mock_http_client',
    'mock_websocket_client',
    'simulate_network_delay',

    # File system mocks
    'MockFileSystem',
    'mock_file_operations',
    'mock_directory_operations',
    'create_temp_workspace',

    # DateTime mocks
    'MockDateTime',
    'fixed_time_context',
    'time_travel_context',
    'mock_trading_hours',

    # Random data mocks
    'MockRandomGenerator',
    'generate_test_scenarios',
    'create_edge_case_data',
    'simulate_error_conditions'
]