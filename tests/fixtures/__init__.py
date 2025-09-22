"""
Mock数据生成器模块 - A股量化因子系统

专注于外部依赖的Mock，遵循"真实优先"策略：
- 使用真实的股票数据库数据
- 使用真实的Hikyuu框架
- 仅对真正的外部依赖进行Mock

新增量化数据Mock生成器：
- 股票数据Mock生成器
- 因子数据Mock生成器
- 信号数据Mock生成器
"""

# 导入外部依赖Mock模块
from .external_api_mock import *
from .network_service_mock import *
from .file_system_mock import *
from .datetime_mock import *
from .random_data_mock import *

# 导入量化数据Mock模块
from .stock_data_mock import *
from .factor_data_mock import *
from .signal_data_mock import *

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
    'simulate_error_conditions',

    # Stock data mocks
    'MockStockDataGenerator',
    'create_sample_stock_pool',
    'create_test_kdata_dataset',
    'create_market_scenario_data',

    # Factor data mocks
    'MockFactorDataGenerator',
    'create_factor_library',
    'create_test_factor_dataset',
    'create_factor_performance_data',

    # Signal data mocks
    'MockSignalDataGenerator',
    'SignalType',
    'SignalStatus',
    'create_sample_signals',
    'create_backtest_signals',
    'create_live_trading_scenario'
]