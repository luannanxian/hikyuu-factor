"""
测试配置文件 - A股量化因子系统

采用"真实优先"策略，最小化mock使用：
- 使用真实但隔离的数据库连接
- 使用真实Hikyuu引擎和最小数据集
- 仅对外部依赖使用mock
- 确保测试反映真实系统行为
"""
import os
import tempfile
import shutil
from pathlib import Path
from datetime import datetime, date
from typing import Generator, Dict, Any
import pytest
import redis
import sqlalchemy as sa
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, Session


# =============================================================================
# 配置和环境设置
# =============================================================================

@pytest.fixture(scope="session")
def test_config() -> Dict[str, Any]:
    """
    测试环境配置fixture

    提供隔离的测试环境配置，避免污染生产配置
    """
    return {
        # 数据库配置（使用真实MySQL数据库）
        "database_url": "mysql+pymysql://remote:remote123456@192.168.3.46:3306/hikyuu_factor_test?charset=utf8mb4",
        "test_database_url": "mysql+pymysql://remote:remote123456@192.168.3.46:3306/hikyuu_factor_test?charset=utf8mb4",

        # Redis配置（使用专用测试数据库）
        "redis_url": "redis://localhost:6379/15",  # DB 15专用于测试
        "redis_timeout": 5,

        # Hikyuu配置
        "hikyuu_data_dir": None,  # 将在temp_data_dir中设置
        "hikyuu_config": {
            "data_source": "memory",  # 内存数据源，快速测试
            "log_level": "ERROR",     # 减少测试日志噪音
        },

        # 测试数据配置
        "test_stocks": ["sh000001", "sz000002", "sh000300"],  # 最小股票池
        "test_date_range": {
            "start": "2023-01-01",
            "end": "2023-12-31"
        },

        # 性能配置
        "timeout": 30,
        "max_memory_mb": 512,
    }


@pytest.fixture(scope="function")
def temp_data_dir() -> Generator[Path, None, None]:
    """
    临时数据目录fixture

    为每个测试创建独立的临时目录，测试完成后自动清理
    """
    temp_dir = Path(tempfile.mkdtemp(prefix="hikyuu_test_"))
    try:
        yield temp_dir
    finally:
        if temp_dir.exists():
            shutil.rmtree(temp_dir, ignore_errors=True)


@pytest.fixture(scope="function")
def cleanup_test_data():
    """
    测试数据清理fixture

    确保测试完成后清理所有临时数据和状态
    """
    created_files = []
    created_dirs = []

    def register_file(filepath: Path):
        created_files.append(filepath)

    def register_dir(dirpath: Path):
        created_dirs.append(dirpath)

    # 提供注册函数给测试使用
    cleanup_data = {
        "register_file": register_file,
        "register_dir": register_dir
    }

    yield cleanup_data

    # 清理注册的文件和目录
    for file_path in created_files:
        if file_path.exists():
            file_path.unlink(missing_ok=True)

    for dir_path in created_dirs:
        if dir_path.exists():
            shutil.rmtree(dir_path, ignore_errors=True)


# =============================================================================
# 数据库Fixtures - 真实轻量级数据库
# =============================================================================

@pytest.fixture(scope="session")
def test_db_url(test_config: Dict[str, Any]) -> str:
    """
    测试数据库URL fixture

    返回用于测试的数据库连接URL，使用SQLite内存数据库确保快速和隔离
    """
    return test_config["database_url"]


@pytest.fixture(scope="session")
def test_db_engine(test_db_url: str) -> Generator[sa.Engine, None, None]:
    """
    测试数据库引擎fixture

    创建真实的MySQL数据库引擎，连接到专用测试数据库
    """
    engine = create_engine(
        test_db_url,
        echo=False,  # 测试时关闭SQL日志
        pool_pre_ping=True,
        pool_size=5,
        max_overflow=10,
        pool_timeout=30,
        pool_recycle=3600,
        connect_args={
            "charset": "utf8mb4",
            "autocommit": False
        }
    )

    try:
        yield engine
    finally:
        engine.dispose()


@pytest.fixture(scope="function")
def test_db_session(test_db_engine: sa.Engine) -> Generator[Session, None, None]:
    """
    测试数据库会话fixture

    为每个测试创建独立的数据库会话，测试完成后回滚所有更改，确保测试间隔离
    """
    SessionLocal = sessionmaker(bind=test_db_engine)
    session = SessionLocal()

    # 开始事务
    transaction = session.begin()

    try:
        yield session
    finally:
        # 回滚事务，确保测试间隔离
        transaction.rollback()
        session.close()


@pytest.fixture(scope="session")
def test_redis_client(test_config: Dict[str, Any]) -> Generator[redis.Redis, None, None]:
    """
    测试Redis客户端fixture

    连接到真实Redis实例的专用测试数据库，确保与生产数据隔离
    """
    try:
        client = redis.from_url(
            test_config["redis_url"],
            socket_timeout=test_config["redis_timeout"],
            decode_responses=True
        )

        # 测试连接
        client.ping()

        yield client

    except redis.ConnectionError:
        pytest.skip("Redis server not available for testing")
    finally:
        # 清理测试数据
        try:
            client.flushdb()  # 只清理测试数据库
        except:
            pass


# =============================================================================
# Hikyuu Framework Fixtures - 真实引擎和数据
# =============================================================================

@pytest.fixture(scope="session")
def hikyuu_engine(test_config: Dict[str, Any]):
    """
    Hikyuu引擎fixture

    使用真实的Hikyuu引擎，但配置最小数据集进行快速测试
    注意：这里先返回一个模拟对象，后续集成真实Hikyuu时替换
    """
    try:
        # 尝试导入Hikyuu
        import hikyuu as hku

        # 配置Hikyuu使用临时数据目录
        import tempfile
        session_temp_dir = Path(tempfile.mkdtemp(prefix="hikyuu_test_session_"))
        config = {
            "tmpdir": str(session_temp_dir),
            "datadir": str(session_temp_dir / "data"),
            "logger": {
                "level": "ERROR"  # 减少测试日志
            }
        }

        # 初始化Hikyuu引擎
        hku.init(config)
        sm = hku.StockManager.instance()

        yield sm

    except ImportError:
        # 如果Hikyuu未安装，返回基础模拟对象
        class MockHikyuuEngine:
            def __init__(self):
                self.stocks = test_config["test_stocks"]

            def get_stock(self, code: str):
                return MockStock(code) if code in self.stocks else None

            def get_stock_list(self):
                return [MockStock(code) for code in self.stocks]

        yield MockHikyuuEngine()


@pytest.fixture(scope="session")
def hikyuu_stock_manager(hikyuu_engine):
    """
    Hikyuu股票管理器fixture

    直接返回Hikyuu引擎（StockManager），保持API一致性
    """
    return hikyuu_engine


@pytest.fixture(scope="function")
def sample_stock_pool(hikyuu_stock_manager, test_config: Dict[str, Any]):
    """
    样本股票池fixture

    提供测试用的最小股票池，包含基础的A股代表股票
    """
    stock_codes = test_config["test_stocks"]

    stock_pool = []
    for code in stock_codes:
        stock = hikyuu_stock_manager.get_stock(code)
        if stock:
            stock_pool.append(stock)

    return stock_pool


# =============================================================================
# 测试数据Fixtures - 真实但最小数据集
# =============================================================================

@pytest.fixture(scope="function")
def minimal_stock_data(sample_stock_pool, test_config: Dict[str, Any]):
    """
    最小股票数据fixture

    生成用于测试的最小股票数据集，包含基本的OHLCV数据
    """
    import pandas as pd
    from datetime import datetime, timedelta

    # 生成测试日期范围
    start_date = datetime.strptime(test_config["test_date_range"]["start"], "%Y-%m-%d")
    end_date = datetime.strptime(test_config["test_date_range"]["end"], "%Y-%m-%d")
    date_range = pd.date_range(start_date, end_date, freq='D')

    stock_data = {}

    for stock in sample_stock_pool:
        # 生成模拟的股票数据
        data = []
        base_price = 10.0  # 基础价格

        for date in date_range:
            # 简单的随机价格生成
            import random
            random.seed(hash(f"{stock}_{date}"))  # 确保可重现

            change = random.uniform(-0.05, 0.05)  # ±5%变化
            base_price *= (1 + change)

            data.append({
                'date': date,
                'open': base_price * 0.99,
                'high': base_price * 1.02,
                'low': base_price * 0.98,
                'close': base_price,
                'volume': random.randint(1000000, 10000000)
            })

        stock_data[getattr(stock, 'code', str(stock))] = pd.DataFrame(data)

    return stock_data


@pytest.fixture(scope="function")
def sample_factor_data(minimal_stock_data):
    """
    样本因子数据fixture

    基于股票数据计算基础因子值（如动量、均值回归等）
    """
    import pandas as pd

    factor_data = {}

    for stock_code, stock_df in minimal_stock_data.items():
        # 计算基础技术因子
        factor_df = stock_df.copy()

        # 移动平均因子
        factor_df['ma_5'] = factor_df['close'].rolling(5).mean()
        factor_df['ma_20'] = factor_df['close'].rolling(20).mean()

        # 动量因子
        factor_df['momentum_5d'] = factor_df['close'].pct_change(5)
        factor_df['momentum_20d'] = factor_df['close'].pct_change(20)

        # 成交量因子
        factor_df['volume_ma_5'] = factor_df['volume'].rolling(5).mean()
        factor_df['volume_ratio'] = factor_df['volume'] / factor_df['volume_ma_5']

        factor_data[stock_code] = factor_df

    return factor_data


@pytest.fixture(scope="function")
def sample_signal_data(sample_factor_data):
    """
    样本信号数据fixture

    基于因子数据生成交易信号
    """
    signal_data = {}

    for stock_code, factor_df in sample_factor_data.items():
        signals = factor_df.copy()

        # 简单的信号生成逻辑
        signals['ma_signal'] = 0  # 默认无信号
        signals.loc[signals['ma_5'] > signals['ma_20'], 'ma_signal'] = 1  # 买入
        signals.loc[signals['ma_5'] < signals['ma_20'], 'ma_signal'] = -1  # 卖出

        # 动量信号
        signals['momentum_signal'] = 0
        signals.loc[signals['momentum_20d'] > 0.1, 'momentum_signal'] = 1
        signals.loc[signals['momentum_20d'] < -0.1, 'momentum_signal'] = -1

        signal_data[stock_code] = signals[['date', 'ma_signal', 'momentum_signal']]

    return signal_data


# =============================================================================
# Mock Fixtures - 仅限外部依赖
# =============================================================================

@pytest.fixture(scope="function")
def mock_external_api():
    """
    外部API Mock fixture

    模拟外部API调用，避免测试依赖网络和第三方服务
    """
    from unittest.mock import Mock, MagicMock

    mock_api = Mock()

    # 模拟常见的API响应
    mock_api.get_market_data.return_value = {
        "status": "success",
        "data": {"market": "open", "timestamp": "2023-12-01T09:30:00"}
    }

    mock_api.get_trading_calendar.return_value = [
        "2023-12-01", "2023-12-04", "2023-12-05"
    ]

    return mock_api


@pytest.fixture(scope="function")
def mock_file_system():
    """
    文件系统Mock fixture

    模拟文件系统操作，避免污染真实文件系统
    """
    from unittest.mock import Mock, patch

    mock_fs = Mock()

    # 模拟文件操作
    mock_fs.read_file.return_value = "mock file content"
    mock_fs.write_file.return_value = True
    mock_fs.exists.return_value = True

    return mock_fs


@pytest.fixture(scope="function")
def fixed_datetime():
    """
    固定时间Mock fixture

    提供固定的时间点，确保时间相关测试的可重现性
    """
    from unittest.mock import patch
    from datetime import datetime

    fixed_time = datetime(2023, 12, 1, 9, 30, 0)  # 固定为交易日开盘时间

    with patch('datetime.datetime') as mock_datetime:
        mock_datetime.now.return_value = fixed_time
        mock_datetime.today.return_value = fixed_time.date()
        yield fixed_time


# =============================================================================
# 辅助类和函数
# =============================================================================

class MockStock:
    """
    模拟股票对象

    当Hikyuu不可用时，提供基础的股票对象模拟
    """
    def __init__(self, code: str):
        self.code = code
        self.name = f"Mock Stock {code}"
        self.market = "SH" if code.startswith("sh") else "SZ"

    def __str__(self):
        return f"MockStock({self.code})"

    def __repr__(self):
        return self.__str__()


# =============================================================================
# 测试标记和跳过条件
# =============================================================================

def pytest_configure(config):
    """
    pytest配置hook

    添加自定义标记和配置测试环境
    """
    # 检查Hikyuu可用性
    try:
        import hikyuu
        config.hikyuu_available = True
    except ImportError:
        config.hikyuu_available = False

    # 检查Redis可用性
    try:
        import redis
        client = redis.Redis(host='localhost', port=6379, db=15)
        client.ping()
        config.redis_available = True
    except:
        config.redis_available = False


def pytest_collection_modifyitems(config, items):
    """
    测试收集修改hook

    根据环境可用性自动跳过某些测试
    """
    skip_hikyuu = pytest.mark.skip(reason="Hikyuu framework not available")
    skip_redis = pytest.mark.skip(reason="Redis server not available")

    for item in items:
        if "requires_hikyuu" in item.keywords and not getattr(config, 'hikyuu_available', False):
            item.add_marker(skip_hikyuu)

        if "requires_redis" in item.keywords and not getattr(config, 'redis_available', False):
            item.add_marker(skip_redis)