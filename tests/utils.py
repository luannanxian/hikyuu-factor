"""
测试工具模块 - A股量化因子系统

提供测试所需的工具函数，包括：
- 自定义断言函数用于量化金融测试
- 数据操作和测试的辅助函数
- 测试对象的Mock工厂
- 数据库测试工具
- Hikyuu测试辅助函数

采用"真实优先"策略，最小化mock使用
"""
import pandas as pd
import numpy as np
import pytest
from datetime import datetime, date, timedelta
from typing import Dict, Any, List, Optional, Union
from pathlib import Path
import sqlalchemy as sa
from sqlalchemy.orm import Session
from unittest.mock import Mock, MagicMock
import random
import time


# =============================================================================
# 自定义断言函数 - 量化金融测试专用
# =============================================================================

def assert_factor_values_equal(actual: Union[pd.Series, pd.DataFrame],
                             expected: Union[pd.Series, pd.DataFrame],
                             tolerance: float = 1e-6) -> None:
    """
    断言因子值相等，支持数值容差

    Args:
        actual: 实际的因子值
        expected: 期望的因子值
        tolerance: 数值容差，默认1e-6

    Raises:
        AssertionError: 当因子值不匹配时
    """
    if isinstance(actual, pd.Series) and isinstance(expected, pd.Series):
        pd.testing.assert_series_equal(actual, expected, rtol=tolerance, atol=tolerance)
    elif isinstance(actual, pd.DataFrame) and isinstance(expected, pd.DataFrame):
        pd.testing.assert_frame_equal(actual, expected, rtol=tolerance, atol=tolerance)
    else:
        raise AssertionError(f"Type mismatch: actual={type(actual)}, expected={type(expected)}")


def assert_signals_equal(actual: Union[pd.Series, pd.DataFrame],
                        expected: Union[pd.Series, pd.DataFrame]) -> None:
    """
    断言交易信号相等

    Args:
        actual: 实际的交易信号
        expected: 期望的交易信号

    Raises:
        AssertionError: 当信号不匹配时
    """
    if isinstance(actual, pd.Series) and isinstance(expected, pd.Series):
        pd.testing.assert_series_equal(actual, expected, check_dtype=False)
    elif isinstance(actual, pd.DataFrame) and isinstance(expected, pd.DataFrame):
        pd.testing.assert_frame_equal(actual, expected, check_dtype=False)
    else:
        raise AssertionError(f"Signal type mismatch: actual={type(actual)}, expected={type(expected)}")


def assert_dataframes_equal(actual: pd.DataFrame,
                           expected: pd.DataFrame,
                           check_column_order: bool = True,
                           tolerance: float = 1e-6) -> None:
    """
    断言DataFrame相等，支持列顺序和数值容差配置

    Args:
        actual: 实际的DataFrame
        expected: 期望的DataFrame
        check_column_order: 是否检查列顺序
        tolerance: 数值容差

    Raises:
        AssertionError: 当DataFrame不匹配时
    """
    pd.testing.assert_frame_equal(
        actual, expected,
        check_like=not check_column_order,
        rtol=tolerance, atol=tolerance
    )


def assert_price_data_valid(price_data: pd.DataFrame,
                           required_columns: List[str] = None) -> None:
    """
    断言价格数据有效性

    Args:
        price_data: 价格数据DataFrame
        required_columns: 必需的列名列表

    Raises:
        AssertionError: 当价格数据无效时
    """
    if required_columns is None:
        required_columns = ['open', 'high', 'low', 'close', 'volume']

    # 检查必需列存在
    missing_columns = set(required_columns) - set(price_data.columns)
    assert not missing_columns, f"Missing required columns: {missing_columns}"

    # 检查数据有效性
    assert not price_data.empty, "Price data should not be empty"

    # 检查价格逻辑关系
    if all(col in price_data.columns for col in ['open', 'high', 'low', 'close']):
        # high >= max(open, close) and low <= min(open, close)
        invalid_high = (price_data['high'] < price_data[['open', 'close']].max(axis=1)).any()
        invalid_low = (price_data['low'] > price_data[['open', 'close']].min(axis=1)).any()

        assert not invalid_high, "High price should be >= max(open, close)"
        assert not invalid_low, "Low price should be <= min(open, close)"

    # 检查成交量非负
    if 'volume' in price_data.columns:
        assert (price_data['volume'] >= 0).all(), "Volume should be non-negative"


def assert_performance_within_bounds(performance_metrics: Dict[str, float],
                                   expected_bounds: Dict[str, tuple]) -> None:
    """
    断言性能指标在预期范围内

    Args:
        performance_metrics: 性能指标字典
        expected_bounds: 期望范围字典，格式为 {metric: (min_val, max_val)}

    Raises:
        AssertionError: 当性能指标超出范围时
    """
    violations = []

    for metric, (min_val, max_val) in expected_bounds.items():
        if metric in performance_metrics:
            actual_val = performance_metrics[metric]
            if not (min_val <= actual_val <= max_val):
                violations.append(
                    f"{metric}: {actual_val} not in range [{min_val}, {max_val}]"
                )

    assert not violations, f"Performance metrics out of bounds: {violations}"


# =============================================================================
# 辅助函数 - 数据操作和测试
# =============================================================================

def create_test_stock_data(stock_codes: List[str],
                          start_date: str = "2023-01-01",
                          end_date: str = "2023-12-31",
                          base_price: float = 10.0,
                          volatility: float = 0.02) -> Dict[str, pd.DataFrame]:
    """
    创建测试用股票数据

    Args:
        stock_codes: 股票代码列表
        start_date: 开始日期
        end_date: 结束日期
        base_price: 基础价格
        volatility: 波动率

    Returns:
        股票数据字典，格式为 {stock_code: DataFrame}
    """
    start_dt = datetime.strptime(start_date, "%Y-%m-%d")
    end_dt = datetime.strptime(end_date, "%Y-%m-%d")
    date_range = pd.date_range(start_dt, end_dt, freq='D')

    stock_data = {}

    for stock_code in stock_codes:
        # 使用股票代码作为随机种子确保可重现
        random.seed(hash(stock_code))

        data = []
        current_price = base_price

        for date in date_range:
            # 模拟价格变化
            daily_return = random.normalvariate(0, volatility)
            current_price *= (1 + daily_return)

            # 生成OHLC数据
            open_price = current_price * random.uniform(0.98, 1.02)
            close_price = current_price
            high_price = max(open_price, close_price) * random.uniform(1.0, 1.03)
            low_price = min(open_price, close_price) * random.uniform(0.97, 1.0)
            volume = random.randint(1000000, 10000000)

            data.append({
                'date': date,
                'open': round(open_price, 2),
                'high': round(high_price, 2),
                'low': round(low_price, 2),
                'close': round(close_price, 2),
                'volume': volume
            })

        stock_data[stock_code] = pd.DataFrame(data)

    return stock_data


def create_test_factor_data(stock_data: Dict[str, pd.DataFrame],
                           factor_names: List[str] = None) -> Dict[str, pd.DataFrame]:
    """
    基于股票数据创建测试因子数据

    Args:
        stock_data: 股票数据字典
        factor_names: 因子名称列表

    Returns:
        因子数据字典，格式为 {stock_code: DataFrame}
    """
    if factor_names is None:
        factor_names = ['momentum_5d', 'momentum_20d', 'ma_ratio', 'volume_ratio']

    factor_data = {}

    for stock_code, stock_df in stock_data.items():
        factor_df = stock_df.copy()

        # 计算各种因子
        if 'momentum_5d' in factor_names:
            factor_df['momentum_5d'] = factor_df['close'].pct_change(5)

        if 'momentum_20d' in factor_names:
            factor_df['momentum_20d'] = factor_df['close'].pct_change(20)

        if 'ma_ratio' in factor_names:
            ma_5 = factor_df['close'].rolling(5).mean()
            ma_20 = factor_df['close'].rolling(20).mean()
            factor_df['ma_ratio'] = ma_5 / ma_20

        if 'volume_ratio' in factor_names:
            volume_ma = factor_df['volume'].rolling(20).mean()
            factor_df['volume_ratio'] = factor_df['volume'] / volume_ma

        factor_data[stock_code] = factor_df

    return factor_data


def setup_test_database_tables(session: Session) -> None:
    """
    为测试设置数据库表

    Args:
        session: SQLAlchemy会话对象
    """
    # 创建测试表结构
    create_tables_sql = """
    CREATE TABLE IF NOT EXISTS test_stocks (
        id INT AUTO_INCREMENT PRIMARY KEY,
        code VARCHAR(10) NOT NULL UNIQUE,
        name VARCHAR(50),
        market VARCHAR(10),
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    );

    CREATE TABLE IF NOT EXISTS test_factors (
        id INT AUTO_INCREMENT PRIMARY KEY,
        stock_code VARCHAR(10) NOT NULL,
        factor_name VARCHAR(50) NOT NULL,
        factor_value DECIMAL(15, 6),
        calc_date DATE NOT NULL,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        INDEX idx_stock_factor_date (stock_code, factor_name, calc_date)
    );

    CREATE TABLE IF NOT EXISTS test_signals (
        id INT AUTO_INCREMENT PRIMARY KEY,
        stock_code VARCHAR(10) NOT NULL,
        signal_type VARCHAR(20) NOT NULL,
        signal_value INT NOT NULL,
        signal_date DATE NOT NULL,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        INDEX idx_stock_signal_date (stock_code, signal_type, signal_date)
    );
    """

    # 执行建表语句
    for statement in create_tables_sql.strip().split(';'):
        if statement.strip():
            session.execute(sa.text(statement))
    session.commit()


def cleanup_test_database(session: Session) -> None:
    """
    清理测试数据库

    Args:
        session: SQLAlchemy会话对象
    """
    # 清理测试数据
    cleanup_sql = """
    TRUNCATE TABLE test_signals;
    TRUNCATE TABLE test_factors;
    TRUNCATE TABLE test_stocks;
    """

    for statement in cleanup_sql.strip().split(';'):
        if statement.strip():
            session.execute(sa.text(statement))
    session.commit()


def wait_for_calculation(max_wait_seconds: int = 30,
                        check_function: callable = None) -> bool:
    """
    等待计算完成

    Args:
        max_wait_seconds: 最大等待秒数
        check_function: 检查函数，返回True表示计算完成

    Returns:
        True表示计算完成，False表示超时
    """
    if check_function is None:
        # 默认等待策略
        time.sleep(1)
        return True

    start_time = time.time()
    while time.time() - start_time < max_wait_seconds:
        if check_function():
            return True
        time.sleep(0.1)

    return False


def compare_factor_results(result1: pd.DataFrame,
                          result2: pd.DataFrame,
                          tolerance: float = 1e-6) -> Dict[str, Any]:
    """
    比较因子计算结果

    Args:
        result1: 第一个结果
        result2: 第二个结果
        tolerance: 数值容差

    Returns:
        比较结果字典
    """
    comparison = {
        'shape_match': result1.shape == result2.shape,
        'columns_match': list(result1.columns) == list(result2.columns),
        'identical': False,
        'differences': {}
    }

    if comparison['shape_match'] and comparison['columns_match']:
        try:
            pd.testing.assert_frame_equal(result1, result2, rtol=tolerance, atol=tolerance)
            comparison['identical'] = True
        except AssertionError as e:
            comparison['differences']['error'] = str(e)

            # 详细差异分析
            numeric_cols = result1.select_dtypes(include=[np.number]).columns
            for col in numeric_cols:
                diff = np.abs(result1[col] - result2[col])
                max_diff = diff.max()
                if max_diff > tolerance:
                    comparison['differences'][col] = {
                        'max_difference': max_diff,
                        'mean_difference': diff.mean(),
                        'positions_different': (diff > tolerance).sum()
                    }

    return comparison


# =============================================================================
# Mock工厂类 - 测试对象创建
# =============================================================================

class MockStockFactory:
    """Mock股票对象工厂"""

    @staticmethod
    def create_stock(code: str, name: str = None, market: str = None):
        """
        创建Mock股票对象

        Args:
            code: 股票代码
            name: 股票名称
            market: 市场代码

        Returns:
            Mock股票对象
        """
        stock = Mock()
        stock.code = code
        stock.name = name or f"股票{code}"
        stock.market = market or ("SH" if code.startswith("sh") else "SZ")
        stock.valid = True

        return stock

    @staticmethod
    def create_stock_list(codes: List[str]) -> List[Mock]:
        """
        创建Mock股票列表

        Args:
            codes: 股票代码列表

        Returns:
            Mock股票对象列表
        """
        return [MockStockFactory.create_stock(code) for code in codes]


class MockFactorFactory:
    """Mock因子对象工厂"""

    @staticmethod
    def create_factor_result(stock_code: str,
                           factor_name: str,
                           factor_values: List[float],
                           dates: List[datetime] = None):
        """
        创建Mock因子结果

        Args:
            stock_code: 股票代码
            factor_name: 因子名称
            factor_values: 因子值列表
            dates: 日期列表

        Returns:
            Mock因子结果对象
        """
        if dates is None:
            dates = pd.date_range(end=datetime.now(), periods=len(factor_values), freq='D')

        factor_result = Mock()
        factor_result.stock_code = stock_code
        factor_result.factor_name = factor_name
        factor_result.values = pd.Series(factor_values, index=dates)
        factor_result.is_valid = True

        return factor_result


class MockSignalFactory:
    """Mock信号对象工厂"""

    @staticmethod
    def create_signal(stock_code: str,
                     signal_type: str,
                     signal_values: List[int],
                     dates: List[datetime] = None):
        """
        创建Mock交易信号

        Args:
            stock_code: 股票代码
            signal_type: 信号类型
            signal_values: 信号值列表 (1买入, 0持有, -1卖出)
            dates: 日期列表

        Returns:
            Mock信号对象
        """
        if dates is None:
            dates = pd.date_range(end=datetime.now(), periods=len(signal_values), freq='D')

        signal = Mock()
        signal.stock_code = stock_code
        signal.signal_type = signal_type
        signal.values = pd.Series(signal_values, index=dates)
        signal.is_valid = True

        return signal


class MockHikyuuKDataFactory:
    """Mock Hikyuu KData对象工厂"""

    @staticmethod
    def create_kdata(stock_code: str,
                    start_date: datetime = None,
                    end_date: datetime = None,
                    frequency: str = "DAY") -> Mock:
        """
        创建Mock Hikyuu KData对象

        Args:
            stock_code: 股票代码
            start_date: 开始日期
            end_date: 结束日期
            frequency: 频率

        Returns:
            Mock KData对象
        """
        if start_date is None:
            start_date = datetime.now() - timedelta(days=365)
        if end_date is None:
            end_date = datetime.now()

        # 生成测试数据
        date_range = pd.date_range(start_date, end_date, freq='D')

        kdata = Mock()
        kdata.stock_code = stock_code
        kdata.start_date = start_date
        kdata.end_date = end_date
        kdata.frequency = frequency
        kdata.size = len(date_range)

        # 模拟数据访问方法
        def get_datetime_list():
            return list(date_range)

        def get_close_list():
            return [10.0 + i * 0.1 for i in range(len(date_range))]

        kdata.get_datetime_list = get_datetime_list
        kdata.get_close_list = get_close_list
        kdata.empty = len(date_range) == 0

        return kdata


# =============================================================================
# 数据库工具函数
# =============================================================================

def create_test_tables(engine: sa.Engine) -> None:
    """
    创建测试表

    Args:
        engine: SQLAlchemy引擎对象
    """
    with engine.connect() as conn:
        # 创建测试表结构
        create_tables_sql = """
        CREATE TABLE IF NOT EXISTS test_stocks (
            id INT AUTO_INCREMENT PRIMARY KEY,
            code VARCHAR(10) NOT NULL UNIQUE,
            name VARCHAR(50),
            market VARCHAR(10),
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );

        CREATE TABLE IF NOT EXISTS test_factors (
            id INT AUTO_INCREMENT PRIMARY KEY,
            stock_code VARCHAR(10) NOT NULL,
            factor_name VARCHAR(50) NOT NULL,
            factor_value DECIMAL(15, 6),
            calc_date DATE NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            INDEX idx_stock_factor_date (stock_code, factor_name, calc_date)
        );
        """

        for statement in create_tables_sql.strip().split(';'):
            if statement.strip():
                conn.execute(sa.text(statement))
        conn.commit()


def truncate_test_tables(session: Session) -> None:
    """
    清空测试表数据

    Args:
        session: SQLAlchemy会话对象
    """
    truncate_sql = """
    TRUNCATE TABLE test_factors;
    TRUNCATE TABLE test_stocks;
    """

    for statement in truncate_sql.strip().split(';'):
        if statement.strip():
            session.execute(sa.text(statement))
    session.commit()


def insert_test_data(session: Session,
                    test_data: Dict[str, List[Dict]]) -> None:
    """
    插入测试数据

    Args:
        session: SQLAlchemy会话对象
        test_data: 测试数据字典
    """
    for table_name, records in test_data.items():
        if records:
            # 构建插入语句
            if table_name == 'test_stocks':
                for record in records:
                    session.execute(
                        sa.text("INSERT INTO test_stocks (code, name, market) VALUES (:code, :name, :market)"),
                        record
                    )
            elif table_name == 'test_factors':
                for record in records:
                    session.execute(
                        sa.text("INSERT INTO test_factors (stock_code, factor_name, factor_value, calc_date) "
                               "VALUES (:stock_code, :factor_name, :factor_value, :calc_date)"),
                        record
                    )

    session.commit()


def verify_database_state(session: Session,
                         expected_counts: Dict[str, int]) -> bool:
    """
    验证数据库状态

    Args:
        session: SQLAlchemy会话对象
        expected_counts: 期望的记录数字典

    Returns:
        True表示状态正确，False表示状态错误
    """
    for table_name, expected_count in expected_counts.items():
        result = session.execute(sa.text(f"SELECT COUNT(*) FROM {table_name}"))
        actual_count = result.scalar()

        if actual_count != expected_count:
            return False

    return True