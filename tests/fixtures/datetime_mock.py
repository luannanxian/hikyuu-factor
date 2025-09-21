"""
日期时间Mock - DateTime Mock

模拟日期时间操作，确保测试的时间确定性和可重现性
"""
import time
from datetime import datetime, date, timedelta, timezone
from typing import Optional, Union, Generator, Dict, Any
from unittest.mock import Mock, patch
from contextlib import contextmanager


class MockDateTime:
    """
    日期时间模拟器

    提供固定时间点和时间旅行功能，确保测试结果的确定性
    """

    def __init__(self, fixed_time: Optional[datetime] = None):
        """
        初始化日期时间模拟器

        Args:
            fixed_time: 固定的时间点，如果为None则使用当前时间
        """
        self.fixed_time = fixed_time or datetime(2024, 1, 15, 9, 30, 0)  # 默认交易日开盘时间
        self.time_offset = timedelta(0)
        self.speed_factor = 1.0  # 时间流逝速度倍数
        self.start_real_time = time.time()
        self.mock_start_time = time.time()

    def now(self, tz: Optional[timezone] = None) -> datetime:
        """
        获取当前时间

        Args:
            tz: 时区信息

        Returns:
            当前模拟时间
        """
        if self.speed_factor == 0:
            # 时间静止
            return self.fixed_time + self.time_offset

        # 计算时间流逝
        real_elapsed = time.time() - self.mock_start_time
        mock_elapsed = timedelta(seconds=real_elapsed * self.speed_factor)

        current_time = self.fixed_time + self.time_offset + mock_elapsed

        if tz:
            current_time = current_time.replace(tzinfo=tz)

        return current_time

    def today(self) -> date:
        """获取今天的日期"""
        return self.now().date()

    def set_time(self, new_time: datetime):
        """
        设置新的固定时间

        Args:
            new_time: 新的时间点
        """
        self.fixed_time = new_time
        self.time_offset = timedelta(0)
        self.mock_start_time = time.time()

    def advance_time(self, delta: timedelta):
        """
        前进时间

        Args:
            delta: 时间增量
        """
        self.time_offset += delta

    def set_speed(self, speed_factor: float):
        """
        设置时间流逝速度

        Args:
            speed_factor: 速度倍数，0表示时间静止，1表示正常速度
        """
        # 保存当前时间状态
        current_mock_time = self.now()
        self.fixed_time = current_mock_time
        self.time_offset = timedelta(0)
        self.speed_factor = speed_factor
        self.mock_start_time = time.time()

    def reset(self, new_time: Optional[datetime] = None):
        """
        重置时间模拟器

        Args:
            new_time: 重置到的时间点
        """
        self.fixed_time = new_time or datetime(2024, 1, 15, 9, 30, 0)
        self.time_offset = timedelta(0)
        self.speed_factor = 1.0
        self.mock_start_time = time.time()


@contextmanager
def fixed_time_context(fixed_time: datetime) -> Generator[MockDateTime, None, None]:
    """
    固定时间上下文管理器

    Args:
        fixed_time: 固定的时间点

    Yields:
        MockDateTime实例
    """
    mock_dt = MockDateTime(fixed_time)

    with patch('datetime.datetime') as mock_datetime:
        mock_datetime.now.return_value = fixed_time
        mock_datetime.today.return_value = fixed_time.date()
        mock_datetime.utcnow.return_value = fixed_time

        with patch('datetime.date') as mock_date:
            mock_date.today.return_value = fixed_time.date()

            with patch('time.time') as mock_time:
                mock_time.return_value = fixed_time.timestamp()

                yield mock_dt


@contextmanager
def time_travel_context(start_time: datetime,
                       end_time: Optional[datetime] = None,
                       steps: int = 10) -> Generator[MockDateTime, None, None]:
    """
    时间旅行上下文管理器

    Args:
        start_time: 开始时间
        end_time: 结束时间，如果为None则只设置开始时间
        steps: 如果指定了结束时间，分成多少步完成

    Yields:
        MockDateTime实例，支持时间前进操作
    """
    mock_dt = MockDateTime(start_time)

    if end_time and steps > 0:
        time_delta = (end_time - start_time) / steps
        mock_dt.step_delta = time_delta
        mock_dt.current_step = 0
        mock_dt.total_steps = steps

        def step_forward():
            if mock_dt.current_step < mock_dt.total_steps:
                mock_dt.advance_time(mock_dt.step_delta)
                mock_dt.current_step += 1

        mock_dt.step_forward = step_forward

    with patch('datetime.datetime') as mock_datetime:
        mock_datetime.now.side_effect = lambda tz=None: mock_dt.now(tz)
        mock_datetime.today.side_effect = lambda: mock_dt.today()
        mock_datetime.utcnow.side_effect = lambda: mock_dt.now(timezone.utc)

        with patch('datetime.date') as mock_date:
            mock_date.today.side_effect = lambda: mock_dt.today()

            with patch('time.time') as mock_time:
                mock_time.side_effect = lambda: mock_dt.now().timestamp()

                yield mock_dt


def mock_trading_hours(trading_date: date = None,
                      market_open: str = "09:30",
                      market_close: str = "15:00",
                      lunch_start: str = "11:30",
                      lunch_end: str = "13:00") -> Dict[str, datetime]:
    """
    模拟单日交易时间段（仅用于时间控制测试）

    注意：实际的交易日历应该从数据库获取真实数据
    此函数仅用于测试时间控制逻辑，不应替代真实交易日历

    Args:
        trading_date: 交易日期
        market_open: 开盘时间
        market_close: 收盘时间
        lunch_start: 午休开始时间
        lunch_end: 午休结束时间

    Returns:
        包含各个时间点的字典
    """
    if trading_date is None:
        trading_date = date(2024, 1, 15)  # 默认工作日

    def parse_time(time_str: str) -> datetime:
        hour, minute = map(int, time_str.split(':'))
        return datetime.combine(trading_date, datetime.min.time().replace(hour=hour, minute=minute))

    return {
        "pre_market": parse_time("09:15"),
        "market_open": parse_time(market_open),
        "lunch_start": parse_time(lunch_start),
        "lunch_end": parse_time(lunch_end),
        "market_close": parse_time(market_close),
        "after_market": parse_time("15:30")
    }


def create_time_series(start_time: datetime,
                      end_time: datetime,
                      frequency: str = "1min") -> list:
    """
    创建时间序列（仅用于测试时间控制）

    注意：实际的交易时间序列应该基于数据库中的真实交易日历生成
    此函数仅用于测试时间相关的逻辑控制

    Args:
        start_time: 开始时间
        end_time: 结束时间
        frequency: 频率 ("1min", "5min", "1h", "1d")

    Returns:
        时间点列表
    """
    freq_mapping = {
        "1min": timedelta(minutes=1),
        "5min": timedelta(minutes=5),
        "15min": timedelta(minutes=15),
        "30min": timedelta(minutes=30),
        "1h": timedelta(hours=1),
        "1d": timedelta(days=1)
    }

    if frequency not in freq_mapping:
        raise ValueError(f"Unsupported frequency: {frequency}")

    delta = freq_mapping[frequency]
    time_series = []
    current_time = start_time

    while current_time <= end_time:
        time_series.append(current_time)
        current_time += delta

    return time_series