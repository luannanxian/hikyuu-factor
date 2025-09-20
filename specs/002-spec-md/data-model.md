# 数据模型设计

## 基于Hikyuu框架的数据架构

### 1. 平台优化配置模型

#### PlatformOptimizer (新增)
```python
from dataclasses import dataclass
from enum import Enum
import platform
import psutil

class PlatformType(Enum):
    APPLE_SILICON = "apple_silicon"
    X86_64 = "x86_64"
    ARM64_LINUX = "arm64_linux"
    GENERIC = "generic"

@dataclass
class OptimizationConfig:
    """平台优化配置"""
    platform_type: PlatformType
    cpu_count: int
    optimal_processes: int
    low_precision_mode: bool
    simd_support: List[str]  # 支持的SIMD指令集

    @classmethod
    def auto_detect(cls) -> 'OptimizationConfig':
        """自动检测平台并生成优化配置"""
        machine = platform.machine().lower()
        system = platform.system().lower()
        cpu_count = psutil.cpu_count(logical=False)

        if system == 'darwin' and machine in ['arm64', 'aarch64']:
            return cls(
                platform_type=PlatformType.APPLE_SILICON,
                cpu_count=cpu_count,
                optimal_processes=min(8, cpu_count),
                low_precision_mode=True,
                simd_support=['ARM_NEON']
            )
        elif machine in ['x86_64', 'amd64']:
            return cls(
                platform_type=PlatformType.X86_64,
                cpu_count=cpu_count,
                optimal_processes=min(16, cpu_count),
                low_precision_mode=False,
                simd_support=['SSE2', 'SSE3', 'SSE41', 'AVX', 'AVX2']
            )
        else:
            return cls(
                platform_type=PlatformType.GENERIC,
                cpu_count=cpu_count,
                optimal_processes=min(8, cpu_count),
                low_precision_mode=False,
                simd_support=['GENERIC']
            )
```

### 2. Hikyuu原生数据模型复用

#### 股票对象 (使用Hikyuu Stock)
```python
# 直接使用Hikyuu的Stock对象
from hikyuu import StockManager, Stock

# 获取股票管理器实例
sm = StockManager.instance()

# 股票池管理逻辑
class StockPoolManager:
    def __init__(self):
        self.sm = sm

    def get_valid_stocks(self, date: datetime) -> List[Stock]:
        """获取有效股票池，排除ST/*ST和上市不足60日的股票"""
        stocks = self.sm.get_stock_list()
        valid_stocks = []

        for stock in stocks:
            # FR-001: 排除ST/*ST股票
            if "ST" in stock.name or "*" in stock.name:
                continue

            # 检查上市时间
            start_date = stock.start_datetime.date()
            trading_days = self._calculate_trading_days(start_date, date.date())
            if trading_days < 60:
                continue

            valid_stocks.append(stock)

        return valid_stocks

    def _calculate_trading_days(self, start: date, end: date) -> int:
        """计算交易日天数"""
        # 使用Hikyuu的交易日历
        from hikyuu import get_stock_type_info
        return len([d for d in get_stock_type_info(1).get_trading_calendar(start, end)])
```

#### K线数据 (使用Hikyuu KData)
```python
# 直接使用Hikyuu的KData和KRecord
from hikyuu import Query, KData, KRecord

class MarketDataService:
    def __init__(self):
        self.sm = StockManager.instance()

    def get_kdata(self, stock_code: str, query: Query) -> KData:
        """获取K线数据，内置Point-in-Time保护"""
        stock = self.sm[stock_code]
        kdata = stock.get_kdata(query)

        # Hikyuu的KData天然支持Point-in-Time
        return kdata

    def validate_kdata_quality(self, kdata: KData) -> List[str]:
        """数据质量检查"""
        issues = []

        for i in range(len(kdata)):
            record = kdata[i]

            # FR-007: 价格一致性检查
            if record.high < max(record.open, record.close, record.low):
                issues.append(f"Invalid high price at {record.datetime}")

            if record.low > min(record.open, record.close, record.high):
                issues.append(f"Invalid low price at {record.datetime}")

            # 价格变动检查
            if i > 0:
                prev_record = kdata[i-1]
                change_pct = abs(record.close - prev_record.close) / prev_record.close
                if change_pct > 0.3:  # 30%阈值
                    issues.append(f"Excessive price change at {record.datetime}: {change_pct:.2%}")

        return issues
```

#### 财务数据 (扩展Hikyuu FINANCE)
```python
# 基于Hikyuu FINANCE指标，增加Point-in-Time约束
from hikyuu import FINANCE

class FinancialDataService:
    def __init__(self):
        self.sm = StockManager.instance()

    def get_financial_at_date(self, stock_code: str, indicator: str,
                            as_of_date: datetime) -> Optional[float]:
        """Point-in-Time财务数据获取"""
        stock = self.sm[stock_code]

        # 构建查询，确保只获取as_of_date之前发布的数据
        # 这需要在数据库层面增加publication_date字段
        finance_data = FINANCE()
        finance_data.set_context(stock, Query(as_of_date))

        # 自定义Point-in-Time逻辑
        return self._get_pit_financial_value(stock_code, indicator, as_of_date)

    def _get_pit_financial_value(self, stock_code: str, indicator: str,
                               as_of_date: datetime) -> Optional[float]:
        """从扩展财务数据表获取Point-in-Time数据"""
        # 查询扩展的财务数据表，包含publication_date
        return self._query_pit_database(stock_code, indicator, as_of_date)
```

### 2. 系统扩展数据模型

#### 因子定义 (Factor Definition)
```python
# 基于Hikyuu指标的因子定义
@dataclass
class FactorDefinition:
    id: str                    # 因子唯一标识
    name: str                  # 因子名称
    category: FactorCategory   # 因子分类
    hikyuu_formula: str        # 基于Hikyuu的计算公式
    economic_logic: str        # 经济逻辑说明

    # 版本控制
    version: str               # 版本号
    created_by: str            # 创建者
    created_at: datetime       # 创建时间
    status: FactorStatus       # 因子状态

    # Hikyuu相关配置
    hikyuu_dependencies: List[str]  # 依赖的Hikyuu指标
    parameters: Dict[str, Any]      # 参数配置

class FactorCategory(Enum):
    MOMENTUM = "momentum"      # 动量因子
    VALUE = "value"           # 价值因子
    QUALITY = "quality"       # 质量因子
    GROWTH = "growth"         # 成长因子
    RISK = "risk"             # 风险因子
    TECHNICAL = "technical"   # 技术因子

# 基于Hikyuu的因子计算服务
class HikyuuFactorCalculator:
    def __init__(self):
        self.sm = StockManager.instance()

    def calculate_time_series_factor(self, factor_def: FactorDefinition,
                                   stock: Stock, query: Query) -> Indicator:
        """使用Hikyuu计算时间序列因子"""
        # 设置上下文
        close = CLOSE()
        close.set_context(stock, query)

        # 解析并执行Hikyuu公式
        # 例如: "MA(CLOSE(), 20) / MA(CLOSE(), 5) - 1"
        result = eval(factor_def.hikyuu_formula, {
            'CLOSE': lambda: close,
            'MA': MA,
            'EMA': EMA,
            'MACD': MACD,
            # ... 其他Hikyuu指标
        })

        return result

    def calculate_cross_section_factor(self, factor_def: FactorDefinition,
                                     stocks: List[Stock], date: datetime) -> Dict[str, float]:
        """计算横截面因子"""
        results = {}

        for stock in stocks:
            try:
                # 为每只股票计算因子值
                query = Query(date, date)
                indicator = self.calculate_time_series_factor(factor_def, stock, query)

                if len(indicator) > 0:
                    results[stock.market_code] = float(indicator[-1])

            except Exception as e:
                # 记录计算失败的股票
                print(f"Failed to calculate {factor_def.id} for {stock.market_code}: {e}")

        return results
```

#### 因子验证报告 (基于Hikyuu回测)
```python
@dataclass
class FactorValidationReport:
    id: str                    # 报告ID
    factor_id: str             # 因子ID

    # 基于Hikyuu的回测结果
    hikyuu_backtest_result: Any  # Hikyuu回测结果对象

    # 可配置的样本划分
    validation_config: ValidationConfig  # 验证配置

    # 分析结果
    ic_metrics: Dict[str, float]
    layered_returns: Dict[str, Any]
    generated_at: datetime

@dataclass
class ValidationConfig:
    """可配置的验证参数"""
    train_period: DateRange    # 训练期（可配置）
    test_period: DateRange     # 测试期（可配置）
    validation_period: DateRange # 验证期（可配置）

    # 默认配置
    @classmethod
    def default_config(cls) -> 'ValidationConfig':
        return cls(
            train_period=DateRange('2010-01-01', '2016-12-31'),
            test_period=DateRange('2017-01-01', '2020-12-31'),
            validation_period=DateRange('2021-01-01', datetime.now().strftime('%Y-%m-%d'))
        )

    # 自定义配置验证
    def validate_periods(self) -> bool:
        """验证期间配置的合理性"""
        if self.train_period.end >= self.test_period.start:
            return False
        if self.test_period.end >= self.validation_period.start:
            return False
        return True

class HikyuuFactorValidator:
    def __init__(self):
        self.sm = StockManager.instance()

    def run_factor_validation(self, factor_def: FactorDefinition,
                            validation_config: ValidationConfig = None) -> FactorValidationReport:
        """使用Hikyuu回测引擎进行因子验证，支持自定义验证周期"""

        # 使用默认配置或自定义配置
        if validation_config is None:
            validation_config = ValidationConfig.default_config()

        # 验证配置的合理性
        if not validation_config.validate_periods():
            raise ValueError("Invalid validation period configuration")

        # 构建基于因子的交易策略
        def create_factor_strategy(factor_def: FactorDefinition):
            # 创建信号生成器
            sg = self._create_factor_signal_generator(factor_def)

            # 创建交易管理器
            tm = crtTM(init_cash=1000000, name=f"Factor_{factor_def.id}")

            # 创建交易系统
            sys = SYS_Simple(tm=tm, sg=sg, mm=MM_FixedPercent(0.95))

            return sys

        # 执行回测
        strategy = create_factor_strategy(factor_def)

        # 运行不同时期的回测
        train_result = self._run_backtest_period(strategy, validation_config.train_period)
        test_result = self._run_backtest_period(strategy, validation_config.test_period)
        validation_result = self._run_backtest_period(strategy, validation_config.validation_period)

        # 生成报告
        return FactorValidationReport(
            id=f"validation_{factor_def.id}_{datetime.now().strftime('%Y%m%d')}",
            factor_id=factor_def.id,
            validation_config=validation_config,
            hikyuu_backtest_result={
                'train': train_result,
                'test': test_result,
                'validation': validation_result
            },
            generated_at=datetime.now()
        )
```

#### 交易信号 (基于Hikyuu交易系统)
```python
@dataclass
class TradingSignal:
    id: str                    # 信号ID
    stock_code: str            # 股票代码
    signal_date: date          # 信号日期
    signal_type: SignalType    # 信号类型
    weight: float              # 权重建议

    # 基于Hikyuu的信号生成信息
    hikyuu_signal_info: Dict[str, Any]  # Hikyuu信号生成器的详细信息
    factor_exposure: Dict[str, float]   # 因子暴露归因

    # 确认状态
    confirmation_status: ConfirmationStatus
    confirmed_by: Optional[str]
    confirmed_at: Optional[datetime]

class HikyuuSignalGenerator:
    def __init__(self):
        self.sm = StockManager.instance()

    def generate_signals(self, strategy_config: Dict[str, Any],
                        confirmation_required: bool = True) -> List[TradingSignal]:
        """基于Hikyuu交易系统生成信号"""

        # FR-003: 强制人工确认
        if not confirmation_required:
            raise ValueError("Human confirmation is mandatory for signal generation")

        # 创建Hikyuu交易系统
        tm = crtTM(**strategy_config['tm_params'])
        sg = self._build_signal_generator(strategy_config['signal_params'])
        mm = self._build_money_manager(strategy_config['mm_params'])

        sys = SYS_Simple(tm=tm, sg=sg, mm=mm)

        # 执行当日信号生成
        today = datetime.now().date()
        signals = []

        for stock_code in strategy_config['stock_universe']:
            stock = self.sm[stock_code]
            query = Query(today, today)

            try:
                # 运行交易系统
                sys.run(stock, query)

                # 从交易系统获取信号
                trades = tm.get_trade_list()
                if trades:
                    latest_trade = trades[-1]
                    signal = TradingSignal(
                        id=f"signal_{stock_code}_{today.strftime('%Y%m%d')}",
                        stock_code=stock_code,
                        signal_date=today,
                        signal_type=self._convert_hikyuu_signal(latest_trade.business),
                        weight=self._calculate_position_weight(latest_trade),
                        hikyuu_signal_info={
                            'trade_info': str(latest_trade),
                            'signal_generator': str(sg),
                            'strategy_params': strategy_config
                        },
                        confirmation_status=ConfirmationStatus.PENDING
                    )
                    signals.append(signal)

            except Exception as e:
                print(f"Failed to generate signal for {stock_code}: {e}")

        return signals

    def _build_signal_generator(self, params: Dict[str, Any]):
        """构建Hikyuu信号生成器"""
        signal_type = params.get('type', 'dual_ma')

        if signal_type == 'dual_ma':
            # 双均线信号
            return SG_Cross(MA(CLOSE(), params['fast_period']),
                          MA(CLOSE(), params['slow_period']))
        elif signal_type == 'factor_based':
            # 基于因子的信号
            return self._create_factor_signal(params['factor_config'])
        else:
            raise ValueError(f"Unsupported signal type: {signal_type}")
```

#### 审计日志 (系统级)
```python
@dataclass
class SystemAuditLog:
    id: str                    # 日志ID
    event_type: str            # 事件类型
    event_data: Dict[str, Any] # 事件数据

    # 操作信息
    user_id: str               # 操作用户
    timestamp: datetime        # 事件时间

    # Hikyuu相关上下文
    hikyuu_context: Optional[Dict[str, Any]]  # Hikyuu运行上下文

    # 不可变性保证
    hash_value: str            # 数据哈希值
    previous_hash: str         # 前一条记录哈希值

class AuditLogger:
    def __init__(self):
        self.logs = []

    def log_hikyuu_operation(self, operation: str, stock: Stock,
                           result: Any, user_id: str):
        """记录Hikyuu相关操作"""
        log_entry = SystemAuditLog(
            id=f"audit_{uuid.uuid4()}",
            event_type=f"hikyuu_{operation}",
            event_data={
                'stock_code': stock.market_code,
                'stock_name': stock.name,
                'operation': operation,
                'result_summary': str(result)[:500]  # 截断长结果
            },
            user_id=user_id,
            timestamp=datetime.now(),
            hikyuu_context={
                'stock_info': {
                    'code': stock.market_code,
                    'type': stock.type,
                    'market': stock.market
                },
                'hikyuu_version': get_version()  # Hikyuu版本信息
            },
            hash_value="",  # 计算后填入
            previous_hash=self.logs[-1].hash_value if self.logs else ""
        )

        # 计算哈希值确保不可变性
        log_entry.hash_value = self._calculate_hash(log_entry)
        self.logs.append(log_entry)

        # FR-009: 持久化审计日志
        self._persist_log(log_entry)
```

## 数据流转架构

### Hikyuu数据流
```
Hikyuu原生数据 (Stock, KData, FINANCE)
    ↓
业务逻辑封装 (StockPoolManager, MarketDataService)
    ↓
因子计算 (HikyuuFactorCalculator)
    ↓
因子验证 (HikyuuFactorValidator + Hikyuu回测引擎)
    ↓
信号生成 (HikyuuSignalGenerator + 交易系统)
    ↓
审计记录 (AuditLogger)
```

### Agent架构集成
```python
class DataManagerAgent:
    """数据管理Agent - 封装Hikyuu数据访问"""
    def __init__(self):
        self.stock_pool_manager = StockPoolManager()
        self.market_data_service = MarketDataService()
        self.financial_data_service = FinancialDataService()

class FactorCalculationAgent:
    """因子计算Agent - 基于Hikyuu指标库"""
    def __init__(self):
        self.calculator = HikyuuFactorCalculator()
        self.factor_registry = {}

class ValidationAgent:
    """验证Agent - 使用Hikyuu回测引擎"""
    def __init__(self):
        self.validator = HikyuuFactorValidator()

class SignalGenerationAgent:
    """信号生成Agent - 基于Hikyuu交易系统"""
    def __init__(self):
        self.signal_generator = HikyuuSignalGenerator()
        self.audit_logger = AuditLogger()
```

## 总结

该数据模型设计充分利用了Hikyuu框架的现有数据结构和计算能力：

1. **复用Hikyuu原生模型**: Stock, KData, FINANCE等核心对象
2. **扩展业务逻辑**: 在Hikyuu基础上添加业务规则和约束
3. **保持系统一致性**: 所有计算基于统一的Hikyuu引擎
4. **简化开发复杂度**: 避免重复造轮子，专注业务价值

这种设计确保了系统的高性能、可靠性和可维护性，同时满足量化因子系统的专业需求。