# 技术研究报告

## 研究概述
本报告包含A股全市场量化因子挖掘与决策支持系统的技术研究结果，重点解决Hikyuu框架集成、Point-in-Time数据实现、Agent架构设计等关键技术问题。

## 1. Hikyuu框架核心API和最佳实践

### 决策: 采用Hikyuu作为计算内核，Agent作为封装层
**理由**:
- C++核心提供高性能计算能力，满足30分钟全市场因子计算要求
- 成熟的事件驱动回测引擎，原生支持Point-in-Time数据访问
- 丰富的技术指标库和交易系统抽象模型

**替代方案考虑**:
- 纯Python实现：性能不足，无法满足大规模计算需求
- 自研C++引擎：开发成本高，成熟度不足
- 其他开源框架：生态不够成熟，文档支持有限

### 核心API使用模式
```python
# 数据获取Agent封装
class DataManagerAgent:
    def __init__(self):
        self.sm = StockManager.instance()

    def get_kdata(self, stock_code, query):
        stock = self.sm[stock_code]
        return stock.get_kdata(query)

# 因子计算Agent封装
class FactorCalculationAgent:
    def calculate_time_series_factor(self, stock, factor_def):
        # 利用Hikyuu技术指标
        close = CLOSE()
        close.set_context(stock, query)
        return MA(close, factor_def.period)
```

## 2. Point-in-Time数据访问实现

### 决策: 基于Hikyuu事件驱动引擎 + 数据库时间戳增强
**理由**:
- Hikyuu回测引擎天然支持Point-in-Time约束
- 策略上下文确保只能访问历史可用数据
- 数据库层publication_timestamp防止财务数据前视偏差

**实现方案**:
```python
# 策略开发规范
def on_bar(stg, bar):
    # 正确：使用策略上下文
    kdata = stg.get_kdata(stg.stock, Query(-20))
    current_time = stg.now()

    # 错误：直接访问可能引入未来数据
    # kdata = stock.get_kdata(Query(-20))

# 数据库设计增强
class FinancialData:
    report_date: datetime      # 报告期
    publication_date: datetime # 实际发布日期
    value: float
    is_available_at: datetime -> bool  # Point-in-Time查询
```

**替代方案考虑**:
- 纯应用层时间控制：复杂度高，容易出错
- 数据库视图方式：查询性能差，不够灵活

## 3. Agent间通信协议设计

### 决策: RESTful API + 异步消息队列混合模式
**理由**:
- RESTful API适合同步查询操作，接口清晰
- 消息队列适合异步任务，解耦Agent依赖
- 支持水平扩展和容错处理

**接口设计**:
```python
# 同步API示例
GET /api/data/kdata?stock=sh000001&start=20200101&end=20201231
POST /api/factor/calculate {factor_id, stock_pool, date_range}
POST /api/signal/generate {strategy_id, confirm=true}

# 异步消息示例
{
  "type": "factor_calculation_request",
  "factor_id": "momentum_20",
  "stock_pool": ["sh000001", "sz000002"],
  "date_range": ["20200101", "20201231"]
}
```

**替代方案考虑**:
- 纯消息队列：调试困难，缺乏即时反馈
- 纯API方式：长任务容易超时，无法解耦
- gRPC：Python生态支持不如REST成熟

## 4. 因子计算并行化策略

### 决策: 平台自适应优化 + 进程级并行 + Hikyuu内存预加载
**理由**:
- Python GIL限制，多进程比多线程更适合CPU密集计算
- Hikyuu支持数据预加载，减少I/O开销
- 可以充分利用多核CPU资源
- **DeepWiki确认**: Hikyuu支持多种SIMD优化(SSE2/3/41, AVX, AVX2, ARM NEON)
- **实际性能基准**: AMD 7950x上全A股市场(1913万条K线)计算20日MA仅需166ms
- **平台差异化**: x86和Apple Silicon需要不同的优化策略

**实现方案**:
```python
import platform
import psutil
from multiprocessing import Pool
import hikyuu as hk

class PlatformOptimizer:
    """平台自适应性能优化器"""

    def __init__(self):
        self.platform_type = self._detect_platform()
        self.cpu_count = psutil.cpu_count(logical=False)
        self._configure_hikyuu_optimization()

    def _detect_platform(self):
        """检测CPU平台类型"""
        machine = platform.machine().lower()
        system = platform.system().lower()

        if system == 'darwin' and machine in ['arm64', 'aarch64']:
            return 'apple_silicon'
        elif machine in ['x86_64', 'amd64']:
            return 'x86_64'
        elif machine in ['arm64', 'aarch64']:
            return 'arm64_linux'
        else:
            return 'generic'

    def _configure_hikyuu_optimization(self):
        """根据平台配置Hikyuu优化选项"""
        if self.platform_type == 'apple_silicon':
            # Apple Silicon优化：ARM NEON + 低精度模式
            hk.set_low_precision_mode(True)  # 启用ARM优化
            self.optimal_processes = min(8, self.cpu_count)  # M系列芯片效率核心
        elif self.platform_type == 'x86_64':
            # x86_64优化：AVX/SSE指令集
            hk.set_low_precision_mode(False)  # 保持高精度
            self.optimal_processes = min(16, self.cpu_count)  # x86高线程数
        else:
            # 通用优化
            self.optimal_processes = min(8, self.cpu_count)

def calculate_factor_parallel(stock_codes, factor_def):
    """平台优化的并行因子计算"""
    optimizer = PlatformOptimizer()

    # 预加载数据 (Hikyuu原生内存映射)
    for code in stock_codes:
        stock = sm[code]
        stock.load_kdata_to_buffer(Query.DAY)

    # 根据平台优化并行计算
    with Pool(processes=optimizer.optimal_processes) as pool:
        results = pool.map(calc_single_factor,
                          [(code, factor_def) for code in stock_codes])

    # 清理内存
    for code in stock_codes:
        stock = sm[code]
        stock.release_kdata_buffer(Query.DAY)

    return results
```

**替代方案考虑**:
- 分布式计算：复杂度高，单机性能已足够
- GPU计算：数据传输开销大，不适合时间序列计算

## 5. MySQL数据库设计和索引优化

### 决策: Hikyuu原生MySQL + 自定义业务表
**理由**:
- 复用Hikyuu成熟的数据存储方案
- 自定义表处理因子和业务数据
- MySQL针对时间序列查询优化良好

**数据库设计**:
```sql
-- 因子定义表
CREATE TABLE factor_definitions (
    id VARCHAR(50) PRIMARY KEY,
    name VARCHAR(100),
    formula TEXT,
    economic_logic TEXT,
    created_at TIMESTAMP,
    version VARCHAR(10),
    INDEX idx_created_version (created_at, version)
);

-- 因子值表（分区优化）
CREATE TABLE factor_values (
    factor_id VARCHAR(50),
    stock_code VARCHAR(10),
    date DATE,
    value DOUBLE,
    PRIMARY KEY (factor_id, stock_code, date),
    INDEX idx_date_factor (date, factor_id),
    INDEX idx_stock_date (stock_code, date)
) PARTITION BY RANGE (YEAR(date)) (
    PARTITION p2020 VALUES LESS THAN (2021),
    PARTITION p2021 VALUES LESS THAN (2022),
    -- ...
);

-- 财务数据增强（Point-in-Time）
CREATE TABLE financial_data_pit (
    stock_code VARCHAR(10),
    indicator VARCHAR(50),
    report_date DATE,
    publication_date DATE,
    value DOUBLE,
    PRIMARY KEY (stock_code, indicator, report_date),
    INDEX idx_pub_date (publication_date),
    INDEX idx_pit_query (stock_code, indicator, publication_date)
);
```

**替代方案考虑**:
- PostgreSQL：Hikyuu原生支持MySQL更好
- 时序数据库：学习成本高，迁移复杂
- NoSQL方案：查询灵活性不足

## 6. 性能优化最佳实践

### 决策: 多层缓存 + 批量处理 + 智能预加载
**具体措施**:

1. **内存缓存策略**
```python
class FactorCache:
    def __init__(self):
        self.cache = {}
        self.cache_stats = {}

    def get_factor_values(self, factor_id, stock_codes, date_range):
        cache_key = f"{factor_id}_{hash(tuple(stock_codes))}_{date_range}"
        if cache_key in self.cache:
            return self.cache[cache_key]

        # 计算并缓存
        values = self._calculate_factor(factor_id, stock_codes, date_range)
        self.cache[cache_key] = values
        return values
```

2. **批量数据转换**
```python
def optimize_cross_section_calculation(stock_pool, date, factor_func):
    # 批量获取数据
    all_data = {}
    for stock_code in stock_pool:
        stock = sm[stock_code]
        kdata = stock.get_kdata(Query(date, date))
        all_data[stock_code] = kdata.to_df()

    # Pandas批量计算
    df = pd.concat(all_data.values(), keys=all_data.keys())
    result = factor_func(df)
    return result
```

3. **智能预加载**
```python
class SmartPreloader:
    def predict_next_request(self, current_request):
        # 基于历史模式预测下次请求
        # 提前加载可能需要的数据
        pass

    def preload_for_factor_batch(self, factor_list, stock_pool):
        # 批量预加载相关数据
        for stock_code in stock_pool:
            stock = sm[stock_code]
            stock.load_kdata_to_buffer(Query.DAY)
```

## 7. 系统架构总体设计

### 决策: 分层微服务架构
```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Python SDK    │    │   Web Frontend  │    │   CLI Interface │
└─────────────────┘    └─────────────────┘    └─────────────────┘
         │                       │                       │
┌─────────────────────────────────────────────────────────────────┐
│                     API Gateway Layer                          │
└─────────────────────────────────────────────────────────────────┘
         │                       │                       │
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│  Data Agent     │    │ Factor Agent    │    │ Signal Agent    │
│  (Hikyuu Core)  │    │ (Computation)   │    │ (Generation)    │
└─────────────────┘    └─────────────────┘    └─────────────────┘
         │                       │                       │
┌─────────────────────────────────────────────────────────────────┐
│                   Storage Layer                                │
│  MySQL (Hikyuu) │ Factor Cache │ Audit Logs │ File System     │
└─────────────────────────────────────────────────────────────────┘
```

**理由**:
- 每个Agent独立部署，便于扩展和维护
- Hikyuu作为计算内核，性能有保障
- 统一API网关，便于接口管理和安全控制

## 总结

通过深入研究，确定了基于Hikyuu框架的Agent架构设计方案，解决了Point-in-Time数据访问、高性能计算、系统可扩展性等关键技术问题。该方案在保证计算性能的同时，提供了良好的系统架构和可维护性，为后续实施提供了坚实的技术基础。