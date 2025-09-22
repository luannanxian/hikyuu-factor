# 集成测试环境配置

## 概述
本配置文件定义了Hikyuu Factor项目的集成测试环境设置，包括数据库配置、Agent配置、性能基准等。

## 数据库配置

### MySQL测试数据库
- **主机**: 192.168.3.46:3306
- **数据库**: hikyuu_factor_test
- **用户**: remote
- **密码**: remote123456
- **字符集**: utf8mb4
- **排序规则**: utf8mb4_unicode_ci

### 连接池配置
```yaml
database:
  host: "192.168.3.46"
  port: 3306
  user: "remote"
  password: "remote123456"
  database: "hikyuu_factor_test"
  charset: "utf8mb4"
  pool_size: 10
  max_overflow: 20
  pool_pre_ping: true
  pool_recycle: 3600
  echo: false  # 测试时不显示SQL
```

## Agent配置

### 通信配置
```yaml
communication:
  timeout: 30
  retry_attempts: 3
  heartbeat_interval: 5
  message_queue_size: 1000
```

### Agent端口配置
```yaml
agents:
  data_manager:
    port: 8001
    max_concurrent_tasks: 50
  factor_calculator:
    port: 8002
    max_concurrent_tasks: 100
  validator:
    port: 8003
    max_concurrent_tasks: 30
  signal_generator:
    port: 8004
    max_concurrent_tasks: 20
```

## 性能基准配置

### 性能目标
```yaml
performance:
  target_throughput: 1000        # 每秒处理量
  max_response_time: 5.0         # 最大响应时间(秒)
  max_memory_usage: 1000         # 最大内存使用(MB)
  min_cpu_efficiency: 0.7        # 最小CPU效率
  max_concurrent_connections: 100 # 最大并发连接数
```

### 性能基准线
```yaml
benchmarks:
  factor_calculation:
    duration: 5.0      # 5秒内完成
    memory_delta: 100  # 增加100MB内存
    avg_cpu: 50.0      # 平均CPU使用50%

  data_processing:
    duration: 2.0      # 2秒内完成
    memory_delta: 50   # 增加50MB内存
    avg_cpu: 30.0      # 平均CPU使用30%

  agent_response:
    duration: 1.0      # 1秒内响应
    memory_delta: 10   # 增加10MB内存
    avg_cpu: 20.0      # 平均CPU使用20%
```

## Hikyuu框架配置

### 数据路径配置
```yaml
hikyuu:
  data_path: "/tmp/hikyuu_test_data"
  block_path: "/tmp/hikyuu_test_blocks"
  version: "2.6.8"
  cache_size: 512  # MB
```

### 计算配置
```yaml
hikyuu_calculation:
  parallel_workers: 4
  chunk_size: 1000
  memory_limit: 2048  # MB
  timeout: 300        # 秒
```

## 测试数据配置

### 股票列表配置
```yaml
test_data:
  small_stock_list:
    - "sh600000"  # 浦发银行
    - "sh600001"  # 邯郸钢铁
    - "sz000001"  # 平安银行
    - "sz000002"  # 万科A

  medium_stock_list:
    count: 100
    pattern: "sh{600000-600099}"

  large_stock_list:
    count: 1000
    pattern: "sh{600000-600999}"
```

### 时间范围配置
```yaml
test_periods:
  short_term:
    start_date: "2024-01-01"
    end_date: "2024-01-31"

  medium_term:
    start_date: "2024-01-01"
    end_date: "2024-06-30"

  long_term:
    start_date: "2024-01-01"
    end_date: "2024-12-31"
```

## 因子配置

### 因子类型配置
```yaml
factor_types:
  momentum:
    - "momentum_5d"
    - "momentum_10d"
    - "momentum_20d"

  technical:
    - "rsi_14d"
    - "macd"
    - "bollinger_bands"

  valuation:
    - "pe_ratio"
    - "pb_ratio"
    - "ps_ratio"

  quality:
    - "roe"
    - "roa"
    - "debt_ratio"
```

### 因子计算参数
```yaml
factor_params:
  momentum:
    periods: [5, 10, 20, 60]
    method: "simple"

  technical:
    rsi_period: 14
    macd_fast: 12
    macd_slow: 26
    macd_signal: 9
    bollinger_period: 20
    bollinger_std: 2

  valuation:
    ttm: true
    quarterly: false
```

## 验证配置

### 数据质量规则
```yaml
validation_rules:
  missing_value_threshold: 0.05    # 5%缺失值阈值
  outlier_std_threshold: 3.0       # 3倍标准差异常值
  correlation_threshold: 0.95      # 95%相关性阈值
  stability_period: 20             # 20期稳定性检查
  ic_threshold: 0.02               # 2% IC阈值
```

### 质量评分权重
```yaml
quality_weights:
  completeness: 0.3      # 完整性权重
  consistency: 0.25      # 一致性权重
  accuracy: 0.25         # 准确性权重
  timeliness: 0.2        # 及时性权重
```

## 信号生成配置

### 策略配置
```yaml
strategies:
  momentum_strategy:
    lookback_period: 20
    threshold: 0.02

  value_strategy:
    pe_threshold: 15
    pb_threshold: 1.5

  multi_factor_strategy:
    factor_weights:
      momentum: 0.4
      value: 0.3
      quality: 0.3
```

### 风险控制配置
```yaml
risk_controls:
  max_position_size: 0.1         # 最大单个仓位10%
  sector_limit: 0.3              # 行业限制30%
  stop_loss: 0.05                # 止损5%
  max_drawdown: 0.15             # 最大回撤15%
  position_limit: 50             # 最大持仓数量
```

## 测试标记配置

### pytest标记
```yaml
pytest_markers:
  integration: "集成测试标记"
  end_to_end: "端到端测试标记"
  performance: "性能测试标记"
  slow: "慢速测试标记(>5秒)"
  database: "数据库相关测试"
  agent_communication: "Agent通信测试"
  requires_mysql: "需要MySQL数据库"
  requires_hikyuu: "需要Hikyuu框架"
```

### 测试分类
```yaml
test_categories:
  unit:
    timeout: 5
    max_memory: 100

  integration:
    timeout: 30
    max_memory: 500

  performance:
    timeout: 300
    max_memory: 1000

  end_to_end:
    timeout: 600
    max_memory: 2000
```

## 日志配置

### 日志级别
```yaml
logging:
  level: "INFO"
  format: "%(asctime)s - %(name)s - %(levelname)s - %(message)s"

  loggers:
    agents: "DEBUG"
    services: "INFO"
    database: "WARNING"
    performance: "INFO"
```

### 日志文件
```yaml
log_files:
  integration_test: "logs/integration_test.log"
  performance_test: "logs/performance_test.log"
  error_log: "logs/error.log"
```

## 监控配置

### 系统监控
```yaml
monitoring:
  cpu_alert_threshold: 80        # CPU使用率告警阈值
  memory_alert_threshold: 80     # 内存使用率告警阈值
  disk_alert_threshold: 90       # 磁盘使用率告警阈值
  network_timeout: 30            # 网络超时时间
```

### 业务监控
```yaml
business_monitoring:
  agent_health_check_interval: 10     # Agent健康检查间隔(秒)
  database_connection_check: 5        # 数据库连接检查间隔(秒)
  factor_calculation_timeout: 300     # 因子计算超时时间(秒)
  signal_generation_timeout: 60       # 信号生成超时时间(秒)
```

## 环境变量

### 必需环境变量
```bash
export HIKYUU_FACTOR_ENV=test
export HIKYUU_FACTOR_CONFIG_PATH=/path/to/config
export HIKYUU_FACTOR_LOG_LEVEL=INFO
export PYTHONPATH=/path/to/hikyuu-factor/src
```

### 可选环境变量
```bash
export HIKYUU_FACTOR_DB_HOST=192.168.3.46
export HIKYUU_FACTOR_DB_PORT=3306
export HIKYUU_FACTOR_DB_USER=remote
export HIKYUU_FACTOR_DB_PASSWORD=remote123456
export HIKYUU_FACTOR_DB_NAME=hikyuu_factor_test
```

## Docker配置

### 测试容器配置
```yaml
docker:
  test_container:
    image: "hikyuu-factor-test:latest"
    memory_limit: "2g"
    cpu_limit: "2"

  mysql_container:
    image: "mysql:8.0"
    memory_limit: "1g"
    environment:
      MYSQL_ROOT_PASSWORD: "root123456"
      MYSQL_DATABASE: "hikyuu_factor_test"
      MYSQL_USER: "remote"
      MYSQL_PASSWORD: "remote123456"
```

## 持续集成配置

### CI/CD流水线
```yaml
ci_cd:
  trigger_branches:
    - "main"
    - "develop"
    - "feature/*"

  test_stages:
    - "unit_tests"
    - "integration_tests"
    - "performance_tests"
    - "end_to_end_tests"

  parallel_jobs: 4
  timeout: 3600  # 1小时
```