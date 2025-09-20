# 快速开始指南

## 系统概述

A股全市场量化因子挖掘与决策支持系统是一个基于Hikyuu框架构建的专业量化分析平台，为投资经理提供因子挖掘、验证和信号生成的完整工具链。系统支持平台自适应优化，能够根据CPU架构(Apple Silicon/x86_64)自动选择最佳性能配置。

## 环境要求

### 硬件要求
- **CPU**: 8核心以上
  - 推荐: Apple M系列芯片(M1/M2/M3) 或 Intel i7/AMD Ryzen 7
  - 支持: ARM NEON (Apple Silicon) 或 AVX2/SSE4 (x86_64)
- **内存**: 32GB以上
- **存储**: SSD 500GB以上可用空间
- **网络**: 稳定的互联网连接

### 软件要求
- **操作系统**: macOS 10.15+ (Apple Silicon优化) 或 Linux (Ubuntu 20.04+推荐)
- **Python**: 3.11+
- **数据库**: MySQL 8.0+
- **其他**: Git, Docker (可选)

## 安装步骤

### 1. 克隆项目仓库
```bash
git clone git@github.com:luannanxian/hikyuu-factor.git
cd hikyuu-factor
```

### 2. 安装Hikyuu框架
```bash
# 方法1: 使用pip安装 (推荐)
pip install hikyuu

# 方法2: 从源码编译安装
# 详见Hikyuu官方文档: https://hikyuu.readthedocs.io
```

### 3. 安装项目依赖
```bash
pip install -r requirements.txt
```

### 4. 配置数据库
```bash
# 创建MySQL数据库
mysql -u root -p
CREATE DATABASE hikyuu_factor CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci;
CREATE USER 'hikyuu_user'@'localhost' IDENTIFIED BY 'your_password';
GRANT ALL PRIVILEGES ON hikyuu_factor.* TO 'hikyuu_user'@'localhost';
FLUSH PRIVILEGES;
```

### 5. 配置Hikyuu数据源
```python
# config/hikyuu_config.py
import hikyuu as hk

# 配置数据驱动
def setup_hikyuu():
    # MySQL配置
    db_config = {
        'host': 'localhost',
        'port': 3306,
        'usr': 'hikyuu_user',
        'pwd': 'your_password',
        'db': 'hikyuu_factor'
    }

    # 初始化Hikyuu
    hk.hikyuu_init(
        data_dir="/path/to/hikyuu/data",
        config_file="/path/to/hikyuu/config.ini"
    )

    return hk.StockManager.instance()
```

## 核心使用流程

### 0. 平台优化检查

#### 检查系统平台信息
```python
import requests

# 获取平台信息和优化配置
platform_info = requests.get('http://localhost:8000/api/v1/system/platform')
print("平台信息:", platform_info.json())

# 示例输出 (Apple Silicon):
# {
#   "status": "success",
#   "data": {
#     "platform_type": "apple_silicon",
#     "cpu_architecture": "arm64",
#     "cpu_count": 8,
#     "simd_support": ["ARM_NEON"]
#   }
# }

# 获取当前优化配置
optimization_config = requests.get('http://localhost:8000/api/v1/system/optimization/config')
print("优化配置:", optimization_config.json())

# 示例输出:
# {
#   "status": "success",
#   "data": {
#     "platform_type": "apple_silicon",
#     "optimal_processes": 8,
#     "low_precision_mode": true,
#     "simd_support": ["ARM_NEON"]
#   }
# }
```

#### 自定义优化配置 (可选)
```python
# 如果需要覆盖自动检测的配置
custom_config = {
    "platform_override": "x86_64",  # 强制使用x86优化
    "process_count": 16,             # 自定义进程数
    "low_precision_mode": False      # 关闭低精度模式
}

response = requests.post('http://localhost:8000/api/v1/system/optimization/config',
                        json=custom_config)
print("配置更新结果:", response.json())
```

### 1. 数据准备和验证

#### 启动数据管理Agent
```bash
python -m src.agents.data_manager --config config/data_agent.yaml
```

#### 执行数据更新
```python
import requests

# 触发数据更新
response = requests.post('http://localhost:8001/api/v1/data/update', json={
    'date': '2025-01-15',
    'data_types': ['market_data', 'financial_data', 'stock_list'],
    'force_update': False
})

task_id = response.json()['task_id']

# 监控更新进度
status_response = requests.get(f'http://localhost:8001/api/v1/data/update/{task_id}/status')
print(status_response.json())
```

#### 数据质量检查
```python
# 执行数据质量检查
quality_check = requests.post('http://localhost:8001/api/v1/data/quality/check', json={
    'check_date': '2025-01-15',
    'check_types': ['price_consistency', 'volume_validity', 'change_threshold'],
    'threshold_config': {
        'max_price_change_pct': 0.3
    }
})

print(f"数据质量检查结果: {quality_check.json()}")
```

### 2. 因子开发和计算

#### 注册新因子
```python
# 创建基于Hikyuu的动量因子
factor_definition = {
    'name': '20日动量因子',
    'category': 'momentum',
    'hikyuu_formula': 'MA(CLOSE(), 5) / MA(CLOSE(), 20) - 1',
    'economic_logic': '短期均线相对长期均线的偏离程度，反映股价的短期动量特征。当短期均线高于长期均线时，表明股价具有上升动量。',
    'parameters': {
        'fast_period': 5,
        'slow_period': 20
    }
}

# 注册因子
response = requests.post('http://localhost:8002/api/v1/factors', json=factor_definition)
factor_id = response.json()['data']['id']
print(f"因子已注册，ID: {factor_id}")
```

#### 计算因子值 (平台优化)
```python
# 为全市场股票计算因子，使用平台自适应优化
calculation_request = {
    'stock_universe': ['sh000001', 'sh000002', 'sz000001', 'sz000002'],  # 示例股票池
    'date_range': {
        'start_date': '2020-01-01',
        'end_date': '2025-01-15'
    },
    # 可选：自定义优化配置
    'optimization_config': {
        'process_count': 8,           # 覆盖自动检测的进程数
        'low_precision_mode': True    # Apple Silicon启用ARM优化
    }
}

response = requests.post(f'http://localhost:8002/api/v1/factors/{factor_id}/calculate',
                        json=calculation_request)
calc_task_id = response.json()['task_id']

# 监控计算进度 (显示平台优化信息)
while True:
    status = requests.get(f'http://localhost:8002/api/v1/tasks/{calc_task_id}/status')
    status_data = status.json()

    if status_data['status'] == 'completed':
        print("因子计算完成")
        print(f"使用平台: {status_data['optimization_used']['platform_type']}")
        print(f"并行进程数: {status_data['optimization_used']['process_count']}")
        print(f"计算耗时: {status_data['elapsed_time']}ms")
        break
    elif status_data['status'] == 'failed':
        print(f"计算失败: {status_data['error_message']}")
        break

    print(f"计算进度: {status_data['progress']}%")
    time.sleep(10)

# 示例输出 (Apple Silicon):
# 因子计算完成
# 使用平台: apple_silicon
# 并行进程数: 8
# 计算耗时: 2840ms  # 比默认配置快约40%
```

#### 获取因子值
```python
# 获取计算结果
factor_values = requests.get(f'http://localhost:8002/api/v1/factors/{factor_id}/values',
                           params={
                               'stock_codes': 'sh000001,sz000002',
                               'query_range': 'Query(-100)'  # 最近100个交易日
                           })

print("因子值:", factor_values.json())
```

### 3. 因子验证

#### 启动验证Agent
```bash
python -m src.agents.validation_agent --config config/validation_agent.yaml
```

#### 执行因子验证
```python
# 使用默认验证配置
validation_request = {
    'factor_id': factor_id,
    'analysis_types': ['ic_analysis', 'layered_returns', 'turnover_analysis']
}

# 或使用自定义验证周期
custom_validation_request = {
    'factor_id': factor_id,
    'validation_config': {
        'train_period': {'start_date': '2015-01-01', 'end_date': '2018-12-31'},
        'test_period': {'start_date': '2019-01-01', 'end_date': '2021-12-31'},
        'validation_period': {'start_date': '2022-01-01', 'end_date': '2025-01-15'}
    },
    'analysis_types': ['ic_analysis', 'layered_returns', 'turnover_analysis']
}

# 首先验证配置的合理性
config_check = requests.post('http://localhost:8003/api/v1/validation/configs/validate',
                           json=custom_validation_request['validation_config'])

if config_check.json()['valid']:
    # 启动验证任务
    response = requests.post('http://localhost:8003/api/v1/validation/start',
                            json=custom_validation_request)
    validation_task_id = response.json()['task_id']

    # 监控验证进度
    while True:
        status = requests.get(f'http://localhost:8003/api/v1/validation/{validation_task_id}/status')
        status_data = status.json()

        print(f"验证进度: {status_data['progress']:.1%}, 当前阶段: {status_data['current_phase']}")

        if status_data['status'] == 'completed':
            print("验证完成")
            break
        elif status_data['status'] == 'failed':
            print(f"验证失败: {status_data['error_message']}")
            break

        time.sleep(30)

    # 获取验证报告
    report_response = requests.get(f'http://localhost:8003/api/v1/validation/{validation_task_id}/report')
    validation_report = report_response.json()['data']

    print(f"因子评分: {validation_report['results']['summary']['overall_score']:.3f}")
    print(f"推荐结果: {validation_report['results']['summary']['recommendation']}")

    # 各阶段IC表现
    for phase in ['train_phase', 'test_phase', 'validation_phase']:
        ic_ir = validation_report['results'][phase]['ic_metrics']['ic_ir']
        print(f"{phase} ICIR: {ic_ir:.3f}")

else:
    print("验证配置不合理:")
    for issue in config_check.json()['issues']:
        print(f"- {issue}")
```

#### 查看预设验证配置
```python
# 获取系统预设的验证配置
presets = requests.get('http://localhost:8003/api/v1/validation/configs/presets')

for preset in presets.json()['data']:
    print(f"配置名称: {preset['name']}")
    print(f"描述: {preset['description']}")
    config = preset['config']
    print(f"训练期: {config['train_period']['start_date']} ~ {config['train_period']['end_date']}")
    print(f"测试期: {config['test_period']['start_date']} ~ {config['test_period']['end_date']}")
    print(f"验证期: {config['validation_period']['start_date']} ~ {config['validation_period']['end_date']}")
    print("---")
```

### 4. 交易信号生成

#### 风控预检查
```python
# 执行风控检查
risk_check = requests.post('http://localhost:8004/api/v1/risk/precheck', json={
    'check_date': '2025-01-15',
    'strategy_config': {
        'strategy_id': 'momentum_strategy_v1',
        'factors': [factor_id],
        'stock_universe': ['sh000001', 'sz000002']
    },
    'check_types': ['data_completeness', 'factor_anomaly', 'position_limits']
})

risk_result = risk_check.json()
if risk_result['risk_level'] in ['high', 'critical']:
    print(f"风险等级过高: {risk_result['risk_level']}")
    for rec in risk_result['recommendations']:
        print(f"建议: {rec}")
```

#### 生成交易信号 (需要人工确认)
```python
# 获取确认令牌 (实际系统中通过安全流程获取)
confirm_token = input("请输入确认令牌: ")

signal_request = {
    'strategy_config': {
        'strategy_id': 'momentum_strategy_v1',
        'stock_universe': ['sh000001', 'sz000002'],
        'signal_params': {
            'type': 'factor_based',
            'factor_config': {
                'factor_id': factor_id,
                'threshold': 0.1
            }
        },
        'tm_params': {
            'init_cash': 1000000,
            'name': 'Momentum Strategy'
        }
    },
    'confirm_token': confirm_token,
    'user_id': 'investment_manager_01'
}

# 生成信号
response = requests.post('http://localhost:8004/api/v1/signals/generate',
                        json=signal_request)

if response.status_code == 202:
    task_id = response.json()['task_id']
    print(f"信号生成任务已启动: {task_id}")
else:
    print(f"信号生成失败: {response.json()}")
```

#### 确认交易信号
```python
# 获取待确认信号
pending_signals = requests.get('http://localhost:8004/api/v1/signals/pending',
                              params={'user_id': 'investment_manager_01'})

for signal in pending_signals.json()['data']:
    print(f"信号ID: {signal['id']}")
    print(f"股票: {signal['stock_code']}")
    print(f"类型: {signal['signal_type']}")
    print(f"权重: {signal['weight']}")

    # 获取详细信息包括因子暴露
    detail = requests.get(f"http://localhost:8004/api/v1/signals/{signal['id']}")
    factor_exposure = detail.json()['data']['factor_exposure']
    print(f"因子暴露: {factor_exposure}")

    # 二次确认
    confirmation_code = input("请输入二次确认码: ")
    confirm_response = requests.post(f"http://localhost:8004/api/v1/signals/{signal['id']}/confirm",
                                   json={
                                       'action': 'confirm',
                                       'user_id': 'investment_manager_01',
                                       'confirmation_code': confirmation_code,
                                       'notes': '确认执行该交易信号'
                                   })

    print(f"确认结果: {confirm_response.json()}")
```

## 系统监控

### 检查Agent状态
```python
import requests

agents = [
    ('数据管理', 'http://localhost:8001/health'),
    ('因子计算', 'http://localhost:8002/health'),
    ('因子验证', 'http://localhost:8003/health'),
    ('信号生成', 'http://localhost:8004/health')
]

for name, url in agents:
    try:
        response = requests.get(url, timeout=5)
        if response.status_code == 200:
            print(f"✅ {name}Agent: 正常")
        else:
            print(f"❌ {name}Agent: 异常 ({response.status_code})")
    except requests.RequestException:
        print(f"❌ {name}Agent: 连接失败")
```

### 性能监控
```python
# 监控因子计算性能
performance_data = requests.get('http://localhost:8002/api/v1/metrics/performance')
metrics = performance_data.json()

print(f"平均计算时间: {metrics['avg_calculation_time']}秒")
print(f"内存使用: {metrics['memory_usage_mb']}MB")
print(f"CPU使用率: {metrics['cpu_usage_percent']}%")

# 检查是否达到性能目标 (30分钟内完成全市场10年计算)
if metrics['last_full_calculation_time'] > 1800:  # 30分钟 = 1800秒
    print("⚠️  性能目标未达成，需要优化")
```

## 故障排除

### 常见问题

#### 1. Hikyuu数据加载失败
```bash
# 检查Hikyuu配置
python -c "import hikyuu as hk; print(hk.StockManager.instance())"

# 重新初始化数据
python tools/init_hikyuu_data.py
```

#### 2. Agent启动失败
```bash
# 检查端口占用
netstat -tulpn | grep :8001

# 查看Agent日志
tail -f logs/data_agent.log
```

#### 3. 因子计算超时
```python
# 检查股票池大小
stocks = requests.get('http://localhost:8001/api/v1/data/stocks')
print(f"股票数量: {len(stocks.json()['data']['stocks'])}")

# 减少股票池进行测试
test_stocks = ['sh000001', 'sh000002']  # 仅使用少量股票测试
```

### 日志查看
```bash
# 查看系统日志
tail -f logs/system.log

# 查看特定Agent日志
tail -f logs/factor_agent.log

# 查看错误日志
grep ERROR logs/*.log
```

## 下一步

1. **扩展因子库**: 参考`docs/factor_development.md`开发更多因子
2. **性能优化**: 查看`docs/performance_tuning.md`进行系统调优
3. **Web界面**: 部署可视化界面 (二期功能)
4. **生产部署**: 参考`docs/production_deployment.md`进行生产环境部署

## 支持和帮助

- **技术文档**: `docs/` 目录下的详细文档
- **API文档**: 访问 `http://localhost:8001/docs` 等查看Swagger文档
- **问题反馈**: 通过GitHub Issues报告问题
- **社区讨论**: 参与项目讨论和经验分享

系统已准备就绪，开始您的量化因子挖掘之旅！