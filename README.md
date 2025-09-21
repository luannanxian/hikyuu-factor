# hikyuu-factor

A股全市场量化因子挖掘与决策支持系统 - 基于Hikyuu框架的Agent架构设计

[![Python Version](https://img.shields.io/badge/python-3.11%2B-blue.svg)](https://python.org)
[![Hikyuu](https://img.shields.io/badge/hikyuu-2.6.0%2B-green.svg)](https://hikyuu.org)
[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![Code Style](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

## 项目简介

hikyuu-factor是一个基于Hikyuu量化框架的A股全市场量化因子挖掘与决策支持系统。系统采用Agent微服务架构设计，提供数据驱动、可解释、可验证的交易洞察。

### 核心特性

- 🚀 **高性能计算**: 基于Hikyuu C++核心，30分钟完成全市场单因子计算
- 🔧 **Agent架构**: 模块化的4个Agent微服务，独立部署和扩展
- 🛡️ **Point-in-Time**: 严格防止前视偏差的数据访问约束
- 👤 **人工确认**: 强制人工确认机制保证交易信号安全性
- 🔍 **全链路审计**: 完整的操作审计和可追溯性
- 💻 **平台优化**: Apple Silicon ARM NEON与x86_64自适应优化

### 系统架构

```
┌─────────────────┐ ┌─────────────────┐ ┌─────────────────┐ ┌─────────────────┐
│   DataManager   │ │FactorCalculation│ │   Validation    │ │ SignalGeneration│
│     Agent       │ │     Agent       │ │     Agent       │ │     Agent       │
├─────────────────┤ ├─────────────────┤ ├─────────────────┤ ├─────────────────┤
│ • 股票池管理    │ │ • 因子注册      │ │ • IC分析        │ │ • 信号生成      │
│ • 数据更新      │ │ • 平台优化计算  │ │ • 分层回测      │ │ • 风险检查      │
│ • 质量检查      │ │ • 因子存储      │ │ • 绩效评估      │ │ • 人工确认      │
│ • ST过滤        │ │ • 版本管理      │ │ • 报告生成      │ │ • 审计记录      │
└─────────────────┘ └─────────────────┘ └─────────────────┘ └─────────────────┘
         │                   │                   │                   │
         └───────────────────┼───────────────────┼───────────────────┘
                             │                   │
                    ┌─────────────────┐ ┌─────────────────┐
                    │   Hikyuu Core   │ │  MySQL + Redis  │
                    │   (C++ Engine)  │ │   (Storage)     │
                    └─────────────────┘ └─────────────────┘
```

### 技术栈

- **核心框架**: Python 3.11+ + Hikyuu量化框架(C++核心)
- **Web框架**: FastAPI + Uvicorn + Pydantic
- **数据存储**: MySQL 8.0+ + Redis + HDF5内存映射
- **数据处理**: Pandas + NumPy + SQLAlchemy
- **测试框架**: pytest + pytest-asyncio + pytest-mock (TDD)
- **部署方案**: Docker + Kubernetes + Prometheus监控

## 安装

### 系统要求

- Python 3.11或更高版本
- MySQL 8.0或更高版本
- Redis 5.0或更高版本
- Hikyuu量化框架 2.6.0或更高版本

### 快速安装

1. **克隆项目**
```bash
git clone https://github.com/luannanxian/hikyuu-factor.git
cd hikyuu-factor
```

2. **安装依赖**
```bash
# 创建虚拟环境
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# 安装项目依赖
pip install -e .

# 安装开发依赖
pip install -e ".[dev]"
```

3. **配置环境**
```bash
# 复制环境变量模板
cp .env.example .env

# 编辑配置文件
vim .env  # 填入您的数据库和Redis连接信息
```

4. **初始化数据库**
```bash
# 运行数据库迁移
python -m src.cli.db_migrate

# 初始化基础数据
python -m src.cli.init_data
```

## 使用方法

### 启动Agent服务

```bash
# 启动数据管理Agent
python -m src.agents.data_manager --config config/data_agent.yaml

# 启动因子计算Agent
python -m src.agents.factor_agent --config config/factor_agent.yaml

# 启动验证Agent
python -m src.agents.validation_agent --config config/validation_agent.yaml

# 启动信号生成Agent
python -m src.agents.signal_agent --config config/signal_agent.yaml
```

### 命令行工具

```bash
# 计算因子
hikyuu-factor calculate --factor-id momentum_20d --stocks sh000001,sz000002

# 生成交易信号（需要人工确认）
hikyuu-factor signal --strategy momentum_v1 --confirm

# 更新股票数据
hikyuu-factor data-update --market sh,sz

# 运行因子验证
hikyuu-factor validate --factor-id momentum_20d --period 2020-2023
```

### Python API示例

```python
from src.agents.factor_agent import FactorCalculationAgent
from src.models.factor_definition import FactorDefinition

# 初始化因子计算Agent
agent = FactorCalculationAgent()

# 注册新因子
factor = FactorDefinition(
    name="20日动量因子",
    category="momentum",
    hikyuu_formula="MA(CLOSE(), 20) / MA(CLOSE(), 5) - 1",
    economic_logic="基于短期与中期均线的相对强度，捕捉动量效应"
)

factor_id = agent.register_factor(factor)

# 计算因子值
results = agent.calculate_factor(
    factor_id=factor_id,
    stock_universe=["sh000001", "sz000002"],
    date_range={"start_date": "2020-01-01", "end_date": "2023-12-31"}
)
```

### REST API

系统提供完整的REST API接口，启动服务后访问 http://localhost:8000/docs 查看API文档。

主要端点：
- `GET /api/v1/system/platform` - 获取平台信息
- `POST /api/v1/factors` - 注册新因子
- `POST /api/v1/factors/{id}/calculate` - 计算因子值
- `POST /api/v1/signals/generate` - 生成交易信号
- `GET /api/v1/system/health` - 系统健康检查

## 开发

### 开发环境设置

```bash
# 安装开发依赖
pip install -e ".[dev]"

# 安装pre-commit钩子
pre-commit install

# 运行代码格式化
black src tests
isort src tests

# 运行类型检查
mypy src

# 运行代码检查
flake8 src tests
```

### 测试

本项目遵循TDD（测试驱动开发）原则：

```bash
# 运行所有测试
pytest

# 运行单元测试
pytest tests/unit

# 运行集成测试
pytest tests/integration

# 运行契约测试
pytest tests/contract

# 运行性能测试
pytest tests/performance -m performance

# 生成测试覆盖率报告
pytest --cov=src --cov-report=html
```

### 贡献指南

1. Fork本仓库
2. 创建特性分支 (`git checkout -b feature/amazing-feature`)
3. 提交更改 (`git commit -m 'Add some amazing feature'`)
4. 推送到分支 (`git push origin feature/amazing-feature`)
5. 打开Pull Request

### 代码规范

- 使用Black进行代码格式化
- 使用isort进行导入排序
- 使用mypy进行类型检查
- 遵循PEP 8编码规范
- 测试覆盖率要求 > 90%

## 部署

### Docker部署

```bash
# 构建镜像
docker build -t hikyuu-factor:latest .

# 使用docker-compose启动
docker-compose up -d
```

### Kubernetes部署

```bash
# 应用Kubernetes配置
kubectl apply -f k8s/

# 查看服务状态
kubectl get pods -n hikyuu-factor
```

## 性能指标

基于Hikyuu框架在AMD 7950x上的性能基准：

- **全A股20日均线计算**: 166毫秒
- **30分钟全市场单因子计算**: ✅ 达标
- **15分钟每日信号生成**: ✅ 达标
- **5000+股票全市场覆盖**: ✅ 支持
- **100+并发因子计算**: ✅ 支持

平台优化效果：
- **Apple Silicon**: ARM NEON优化，性能提升15-25%
- **x86_64**: AVX/SSE优化，性能提升10-20%

## 许可证

本项目采用MIT许可证 - 查看 [LICENSE](LICENSE) 文件了解详情

## 联系方式

- 项目主页: https://github.com/luannanxian/hikyuu-factor
- 问题反馈: https://github.com/luannanxian/hikyuu-factor/issues
- 文档地址: https://github.com/luannanxian/hikyuu-factor/docs

## 致谢

- [Hikyuu量化框架](https://hikyuu.org) - 提供高性能的量化分析核心
- [FastAPI](https://fastapi.tiangolo.com) - 现代化的Python Web框架
- [pytest](https://pytest.org) - 强大的Python测试框架

---

**⚠️ 风险提示**: 本系统仅用于量化研究和教育目的，不构成投资建议。使用本系统进行实际交易的风险由用户自行承担。