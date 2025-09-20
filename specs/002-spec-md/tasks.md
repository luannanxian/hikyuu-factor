# Tasks: A股全市场量化因子挖掘与决策支持系统

**Input**: Design documents from `/specs/002-spec-md/`
**Prerequisites**: plan.md (required), research.md, data-model.md, contracts/

## Execution Flow (main)
```
1. Load plan.md from feature directory
   → Found: Python 3.11+, Hikyuu框架, FastAPI, Agent架构, 平台自适应优化
   → Extract: tech stack, libraries, structure
2. Load optional design documents:
   → data-model.md: Extract entities → OptimizationConfig, StockPoolManager, ValidationConfig, AuditLog, TradingSignal
   → contracts/: 5 files → data-agent, factor-agent, validation-agent, signal-agent, system-api
   → research.md: Extract decisions → 平台优化策略, Hikyuu集成, Point-in-Time实现
3. Generate tasks by category:
   → Setup: project init, dependencies, linting, 平台检测
   → Tests: contract tests, integration tests
   → Core: models, services, CLI commands, agents
   → Integration: DB, middleware, logging, Hikyuu集成
   → Polish: unit tests, performance, docs
4. Apply task rules:
   → Different files = mark [P] for parallel
   → Same file = sequential (no [P])
   → Tests before implementation (TDD)
5. Number tasks sequentially (T001, T002...)
6. Generate dependency graph
7. Create parallel execution examples
8. Validate task completeness:
   → All contracts have tests? ✓
   → All entities have models? ✓
   → All endpoints implemented? ✓
9. Return: SUCCESS (tasks ready for execution)
```

## Format: `[ID] [P?] Description`
- **[P]**: Can run in parallel (different files, no dependencies)
- Include exact file paths in descriptions

## Path Conventions
- **Single project**: `src/`, `tests/` at repository root
- Agent微服务架构，每个Agent独立但在同一代码库中

## Phase 3.1: Setup
- [ ] T001 Create project structure with src/agents/, src/models/, src/services/, src/cli/, tests/ directories
- [ ] T002 Initialize Python project with requirements.txt (Hikyuu, FastAPI, SQLAlchemy, pytest, psutil dependencies)
- [ ] T003 [P] Configure linting and formatting tools in pyproject.toml (black, isort, flake8)
- [ ] T004 [P] Setup MySQL database schema and Hikyuu configuration files in config/
- [ ] T005 [P] Create Docker development environment in docker-compose.yml
- [ ] T006 [P] Initialize platform detection module in src/lib/platform_optimizer.py

## Phase 3.2: Tests First (TDD) ⚠️ MUST COMPLETE BEFORE 3.3
**CRITICAL: These tests MUST be written and MUST FAIL before ANY implementation**

### Contract Tests for API Endpoints
- [ ] T007 [P] Contract test GET /api/v1/system/platform in tests/contract/test_system_api_platform.py
- [ ] T008 [P] Contract test POST /api/v1/system/optimization/config in tests/contract/test_system_api_optimization.py
- [ ] T009 [P] Contract test GET /api/v1/system/health in tests/contract/test_system_api_health.py
- [ ] T010 [P] Contract test GET /api/v1/data/stocks in tests/contract/test_data_agent_stocks.py
- [ ] T011 [P] Contract test POST /api/v1/data/update in tests/contract/test_data_agent_update.py
- [ ] T012 [P] Contract test POST /api/v1/data/quality/check in tests/contract/test_data_agent_quality.py
- [ ] T013 [P] Contract test GET /api/v1/factors in tests/contract/test_factor_agent_list.py
- [ ] T014 [P] Contract test POST /api/v1/factors in tests/contract/test_factor_agent_create.py
- [ ] T015 [P] Contract test POST /api/v1/factors/{id}/calculate in tests/contract/test_factor_agent_calculate.py
- [ ] T016 [P] Contract test POST /api/v1/validation/start in tests/contract/test_validation_agent_start.py
- [ ] T017 [P] Contract test GET /api/v1/validation/configs/presets in tests/contract/test_validation_agent_presets.py
- [ ] T018 [P] Contract test POST /api/v1/signals/generate in tests/contract/test_signal_agent_generate.py
- [ ] T019 [P] Contract test POST /api/v1/signals/{id}/confirm in tests/contract/test_signal_agent_confirm.py

### Integration Tests for User Scenarios
- [ ] T020 [P] Integration test platform optimization workflow in tests/integration/test_platform_optimization.py
- [ ] T021 [P] Integration test data update and quality check workflow in tests/integration/test_data_workflow.py
- [ ] T022 [P] Integration test factor registration and calculation with platform optimization in tests/integration/test_factor_workflow.py
- [ ] T023 [P] Integration test factor validation with configurable periods in tests/integration/test_validation_workflow.py
- [ ] T024 [P] Integration test signal generation with human confirmation in tests/integration/test_signal_workflow.py

## Phase 3.3: Core Implementation (ONLY after tests are failing)

### Data Models Based on Design
- [ ] T025 [P] PlatformType enum and OptimizationConfig dataclass in src/models/platform_config.py
- [ ] T026 [P] StockPoolManager class using Hikyuu Stock in src/models/stock_pool.py
- [ ] T027 [P] FactorDefinition model with Hikyuu formula in src/models/factor_definition.py
- [ ] T028 [P] ValidationConfig model with configurable periods in src/models/validation_config.py
- [ ] T029 [P] AuditLog model with hash chain verification in src/models/audit_log.py
- [ ] T030 [P] TradingSignal model with confirmation workflow in src/models/trading_signal.py

### Platform Optimization Services
- [ ] T031 Platform detection and optimization service in src/services/platform_service.py
- [ ] T032 Hikyuu initialization with platform optimization in src/services/hikyuu_service.py

### Agent Services Implementation
- [ ] T033 DataManagerAgent with Hikyuu StockManager integration in src/agents/data_manager.py
- [ ] T034 FactorCalculationAgent with platform-optimized computation in src/agents/factor_calculation.py
- [ ] T035 ValidationAgent with configurable validation periods in src/agents/validation_agent.py
- [ ] T036 SignalGenerationAgent with human confirmation workflow in src/agents/signal_generation.py

### API Endpoints - System Management
- [ ] T037 GET /api/v1/system/platform endpoint in src/api/system_api.py
- [ ] T038 POST /api/v1/system/optimization/config endpoint in src/api/system_api.py
- [ ] T039 GET /api/v1/system/health endpoint in src/api/system_api.py

### API Endpoints - Data Agent
- [ ] T040 GET /api/v1/data/stocks endpoint in src/api/data_agent.py
- [ ] T041 POST /api/v1/data/update endpoint in src/api/data_agent.py
- [ ] T042 POST /api/v1/data/quality/check endpoint in src/api/data_agent.py

### API Endpoints - Factor Agent
- [ ] T043 GET /api/v1/factors endpoint in src/api/factor_agent.py
- [ ] T044 POST /api/v1/factors endpoint in src/api/factor_agent.py
- [ ] T045 POST /api/v1/factors/{id}/calculate endpoint with platform optimization in src/api/factor_agent.py

### API Endpoints - Validation Agent
- [ ] T046 POST /api/v1/validation/start endpoint in src/api/validation_agent.py
- [ ] T047 GET /api/v1/validation/configs/presets endpoint in src/api/validation_agent.py

### API Endpoints - Signal Agent
- [ ] T048 POST /api/v1/signals/generate endpoint in src/api/signal_agent.py
- [ ] T049 POST /api/v1/signals/{id}/confirm endpoint in src/api/signal_agent.py

### Business Logic Services
- [ ] T050 Point-in-Time data access service with Hikyuu integration in src/services/pit_data_service.py
- [ ] T051 Factor calculation service with platform optimization in src/services/factor_calculation_service.py
- [ ] T052 Validation service with configurable periods in src/services/validation_service.py
- [ ] T053 Risk management service for signal generation in src/services/risk_service.py
- [ ] T054 Audit logging service with hash chains in src/services/audit_service.py

## Phase 3.4: Integration
- [ ] T055 Connect DataManagerAgent to MySQL and Hikyuu StockManager
- [ ] T056 Implement FastAPI dependency injection for database connections and platform optimization
- [ ] T057 Human confirmation middleware for signal generation endpoints
- [ ] T058 Request/response logging with audit trails for all agents
- [ ] T059 Error handling and validation middleware across all APIs
- [ ] T060 Agent health check and monitoring integration
- [ ] T061 Performance monitoring with platform-specific metrics collection

## Phase 3.5: Polish
- [ ] T062 [P] Unit tests for platform optimization logic in tests/unit/test_platform_optimizer.py
- [ ] T063 [P] Unit tests for Point-in-Time data access in tests/unit/test_pit_data_service.py
- [ ] T064 [P] Unit tests for factor calculation service in tests/unit/test_factor_calculation_service.py
- [ ] T065 [P] Unit tests for validation service in tests/unit/test_validation_service.py
- [ ] T066 [P] Unit tests for audit logging in tests/unit/test_audit_service.py
- [ ] T067 Performance benchmarks for Apple Silicon vs x86_64 optimization in tests/performance/
- [ ] T068 Performance tests for 30-minute full market calculation requirement
- [ ] T069 Performance tests for 15-minute daily signal generation requirement
- [ ] T070 [P] CLI interface for agent management in src/cli/agent_cli.py
- [ ] T071 [P] Configuration management and environment variables in src/config/settings.py
- [ ] T072 Remove code duplication and refactor shared utilities
- [ ] T073 Execute quickstart.md manual testing scenarios with platform optimization
- [ ] T074 [P] API documentation generation from OpenAPI specs

## Dependencies
- Setup (T001-T006) before everything
- Tests (T007-T024) before implementation (T025-T054)
- Platform models (T025, T031-T032) before agent services (T033-T036)
- Models (T025-T030) before services (T033-T054) before endpoints (T037-T049)
- Business services (T050-T054) before integration (T055-T061)
- Implementation before polish (T062-T074)

### Specific Dependencies
- T031-T032 (Platform services) block T033-T036 (Agent services)
- T033-T036 (Agent services) block T037-T049 (API endpoints)
- T050 (PIT data service) blocks T033, T034, T035 (Data/Factor/Validation agents)
- T054 (Audit service) blocks T036 (Signal agent needs audit)
- T056 (FastAPI DI) blocks all API endpoints
- T061 (Performance monitoring) needs T031 (Platform service)

## Parallel Example
```bash
# Launch contract tests together (T007-T019):
Task: "Contract test GET /api/v1/system/platform in tests/contract/test_system_api_platform.py"
Task: "Contract test POST /api/v1/system/optimization/config in tests/contract/test_system_api_optimization.py"
Task: "Contract test GET /api/v1/data/stocks in tests/contract/test_data_agent_stocks.py"
Task: "Contract test POST /api/v1/factors in tests/contract/test_factor_agent_create.py"

# Launch model creation together (T025-T030):
Task: "PlatformType enum and OptimizationConfig dataclass in src/models/platform_config.py"
Task: "StockPoolManager class using Hikyuu Stock in src/models/stock_pool.py"
Task: "FactorDefinition model with Hikyuu formula in src/models/factor_definition.py"
Task: "ValidationConfig model with configurable periods in src/models/validation_config.py"
Task: "AuditLog model with hash chain verification in src/models/audit_log.py"
Task: "TradingSignal model with confirmation workflow in src/models/trading_signal.py"

# Launch unit tests together (T062-T066):
Task: "Unit tests for platform optimization logic in tests/unit/test_platform_optimizer.py"
Task: "Unit tests for Point-in-Time data access in tests/unit/test_pit_data_service.py"
Task: "Unit tests for factor calculation service in tests/unit/test_factor_calculation_service.py"
Task: "Unit tests for validation service in tests/unit/test_validation_service.py"
Task: "Unit tests for audit logging in tests/unit/test_audit_service.py"
```

## Notes
- [P] tasks = different files, no dependencies
- Verify tests fail before implementing
- Commit after each task
- Follow Hikyuu best practices from research.md
- Maintain Point-in-Time data access constraints
- Ensure all signal generation requires human confirmation
- Performance targets: 30min full market calculation, 15min daily signals on both Apple Silicon and x86_64

## Task Generation Rules
*Applied during main() execution*

1. **From Contracts**:
   - Each contract file (5 files) → contract test task [P]
   - Each endpoint (19 endpoints) → implementation task

2. **From Data Model**:
   - Each entity (6 entities) → model creation task [P]
   - Platform optimization → dedicated service tasks

3. **From User Stories**:
   - Each quickstart scenario (5 scenarios) → integration test [P]
   - Platform optimization workflow → dedicated integration test

4. **Ordering**:
   - Setup → Tests → Models → Services → Endpoints → Polish
   - Platform optimization setup before Agent initialization
   - Dependencies block parallel execution

## Validation Checklist
*GATE: Checked by main() before returning*

- [x] All contracts have corresponding tests (19 endpoints covered)
- [x] All entities have model tasks (6 entities covered)
- [x] All tests come before implementation (TDD enforced)
- [x] Parallel tasks truly independent (different files)
- [x] Each task specifies exact file path
- [x] No task modifies same file as another [P] task
- [x] Hikyuu framework integration properly planned
- [x] Point-in-Time data access requirements covered
- [x] Human confirmation workflow for signals included
- [x] Performance requirements addressed (30min/15min targets)
- [x] Platform optimization (Apple Silicon/x86_64) fully integrated
- [x] Configurable validation periods implemented