
# Implementation Plan: A股全市场量化因子挖掘与决策支持系统

**Branch**: `002-spec-md` | **Date**: 2025-09-22 | **Spec**: [spec.md](./spec.md)
**Input**: Feature specification from `/specs/002-spec-md/spec.md`

## Execution Flow (/plan command scope)
```
1. Load feature spec from Input path
   → If not found: ERROR "No feature spec at {path}"
2. Fill Technical Context (scan for NEEDS CLARIFICATION)
   → Detect Project Type from context (web=frontend+backend, mobile=app+api)
   → Set Structure Decision based on project type
3. Fill the Constitution Check section based on the content of the constitution document.
4. Evaluate Constitution Check section below
   → If violations exist: Document in Complexity Tracking
   → If no justification possible: ERROR "Simplify approach first"
   → Update Progress Tracking: Initial Constitution Check
5. Execute Phase 0 → research.md
   → If NEEDS CLARIFICATION remain: ERROR "Resolve unknowns"
6. Execute Phase 1 → contracts, data-model.md, quickstart.md, agent-specific template file (e.g., `CLAUDE.md` for Claude Code, `.github/copilot-instructions.md` for GitHub Copilot, `GEMINI.md` for Gemini CLI, `QWEN.md` for Qwen Code or `AGENTS.md` for opencode).
7. Re-evaluate Constitution Check section
   → If new violations: Refactor design, return to Phase 1
   → Update Progress Tracking: Post-Design Constitution Check
8. Plan Phase 2 → Describe task generation approach (DO NOT create tasks.md)
9. STOP - Ready for /tasks command
```

**IMPORTANT**: The /plan command STOPS at step 7. Phases 2-4 are executed by other commands:
- Phase 2: /tasks command creates tasks.md
- Phase 3-4: Implementation execution (manual or via tools)

## Summary
A股全市场量化因子挖掘与决策支持系统，专注于提供数据驱动、可解释、可验证的交易洞察，支持因子验证、信号生成和人工确认的完整工作流。技术方法基于Hikyuu量化框架的高性能Agent架构实现，确保30分钟内完成全市场因子计算和严格的时点数据访问。

## Technical Context
**Language/Version**: Python 3.11+
**Primary Dependencies**: Hikyuu>=2.6.8, FastAPI, Pandas, NumPy, MySQL
**Storage**: MySQL (分区表), 本地文件缓存 (factors/, signals/, audit/)
**Testing**: pytest, 集成测试, 合约测试, TDD方法论
**Target Platform**: Linux/macOS 服务器, Apple Silicon优化
**Project Type**: single - Agent微服务架构
**Performance Goals**: 30分钟全市场单因子计算, 15分钟每日信号生成, <200ms API响应
**Constraints**: 严格时点数据访问, 强制人工确认, 不可变审计日志, ST股票自动过滤
**Scale/Scope**: 5000只A股, 10年历史数据, 多因子并行计算, 可扩展因子库

## Constitution Check
*GATE: Must pass before Phase 0 research. Re-check after Phase 1 design.*

### I. Hikyuu-First Principle
- ✅ **PASS**: 优先使用Hikyuu框架的C++性能引擎
- ✅ **PASS**: 利用Hikyuu FINANCE和MF(多因子)高级功能
- ✅ **PASS**: 避免重复实现已有的技术指标计算

### II. Agent-Based Architecture
- ✅ **PASS**: 采用Agent微服务架构(DataManager, FactorCalculator, Validator, SignalGenerator)
- ✅ **PASS**: 每个Agent独立可测试、可部署
- ✅ **PASS**: RESTful API接口标准化

### III. Test-First (NON-NEGOTIABLE)
- ✅ **PASS**: TDD方法论强制要求
- ✅ **PASS**: 合约测试优先于实现
- ✅ **PASS**: 集成测试覆盖端到端工作流
- ✅ **PASS**: 单元测试覆盖关键算法

### IV. Point-in-Time Data Integrity
- ✅ **PASS**: 严格时点数据访问，消除前视偏差
- ✅ **PASS**: 不可变审计日志记录所有操作
- ✅ **PASS**: 版本控制确保研究可重现性

### V. Human-in-Loop Safety
- ✅ **PASS**: 强制人工确认交易信号
- ✅ **PASS**: 可解释的因子暴露归因
- ✅ **PASS**: 风险检查和异常警报机制

### VI. Performance & Scalability
- ✅ **PASS**: 30分钟全市场因子计算目标
- ✅ **PASS**: Apple Silicon和x86_64平台优化
- ✅ **PASS**: 并行计算和内存优化策略

## Project Structure

### Documentation (this feature)
```
specs/[###-feature]/
├── plan.md              # This file (/plan command output)
├── research.md          # Phase 0 output (/plan command)
├── data-model.md        # Phase 1 output (/plan command)
├── quickstart.md        # Phase 1 output (/plan command)
├── contracts/           # Phase 1 output (/plan command)
└── tasks.md             # Phase 2 output (/tasks command - NOT created by /plan)
```

### Source Code (repository root)
```
# Option 1: Single project (DEFAULT)
src/
├── models/
├── services/
├── cli/
└── lib/

tests/
├── contract/
├── integration/
└── unit/

# Option 2: Web application (when "frontend" + "backend" detected)
backend/
├── src/
│   ├── models/
│   ├── services/
│   └── api/
└── tests/

frontend/
├── src/
│   ├── components/
│   ├── pages/
│   └── services/
└── tests/

# Option 3: Mobile + API (when "iOS/Android" detected)
api/
└── [same as backend above]

ios/ or android/
└── [platform-specific structure]
```

**Structure Decision**: [DEFAULT to Option 1 unless Technical Context indicates web/mobile app]

## Phase 0: Outline & Research
1. **Extract unknowns from Technical Context** above:
   - For each NEEDS CLARIFICATION → research task
   - For each dependency → best practices task
   - For each integration → patterns task

2. **Generate and dispatch research agents**:
   ```
   For each unknown in Technical Context:
     Task: "Research {unknown} for {feature context}"
   For each technology choice:
     Task: "Find best practices for {tech} in {domain}"
   ```

3. **Consolidate findings** in `research.md` using format:
   - Decision: [what was chosen]
   - Rationale: [why chosen]
   - Alternatives considered: [what else evaluated]

**Output**: research.md with all NEEDS CLARIFICATION resolved

## Phase 1: Design & Contracts
*Prerequisites: research.md complete*

1. **Extract entities from feature spec** → `data-model.md`:
   - Entity name, fields, relationships
   - Validation rules from requirements
   - State transitions if applicable

2. **Generate API contracts** from functional requirements:
   - For each user action → endpoint
   - Use standard REST/GraphQL patterns
   - Output OpenAPI/GraphQL schema to `/contracts/`

3. **Generate contract tests** from contracts:
   - One test file per endpoint
   - Assert request/response schemas
   - Tests must fail (no implementation yet)

4. **Extract test scenarios** from user stories:
   - Each story → integration test scenario
   - Quickstart test = story validation steps

5. **Update agent file incrementally** (O(1) operation):
   - Run `.specify/scripts/bash/update-agent-context.sh claude` for your AI assistant
   - If exists: Add only NEW tech from current plan
   - Preserve manual additions between markers
   - Update recent changes (keep last 3)
   - Keep under 150 lines for token efficiency
   - Output to repository root

**Output**: data-model.md, /contracts/*, failing tests, quickstart.md, agent-specific file

## Phase 2: Task Planning Approach
*This section describes what the /tasks command will do - DO NOT execute during /plan*

**Task Generation Strategy**:
基于Phase 1的设计文档生成详细的实施任务列表：

1. **从API合约生成合约测试任务**
   - data_manager_api.yaml → 数据管理Agent合约测试 [P]
   - factor_calculation_api.yaml → 因子计算Agent合约测试 [P]
   - validation_api.yaml → 验证Agent合约测试 [P]
   - signal_generation_api.yaml → 信号生成Agent合约测试 [P]

2. **从数据模型生成模型实现任务**
   - PlatformOptimizer模型创建 [P]
   - StockPoolManager模型创建 [P]
   - HikyuuFactorCalculator模型创建 [P]
   - FactorDefinition和ValidationConfig模型创建 [P]

3. **从quickstart.md生成集成测试任务**
   - 数据准备和验证端到端测试
   - 因子开发和计算集成测试
   - 因子验证工作流测试
   - 交易信号生成集成测试

4. **Agent实现任务**
   - DataManagerAgent实现 (基于合约测试)
   - FactorCalculationAgent实现 (基于合约测试)
   - ValidationAgent实现 (基于合约测试)
   - SignalGenerationAgent实现 (基于合约测试)

**Ordering Strategy**:
遵循TDD原则和依赖关系：
1. **Phase 2.1**: 测试优先 - 合约测试必须先于实现
2. **Phase 2.2**: 依赖排序 - 模型 → 服务 → Agent → 集成
3. **Phase 2.3**: 并行标记 - [P]标记独立文件，可并行执行

**Estimated Output**:
- 约30个编号任务，按TDD和依赖关系排序
- 15个合约测试任务 [P]
- 8个模型实现任务 [P]
- 4个Agent实现任务
- 3个集成测试任务

**IMPORTANT**: 此阶段由/tasks命令执行，/plan命令仅描述方法

## Phase 3+: Future Implementation
*These phases are beyond the scope of the /plan command*

**Phase 3**: Task execution (/tasks command creates tasks.md)  
**Phase 4**: Implementation (execute tasks.md following constitutional principles)  
**Phase 5**: Validation (run tests, execute quickstart.md, performance validation)

## Complexity Tracking
*Fill ONLY if Constitution Check has violations that must be justified*

| Violation | Why Needed | Simpler Alternative Rejected Because |
|-----------|------------|-------------------------------------|
| [e.g., 4th project] | [current need] | [why 3 projects insufficient] |
| [e.g., Repository pattern] | [specific problem] | [why direct DB access insufficient] |


## Progress Tracking
*This checklist is updated during execution flow*

**Phase Status**:
- [x] Phase 0: Research complete (/plan command)
- [x] Phase 1: Design complete (/plan command)
- [x] Phase 2: Task planning complete (/plan command - describe approach only)
- [ ] Phase 3: Tasks generated (/tasks command)
- [ ] Phase 4: Implementation complete
- [ ] Phase 5: Validation passed

**Gate Status**:
- [x] Initial Constitution Check: PASS
- [x] Post-Design Constitution Check: PASS
- [x] All NEEDS CLARIFICATION resolved
- [x] Complexity deviations documented

---
*Based on Constitution v2.1.1 - See `/memory/constitution.md`*
