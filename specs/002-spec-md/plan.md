
# Implementation Plan: A股全市场量化因子挖掘与决策支持系统

**Branch**: `002-spec-md` | **Date**: 2025-09-20 | **Spec**: [spec.md](./spec.md)
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
基于Hikyuu框架构建的A股全市场量化因子挖掘与决策支持系统，提供数据驱动、可解释、可验证的交易洞察。系统采用Agent架构设计，支持因子计算、验证和信号生成的完整工具链。基于DeepWiki确认，Hikyuu框架在AMD 7950x上计算全A股市场20日均线仅需166毫秒，完全满足30分钟全市场因子计算和15分钟每日信号生成的性能要求。

## Technical Context
**Language/Version**: Python 3.11+ (ARM NEON优化支持Apple Silicon)
**Primary Dependencies**: Hikyuu量化框架(C++核心), FastAPI, SQLAlchemy, Pandas, NumPy, MySQL连接池
**Storage**: MySQL 8.0+ (Hikyuu原生数据结构), HDF5内存映射, Redis (因子缓存), 文件系统 (审计日志)
**Testing**: pytest, pytest-asyncio, pytest-mock (TDD强制要求)
**Target Platform**: macOS (Apple Silicon ARM优化), Linux (生产环境), Docker容器化部署
**Project Type**: single (Agent微服务架构)
**Performance Goals**: 30分钟全市场单因子计算, 15分钟每日信号生成 (基于Hikyuu 166ms基准性能)
**Constraints**: Point-in-Time数据访问约束, 强制人工确认交易信号, 不可变审计日志链
**Scale/Scope**: 5000+A股股票全市场覆盖, 100+并发因子计算, 4个Agent微服务模块

## Constitution Check
*GATE: Must pass before Phase 0 research. Re-check after Phase 1 design.*

**基于constitution.md的检查门控**:
- **TDD强制要求**: ✓ 所有实现必须先写测试，测试失败后再实现
- **Agent架构设计**: ✓ 每个Agent独立部署，通过RESTful API通信
- **性能约束**: ✓ 基于Hikyuu 166ms基准，满足30分钟/15分钟性能目标
- **简化原则**: ✓ 基于成熟的Hikyuu框架，避免重复造轮子
- **技术选择**: ✓ CPU优化路径(多线程+SIMD+ARM NEON)，无需GPU加速

**PASS**: 无constitutional违规项目

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

**Structure Decision**: Option 1 (Single project) - Agent微服务架构，每个Agent独立但在同一代码库中，基于Hikyuu性能优化

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
- Load `.specify/templates/tasks-template.md` as base
- Generate tasks from Phase 1 design docs (contracts, data model, quickstart)
- **平台优化任务**: 每个Agent添加平台检测和优化配置任务 [P]
- Each contract → contract test task [P] (包含新的system-api.yaml)
- Each entity → model creation task [P] (包含OptimizationConfig模型)
- Each user story → integration test task
- Implementation tasks to make tests pass

**新增任务类别**:
- **Platform Setup**: 平台检测、优化配置管理
- **Performance Tests**: 针对Apple Silicon/x86_64的性能基准测试
- **Agent Integration**: 平台优化在各Agent中的集成

**Ordering Strategy**:
- TDD order: Tests before implementation
- Platform setup before Agent initialization
- Dependency order: Models before services before UI
- Mark [P] for parallel execution (independent files)

**Estimated Output**: 35-40 numbered, ordered tasks in tasks.md (增加了平台优化相关任务)

**IMPORTANT**: This phase is executed by the /tasks command, NOT by /plan

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
- [x] Phase 0: Research complete (/plan command) - 平台自适应优化策略已确定
- [x] Phase 1: Design complete (/plan command) - 数据模型、API契约、快速开始指南已更新
- [x] Phase 2: Task planning complete (/plan command - describe approach only) - 任务生成策略已规划
- [x] Phase 3: Tasks generated (/tasks command) - 74个具体实现任务已生成
- [ ] Phase 4: Implementation complete
- [ ] Phase 5: Validation passed

**Gate Status**:
- [x] Initial Constitution Check: PASS
- [x] Post-Design Constitution Check: PASS - 平台优化架构符合简化原则
- [x] All NEEDS CLARIFICATION resolved - Hikyuu性能特征已通过DeepWiki确认
- [x] Complexity deviations documented - 无需额外复杂性

---
*Based on Constitution v2.1.1 - See `/memory/constitution.md`*
