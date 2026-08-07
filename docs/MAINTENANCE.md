# 仓库维护准则（Git 提交 + 文档管理）

> 本文档定义 `DEIMv2_DAOD` 仓库的 git 提交与文档维护规范，是 agent（Sisyphus）与人类协作者之间的协作契约。
> **后续将据此沉淀为可复用的 skill**；在沉淀之前，本文是唯一权威来源。
> 修改本文件需显式授权（见 §3.1）。

---

## 1. 角色与职责

| 职责 | Agent（Sisyphus） | 人类协作者 |
|------|-------------------|-----------|
| 起草 commit message | ✅ 负责 | 可修改 |
| 审核 commit message | 提议待审 | ✅ 最终决定 |
| 执行 `git commit` | ✅ 执行（审核通过后） | — |
| 执行 `git push` | ❌ 禁止 | ✅ 专属 |
| 新增文档 | ✅ 可自主 | — |
| 修改既有文档 | ❌ 需授权 | ✅ 决定 |
| 修改 production 代码 | 视任务而定 | 视任务而定 |
| 运行测试 / 调查 | ✅ 自主 | — |

**核心原则**：agent 负责起草与执行，人类负责审批与对外发布。任何影响共享系统（git 历史、远程、既有文档内容）的操作都必须经过人类显式批准。

---

## 2. Git 提交准则

### 2.1 审批门（硬性约束）

> **git commit 内容由 agent 编辑，但必须在人类审核通过后才能 commit。**

- agent 起草 commit message 与拆分方案 → 提交人类审核
- 人类回复「审核通过」或给出修改意见后，agent 才执行 `git commit`
- **未收到明确批准前，agent 不得 commit**（即便续跑指令催促也不可绕过此门）

### 2.2 原子提交（最小变更）

- **每次 commit 只做一件事**：一个主题、一个逻辑变更
- 拆分维度：按主题（feature A vs feature B）、按性质（代码 vs 文档 vs 测试）、按文件归属
- 反例：把 spec 修订 + 实现代码 + 不相关设计文档塞进一个 commit
- 正例：`feat: 新增 X 函数` 与 `docs: 新增 Y 设计` 分开提交

### 2.3 Commit Message 格式

采用 **Conventional Commits**，描述用中文：

```
<type>: <中文描述>
```

**允许的 type：**

| type | 用途 |
|------|------|
| `feat` | 新功能 / 新实现 |
| `fix` | bug 修复 |
| `refactor` | 重构（行为不变） |
| `test` | 新增 / 调整测试 |
| `docs` | 文档新增 / 修订 |
| `chore` | 杂项（忽略规则、构建配置等） |

**格式约定（规范化）：**
- 半角冒号 + 一个空格：`feat: 描述`（不用全角 `：`，不留空格）
- 单行 summary，**不加 body**（仓库既有惯例）
- 描述用中文，可含专有名词 / 文件路径（用反引号标注）
- 描述应精炼到能独立回答「这个 commit 做了什么」

**示例：**
```
feat: 新增 shifted 角度编解码纯函数（decoder 私有编码）
docs: 简化 shifted-angle 设计为双模式（proportional|shifted），移除 shifted_direct 与 scope A/B
fix: 修复混合精度训练下 ema 不更新的问题
```

### 2.4 提交前验证

**证据优先，禁止「应该能过」的断言。**

- **行为变更**（feat/fix/refactor）：运行相关测试，确认 GREEN 后再提交
- **新增测试**：至少运行该测试文件全量，确认既有用例不回归
- **文档**：无需测试，但确认链接 / 路径正确
- **RED 测试处理**：不单独提交 RED（失败）测试。TDD 流程下，RED 测试应与对应实现配对提交，或在实现完成的同一个 Unit commit 中提交

### 2.5 暂存纪律

- **精确暂存**：`git add <具体文件>`，一次只 add 该 commit 涉及的文件
- **禁止** `git add -A` / `git add .` / `git add <dir>`（除非目录内所有文件都属于本 commit）
- 暂存后用 `git status --short` 复核，确认无意外文件入列
- 特别注意 `.omo/`、`__pycache__/`、日志文件等不应入库的产物

### 2.6 禁止操作

agent **永远不**执行以下操作（除非人类显式要求）：

- `git push` / `git push --force`
- `git revert` / `git reset --hard`（破坏性）
- `git commit --amend`（除非人类要求修正刚刚的 commit）
- 跳过 git hooks（`--no-verify`）
- 交互式 rebase（`-i`）
- 修改 git config
- 任何影响远程 / 共享分支的操作

---

## 3. 文档维护准则

### 3.1 改动许可（硬性约束）

> **不可改动原有文档的内容和位置，除非经过人类授权。**
> **可以新增描述性或其他文档。**

- **既有文档**（已入库的 `.md`）：内容修改、重命名、移动、删除 → 必须授权
- **新增文档**：agent 可自主创建（遵循 §3.2-§3.4 的归属与命名规则）
- 修改本文件（`MAINTENANCE.md`）同样需要授权

### 3.2 双轨文档体系

仓库维护两套互补的文档系统：

| 系统 | 角色 | 路径 | 内容 |
|------|------|------|------|
| **Superpowers** | 工作区 | `docs/superpowers/` | agent 工作流制品：探索、设计草稿、实施计划、评审、实验结果 |
| **OpenSpec** | 决策系统 | `openspec/` | 正式决策、规范、变更提案 |

**一条线不替代另一条**：一个正式决策必须有 `openspec/` 文档，但其探索过程（评审、计划迭代）可放在 `docs/superpowers/`。

### 3.3 命名与归属

#### `docs/superpowers/` 子目录

| 子目录 | 放什么 | 命名 |
|--------|--------|------|
| `design/` | 脑暴 / 设计草稿 | `YYYY-MM-DD-<主题>-design.md` |
| `plans/` | 实施计划（agent 执行用） | `YYYY-MM-DD-<主题>.md` 或 `-plan` |
| `specs/` | 已批准的设计规范 | `YYYY-MM-DD-<主题>-design.md` |
| `review/` | 代码评审、方案评审、实验结果、流程笔记 | `YYYY-MM-DD-<主题>.md` |

> 持久参考文档（如本文件）不放日期前缀，置于 `docs/` 顶层。

#### `openspec/` 子目录

| 子目录 / 文件 | 放什么 |
|---------------|--------|
| `changes/<name>/proposal.md` | 正式变更提案 |
| `changes/<name>/design.md` | 正式架构设计 |
| `changes/<name>/tasks.md` | 正式任务列表 |
| `changes/<name>/specs/` | 正式能力规范 |
| `changes/<name>/analysis/` | 深度分析报告 |
| `specs/` | 跨变更的稳定规范 |

### 3.4 INDEX.md 同步

`docs/superpowers/INDEX.md` 自身规定：

> **本索引必须覆盖 `docs/superpowers/**/*.md` 所有 Markdown 文件；故意排除的文件需在索引中注明原因。**

- 新增 / 删除 `docs/superpowers/` 下任何 `.md` 时，**必须同步更新 INDEX.md**
- 索引维护应作为**独立的原子 commit**，不与 feature 工作混合
- 每个索引条目需一句话准确描述（需读文件取摘要，不可臆造）
- `openspec/INDEX.md` 同理（覆盖 `openspec/` 下文档）

### 3.5 工作流归属判断

**放 superpowers：** 探索、脑暴、计划、评审、实验记录、流程笔记
**放 openspec：** 已接受的正式决策、规范、架构设计、任务分解、深度分析

不确定时，问一句：**「这是过程产物，还是已定决策？」** 过程 → superpowers；决策 → openspec。

---

## 4. 工作流集成

### 4.1 TDD 计划与 commit 边界

当实施遵循 superpowers TDD 计划（如 `docs/superpowers/plans/*.md`）时：

- 计划中的 **Review Unit = 一个 commit 边界**
- 每个 Unit 的 commit message 通常已在计划中预先定义
- agent 按 Unit 顺序提交，不跨 Unit 合并

### 4.2 Gate 顺序（每个 Unit）

```
1. AI Test Gate（RED）  —— agent 写失败测试，验证预期失败
2. User Implementation Gate —— 人类写 production 代码
3. AI Green/Review Gate —— agent 运行 GREEN + 代码 review
4. Commit Gate          —— 审批通过后，agent 提交
```

**禁止跳过 Gate**：未通过 GREEN 验证的 Unit 不进入 Commit Gate。

### 4.3 提交顺序原则

多个独立 commit 按依赖顺序提交：
- 被依赖方先行（如 spec 修订 → 实现该 spec 的代码 → 引用该 spec 的计划）
- 同层无依赖时，按主题重要性 / 逻辑顺序排列
- 文档与代码分属不同 commit，即使同一 feature

---

## 5. 决策权限矩阵（速查）

| 操作 | agent 可自主 | 需人类批准 |
|------|:-----------:|:---------:|
| 读取文件 / 运行测试 / 调查 | ✅ | |
| 起草 commit message | ✅ | |
| 新增文档（遵循归属规则） | ✅ | |
| 精确暂存 `git add <file>` | ✅ | |
| 执行 `git commit` | | ✅（审核门） |
| 修改既有文档内容 / 位置 | | ✅ |
| `git push` | | 人类专属 |
| 破坏性 git 操作（revert/reset --hard/amend） | | ✅ |
| 修改 production 代码 | 视任务授权 | |
| 跳过 / 改动 git hooks / config | | ✅ |

---

## 6. 现状基线（2026-08-07）

记录创建时的仓库状态，作为后续演进的参照点：

- **分支**：`dev`
- **commit message 语言**：中文描述 + 英文 type
- **既有 colon 风格不一致**：历史 commit 混用半角 `:` 与全角 `：`、有无空格；本规范统一为半角冒号 + 空格
- **INDEX.md 漂移**：`docs/superpowers/` 下 49 个 `.md`，INDEX 仅索引 17 个（specs 12/13 缺、plans 16/22 缺、review 4/13 缺）——已知欠债，待独立维护 commit 补齐
- **既有 commit 不含 body**：单行 summary 为主
- **production 代码署名**：部分计划规定「production 代码由人类编写，agent 只写测试」（TDD 流程）

---

## 附：演变为 skill 的路径

本文档设计为可直接沉淀为 skill 的种子。skill 化时的映射建议：

- **触发条件**：「commit」「提交」「整理提交」「文档维护」「更新索引」「openspec」「superpowers」等
- **核心约束**：§2.1 审批门 + §3.1 改动许可 为硬性规则，skill 不可绕过
- **检查清单**：每次 commit 前走 §2.4 验证清单；每次文档变更前走 §3 归属判断
- **决策树**：§5 权限矩阵作为「能否自主执行」的第一判断依据
