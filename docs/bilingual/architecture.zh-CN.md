# BVP Solver 软件架构

[Русская версия](architecture.ru.md) · [双语索引](README.md)

## 1. 分层边界

```text
GUI / JSON
    ↓
Dataset / adapters
    ↓
BVPProblem + SolverConfig
    ↓
bvp_core public API
    ↓
restricted expression parser → BVPSolver → BVPResult
    ↓
SolveOutcome / SolveRecord → Plot / Export / GUI
```

`bvp_core` 是数值事实来源，不导入 `main.py`、PyQt5 或 Matplotlib。`main.py` 负责旧 `Dataset`
兼容、窗口、worker、绘图与交互。导入核心包不会创建 `QApplication`、窗口、日志文件或输出。

## 2. 核心模型与求解入口

`BVPProblem` 描述方程、变量、边界条件、已知/未知初值和时间区间；`SolverConfig` 描述算法与
容差；`solve_bvp_problem()` 返回不可变 `BVPResult`。模型不会在求解中原地修改，结果数组为
防御性只读副本。模型错误抛出 `BVPValidationError`，预期数值失败返回 `success=False` 的结果。

核心职责主要分布在 `models.py`、`tolerances.py`、`expressions.py`、`solver.py`、`api.py`、
`results.py`、`observability.py` 与 `serialization.py`。

## 3. 请求与 GUI 生命周期

GUI 点击求解时冻结不可变 `SolveRequest`，其中包含唯一 `request_id`、问题、配置、任务身份和
显示元数据。worker 返回 `SolveOutcome`；只有成功结果可形成 `SolveRecord`。GUI 只接受与当前
活动请求一致且尚未处理的终态，忽略过期或重复信号。

绘图与导出读取记录自身冻结的变量、时间、曲线和来源，而不读取后来被编辑的输入。因而同一
方程只改初始猜测、或修改方程数值、边界目标、已知初值、时间区间、求解方法、IVP 方法、容差
后得到的兼容结果，都可以独立保留并共同显示。坐标语义不兼容的记录会明确分组或排除。

## 4. 表达式安全边界

```text
string → ast.parse(mode="eval") → AST whitelist
       → direct SymPy construction → lambdify(modules="numpy")
```

生产字符串入口不使用 Python `eval`、`exec`、`parse_expr` 或 `sympify`。上下文白名单控制符号、
函数、常量、运算和复杂度，并拒绝属性访问、下标、lambda、容器、未知名称与私有名称。语法
安全不等于数值正确：例如合法的 `1/(t-0.5)` 仍会在跨越奇点时导致 IVP 失败。

## 5. 可观测性与运行关联

每次真实求解创建 `RunContext`，最终形成不可变 `RunMetadata` 和 `SolverCounters`。`run_id` 标识
一次运行，`request_id` 标识 GUI 请求，`problem_signature` 是规范化问题 JSON 的稳定 SHA-256。
计数器记录 IVP、nfev、`Phi`、Jacobian、优化器与延拓工作量。

核心通过 `bvp_core.events` 发出结构化 `logging` 事件，并只安装 `NullHandler`。应用方可配置
handler；默认不会创建日志文件，这里的“日志能力”不等于“磁盘上必有日志”。

## 6. 导出契约

JSON 与 TXT 共享 `CanonicalExportRecord`，schema 为 `bvp-result-v1`。记录来自原始请求快照和
对应运行，不读取导出时的 GUI 编辑器。JSON 使用 UTF-8、`ensure_ascii=False`、
`allow_nan=False`；非有限诊断显式规范化。两种格式都先写同目录临时文件、flush/`fsync`，再用
`os.replace` 原子替换。TXT 是摘要，JSON 是机器可读事实来源。

## 7. 失败与取消传播

IVP、有限性、优化、延拓和边界验收各有明确失败状态。失败结果保留诊断而不进入成功历史。
取消采用协作式检查点：解析前后、算法阶段、延拓步和最终验收附近均可响应，但正在执行的单次
SciPy `solve_ivp` 无法立即被 Python 线程强制中断。窗口关闭会请求取消并有限等待，不使用
`QThread.terminate()`。

## 8. 依赖与维护边界

CI workflow 覆盖 Windows/Ubuntu Python 3.11，具体见[测试文档](testing.zh-CN.md)。官方 GitHub
Actions 的 Node.js 20→24 注释记录在 [`docs/maintenance.md`](../maintenance.md)，当前为
`DEFERRED`、`LOW_NON_BLOCKING`，不属于算法或测试缺陷。仓库尚无 `LICENSE`，发布准备仍待
许可证决定。
