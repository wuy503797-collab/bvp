# BVP Solver v2.10 架构附录

## 1. 包依赖边界

```text
main.py (PyQt5 GUI, Dataset compatibility, worker, plot)
    │
    ├── bvp_core.adapters
    ├── bvp_core.requests
    ├── bvp_core.serialization
    └── bvp_core public API
            │
            ├── models / tolerances
            ├── expression_policy → expressions
            ├── solver → results / performance
            └── observability / serialization
```

约束：

- `main.py` 可以依赖 `bvp_core`；
- `bvp_core` 不导入 `main.py`；
- `bvp_core` 不导入 PyQt5，不创建窗口、绘图或 `QApplication`；
- 无 GUI 进程可只通过公共 API 求解并取得 `BVPResult`；
- JSON/旧 `Dataset` 在 adapter 边界转换成不可变 core 模型。

## 2. 求解数据流

```text
BVPProblem + SolverConfig
          ↓ validate
restricted expression parsing
          ↓ lambdify
       BVPSolver
          ↓
  shooting / continuation
          ↓
candidate parameters
          ↓
independent validation IVP
          ↓
original boundary residual
          ↓
BVPResult + RunMetadata + SolverCounters
```

### Shooting

```text
p → _Phi(p) → solve_ivp → boundary residual
                 ↓
        root(method="hybr")
                 ↓ fallback if needed
          least_squares
                 ↓
          candidate p
                 ↓
  validation solve_ivp + original residual
```

### Continuation

```text
fixed homotopy steps μ=0…1
          ↓
damped Newton at each μ
          ↓
forward-difference Jacobian
          ↓
solve / least-squares linear step
          ↓
failure propagation or next μ
          ↓
original-problem validation IVP
```

当前没有 analytic parameter Jacobian、adaptive continuation 或 pseudo-arclength。

## 3. 成功与失败传播

```text
algorithm candidate
    ├── IVP did not reach end ─────────────→ ivp_failed
    ├── p/t/y/residual non-finite ─────────→ non_finite_result
    ├── algorithm/optimizer invalid ───────→ optimizer_failed / continuation_failed
    ├── boundary component rejected ───────→ boundary_residual_too_large
    └── all four acceptance gates true ────→ success
```

```text
success = ivp_success
       AND finite_success
       AND boundary_success
       AND algorithm_success
```

失败 `BVPResult` 可携带积分器诊断与计数，但不能进入 `SolveRecord`。无数值结果的异常与取消由
`SolveOutcome` 表达。

## 4. 请求与 GUI 生命周期

```text
editable GUI state
       ↓ snapshot
immutable SolveRequest(request_id, problem, config, display metadata)
       ↓
RunContext(run_id, request_id, problem_signature)
       ↓
SolverWorker + CancellationToken
       ↓
completed / failed / cancelled signal
       ↓ request-id and duplicate guards
SolveOutcome
       ↓ only successful outcomes
SolveRecord(frozen request + result + frozen plot samples)
       ↓
history / plot / export
```

- stale signal：`request_id` 不是当前活动请求时忽略；
- duplicate signal：请求已进入 processed set 时忽略；
- task switch：编辑器变化不修改已冻结请求；
- cancellation：多个检查点合作终止，单次 `solve_ivp` 内不能抢占；
- window close：先取消并等待 worker，仍运行则拒绝关闭；
- plot：使用 `SolveRecord.primary_plot_t/y` 冻结副本，不读取后来变化的 dense solution；
- variants：初始猜测、ODE、边界条件、终止时间、积分方法、求解方法和容差不同的结果保持各自
  来源及曲线，可在坐标语义兼容时共同显示。

## 5. 表达式流

```text
ODE / boundary / auxiliary / scalar string
        ↓
context-specific symbol table
        ↓
ast.parse(mode="eval")        # parse syntax only
        ↓
node/operator/function/name/complexity policy
        ↓
recursive direct SymPy construction
        ↓
lambdify(modules="numpy")
        ↓
numeric callable
```

安全边界先验证完整 AST，再构造 SymPy。禁止属性、下标、容器、lambda、比较、布尔表达式、
comprehension、关键字/星号调用、私有名称和未知符号。输入路径不存在 `eval`、`exec`、
`parse_expr` 或 `sympify` 的旁路。

## 6. 容差与验收流

```text
SolverConfig
   ├── IVP tolerances ─────────────→ solve_ivp
   ├── root / least-squares tol ───→ optimizer
   ├── continuation residual tol ──→ Newton step acceptance
   ├── Jacobian relative step ─────→ finite differences
   └── boundary atol/rtol/scales ──→ final acceptance only
```

逐分量最终门槛：

\[
|\Phi_i|\le \text{boundary\_atol}
+\text{boundary\_rtol}\cdot\text{boundary\_scales}_i.
\]

## 7. 可观测性流

```text
RunContext
   ├── correlation: run_id / request_id / problem_signature
   ├── structured solver events via logging
   ├── elapsed / terminal status
   └── SolverCounters snapshot
                ↓
           RunMetadata
                ↓
       BVPResult / SolveOutcome / Export
```

核心不配置全局日志 handler，也不在默认路径输出用户表达式正文。

## 8. 导出流

```text
SolveOutcome or SolveRecord
          ↓
build_canonical_export_record
          ↓
CanonicalExportRecord (bvp-result-v1)
          ├── strict JSON (machine fact source)
          └── TXT summary (human-readable)
                    ↓
same-directory temporary UTF-8 file
          ↓ flush + fsync
          ↓ os.replace
atomic target replacement
```

问题和配置来自请求快照。JSON 禁止非标准 NaN/Infinity token，非有限值先变成显式结构。

## 9. 主要契约测试

| 边界 | 代表性验证 |
|---|---|
| import | core 不加载 GUI、`main` 或 `QApplication` |
| model | 维数、索引、初值、容差和 scale 预验证 |
| solver | 正向、无根、奇异、取消和计数器 |
| expression | 允许语言、攻击输入、持久化任务兼容 |
| request | 不可变、唯一 ID、signature、plot partition |
| GUI | stale/duplicate signal、close、取消、历史和多曲线 |
| export | 成功/失败/取消、严格 JSON、原子替换 |
| integration | 公共 API 到规范导出的端到端事实一致性 |
