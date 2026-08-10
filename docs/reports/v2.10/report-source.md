# BVP Solver v2.10 技术报告：边值问题数值求解器的可靠性、工程化与验证

*BVP Solver v2.10: Numerical Reliability, Engineering and Validation*

## 版本声明

本文档描述 BVP Solver v2.10 对应代码快照下的设计、实现、验证结果和已知限制。

软件版本：v2.10<br>
对应 Git commit：`2ac1a20`<br>
进入 `main` 的 merge commit：`1413ba43`

该版本已经通过本地验证、独立干净环境验证及 Windows / Ubuntu GitHub Actions 持续集成验证。

本项目仍持续迭代。本文档属于版本历史记录，不代表项目未来的最终状态。本文档产生于软件
快照之后，报告文档提交不改变上述软件版本绑定；在文档尚未提交时，
`report_document_commit=null`。

## 1. 摘要

BVP Solver 是面向教学与数值实验的常微分方程边值问题（BVP）求解器。v2.10 提供打靶法和
参数延拓两种求解路径、PyQt5 图形界面、JSON 任务适配器以及稳定的无 GUI 公共 API。项目从
课程原型逐步增加了严格的数值成功判据、解析解与制造解验证、受限数学表达式语言、分离的
容差语义、不可变请求快照、结构化运行元数据、统一原子导出、性能计数和跨平台 CI。

v2.10 的可靠性核心不是把优化器的终止标志直接当作答案，而是对候选参数执行独立构造的
最终 IVP，并重新计算原始边界残差；只有积分、有限性、边界验收和算法状态全部成立，结果才
进入成功历史、绘图和导出。本阶段实测 285 项 pytest 全部通过且 0 warnings，全部专项验证
脚本通过，Phase 11 分支与合并后 `main` 的远端 CI 均通过。仓库仍未确定 LICENSE，因此当前
状态是 `release_readiness_status=PENDING_LICENSE`，不能据此推断复制、修改或分发授权。

本项目不是生产级通用数值平台，也不适合安全关键用途。有限差分 Jacobian、固定延拓步、
协作式取消和缺少部分任务解析真值等限制仍需在使用结果时明确考虑。

## 2. 从 v2.0 到 v2.10 的工程演进

版本号、内部阶段与提交使用仓库既有映射，不以报告编写时间重新编号。

| 版本 | 内部阶段 | 绑定 commit | 主要演进 |
|---|---:|---|---|
| v2.0 | 课程快照 | `a4ef249` | 集成绘图、双语 GUI、JSON 任务库和辅助输出 |
| 未正式分配版本 | Phase 1 | `1f62eac` | 自动化基线与严格已知缺陷测试 |
| v2.1 | Phase 2 | `b9cb298` | 统一结果验收，修复错误成功和失败传播 |
| v2.2 | Phase 3 | `fdcc672` | 解析解、制造解、容差敏感性和数值一致性验证 |
| v2.3 | Phase 4 | `525fac1` | 稳定的无 GUI Core API 与结果模型 |
| v2.4 | Phase 5 | `eeee178` | 不可变 GUI 请求快照、结果溯源和线程生命周期 |
| v2.5 | Phase 6 | `5f480c3` | 解析器和求解器迁入 `bvp_core`，消除对 `main.py` 的反向依赖 |
| v2.6 | Phase 7 | `e732b4e` | 受限表达式语言、安全策略和统一错误边界 |
| v2.7 | Phase 8 | `b11dc1a` | 显式求解容差和尺度化边界验收 |
| v2.8 | Phase 9 | `2b1c6d7` | 结构化运行元数据、统一导出契约和原子写入 |
| v2.9 | Phase 10 | `7cea5d3` | 性能计数、可复现基准和冗余计算清理 |
| v2.10 | Phase 11 | `2ac1a20` | 干净环境、跨平台 CI、GUI 回归和发布准备 |

v2.0 是原始课程时期报告所绑定的历史快照；v2.10 是经过多轮可靠性与工程化迭代后的软件
快照。后续版本发现并修正了早期实现中的若干正确性、架构、安全和可复现性问题，但 v2.0
原始报告及其 PDF 保持历史原貌，不被当前能力反向改写。

## 3. 数学模型

考虑一阶常微分方程组

\[
\dot{x}=f(t,x),\qquad t\in[t_0,T],\qquad x(t)\in\mathbb{R}^n.
\]

一部分初值已知，另一部分由参数向量表示：

\[
p\in\mathbb{R}^k.
\]

给定由起点和终点状态构成的边界残差函数，定义

\[
\Phi(p)=R\!\left(x(t_0;p),x(T;p)\right).
\]

求解目标为

\[
\Phi(p)=0.
\]

当前实现要求边界残差数量等于未知初始参数数量，即 `len(R)=k`。该维数条件在数值求解前
验证。即使维数匹配，问题仍可能多解、无解、在积分区间包含奇异点、对初始猜测敏感或具有
刚性；软件的成功状态只证明给定数值路径通过既定验收，不证明解的唯一性或枚举完整性。

## 4. 打靶法

打靶法的真实执行流程为：

```text
未知参数 p
→ 组装完整初值
→ solve_ivp
→ 计算 Φ(p)
→ scipy.optimize.root(method="hybr")
→ 必要时 scipy.optimize.least_squares fallback
→ 候选参数
→ 独立的最终 solve_ivp
→ 重新计算原始边界残差
→ 最终验收
```

`root` 或 `least_squares` 的成功字段是优化过程诊断，不是 BVP 的数学验收。优化器可能因步长、
梯度或迭代停止条件终止，但候选点仍可能具有不可接受的边界残差；也可能在候选点评估时积分
失败或产生非有限值。因此：

```text
optimizer_success != BVP success
```

最终验证重新积分，不复用优化器内部最后一次可能不完整的轨迹，并基于原始边界条件重新计算
残差。这一设计使“优化过程结束”和“边值问题可接受”成为两个显式层次。

## 5. 参数延拓

当前参数延拓构造从初始猜测处容易满足的残差到原始残差的固定步同伦问题，并在每个延拓步
使用阻尼 Newton 迭代。Jacobian 采用前向有限差分；线性系统优先直接求解，必要时使用最小
二乘。阻尼候选依次减小步幅，以寻求残差下降。

延拓步骤、Newton 迭代、有限差分参数扰动和 IVP 积分中的失败会显式传播；完成全部同伦步骤
后仍需对原始问题执行独立 IVP 和最终边界验收。v2.10 没有实现伪弧长延拓、自适应延拓步长或
解析参数 Jacobian，不应将固定步同伦描述为这些更高级算法。

## 6. 最终成功判据

v2.10 的最终状态按下式组合：

```text
success =
    ivp_success
    AND finite_success
    AND boundary_success
    AND algorithm_success
```

- `ivp_success`：独立最终 IVP 成功到达终点；
- `finite_success`：参数、采样时间、状态和边界残差均有限且形状有效；
- `boundary_success`：积分和有限性成立，并通过逐分量边界容差；
- `algorithm_success`：打靶或延拓算法没有以失败状态退出；
- `optimizer_success`：仅用于诊断优化器行为，不替代上述组合判据。

失败结果保留结构化状态，例如 `ivp_failed`、`non_finite_result`、
`optimizer_failed`、`boundary_residual_too_large` 或 `continuation_failed`，不会进入成功记录。

## 7. 容差体系与尺度化边界验收

求解配置将不同数值过程的容差语义显式拆分：

| 配置 | 作用 |
|---|---|
| `ivp_rtol`, `ivp_atol` | `solve_ivp` 相对/绝对误差控制 |
| `root_tol` | `scipy.optimize.root` 的停止容差 |
| `least_squares_ftol`, `least_squares_xtol`, `least_squares_gtol` | fallback 的函数值、参数和梯度停止条件 |
| `continuation_residual_tol` | 延拓 Newton 残差阈值 |
| `jacobian_relative_step` | 前向有限差分参数步长语义 |
| `boundary_atol`, `boundary_rtol` | 最终边界验收绝对/相对阈值 |
| `boundary_scales` | 最终边界逐分量尺度 |

兼容字段 `eps` 仍可映射到旧调用预期，但新配置允许各过程独立设置。最终逐分量判据为

\[
|\Phi_i|\le a_{\mathrm{tol}}+r_{\mathrm{tol}}s_i.
\]

`boundary_scales` 只参与最终验收，不改变优化器收到的原始残差，也不会把不同量纲的约束在
优化阶段静默重加权。尺度需由用户根据问题量纲明确提供；缺省值不能替代领域判断。

## 8. 受限数学表达式语言

ODE、边界条件、辅助输出和 GUI 标量初值共用受限解析边界：

```text
用户字符串
→ ast.parse(mode="eval")
→ 完整 AST 白名单与复杂度验证
→ 逐节点直接构造 SymPy 表达式
→ lambdify(modules="numpy")
```

生产输入路径不使用 `parse_expr`、`sympify`、Python `eval` 或 `exec`。策略只允许明确列出的
算术/一元运算符、数学函数、常量与上下文符号；拒绝属性、下标、关键字参数、星号参数、私有
名称和未知名称，并限制字符数、AST 节点、嵌套深度、数字位数、函数调用数、幂运算数和指数
绝对值。

安全解析通过只说明字符串满足项目数学语言，不等于 ODE 数值积分必然成功，也不等于 BVP
存在解。语法安全、数值可积性和边值问题可解性是三个不同层次。

## 9. 软件架构

```text
GUI / JSON
    ↓
Dataset / adapters
    ↓
BVPProblem + SolverConfig
    ↓
bvp_core public API
    ↓
Restricted Expression Parser
    ↓
BVPSolver
    ↓
BVPResult
    ↓
SolveOutcome / SolveRecord
    ↓
Plot / Export / GUI
```

`bvp_core` 持有问题模型、配置、解析器、求解器、结果、请求、可观测性、性能计数与序列化契约。
核心包不导入 `main`，不导入 PyQt5，也不创建 `QApplication`；公共 API 可在无 GUI 进程中
解析和求解。`main.py` 负责旧 `Dataset` 兼容、界面、工作线程、绘图和用户交互。

更详细依赖、数据流与失败传播见 [architecture.md](architecture.md)。

## 10. 请求生命周期与绘图隔离

每次求解先冻结为不可变 `SolveRequest`，包含 `request_id`、`BVPProblem`、`SolverConfig`、来源
任务和显示元数据。`problem_signature` 使用稳定 SHA-256 绑定问题事实；运行时再由 `RunContext`
生成独立 `run_id`。终态通过 `SolveOutcome` 区分 completed、failed 和 cancelled，成功结果才可
成为 `SolveRecord`。

GUI 只接受与当前活动 `request_id` 一致的终态；旧工作线程信号被忽略，重复终态信号只处理
一次。切换任务或继续编辑输入不会改变已经启动的请求快照。关闭窗口时请求取消并等待工作
线程退出，超时则拒绝关闭，避免界面对象销毁后后台线程继续回调。

绘图使用每个 `SolveRecord` 冻结的曲线与来源，而不是当前编辑器内容或可变求解器 dense output。
回归测试验证同一任务仅改变初始猜测的多个分支可同时勾选并显示；还覆盖 ODE、边界条件、
终止时间、IVP 方法、求解方法和容差配置变化后的多记录保持独立。只有状态维数、变量名和时间
区间等绘图坐标语义不兼容的记录才被分组排除。

## 11. 取消语义

取消是 cooperative cancellation。API、工作线程、打靶和延拓循环设置多个检查点，取消状态
保留独立结果与运行元数据。但是单次 `solve_ivp` 正在执行时不能被 Python 线程立即强制终止；
请求只能在控制返回下一个检查点时结束。因此 v2.10 不提供实时强制取消保证。

## 12. 可观测性

`RunContext`、`RunMetadata` 和 `SolverCounters` 将一次运行绑定到 `run_id`、`request_id` 与
`problem_signature`。计数器记录 IVP 调用、函数评估、`Phi`、Jacobian、优化器与延拓工作量，
成功、失败和取消路径都保留可用快照。

核心使用标准库 `logging` 产生带相关标识的结构化事件，但不调用 `basicConfig`，导入和默认
求解不会污染 stdout/stderr。展示层或调用方决定日志 handler 与级别。运行元数据避免记录
原始用户表达式正文，降低日志泄露输入内容的风险。

## 13. 统一导出契约

JSON 和 TXT 导出共享 `CanonicalExportRecord`，schema 为 `bvp-result-v1`。问题和配置来自
`SolveRequest` 快照，运行信息来自对应结果/终态，而不是导出时的当前 GUI 输入。成功、失败和
取消拥有不同 `record_type`，但共用来源、配置与运行事实结构。

文本使用 UTF-8。JSON 使用 `allow_nan=False`，非有限数值在规范化阶段显式表示，避免输出非
标准 `NaN`/`Infinity` token。写入先在目标同目录创建临时文件，flush 后执行 `fsync`，再用
`os.replace` 原子替换；异常时清理临时文件。TXT 是人类可读摘要，JSON 是机器可读事实来源。

## 14. 自动化测试与验证体系

本报告阶段重新执行完整测试，结果为：

```text
285 passed in 14.13s
warnings = 0
```

由于 Windows 临时目录权限与仓库卫生测试的自检查范围，最终计数使用仓库外专用
`--basetemp` 复核；这只隔离测试临时物，不修改测试门槛或产品代码。覆盖类别包括 core、
numerical、GUI、integration、expression security、tolerances、observability、performance、
CI/repository hygiene。

专项脚本分别验证数值基线、解析/制造解、表达式边界、容差语义、运行元数据与导出、确定性
性能契约、仓库卫生和统一 CI 入口。命令与关键证据见 [validation-evidence.md](validation-evidence.md)。

## 15. 解析解与制造解验证

所有误差数据来自本报告阶段重新运行 `scripts/run_numerical_validation.py`，采样点数为 201。

| 问题 / 方法 | 参数误差 | 最大状态误差 | RMS 状态误差 | 边界残差范数 | 最大 ODE defect |
|---|---:|---:|---:|---:|---:|
| 指数 `x'=x` / shooting | `2.01665351085e-09` | `5.28216670403e-09` | `1.78454824072e-09` | `4.44089209850e-16` | `1.16312365989e-09` |
| 指数 `x'=x` / continuation | `2.01679262180e-09` | `5.28181587356e-09` | `1.78447031157e-09` | `3.77919917582e-13` | `1.16312405020e-09` |
| 简谐振子 / shooting | `2.39334307928e-09` | `3.24697235765e-09` | `1.48556439051e-09` | `4.44089209850e-16` | `6.22137192458e-10` |
| 制造三次多项式 / shooting | `0` | `2.22044604925e-15` | `5.07278600450e-16` | `2.91433543964e-16` | `1.73472347598e-15` |

指数问题的解析解用于同时比较 shooting 与 continuation；简谐振子满足
\(x'=v,\ v'=-x\)；制造解通过先指定多项式真解再构造与之匹配的方程和边界条件，为状态、
参数和边界提供数值 oracle。重复性、方法比较和容差敏感性检查也全部通过。

## 16. ODE 数值一致性指标

对 dense output 采样区间，验证脚本计算 Simpson 型 defect：

\[
D_i=y(t_{i+1})-y(t_i)-\frac{h_i}{6}
\left[f_i+4f_{i+1/2}+f_{i+1}\right].
\]

它独立于边界残差和求解器 `success` 字段构造，可发现“终点边界看似满足但轨迹与 ODE 不
一致”的问题；但采样仍使用同一次 `solve_ivp` 的 dense output。因此它是独立构造的数值
一致性指标，不是完全独立于积分器的外部验证，也不是严格后验误差界。

## 17. 26.1 双体问题与多分支

当前基线重新确认同一终端约束下的两个数值分支。

**Shooting**

```text
p_opt = [-2.380789650244e-08, 5.000000147112e-01]
boundary residual = [-1.402100657799e-11, -3.174704943376e-11]
boundary residual norm = 3.47053853632e-11
```

**Continuation**

```text
p_opt = [4.51078122246e-01, -2.99418638352e-01]
boundary residual = [-3.958611216603e-12, 9.391820654514e-12]
boundary residual norm = 1.01920016665e-11
```

两组参数都通过相同终端约束，说明该算例存在由不同初始猜测/数值路径得到的多分支现象；这
不是对全部解分支的枚举或唯一性证明。GUI 的记录与绘图隔离允许两种结果同时保留和比较。

## 18. 负向测试

### 18.1 无实根

构造边界方程 `x(T)^2+1=0`。shooting 返回
`success=False`、`status=boundary_residual_too_large`、边界残差范数 `1`；continuation 返回
`success=False`、`status=continuation_failed`、边界残差范数 `1`。该测试直接验证优化器终止
不等于有效 BVP 解。

### 18.2 奇异 IVP

对 `x'=1/(t-0.5)`、区间 `[0,1]` 的当前实测诊断为：

```text
success = False
status = ivp_failed
ivp_status = -1
ivp_t_final = 0.499999999999952
boundary_residual_norm = inf
```

积分器在奇点前停止，失败状态和已到达时间被保留；程序没有把不完整轨迹包装成成功结果。

### 18.3 过定约束

当一个未知初始参数对应两个边界残差时，模型验证返回清晰的维数错误，并在数值求解前拒绝：

```text
unknown count != boundary residual count
```

## 19. 基于证据的性能优化

Phase 10 删除了 `_dPhi_dp` 中未参与最终有限差分 Jacobian 的冗余变分矩阵积分，并在同一
Newton 步中安全复用已经计算的 `Phi_current`。有限差分公式、扰动方向和最终验收均未改变。

26.1 continuation 的确定性工作量证据为：

| 指标 | 优化前参考路径 | v2.10 路径 |
|---|---:|---:|
| IVP calls | 874 | 600 |
| total nfev | 131732 | 94200 |
| `Phi` calls | 736 | 599 |
| Jacobian calls | 137 | 137 |
| 每次 Jacobian 的变分 IVP | 1 | 0 |

Jacobian 对照最大差异为 0，26.1 continuation 的最终残差范数仍为
`1.01920016665e-11`。Phase 10 历史记录机器上的中位耗时由约 0.956 s 降至 0.617 s，约
35.5%；本报告阶段当前机器重测为约 1.545 s 降至 1.108 s，约 28.3%。耗时受硬件、负载和
依赖版本影响，不是跨机器性能保证。CI 只检查数值等价、调用计数与冗余工作消失，不使用绝对
耗时或固定提升百分比作为通过门槛。

## 20. 干净环境、CI 与 Warning 状态

Phase 11 在当前开发环境、独立范围依赖环境和独立精确参考环境中执行依赖安装与统一验证；
`pip check`、导入、`compileall`、pytest 和全部专项脚本均通过。GitHub Actions 实际运行覆盖：

- Windows / Python 3.11 范围依赖；
- Ubuntu / Python 3.11 范围依赖；
- Ubuntu / Python 3.11 精确直接依赖参考环境；
- `QT_QPA_PLATFORM=offscreen` 的 GUI 测试，未跳过；
- 不使用 `continue-on-error`。

Phase 11 分支 CI run `31305225622` 为 PASS；合入 `main` 后 run `31310177928` 也为 PASS。
本报告阶段统一验证再次得到 `285 passed`。pytest 输出为 0 warnings。原 Matplotlib
experimental toolmanager 警告通过删除全局 `rcParams["toolbar"]="toolmanager"` 配置解决，
界面继续使用 `NavigationToolbar2QT`；仓库没有用全局 warning filter 隐藏问题。

## 21. v2.0 与 v2.10 核心对比

| 项目 | v2.0 历史快照 | v2.10 历史快照 |
|---|---|---|
| 自动化测试 | 未建立当前分层测试体系 | 285 项，覆盖 core/numerical/GUI/integration/CI |
| 最终成功判据 | 后续审计发现会发生错误成功 | IVP、有限性、边界和算法状态联合验收 |
| 数值正确性证据 | 未建立解析解/制造解自动验证 | 解析解、制造解、defect、容差敏感性 |
| Core / GUI | 主要实现集中在 `main.py` | 无 GUI `bvp_core` API，不反向依赖 `main` |
| 表达式安全 | 受限 AST 边界未建立 | 白名单 AST，直接构造 SymPy |
| GUI 生命周期 | 不可变请求与终态协议未建立 | 请求快照、溯源、旧/重复信号防护、记录隔离 |
| 容差 | 独立语义未建立 | IVP、优化、延拓、Jacobian、边界验收拆分 |
| logging | 结构化运行关联未建立 | `run_id` / `request_id` / signature 与事件 |
| export | 统一规范与原子替换未建立 | `bvp-result-v1`，JSON/TXT 同源、严格 JSON、原子写入 |
| performance | 确定性计数基线未建立 | 计数器、等价性基准和冗余工作清理 |
| CI | not established | Windows/Ubuntu/精确参考环境远端 PASS |
| 干净环境 | not established | 两类独立干净环境验证 |

表中 v2.0 只使用历史报告、真实代码快照和后续历史审计能够支持的结论；无法建立的数据标为
`not established`，不把 v2.10 能力反向写入课程时期。

## 22. 当前限制与适用边界

- 参数 Jacobian 仍是前向有限差分，可能受尺度与舍入误差影响；
- continuation 使用固定步数，未实现自适应步长或 pseudo-arclength；
- 单次 `solve_ivp` 执行中不能即时强制取消；
- `boundary_scales` 需要用户按问题量纲明确提供；
- GUI 暂无覆盖全部高级字段的完整容差编辑器；
- 教材任务 26.2–26.5 没有解析真值，只能使用多指标一致性证据；
- 26.1 已验证两个分支，但未枚举全部可能解；
- Simpson 型 ODE defect 是数值一致性指标，不是严格误差界；
- 固定步打靶/延拓对初始猜测、奇异性和刚性仍可能敏感；
- 项目主要用于教学和数值实验，不适合安全关键用途；
- 仓库 LICENSE 尚未决定，不可推断代码使用授权；
- 重复任务 JSON、空兼容文件等部分旧兼容资产尚未单独清理。

## 23. 报告冻结、资产与发布状态

本报告的 `snapshot_status=historical_snapshot`。未来软件产生新版本后，不把新能力回写到 v2.10。
不改变技术事实的排版、链接和错别字可修正；技术事实修订应增加 `report_revision` 并记录
`revision_reason`。

本阶段没有生成 PDF，没有创建 tag 或 GitHub Release，没有选择 LICENSE，也没有提交或推送。
仓库根目录的 7 张用户截图未移动、未重命名、未纳入本报告；建议使用位置仅记录在
[assets/README.md](assets/README.md)，须由用户后续确认。

## 24. 结论

v2.10 将课程型 BVP 原型推进为一个可验证、可追溯且可在 Windows/Linux 重现的教学与数值
实验软件快照。最关键的工程结论是：优化器结束不等于边值问题成立。通过独立最终积分、边界
验收、负向测试、数值 oracle、请求快照、安全解析、运行元数据和跨平台 CI，项目为每一次
“成功”建立了可检查的证据链，同时如实保留有限差分、固定步延拓、协作式取消和许可证未定等
边界。

```text
project_version=v2.10
report_snapshot_status=PASS
remote_ci_status=PASS
release_readiness_status=PENDING_LICENSE
```
