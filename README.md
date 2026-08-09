# BVP Solver — 通用常微分方程边值问题求解器 / Универсальный решатель краевых задач

[![CI](https://github.com/wuy503797-collab/bvp/actions/workflows/ci.yml/badge.svg?branch=main)](https://github.com/wuy503797-collab/bvp/actions/workflows/ci.yml)

这是一个使用 Python、SciPy、SymPy、PyQt5 和 Matplotlib 开发的通用一阶常微分方程系统边值问题（Boundary Value Problem，BVP）教学与实验型求解器。程序提供打靶法和参数延拓法、PyQt5 GUI、无窗口 Python API、JSON 任务加载、交互绘图和原子结果导出。

项目仍处于早期迭代阶段。当前已建立统一的结果验收、26.1 双体问题基线，以及若干解析解和制造解验证。现有证据只覆盖文中列出的具体问题，因此本项目仍不应被视为可用于生产或高风险计算的可信数值软件发布版。

## 核心功能与正确性边界

- 不可变 `BVPProblem`、`SolverConfig`、`BVPResult` 以及无 Qt 副作用的核心 API；
- 打靶、root→least-squares fallback 和固定流程的参数延拓；
- 受限 AST-to-SymPy 数学表达式语言，不执行任意 Python/SymPy 代码；
- 分离的 IVP、非线性求解、Jacobian 和最终尺度化边界验收容差；
- GUI 请求快照、结果溯源、协作式取消和离屏生命周期测试；
- 结构化日志、运行元数据、性能计数、统一 JSON/TXT schema 和原子写入。

通过表达式安全检查、优化器收敛或某一组基线都不等价于证明任意 BVP 有解或数值结果可信。最终结果仍必须通过 IVP、有限性、算法状态和逐分量边界验收。

## 正式验证环境

当前正式验证基线为 Python 3.11。第十一阶段在独立于仓库现有 `venv/` 的范围环境和精确直接依赖参考环境中复验；GitHub Actions 工作流配置为在 `windows-latest` 与 `ubuntu-latest` 上运行 Python 3.11。其他 Python 版本尚不属于正式支持声明。

| 组件 | 版本 |
| --- | --- |
| Python | 3.11.9 |
| NumPy | 2.4.4 |
| SciPy | 1.17.1 |
| SymPy | 1.14.0 |
| Matplotlib | 3.10.9 |
| PyQt5 | 5.15.11 |
| pytest | 9.0.3 |

`requirements.txt` 只声明项目直接运行依赖的兼容范围；`requirements-dev.txt` 在此基础上增加 pytest。`requirements-lock.txt` 固定当前 Python 3.11 参考环境中的直接依赖和 pytest，但传递依赖仍由 pip 解析，因此它不是带哈希的完整传递依赖锁。范围约束不代表其中每一种可能组合都经过 CI。

## 快速开始与干净环境安装

不要把仓库已有的 `venv/` 当作复现证据。PowerShell 中可创建新的 Python 3.11 环境：

```powershell
py -3.11 -m venv .venv-clean
.\.venv-clean\Scripts\python.exe -m pip install --upgrade pip
.\.venv-clean\Scripts\python.exe -m pip install -r requirements-dev.txt
.\.venv-clean\Scripts\python.exe -m pytest -q
```

Linux 或 macOS 中可使用：

```bash
python3.11 -m venv .venv-clean
.venv-clean/bin/python -m pip install --upgrade pip
.venv-clean/bin/python -m pip install -r requirements-dev.txt
.venv-clean/bin/python -m pytest -q
```

Windows Git Bash 使用虚拟环境的 `Scripts` 目录：

```bash
python -m venv .venv-clean
.venv-clean/Scripts/python.exe -m pip install --upgrade pip
.venv-clean/Scripts/python.exe -m pip install -r requirements-dev.txt
.venv-clean/Scripts/python.exe -m pytest -q
```

只运行 GUI 或核心 API 时安装直接运行依赖：

```bash
python -m pip install -r requirements.txt
```

开发和测试依赖：

```bash
python -m pip install -r requirements-dev.txt
```

如需安装当前精确直接依赖参考版本：

```bash
python -m pip install -r requirements-lock.txt
```

虚拟环境目录 `venv/`、`.venv/` 或 `.venv-*/` 属于本地环境，不应提交到仓库。安装后建议运行 `python -m pip check`。

## 启动 GUI

```bash
python main.py
```

程序启动后通过“加载”按钮选择仓库中的任务 JSON，例如 `task1.json` 或 `tasks.json`。

### 后台求解与结果来源

用户点击“求解”时，GUI 会创建带唯一 `request_id` 的不可变 `SolveRequest`，其中保存
当时的问题、求解配置、任务身份、变量名、初值角色和辅助表达式。求解开始后切换、
编辑、重排或删除任务都不会改变这份请求。每个成功结果以 `SolveRecord` 保存并永久
绑定自己的请求快照；失败和取消保留请求及诊断，但不会进入成功历史。

绘图和导出使用结果自身的变量名与问题元数据，不再读取完成时的当前编辑器。每个成功
记录还会在完成时保存只读的主曲线采样，因此后续编辑输入或修改 SciPy dense-output 对象
都不能改变旧曲线。

绘图默认以最新成功记录为基准。同一稳定任务身份下，只要状态维数和变量含义不变，修改
初始猜测、方程数值、边界目标、已知初值、积分区间、辅助量、求解方法、IVP 方法、容差或
延拓步数后得到的结果都可同时保留并默认勾选。不同任务仍按问题快照隔离，避免把无关题目
自动混入；不兼容记录会明确显示原因。解列表显示请求/问题签名、`guess`、方法和容差，完整
`problem_signature` 仍区分各次请求，导出和运行关联没有被弱化。

“取消”采用协作式检查点，不会强制终止线程。取消请求会在解析、求解阶段边界、打靶
fallback、延拓步骤/Newton 迭代和最终验收附近被检查；正在执行的单次 SciPy
`solve_ivp` 调用不能从外部立即中断，因此可能要等它返回后才在最近检查点响应。关闭
窗口时会先请求取消并有限等待线程安全退出；若等待超时，窗口保持打开并保留诊断，
不会调用 `QThread.terminate()`。

GUI worker 继续消费不可变 `SolveRequest`，但数值调用现在统一经过
`bvp_core.solve_bvp_problem()`。`main.py` 只保留 Dataset/JSON 兼容、Qt worker、窗口、
绘图和导出职责，不再包含表达式解析器或数值求解算法的真实定义。

## 无 GUI 核心 API

脚本和测试可通过 `bvp_core` 使用稳定的输入、配置和结果模型，无需创建
`QApplication`、窗口、图形或输出文件：

```python
from bvp_core import BVPProblem, SolverConfig, solve_bvp_problem

problem = BVPProblem(
    name="Scalar exponential",
    odes=["x"],
    var_names=["x"],
    boundary_conditions=["x0_T - E"],
    known_indices=[],
    unknown_indices=[0],
    known_values={},
    initial_guess=[0.5],
    t_start=0.0,
    t_end=1.0,
)
config = SolverConfig(
    method="shooting",
    ivp_method="RK45",
    eps=1e-8,
    boundary_atol=1e-8,
    boundary_rtol=0.0,
)
result = solve_bvp_problem(problem, config)

if result.success:
    print(result.p_opt, result.boundary_residual_norm)
else:
    print(result.status, result.message)
```

`BVPProblem` 只描述数学问题，`SolverConfig` 只描述当前实现实际使用的
求解控制。两者以及 `BVPResult` 均不会被求解过程原地修改；结果数组是只读副本，
旧字典消费者可显式调用 `result.to_dict()` 获得防御性副本。问题定义或配置错误会在
数值求解前抛出 `BVPValidationError`；预期内的数值失败则返回
`BVPResult(success=False)`，并保留状态、IVP 诊断和最终边界残差。

导入或实际调用 `bvp_core` 均不会导入 `main.py`、PyQt5、窗口或绘图后端。核心实现的
职责分布如下：

- `bvp_core.expressions`：现有 SymPy 表达式解析、ODE/Jacobian、边界条件和辅助表达式；
- `bvp_core.exceptions`：带 SciPy IVP 状态、消息、终止位置和候选参数的失败类型；
- `bvp_core.solver`：打靶、root→least_squares fallback、参数延拓和最终数值验收；
- `bvp_core.api`：将公共模型送入核心求解器并返回不可变 `BVPResult`。

`Dataset` 继续作为 GUI 和旧 JSON 的兼容输入层，由 `bvp_core.adapters` 转换为
`BVPProblem + SolverConfig`；它不是核心求解器的事实来源。`main.py` 仍兼容导出
`SymPyParser`、`IVPIntegrationError` 和可用旧 Dataset 构造方式调用的 `BVPSolver`
入口，但数值算法只有 `bvp_core.solver.BVPSolver` 一份真实实现。

外部表达式现在必须通过下述受限数学语言边界；这不是“安全执行任意 SymPy/Python
表达式”的承诺。取消仍采用协作式检查点；单次正在执行的 `solve_ivp` 不能被立即中断。

## 容差配置

核心 API 将不同数值过程的容差分开处理：

- `ivp_rtol`、`ivp_atol` 传给 `solve_ivp`，控制内部自适应积分的局部误差估计；
- `root_tol` 传给 `scipy.optimize.root`，控制根求解终止；
- `least_squares_ftol`、`least_squares_xtol`、`least_squares_gtol` 分别传给
  `scipy.optimize.least_squares` 的对应终止条件；
- `continuation_residual_tol` 是参数延拓各 Newton 子问题的残差收敛门槛；
- `jacobian_relative_step` 控制现有参数有限差分 Jacobian 的相对扰动基值；
- `boundary_atol`、`boundary_rtol` 和 `boundary_scales` 只控制最终边界验收，不改变
  原始边界残差，也不作为优化权重。

为兼容旧 Python 调用和任务 JSON，`eps` 暂时保留。只提供 `eps` 时，有效映射为：

```text
ivp_rtol                    = eps
ivp_atol                    = eps / 10
root_tol                    = eps
least_squares_ftol          = eps
least_squares_xtol          = eps
least_squares_gtol          = eps
continuation_residual_tol   = eps
jacobian_relative_step      = sqrt(eps)
```

显式字段只覆盖自己的过程，其他缺省字段仍从 `eps` 推导。例如
`SolverConfig(eps=1e-6, ivp_rtol=1e-8)` 不会改变 `ivp_atol=1e-7` 或
`root_tol=1e-6`。若 `eps=None`，上述八个过程字段必须全部显式给出。所有容差必须为
正有限数；`boundary_rtol` 可为零。有效值、逐字段来源和整体
`tolerance_mode=legacy|explicit` 会写入结果元数据。GUI 的基础“精度”输入暂时仍创建
legacy 模式；高级显式配置通过 JSON 或 Python API 提供。

全显式配置示例：

```python
config = SolverConfig(
    method="shooting",
    eps=None,
    ivp_rtol=1e-8,
    ivp_atol=1e-9,
    root_tol=1e-8,
    least_squares_ftol=1e-8,
    least_squares_xtol=1e-8,
    least_squares_gtol=1e-8,
    continuation_residual_tol=1e-8,
    jacobian_relative_step=1e-4,
    boundary_atol=1e-8,
    boundary_rtol=0.0,
)
```

最终边界验收逐分量计算：

```text
threshold_i = boundary_atol + boundary_rtol * boundary_scales_i
component_success_i = abs(residual_i) <= threshold_i
boundary_success = all(component_success_i)
```

`boundary_scales` 未提供时每个分量取 `1.0`；提供时长度必须等于边界条件数量，且每项
为正有限数。程序不会从残差当前值、优化器 cost 或状态幅值自动推断尺度。结果继续保留
`boundary_residual_norm`，并新增尺度、阈值、逐分量状态、尺度化比值及其最大值。优化器
成功仍不等于 BVP 成功，算法、IVP、有限性和最终边界验收必须同时有效。

输出或验证用的固定采样点不是 `solve_ivp` 的内部自适应网格。收紧某一容差也不保证
观测误差或函数评估次数严格单调变化；它们还会受到其他误差源、算法分支和浮点平台影响。
可运行独立的 legacy/explicit 等价性、过程独立性、尺度化验收和 JSON 兼容检查：

```bash
python scripts/run_tolerance_validation.py
```

## 运行可观测性与统一导出

每次真正执行求解都会创建一个新的 UUID `run_id`，即使数学问题和配置完全相同也不会
复用。GUI 的 `request_id` 标识一次不可变用户请求；它与 `run_id` 不同，未来同一请求若
重试，可以关联新的运行。`problem_signature` 是规范化问题定义 JSON 的 SHA-256，跨
进程稳定，不使用 Python `hash()`，也不包含路径、时间或 Qt 对象。

公共 API 调用方式保持不变：

```python
result = solve_bvp_problem(problem, config)
print(result.run_metadata.run_id)
print(result.run_metadata.elapsed_seconds)
```

`RunMetadata` 在成功和数值失败结果中均为不可变对象，记录 UTC 开始/结束时间、由
`time.perf_counter()` 测得的耗时、最终状态、实际求解方法、有效容差、Python 与
NumPy/SciPy/SymPy 版本，以及真实可取得的 IVP、root、least-squares 和延拓诊断。
取消和没有 `BVPResult` 的编程异常通过 `SolveOutcome.run_metadata` 保留同样的运行关联。
一次机器上的单次耗时只是诊断事实，不是性能承诺或基准结果。

核心包使用标准库 `logging` 的 `bvp_core.events` logger，并仅安装 `NullHandler`：导入或
调用 `bvp_core` 不会调用 `logging.basicConfig()`、创建日志文件或主动输出到
stdout/stderr。应用可以显式安装自己的 handler；每条记录的 `solver_event` 属性包含
结构化事件字典：

```python
import logging

handler = logging.StreamHandler()
logging.getLogger("bvp_core.events").addHandler(handler)
logging.getLogger("bvp_core.events").setLevel(logging.INFO)
```

INFO 记录一次求解的关键阶段，WARNING 记录可恢复异常、辅助输出失败和过期/重复 GUI
信号，ERROR 只用于导出或内部编程错误；逐延拓步骤属于 DEBUG，不会在默认 INFO 刷屏。
事件细节会截断长字符串，并脱敏 Path、opaque 对象和已知本机路径；默认日志不包含环境
变量、用户目录、完整解释器路径、Qt repr、原始 SciPy 对象或完整恶意表达式。完整技术
traceback 只保留在 worker 内部诊断和 DEBUG 技术日志中，普通 GUI 只显示简化消息。

结果导出统一先构造不可变 `CanonicalExportRecord`，其 schema 为独立于应用版本的：

```text
bvp-result-v1
```

记录包含产生结果的 `SolveRequest` 问题与配置快照、显式和有效容差、运行元数据、结果、
IVP/优化器/算法/边界诊断、来源信息及辅助输出。导出历史结果不会重新读取当前 GUI
编辑器，因此任务被编辑或删除后，原结果的数学内容仍保持不变。完整问题表达式属于用户
主动定义的快照，可以进入导出；这不同于错误日志中受限的表达式预览。

JSON 是机器可读事实来源，使用 UTF-8、`ensure_ascii=False`、`allow_nan=False` 和稳定
字段顺序；非有限失败诊断显式写为 `{"non_finite": ...}`，不会输出非标准 NaN/Infinity
或伪装成 0。TXT 从同一个规范记录生成，是包含 schema、运行关联、配置、有效容差、
关键诊断、数值表和辅助输出的人类可读摘要，不承诺可无损重新导入或执行。

JSON、TXT 以及 GUI 任务库保存均采用同目标目录临时文件、flush、fsync 和 `os.replace`
原子替换；写入或序列化失败时已有目标文件保持不变，临时文件会尽可能清理。GUI 仍只
导出成功历史；失败和取消可通过核心规范记录 API 生成明确的
`failure_diagnostic`/`cancelled` 诊断，绝不会伪装为成功结果。

可单独运行第九阶段的结构化事件、运行关联、统一导出、原子写入和隐私边界验收：

```bash
python scripts/run_observability_validation.py
```

## 性能计数与本机基准

求解器为每次运行生成不可变 `performance_counters`，分别统计项目层 `Phi`、有限差分
Jacobian、普通/最终验收 IVP、SciPy IVP `nfev`、root 与 least-squares 残差调用、延拓
步、Newton 检查与更新以及阻尼试算。`RunMetadata.elapsed_seconds` 和
`api_elapsed_seconds` 都表示公共 API 总时间；`core_elapsed_seconds` 只覆盖表达式已经准备
完成后的数值求解与最终验收。GUI 线程调度、辅助输出、绘图和界面更新不属于这两个核心
基准。输出的固定采样点也不是 `solve_ivp` 的内部自适应网格。

第十阶段确认旧 `_dPhi_dp` 每次都积分状态变分矩阵 `XT`，但最终 Jacobian 完全由前向
有限差分生成，`XT` 没有参与返回值。现已删除这段冗余变分积分；这不代表实现了解析
Jacobian。当前参数 Jacobian 仍使用原来的前向扰动方向和步长。延拓还会把同一 Newton
迭代刚计算的 `Phi_current` 作为显式局部基准传入，参数快照不一致时拒绝复用，不建立跨
Newton 或跨运行缓存。固定延拓步数、Newton 上限和原有阻尼策略均未改变。

最终验收仍独立执行一次 dense-output IVP 并重新计算边界残差。虽然某些路径可能刚计算过
相同参数，这一步承担独立正确性验收，当前没有足够证据安全复用。每个公共 API 求解对
ODE 的解析、lambdify 和边界编译各执行一次；没有建立跨任务全局表达式缓存。结果数组
继续在 `BVPResult` 边界防御性复制并设为只读，未采用可能破坏不可变性的零复制技巧。
默认日志级别未启用时，求解器会在构造结构化事件前短路；关键 INFO 事件和显式启用的
DEBUG 事件保持原语义。

在本仓库当前 Windows AMD64、Python 3.11.9、NumPy 2.4.4、SciPy 1.17.1、SymPy
1.14.0 环境的一次同进程对比中，26.1 参数延拓的确定性统计为：

```text
                         修改前      修改后
IVP 调用                    874         600
总 IVP nfev              131732       94200
Phi 评估                    736         599
Jacobian 评估               137         137
冗余变分 IVP                 137           0
核心中位耗时（3 次）       0.956 s     0.617 s
```

时间只代表这一次机器与依赖组合，不是跨机器性能承诺，也不进入 pytest 的固定毫秒门槛。
性能回归以调用次数不增加、数值数组等价和失败语义保持为主要判据。可运行预热后轻量案例
5 次、26.1 延拓 3 次的同进程旧工作量参考/当前实现对比：

```bash
python scripts/run_performance_validation.py
```

## 自动化测试

```bash
python -m pytest -q
```

当前测试集合为 285 项，默认运行目标为 0 warnings。此前唯一警告来自全局启用 Matplotlib
实验性 `toolmanager`；绘图组件实际使用稳定的 `NavigationToolbar2QT`，因此已删除该全局
设置，没有增加 warning ignore 或改变绘图入口。

测试分为：

- 正向数值基线：确认当前 26.1 双体问题仍能得到有限结果和合格的终端边界残差；
- 正确性回归：确认无解问题被拒绝、IVP 失败被传播、输入维数被提前验证；
- 核心 API：确认模型不可变、结果数组防御性复制、无窗口导入、失败语义和旧接口兼容；
- 核心迁移契约：确认表达式数值行为、求解器字典字段、唯一实现和实际求解无 main/Qt；
- GUI 生命周期：确认请求快照、结果溯源、过期/重复信号、协作式取消和安全关闭；
- 数值验证：用解析解和制造解检查未知参数、全部状态、边界条件和 ODE 积分缺陷。

单独运行离屏 GUI 生命周期测试：

```bash
python -m pytest -v tests/gui
```

单独运行核心 API 测试：

```bash
python -m pytest -v tests/core
```

单独运行数值验证测试：

```bash
python -m pytest -v tests/numerical
```

## 持续集成与发布前验证

统一验证入口会输出 Python/依赖版本，并依次执行 `pip check`、无警告导入、`compileall`、
完整 pytest、数值基线、解析/制造解、表达式安全、容差、可观测性/导出、确定性性能契约和
仓库卫生检查：

```bash
python scripts/run_ci_validation.py
```

`.github/workflows/ci.yml` 在 push 和 pull request 上运行。范围依赖 job 覆盖
`windows-latest` 与 `ubuntu-latest` 的 Python 3.11；精确直接依赖参考 job 另在 Ubuntu
安装 `requirements-lock.txt`。GUI 测试不被跳过，Qt 使用 `QT_QPA_PLATFORM=offscreen`。
工作流不使用 `continue-on-error`，也不把固定耗时或“至少提升某百分比”作为性能门槛。
CI 性能模式只验证数值等价、Jacobian、Phi 复用、变分 IVP 为零、计数器一致和元数据导出。

当前第十一阶段状态严格区分本地证据、远端证据和发布许可：

```text
phase_eleven_local_status=PASS
release_readiness_status=PENDING_REMOTE_CI_AND_LICENSE
```

本地环境、两个独立干净环境、285 项测试和全部验证脚本已经通过。以上结论不等于
GitHub Actions 已通过；只有该提交推送后，Windows、Ubuntu 范围依赖和 Ubuntu 精确依赖
三个远端任务实际成功，才能记录远端 CI 通过。仓库尚无 `LICENSE`，这不是代码缺陷，
但在许可证确定前不创建正式 release 或版本 tag，也不宣称他人已经获得复制、修改或分发许可。

本地完整性能脚本仍保留预热、重复中位数和 profile；CI 调用轻量确定性模式：

```bash
python scripts/run_performance_validation.py --ci
```

仓库卫生检查为只读操作，不会自动删除文件。它检查被跟踪的环境/缓存/临时文件、有限的
疑似秘密模式、本机绝对路径、README 相对链接、大小写和大文件，并分类已知旧文件：

```bash
python scripts/check_repository_hygiene.py
```

当前课程与项目报告 PDF 是用户有意保留但被 Git 忽略的项目材料；重复任务 JSON 和空
`gui.py` 留待独立清理决策，不在 CI 阶段删除。贡献要求见
[CONTRIBUTING.md](CONTRIBUTING.md)，安全边界和报告方式见 [SECURITY.md](SECURITY.md)。

仓库当前没有 `LICENSE`。这意味着许可证仍需仓库所有者在发布前决定；README 不据此
声称项目已经采用某种开源许可证，CI 也不会因缺少许可证失败。

## 表达式语言与安全边界

GUI 和 JSON 中的 ODE、边界条件、辅助输出以及 GUI 初值常量都先经过同一个受限解析
入口。输入先以 Python AST 的 `eval` 语法模式生成纯语法树（不会调用 Python `eval`），
完整通过白名单与复杂度检查后，才由逐节点转换器直接构造 SymPy 对象。未知名称不会
自动变成 Symbol 或 Function，也没有第二条 `parse_expr`/`sympify` 外部字符串入口。

支持的基础语法为十进制整数、小数、科学计数法、括号、二元 `+ - * / **`、一元
`+ -` 和显式函数调用。不支持 `^` 代替幂，也不支持隐式乘法。函数白名单为：
`sin`、`cos`、`tan`、`asin`、`acos`、`atan`、`atan2`、`sinh`、`cosh`、`tanh`、
`exp`、`log`、`sqrt`、`abs`、`Abs`、`sign`。常量仅为区分大小写的 `pi` 和 `E`；
`e`、`inf`、`nan`、`oo`、`zoo` 不在白名单中。
幂指数必须是数值常量或常数分数（例如 `2`、`3/2`），不接受变量指数。

符号权限按上下文隔离：

- ODE：`t` 和当前任务的状态变量；
- 边界条件：现有索引别名 `x0_0 ... xN_0` 与 `x0_T ... xN_T`；
- 辅助输出：`t` 和产生该结果的请求快照中的状态变量；
- GUI 初值常量：数学函数和常量，不允许任务状态符号。

状态变量和辅助输出名称必须由 ASCII 字母开头，后续只能是 ASCII 字母、数字或单下划线。
空名、重复名、Python 关键字、下划线开头、包含双下划线，以及与时间、边界别名、函数或
常量冲突的名称都会被拒绝。

输入保护上限为：512 个字符、256 个 AST 节点、32 层嵌套、单个数值字面量 32 位数字、
32 次函数调用、16 次幂运算，且常量指数绝对值不超过 64。这些值是资源保护限制，不是
数学理论限制。属性、下标、字符串/字节串、容器、Lambda、推导式、生成器、条件表达式、
赋值表达式、任意函数调用和导入路径均在 SymPy 转换前拒绝。

通过安全解析只说明表达式属于该语言，不说明 BVP 一定有解，也不保证表达式在整个积分
区间都有限。例如 `1 / (t - 0.5)` 语法合法，但仍会由既有 IVP 数值失败机制处理。单个
辅助表达式的安全或数值失败仍只写入该结果的 `auxiliary_errors`，不会丢弃已验收的主解。

可单独运行表达式安全与任务库兼容验收：

```bash
python scripts/run_expression_security.py
```

## 终端基线脚本

```bash
python scripts/run_baseline.py
```

脚本默认只向终端输出，不创建 JSON、TXT、图片或项目缓存。输出包括环境版本、初始猜测、求解参数、IVP 状态、有限性、独立边界残差、运行时间和已知缺陷状态。

脚本不会因为求解器返回 `success=True` 就直接判定数值结果有效；最终是否接受由独立边界残差检查决定。

## 数值验证

运行可重复的解析解、制造解和容差敏感性报告：

```bash
python scripts/run_numerical_validation.py
```

验证使用 SciPy dense output 在 201 个固定验证点上采样。这些点是独立的验证网格，不是 `solve_ivp` 的内部自适应步节点。

当前验证问题：

- 标量指数问题：`x'=x`，`x(1)=e`，精确解 `x(t)=exp(t)`，精确未知初值 `x(0)=1`；
- 简谐振子：`x'=v, v'=-x`，`x(0)=0, x(pi/2)=1`，精确解为 `sin(t), cos(t)`，精确未知初值 `v(0)=1`；
- 制造解：`y'=v, v'=6t`，精确解 `y=t^3-2t+1, v=3t^2-2`，精确未知初值 `v(0)=-2`。

在 Python 3.11.9、SciPy 1.17.1 和 `eps=1e-8` 下的一次实测结果：

| 问题 | 方法 | 参数误差 | 最大状态误差 | RMS 状态误差 | 边界残差范数 | 最大 ODE 缺陷 |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| 指数 | 打靶 | 2.02e-9 | 5.28e-9 | 1.78e-9 | 4.44e-16 | 1.16e-9 |
| 指数 | 参数延拓 | 2.02e-9 | 5.28e-9 | 1.78e-9 | 3.78e-13 | 1.16e-9 |
| 简谐振子 | 打靶 | 2.39e-9 | 3.25e-9 | 1.49e-9 | 4.44e-16 | 6.22e-10 |
| 制造多项式 | 打靶 | 0 | 2.22e-15 | 5.07e-16 | 2.91e-16 | 1.73e-15 |

状态误差使用数值状态与精确状态在固定验证点上的差。ODE 缺陷使用相邻验证点上的 Simpson 积分估计：

```text
D_i = y(t_{i+1}) - y(t_i)
      - h_i/6 * [f_i + 4 f(t_mid, y(t_mid)) + f_{i+1}]
```

报告同时给出最大分量缺陷、RMS 缺陷和单位区间长度最大缺陷。该指标不复制边界残差或求解器 `success` 字段，而是重新调用原始 ODE 构造数值一致性检查；但其中点状态仍来自同一个 `solve_ivp` dense output，因此它是独立构造的数值一致性指标，不是完全独立于求解器的外部验证。

指数问题的容差敏感性实测：

| eps | 参数误差 | 最大状态误差 | RMS 状态误差 | 边界残差范数 | 最大 ODE 缺陷 |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 1e-4 | 4.41e-6 | 9.97e-6 | 4.56e-6 | 1.02e-14 | 4.74e-7 |
| 1e-6 | 1.56e-7 | 2.82e-7 | 1.24e-7 | 4.44e-16 | 3.28e-8 |
| 1e-8 | 2.02e-9 | 5.28e-9 | 1.78e-9 | 4.44e-16 | 1.16e-9 |

这张历史表使用 legacy 模式，所以 `eps` 按上述兼容映射同时影响 IVP、非线性求解、
延拓残差和 Jacobian 扰动。它只是一项容差敏感性实验，不是内部网格收敛实验，也不用于
拟合理论收敛阶；浮点舍入和 dense output 误差会形成误差平台，更严格的容差也不保证
误差严格单调下降。

## 当前正向数值基线

### 26.1 独立打靶实现

当前一次实测结果：

```text
初始猜测                    [-0.5, 0.5]
采样点数量                  200
求得初始速度                约 [0.0, 0.50000001]
边界残差范数                3.47e-11
独立验收阈值                1e-8
```

### 26.1 主程序参数延拓实现

当前一次实测结果：

```text
初始猜测                    [0.5, -0.5]
求得初始速度                约 [0.45107812, -0.29941864]
边界残差范数                1.02e-11
独立验收阈值                1e-8
```

两组不同初始速度都满足同一终端位置，形成当前多解分支的正向基线。测试不对运行时间或初始速度的过多小数位作断言。

当前只对上述 26.1 路径建立了正向自动基线，不应据此声称所有 JSON 任务均已验证正确。

## 第二阶段已关闭缺陷

- `BVP-P0-001`：无实根打靶问题现在返回 `success=False` 并保留最终残差；
- `BVP-P0-002`：参数延拓失败不再无条件返回成功；
- `BVP-P0-003`：GUI 不再接受无效结果作为成功历史、绘图或导出数据；
- `BVP-P1-001`：边界条件与未知参数维数在调用 SciPy 前校验；
- `BVP-P1-002`：奇异 IVP 返回结构化失败和底层诊断。

当前测试中不再保留这些编号的 `xfail`。数值验证通过并不证明未测试的任务或所有 BVP 都正确。

## 项目架构

```text
bvp_core/                    无窗口问题模型、表达式、求解、验收、元数据与导出
main.py                      Dataset/旧 JSON 兼容、Qt worker、主窗口和绘图
validation_metrics.py        解析解/制造解采样与独立构造的数值一致性指标
scripts/run_*.py             可执行阶段验证入口
scripts/check_repository_hygiene.py
tests/core/                  核心 API、模型、安全、容差、性能和导出契约
tests/gui/                   Qt 离屏请求、线程、关闭、绘图与警告回归
tests/numerical/             解析解、制造解和容差敏感性
.github/workflows/ci.yml     Python 3.11 Windows/Linux 持续集成
```

`solver.py` 是仍被 26.1 独立基线使用的兼容实现；它不是 `bvp_core` 求解器的第二份公共
事实来源。`task_error_test.json` 属于负向测试数据，其余任务 JSON 是示例和兼容性夹具。

## 已知限制与开发计划

- 当前正式验证只覆盖 Python 3.11、文档列出的解析/制造问题和 26.1 基线；
- 当前 Jacobian 是参数前向有限差分，不是解析或自动微分 Jacobian；
- 参数延拓使用固定流程，不能保证跨越所有奇点、转折点或找到全部分支；
- 协作式取消不能中断正在执行的单次 SciPy `solve_ivp`；
- 精确参考文件没有锁定或哈希全部传递依赖；
- GitHub 托管的 Windows/Linux 状态以真实 CI badge 和工作流记录为准；
- 正式公开发布前仍需由仓库所有者决定许可证，并另行判断空 `gui.py` 和重复示例数据是否清理。

后续迭代应优先依据 CI 和数值验证发现的问题推进，不因平台波动随意放宽既有数学门槛。

## 版本化报告 / Versioned Reports

本项目持续迭代，并保留不同软件版本对应的历史技术报告。每份报告只描述其绑定代码快照
当时的实现、证据和限制，不会被后续版本覆盖。

- [查看版本化报告](docs/reports/README.md)
- [查看 Version ↔ Commit ↔ Phase 演进](docs/evolution/version-history.md)

## 文档说明

仓库中的旧项目报告用于了解对应历史版本。历史报告正文默认冻结；其依赖版本、数值算法、
测试证据和已知限制不得用当前状态回填。了解当前开发状态时，应以当前源码、自动化测试和
可复现运行结果为准。本项目不存在“最终报告”，未来仍可能继续产生新的版本报告。

## 作者

Ли Хунюй, 313 группа, ВМК МГУ, кафедра ОУ, 2026.

教师：Аввакумов С.Н.、Орлов М.С.
