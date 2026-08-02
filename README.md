# BVP Solver — 通用常微分方程边值问题求解器 / Универсальный решатель краевых задач

这是一个使用 Python、SciPy、SymPy、PyQt5 和 Matplotlib 开发的边值问题（Boundary Value Problem，BVP）求解与可视化项目。程序目前提供参数延拓法和打靶法，支持从 JSON 加载任务、交互绘图以及结果导出。

项目仍处于早期迭代阶段。当前已建立统一的结果验收、26.1 双体问题基线，以及若干解析解和制造解验证。现有证据只覆盖文中列出的具体问题，因此本项目仍不应被视为可用于生产或高风险计算的可信数值软件发布版。

## 已验证环境

当前宿主环境实际验证使用：

| 组件 | 版本 |
| --- | --- |
| Python | 3.11.9 |
| NumPy | 2.4.4 |
| SciPy | 1.17.1 |
| SymPy | 1.14.0 |
| Matplotlib | 3.10.9 |
| PyQt5 | 5.15.11 |
| pytest | 9.0.3 |

`requirements.txt` 中的范围是安装约束，不代表其中每一种版本组合都已经通过 CI 或全新机器验证。`requirements-lock.txt` 只记录当前已实际运行成功的精确组合，也尚未在全新机器上完成重建验证。

## 环境安装与激活

运行依赖：

```bash
pip install -r requirements.txt
```

开发和测试依赖：

```bash
pip install -r requirements-dev.txt
```

如需复现当前已验证的精确包版本：

```bash
pip install -r requirements-lock.txt
```

Git Bash 激活现有虚拟环境：

```bash
source venv/Scripts/activate
```

PowerShell 激活现有虚拟环境：

```powershell
.\venv\Scripts\Activate.ps1
```

虚拟环境目录 `venv/` 或 `.venv/` 属于本地环境，不应提交到仓库。

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

绘图和导出使用结果自身的变量名与问题元数据，不再读取完成时的当前编辑器。绘图默认
以最新成功记录为基准，只叠加状态维数、变量含义、方程、边界条件和时间区间均兼容的
历史记录；不兼容记录会被明确隔离并显示原因，而不是静默跳过。

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

解析器迁移只保证现有合法表达式、变量顺序、数组形状和主要错误行为保持不变，并不
代表已经完成针对任意不可信表达式的安全加固。取消仍采用协作式检查点；单次正在执行
的 `solve_ivp` 不能被立即中断。

## 自动化测试

```bash
python -m pytest -q
```

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

`eps` 同时影响 IVP 和非线性求解过程，所以这里只称为容差敏感性实验；它不是内部网格收敛实验，也不用于拟合理论收敛阶。浮点舍入和 dense output 误差会形成误差平台。

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

## 文档说明

仓库中的旧项目报告用于了解项目背景。报告描述、依赖版本和数值算法细节可能与当前代码不完全一致，应以当前源码、自动化测试和可复现运行结果为准。代码和验证体系稳定后再单独更新报告。

## 作者

Ли Хунюй, 313 группа, ВМК МГУ, кафедра ОУ, 2026.

教师：Аввакумов С.Н.、Орлов М.С.
