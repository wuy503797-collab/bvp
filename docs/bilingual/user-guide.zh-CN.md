# BVP Solver 用户指南

[Русская версия](user-guide.ru.md) · [双语索引](README.md)

## 1. 安装

正式验证基线为 Python 3.11。建议创建独立环境并安装开发依赖：

```powershell
py -3.11 -m venv .venv-clean
.\.venv-clean\Scripts\python.exe -m pip install -r requirements-dev.txt
.\.venv-clean\Scripts\python.exe -m pip check
```

只运行 GUI/API 时可安装 `requirements.txt`；精确直接依赖参考环境使用
`requirements-lock.txt`。这些文件不构成带哈希的完整传递依赖锁。

## 2. 启动与选择语言

```bash
python main.py
```

GUI 当前默认俄语，并提供中文/俄语菜单切换。界面仍有少量中文、俄文、英文硬编码混用，因此
文档不会把它描述为完整本地化。本阶段没有改变界面语言实现。

## 3. 加载与保存任务

点击“加载/Загрузить”可打开 JSON。推荐从
[`examples/tasks/26_1_two_body.json`](../../examples/tasks/26_1_two_body.json) 或
[`examples/tasks/textbook_problems.json`](../../examples/tasks/textbook_problems.json) 开始。
正常教材任务位于 `examples/tasks/`；`tests/fixtures/tasks/` 是自动化测试故意使用的负向输入，
不是普通用户示例。默认保存名为 `bvp_task_collection.json`。

## 4. 编辑数学问题

设置系统维数、变量名、每个 ODE 右端、终端时间 `T`、已知初值、未知初值的猜测、边界条件、
IVP 方法和 shooting/continuation。边界别名使用 `x0_0`、`x0_T` 等索引形式；ODE 与辅助输出可
使用变量名。表达式只支持项目白名单语法，幂使用 `**`，不使用 `^` 或隐式乘法。

未知初值数量必须等于边界条件数量。`eps` 是兼容入口；完整高级容差主要通过 JSON 或 Python
API 设置，GUI 目前没有覆盖全部字段的编辑器。

## 5. 求解、取消与诊断

点击“求解/Решить”后输入被冻结成一个 `SolveRequest`。求解期间再编辑或切换任务不会改变已
启动请求。取消是协作式的：正在执行的单次 `solve_ivp` 可能先完成或失败，程序才在下一检查点
响应。失败时查看状态、IVP 诊断、边界残差和消息；优化器显示成功并不代表 BVP 已通过验收。

## 6. 多结果与绘图

每次成功结果都有独立快照。勾选多个兼容记录可共同显示，包括同一方程只改初始猜测得到的
不同 26.1 分支，也包括改变方程数值、边界目标、已知初值、终止时间、方法或容差后的结果。
旧曲线不会因当前编辑器或新 dense output 改变。维数、变量语义或时间坐标不兼容时，界面会
排除并给出诊断。

## 7. 导出结果

JSON 和 TXT 都从所选结果自己的请求快照导出，schema 为 `bvp-result-v1`。JSON 是严格 UTF-8
机器记录，TXT 是同源的人类可读摘要。写入采用临时文件和原子替换；导出历史结果不会读取
当前编辑器，也不会静默生成非标准 NaN/Infinity。

## 8. 无窗口 API

```python
from bvp_core import BVPProblem, SolverConfig, solve_bvp_problem

problem = BVPProblem(
    name="Scalar exponential",
    odes=["x"], var_names=["x"],
    boundary_conditions=["x0_T - E"],
    known_indices=[], unknown_indices=[0], known_values={},
    initial_guess=[0.5], t_start=0.0, t_end=1.0,
)
result = solve_bvp_problem(problem, SolverConfig(method="shooting", eps=1e-8))
print(result.success, result.p_opt, result.boundary_residual_norm)
```

导入 `bvp_core` 不需要 `QApplication`。定义错误抛出 `BVPValidationError`，预期数值失败返回
`BVPResult(success=False)`。

## 9. 使用边界

不要把某次成功当成唯一性证明，也不要认为改变初始猜测后必须得到同一分支。26.2–26.5 没有
解析真值；ODE defect 也不是严格后验误差界。软件用于教学和数值实验，不用于安全关键决策。
许可证尚未决定，仓库内容的使用授权不能从缺少 `LICENSE` 推断。
