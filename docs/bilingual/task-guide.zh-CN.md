# BVP Solver 教材任务指南

[Русская версия](task-guide.ru.md) · [双语索引](README.md)

本文逐项读取当前 `examples/tasks/*.json`。方程、边界条件、猜测和方法均以仓库文件为准，不从
旧报告或记忆回填。

## 1. 任务资产边界

`examples/tasks/` 存放可由 GUI/API 加载的用户示例：26.1–26.5 的独立文件，以及包含 26.1–26.4
的 `textbook_problems.json`。`tests/fixtures/tasks/` 只存放故意奇异或维数错误的负向输入，不能
作为普通示例推荐。任务资产迁移不改变方程数学。

## 2. 26.1 双体问题

文件：[`26_1_two_body.json`](../../examples/tasks/26_1_two_body.json)。状态为

\[
(x,y,v_x,v_y),\quad x'=v_x,\quad y'=v_y,
\]

\[
v_x'=-\frac{x}{r^3},\quad v_y'=-\frac{y}{r^3},\quad
r=\sqrt{x^2+y^2}.
\]

已知初值 `x(0)=2`、`y(0)=0`，终端约束为

\[
x(7)=1.0738644361,\qquad y(7)=-1.0995343576.
\]

未知量是 `vx(0), vy(0)`，恰有两个未知参数和两个边界残差。JSON 默认猜测 `[0.5,-0.5]`、
`eps=1e-8`、`RK45`、continuation 50 步。shooting 与 continuation 可落到不同但都满足边界的
轨道分支；当前证据不枚举全部分支，也不保证任意猜测都收敛。

## 3. 26.2 极限环

文件：[`26_2_limit_cycle.json`](../../examples/tasks/26_2_limit_cycle.json)。变量名为
`x1,x2,x3,x4`，方程为

\[
x_1'=x_3x_2,\quad x_2'=x_3(-x_1+\sin x_2),\quad x_3'=0,\quad x_4'=0.
\]

`x2(0)=0` 已知；`x1(0),x3(0),x4(0)` 未知。三个边界残差对应 `x1(0)=x4(0)`、
`x1(1)=x4(0)`、`x2(1)=0`。默认猜测 `[2,6.283185,2]`，continuation 100 步。该示例没有仓库内
解析真值，成功只能按数值验收与问题结构解释。

## 4. 26.3 三重积分器

文件：[`26_3_triple_integrator.json`](../../examples/tasks/26_3_triple_integrator.json)。六维系统
包含三重积分器状态与三个伴随量：

\[
x_0'=x_1,\quad x_1'=x_2,\quad
x_2'=\tfrac12\left(\sqrt{10^{-10}+(x_5+1)^2}-\sqrt{10^{-10}+(x_5-1)^2}\right),
\]

\[
x_3'=0,\quad x_4'=-x_3,\quad x_5'=-x_4.
\]

`x0(0)=1,x1(0)=0,x2(0)=0`，未知 `x3(0),x4(0),x5(0)`；在 `T=3.275` 要求
`x0(T)=x1(T)=x2(T)=0`。辅助输出 `u` 与 `x2'` 的平滑控制表达式相同。默认 continuation 50 步。

## 5. 26.4 月牙域时间最优示例

文件：[`26_4_lunula.json`](../../examples/tasks/26_4_lunula.json)。变量 `x0,x1` 是主状态，
`x2,x3` 是伴随类变量，常量状态 `x4` 缩放归一化区间上的动力学。右端包含
`sqrt(x2**2+x3**2)` 归一化与 `1e-6` 平滑符号近似，完整表达式以 JSON 为准。

已知 `x0(0)=4,x1(0)=1`；未知 `x2(0),x3(0),x4(0)`。在归一化 `T=1` 上要求

\[
x_0(T)=0,\quad x_1(T)=0,\quad x_2(T)^2+x_3(T)^2=1.
\]

默认猜测 `[-0.54,-0.13,4.9]`、`eps=1e-6`，使用 shooting。分母接近零与平滑尺度会影响数值
行为；本项目没有为该任务提供解析解。

## 6. 26.5 二维时间最优控制

文件：[`26_5_time_optimal.json`](../../examples/tasks/26_5_time_optimal.json)。状态为
`x,y,vx,vy,psi1,psi2,psi3,psi4,T`。JSON 的独立积分区间终点为 1，而常量状态 `T` 是未知实际
时间尺度，所有八个动力学方程都乘以它，且 `T'=0`。

已知 `x(0)=y(0)=vx(0)=vy(0)=0`；未知四个伴随初值和 `T(0)`。终端要求位置 `(1,1)`、速度
为零，并满足 JSON 中的横截条件

\[
T\left(\sqrt{\psi_3^2+\psi_4^2}-\psi_3-2\psi_4\right)-1=0.
\]

默认猜测 `[0,0,1,1,3]`、`eps=1e-7`、continuation 200 步。控制方向在方程中通过
`sqrt(1e-10+psi3**2+psi4**2)` 正则化。

## 7. 负向 fixture

[`singular_ivp.json`](../../tests/fixtures/tasks/singular_ivp.json) 使用
`x'=1/(t-0.5)` 穿过奇点，用于确认 IVP 失败传播。
[`overdetermined.json`](../../tests/fixtures/tasks/overdetermined.json) 只有一个未知初值却有两个
边界残差，用于确认 `unknown count != boundary residual count` 在数值求解前被拒绝。无实根
`x(T)^2+1=0` 由自动化测试构造，验证优化器结束不能制造 BVP 成功。

## 8. 修改任务时的检查

变量名、ODE 数量和初值映射必须同维；未知初值数量必须等于边界残差数量；表达式必须通过
受限 AST 语言。保存新用户示例到 `examples/tasks/`，只把故意无效的自动化输入放到
`tests/fixtures/tasks/`。修改数值后应把它视为新的求解请求，不应改写旧 `SolveRecord` 的曲线。

## 9. 证据解释

26.1 有两条已验证数值分支与明确终端残差证据，但没有全分支枚举。26.2–26.5 没有解析真值，
因此需要边界残差、有限性、ODE 一致性、重复性、方法敏感性与问题特定约束组合判断。任何单一
`success` 字段、漂亮图像或优化器消息都不能独自证明数值正确。
