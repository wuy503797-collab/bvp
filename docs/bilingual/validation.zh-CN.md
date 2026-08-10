# BVP Solver 数值验证体系

[Русская версия](validation.ru.md) · [双语索引](README.md)

## 1. 验证原则

求解器返回 `success=True`、优化器停止、边界残差很小和图像平滑分别是不同证据。验证体系组合
解析 oracle、制造解、独立最终 IVP、逐分量边界验收、ODE 数值一致性、负向输入、容差敏感性和
确定性性能计数。它只支持已运行的问题与阈值，不证明任意 BVP 正确。

## 2. 解析解

标量指数问题 `x'=x, x(1)=e` 的精确解是 `x(t)=exp(t)`，未知 `x(0)=1`；shooting 与
continuation 都比较参数、全轨迹和边界。简谐振子

\[
x'=v,\qquad v'=-x,qquad x(0)=0,\quad x(\pi/2)=1
\]

的真解是 `x=sin(t), v=cos(t)`，未知 `v(0)=1`。这些测试检测只看终点无法发现的轨迹错误。

## 3. 制造解

先指定

\[
y(t)=t^3-2t+1,\qquad v(t)=3t^2-2,
\]

再构造 `y'=v, v'=6t` 及匹配边界。精确未知参数为 `v(0)=-2`。制造解同时检查参数、状态、
边界和 ODE，不依赖教材任务恰好有闭式解。

## 4. 指标与 ODE defect

验证在 201 个固定点采样，报告参数误差、最大状态误差、RMS 状态误差、边界残差范数、最大
分量 defect、RMS defect 和单位区间最大 defect。相邻点上的 Simpson 型指标为

\[
D_i=y(t_{i+1})-y(t_i)-\frac{h_i}{6}
\left[f_i+4f(t_{i+1/2},y(t_{i+1/2}))+f_{i+1}\right].
\]

它不复用边界残差或求解器 `success` 字段，而是重新调用 ODE，因而是独立构造的数值一致性
指标。但中点状态仍来自同一个 `solve_ivp` dense output，所以它不是完全独立于积分器的外部
验证，也不是严格后验误差界。

## 5. 容差与边界验收

验证分别检查 `ivp_rtol/atol`、`root_tol`、least-squares 三个容差、
`continuation_residual_tol`、`jacobian_relative_step` 和边界验收字段没有被错误串联。逐分量条件为

\[
|\Phi_i|\leq a_{\mathrm{tol}}+r_{\mathrm{tol}}s_i.
\]

尺度只参与最终验收，不改变原始残差。legacy `eps` 与显式等价测试证明兼容映射；敏感性实验
观察误差趋势，但不要求浮点误差或调用次数严格单调。

## 6. 负向验证

- 无实根：`x(T)^2+1=0` 必须返回失败与残差，而不是接受优化器终止；
- 奇异 IVP：`x'=1/(t-0.5)` 在 `[0,1]` 的积分必须保留停止位置和 SciPy 诊断；
- 过定输入：一个未知参数配两个边界残差必须在 SciPy 前因维数不匹配而拒绝；
- 非有限候选、延拓失败和最终残差超限均不得进入成功历史、绘图或成功导出。

## 7. 26.1 与多分支

26.1 的 shooting 基线得到接近 `[0,0.50000001]` 的初始速度，continuation 得到接近
`[0.45107812,-0.29941864]` 的另一组初始速度；两者终端残差范数均小于 `1e-8`。这证明至少两条
已验证数值分支可以满足同一终端位置，不证明唯一性，也不证明已经枚举所有分支。

## 8. 性能一致性证据

删除无效变分矩阵积分前后，26.1 continuation 的 IVP 调用从 874 降为 600，`nfev` 从 131732
降为 94200，Jacobian 数组最大差异为 0。`Phi_current` 复用与冗余工作删除没有改变前向有限
差分公式。耗时只作为特定机器诊断，不是跨机器性能保证。

## 9. 可复现入口与限制

```bash
python scripts/run_numerical_validation.py
python scripts/run_tolerance_validation.py
python scripts/run_performance_validation.py --ci
```

这些入口不为 26.2–26.5 提供不存在的解析真值。当前验证仍受前向有限差分、固定步延拓、
dense output 共享、浮点平台和覆盖问题有限等边界约束。软件只适合教学与数值实验。
