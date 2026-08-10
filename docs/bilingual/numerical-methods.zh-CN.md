# BVP Solver 数值方法

[Русская версия](numerical-methods.ru.md) · [双语索引](README.md)

## 1. 边值问题参数化

考虑

\[
\dot{x}=f(t,x),\quad t\in[t_0,T],\quad p\in\mathbb{R}^k,
\]

其中 `p` 表示未知初始状态分量。积分得到 `x(t;p)` 后，边界残差为

\[
\Phi(p)=R\!\left(x(t_0;p),x(T;p)\right),\qquad \Phi(p)=0.
\]

当前实现只接受 `len(Phi)=len(p)`。这是一项模型维数前置条件，不是存在性或唯一性定理。

## 2. 打靶法

真实执行链为：

```text
p → IVP → boundary residual → root → least_squares fallback
  → candidate → independent final IVP → final boundary acceptance
```

`scipy.optimize.root(method="hybr")` 首先尝试求根；未形成可接受候选时可进入
`scipy.optimize.least_squares`。无论优化器如何结束，候选都要重新积分并按原始边界条件验收。

## 3. 参数延拓法

参数延拓从初始猜测处容易满足的残差构造到原始残差的同伦，以固定步数推进。每一步采用阻尼
Newton；参数 Jacobian 用前向有限差分，线性系统必要时用最小二乘。延拓步、Newton 迭代、
有限差分扰动或 IVP 的失败均向上传播，不能被包装为成功。

当前没有自适应延拓步长、pseudo-arclength 或解析/自动微分 Jacobian。固定步延拓可能在奇点、
转折点或刚性区域失败，也不保证发现所有分支。

## 4. 最终成功判据

```text
success = ivp_success AND finite_success AND boundary_success AND algorithm_success
optimizer_success != mathematical BVP success
```

`ivp_success` 要求最终独立积分到达终点；`finite_success` 要求参数、时间、状态和残差有限且形状
正确；`boundary_success` 要求逐分量边界验收通过；`algorithm_success` 要求打靶或延拓没有失败。
优化器 success 只保留为诊断字段。

## 5. 容差语义

| 字段 | 作用 |
|---|---|
| `ivp_rtol`, `ivp_atol` | `solve_ivp` 的相对/绝对局部误差控制 |
| `root_tol` | `root` 终止控制 |
| `least_squares_ftol`, `least_squares_xtol`, `least_squares_gtol` | fallback 的函数、参数和梯度终止条件 |
| `continuation_residual_tol` | 延拓 Newton 残差阈值 |
| `jacobian_relative_step` | 参数前向有限差分相对步长 |
| `boundary_atol`, `boundary_rtol`, `boundary_scales` | 最终边界验收 |

第 `i` 个分量必须满足

\[
|\Phi_i|\leq \texttt{boundary\_atol}
+\texttt{boundary\_rtol}\,s_i.
\]

`boundary_scales` 不改变优化器看到的原始残差。兼容字段 `eps` 可映射旧配置；显式字段只覆盖
自身过程。收紧某一容差不保证观测误差或函数调用次数严格单调变化。

## 6. 有限差分与性能事实

Phase 10 证明 `_dPhi_dp` 的变分矩阵积分未参与最终 Jacobian，随后删除该冗余工作，并在同一
Newton 步复用 `Phi_current`。前向差分公式与扰动方向没有改变，Jacobian 对照最大差异为 0。

```text
26.1 continuation: IVP calls 874 → 600
total nfev:         131732 → 94200
```

这些是确定性调用计数证据。历史机器上的耗时改善不是跨机器保证，CI 也不使用固定毫秒或固定
百分比作为门槛。

## 7. 多解、无解与奇异性

同一 BVP 可因初始猜测和数值路径得到不同有效分支。26.1 已验证 shooting 与 continuation 的
两个分支，但不保证枚举全部解。`x(T)^2+1=0` 的无实根问题必须失败；`x'=1/(t-0.5)` 在
`[0,1]` 上的奇异积分必须传播 IVP 失败；未知数与边界残差不等的过定输入必须在求解前拒绝。

## 8. 方法适用边界

当前方法适合教学、算法观察与受控数值实验。前向差分可能受尺度与舍入影响，固定步延拓可能
错过分支，单次 `solve_ivp` 不能立即取消。对没有解析真值的 26.2–26.5，只能组合边界、轨迹、
有限性、重复性和物理/问题特定指标，不能宣称获得数学证明。
