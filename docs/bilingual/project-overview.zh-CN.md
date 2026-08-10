# BVP Solver 项目概览

[Русская версия](project-overview.ru.md) · [双语索引](README.md)

## 1. 项目定位

BVP Solver 是使用 Python、SciPy、SymPy、PyQt5 与 Matplotlib 开发的常微分方程边值问题教学与
数值实验软件。当前项目版本仍是 `v2.10`；Phase 12–14 的报告、资产整理与双语文档没有形成新的
数值软件能力，因此不创建 `v2.11`。

项目提供打靶法、参数延拓法、PyQt5 GUI、无窗口 `bvp_core` API、受限数学表达式、JSON 任务、
多结果绘图、结构化运行元数据及统一 JSON/TXT 导出。它不是生产级或安全关键计算工具。

## 2. 当前事实与历史快照

当前文档描述 `main` 的状态：307 项 pytest、Windows/Ubuntu CI、用户示例位于 `examples/tasks/`，
负向 fixture 位于 `tests/fixtures/tasks/`。历史 v2.10 软件快照仍绑定 `2ac1a20`，冻结报告仍记录
该时期的 285 项测试。当前事实不得回写历史报告，历史数字也不能替代当前审计。

## 3. 数学对象

对一阶系统

\[
\dot{x}=f(t,x),\qquad x(t)\in\mathbb{R}^n,
\]

用未知初始参数 `p∈R^k` 补全初值，并定义

\[
\Phi(p)=R\!\left(x(t_0;p),x(T;p)\right).
\]

目标是求 `Phi(p)=0`。实现要求未知参数数量与边界残差数量相等；维数匹配仍不保证有解、唯一、
非奇异或对初始猜测不敏感。

## 4. 可信成功的边界

最终成功不是 SciPy 优化器状态的别名，而是四类证据的合取：

```text
success = ivp_success AND finite_success AND boundary_success AND algorithm_success
```

候选参数必须经过独立的最终 IVP、有限性检查、原始边界残差重算和逐分量验收。表达式通过安全
解析也只证明语法属于受限语言，不证明 ODE 可积或 BVP 有解。

## 5. 工程能力

- `bvp_core` 不依赖 Qt 或 `main.py`，可无窗口导入和求解；
- `SolveRequest`、`SolveOutcome`、`SolveRecord` 隔离请求、终态和历史曲线；
- `run_id`、`request_id`、`problem_signature` 建立运行关联；
- `bvp-result-v1` 使 JSON/TXT 同源，并采用 UTF-8、严格 JSON 与原子替换；
- 解析解、制造解、负向输入、容差、性能计数和离屏 GUI 测试形成多层证据；
- CI 覆盖 Windows、Ubuntu 与 Ubuntu 精确依赖参考环境。

## 6. 当前限制

中俄版本共同声明以下限制：

1. 参数 Jacobian 使用前向有限差分；
2. 参数延拓采用固定步数，没有 pseudo-arclength；
3. 当前一次 `solve_ivp` 调用不能被立即取消；
4. 26.2–26.5 没有解析真值；
5. 26.1 不枚举全部解分支；
6. ODE defect 不是严格后验误差界；
7. GUI 对高级容差的编辑能力有限；
8. 项目仅用于教学与数值实验；
9. 不得用于安全关键用途；
10. 项目许可证尚未决定。

## 7. 语言与发布状态

文档支持 `zh-CN` 和 `ru-RU`。GUI 已有中文/俄语字典与切换入口，但审计发现仍有中文、俄文、
英文硬编码混用，故状态为“部分双语”，而不是完整 GUI i18n。本阶段没有修改 GUI。

许可证尚未决定，`release_readiness_status=PENDING_LICENSE`。这不是代码缺陷，但在许可证确定前
不创建正式 tag 或 Release，也不宣称他人已获得复制、修改或分发授权。
