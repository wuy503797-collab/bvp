# BVP Solver v2.10 答辩提纲

## 答辩定位

主题：如何把一个能运行的课程 BVP 求解器，迭代成结果可验收、行为可追溯、跨平台可复现的
教学与数值实验软件。

核心论点：

```text
optimizer success != mathematical success
```

版本边界：本材料只描述 v2.10（commit `2ac1a20`），不把报告文档工作当成新软件版本。

## 建议演示结构（12–15 分钟）

### 1. 项目背景（1 分钟）

- 输入 ODE、边界条件、已知/未知初值和初始猜测；
- 以 shooting 或 continuation 求未知初始参数；
- 提供 PyQt5 GUI、JSON 任务和无 GUI Core API；
- 定位为教学与数值实验，不宣称生产级或安全关键适用性。

### 2. v2.0 课程原型（1 分钟）

- 已具备集成绘图、双语界面、任务库和辅助输出；
- 求解、界面和数据职责集中，自动化数值 oracle 未建立；
- 原始报告绑定 `a4ef249`，保持历史原貌。

### 3. 最严重的正确性缺陷（2 分钟）

用无实根 `x(T)^2+1=0` 解释：优化器可以结束，但不存在实数边界根。旧逻辑若只读优化器
标志就可能报告错误成功。

展示修复后的四门验收：

```text
success = ivp_success
       AND finite_success
       AND boundary_success
       AND algorithm_success
```

候选参数还要经过独立 `solve_ivp` 与原始边界残差复算。当前无实根残差范数为 1，明确返回
失败；奇异 IVP 在 `t≈0.5` 前停止，返回 `ivp_failed`。

### 4. 自动化测试与数值验证（2 分钟）

- 285 项测试，0 warnings；
- 指数、简谐振子解析解；
- 三次多项式制造解，最大状态误差 `2.22044604925e-15`；
- dense-output Simpson defect 是独立构造的一致性指标，不是严格误差界；
- 正向、无解、奇异、过定约束、多分支和容差敏感性一起构成证据链。

### 5. 架构与 API（1 分钟）

- `BVPProblem + SolverConfig → bvp_core API → BVPResult`；
- 核心不导入 `main`、PyQt5 或创建 `QApplication`；
- GUI/JSON 只在 adapter 边界转换；
- 结果、请求、运行信息和导出具有明确数据模型。

### 6. 表达式安全（1 分钟）

- `ast.parse(mode="eval")` 只生成语法树，不调用 Python `eval`；
- AST 白名单、函数/常量/符号白名单和复杂度限制；
- 逐节点直接构造 SymPy，生产路径无 `parse_expr`、`sympify`、`eval`、`exec`；
- 安全语法不保证数值可解，边界分层明确。

### 7. GUI 生命周期与绘图（1.5 分钟）

- 每次求解冻结 `SolveRequest`，用 `request_id` 和 `problem_signature` 溯源；
- stale/duplicate signal 防护，成功/失败/取消分离；
- 绘图读取记录的冻结曲线，不读取后来改变的输入或 dense solution；
- 同一任务不同初始猜测以及 ODE、边界、时间、方法、容差变化的曲线都能保持独立并共同显示；
- 取消为 cooperative cancellation，单次 `solve_ivp` 内不能即时抢占。

### 8. 容差、可观测性与导出（1.5 分钟）

- IVP、root、least-squares、continuation、Jacobian 和 boundary tolerance 拆分；
- `|Φ_i|≤atol+rtol·scale_i`，scale 只用于验收；
- `run_id / request_id / problem_signature` 关联日志、结果和导出；
- JSON/TXT 共享 `bvp-result-v1` 事实来源，UTF-8、严格 JSON、fsync + 原子替换。

### 9. 性能热点（1 分钟）

- 先用计数器证明 `_dPhi_dp` 的变分矩阵积分未参与 Jacobian；
- 删除冗余积分并复用 `Phi_current`，有限差分不变；
- 26.1 continuation：IVP `874→600`，nfev `131732→94200`，Jacobian 对照差 0；
- 历史机器约 35.5%，当前复测约 28.3%，不作跨机器承诺，CI 不以耗时过关。

### 10. 干净环境、CI 与限制（1 分钟）

- 本地与两个独立干净环境通过；
- Windows、Ubuntu、Ubuntu 精确依赖远端 jobs 均 PASS，GUI offscreen 未跳过；
- 当前限制：前向差分、固定延拓步、无 pseudo-arclength、取消非抢占、部分任务无真值；
- LICENSE 未定，`release_readiness_status=PENDING_LICENSE`。

## 建议现场演示顺序

1. 运行 26.1 shooting 与 continuation，展示两个满足边界的分支；
2. 同时勾选两个记录，强调曲线与请求快照绑定；
3. 运行无实根任务，展示失败而非“优化器成功”；
4. 打开 JSON 导出，定位 schema、request/run ID、容差和残差；
5. 展示 CI 三个平台/依赖 job 的成功状态。

截图只预留位置；根目录现有 7 张截图未自动移动或采用。

## 结尾要点

- 做数值软件时，“能画图”和“数学上可接受”不是同一件事；
- 测试不仅验证代码分支，也需要解析解、制造解、负向问题和一致性指标；
- GUI 的异步行为必须用不可变快照与来源标识管理；
- 性能优化先证明无效工作，再证明数值等价；
- v2.10 是一个稳定工程快照，项目仍持续迭代。
