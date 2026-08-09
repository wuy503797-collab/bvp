# BVP Solver v2.10 验证证据附录

## 证据边界

本附录记录为 v2.10 报告重新取得的摘要证据，不复制完整终端日志。软件快照绑定
`2ac1a20`，报告阶段测试日期为 2026-08-09。计时仅代表当次机器与负载，确定性计数和数值
结果才是性能验收依据。

PowerShell 会话没有继承 VS Code/Git Bash 的项目 Python PATH，因此验证使用用户已确认可用的
项目虚拟环境。pytest 的最终计数使用仓库外专用 `--basetemp`，避免 Windows 默认临时目录权限
以及仓库卫生自检扫描自身 fixture 的干扰；没有改变产品代码、测试或通过门槛。

## 验证矩阵

| 命令 | 目的 | 关键证据 | 状态与限制 |
|---|---|---|---|
| `python -m pytest -q` | 全量自动化回归 | `285 passed in 14.13s`，0 warnings | PASS；使用仓库外 basetemp |
| `python scripts/run_baseline.py` | 正向/负向求解基线 | 26.1 两方法通过；无实根两方法拒绝 | PASS |
| `python scripts/run_numerical_validation.py` | 解析解、制造解、defect、重复性与容差敏感性 | 最大状态误差从 `2.22e-15` 到 `5.29e-09` 量级 | PASS；defect 不是严格误差界 |
| `python scripts/run_expression_security.py` | 允许/拒绝表达式和现有任务兼容 | 允许集、攻击输入、全部任务表达式均符合预期 | PASS；安全解析不保证数值可解 |
| `python scripts/run_tolerance_validation.py` | legacy 映射、显式容差、尺度化验收 | 显式等价差 0；IVP、边界验收语义独立 | PASS |
| `python scripts/run_observability_validation.py` | 事件、关联、元数据、严格导出、隐私 | 成功/失败/取消信息与 JSON/TXT/原子写入通过 | PASS |
| `python scripts/run_performance_validation.py` | 删除工作量与等价性证据 | IVP `874→600`，nfev `131732→94200`，Jacobian 差 0 | PASS；计时不作为跨机器保证 |
| `python scripts/check_repository_hygiene.py` | 跟踪文件、秘密、路径、链接和大文件 | 相关检查计数均为 0；已知旧资产被分类 | PASS；7 张未跟踪截图使工作树非 clean，但不属于跟踪污染 |
| `python scripts/run_ci_validation.py` | 与 CI 同构的统一本地入口 | pip check、导入、compileall、285 tests、全部专项脚本 | PASS，约 79.7 s |

## 数值基线详情

### 26.1 shooting

```text
p_opt=[-2.380789650244e-08, 5.000000147112e-01]
boundary_residual=[-1.402100657799e-11, -3.174704943376e-11]
boundary_residual_norm=3.47053853632e-11
acceptance_threshold=1e-8
```

### 26.1 continuation

```text
p_opt=[0.451078122246, -0.299418638352]
boundary_residual=[-3.958611216603e-12, 9.391820654514e-12]
boundary_residual_norm=1.01920016665e-11
acceptance_threshold=1e-8
```

### 无实根

```text
shooting: success=False, status=boundary_residual_too_large, residual_norm=1
continuation: success=False, status=continuation_failed, residual_norm=1
```

### 奇异 IVP

```text
success=False
status=ivp_failed
ivp_status=-1
ivp_t_final=0.499999999999952
boundary_residual_norm=inf
```

## 数值正确性详情

| 问题 / 方法 | `p_opt` | 参数误差 | 最大状态误差 | RMS | 边界残差 | 最大 defect |
|---|---:|---:|---:|---:|---:|---:|
| 指数 / shooting | `0.999999997983` | `2.01665351085e-09` | `5.28216670403e-09` | `1.78454824072e-09` | `4.44089209850e-16` | `1.16312365989e-09` |
| 指数 / continuation | `0.999999997983` | `2.01679262180e-09` | `5.28181587356e-09` | `1.78447031157e-09` | `3.77919917582e-13` | `1.16312405020e-09` |
| 简谐振子 / shooting | `1.000000002393` | `2.39334307928e-09` | `3.24697235765e-09` | `1.48556439051e-09` | `4.44089209850e-16` | `6.22137192458e-10` |
| 制造三次多项式 / shooting | `-2` | `0` | `2.22044604925e-15` | `5.07278600450e-16` | `2.91433543964e-16` | `1.73472347598e-15` |

## 性能证据详情

当前重新验证的 26.1 continuation 对照：

```text
before_median_seconds≈1.5447018
after_median_seconds≈1.108176
observed_improvement≈28.3%
IVP calls=874→600
total nfev=131732→94200
Phi calls=736→599
Jacobian calls=137→137
variational IVP per Jacobian=1→0
Jacobian max difference=0
final boundary residual norm=1.01920016665e-11
```

Phase 10 的历史机器观测约为 `0.956 s→0.617 s`（约 35.5%）。两组计时环境不同；报告保留
两者而不挑选更好看的数字。CI 的 `--ci` 模式只验证确定性等价和计数契约。

## 远端 CI 证据

| 范围 | Commit | Run | 结论 |
|---|---|---|---|
| Phase 11 分支 | `2ac1a20` | [31305225622](https://github.com/wuy503797-collab/bvp/actions/runs/31305225622) | PASS，3 jobs |
| 合并后 `main` | `1413ba43` | [31310177928](https://github.com/wuy503797-collab/bvp/actions/runs/31310177928) | PASS，3 jobs |

三个 job 为 Windows/Python 3.11 范围依赖、Ubuntu/Python 3.11 范围依赖和 Ubuntu 精确直接
依赖参考环境。GUI 离屏测试未跳过，workflow 没有 `continue-on-error`。

## 仓库与版本不变量

```text
v2.10 commit=2ac1a20bf337bffd816582d1670af096c9f87dc0
main merge commit=1413ba43b4f919f4bfe21e0a1159e2fafb0dab7f
phase history preserved=true
remote_ci_status=PASS
license_status=license_decision_required
```

原始 v2.0 PDF 的 SHA-256 基准为
`5ed71322f97f78ec1a3b0a15e3d7d2af19a70da58a3ab26c9f57a2d209ece4f2`。报告回归阶段必须再次
确认该值未变。用户的 7 张根目录截图不属于本附录的自动处理范围。
