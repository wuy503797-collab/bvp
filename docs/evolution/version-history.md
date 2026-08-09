# BVP Version History

本文件建立 Project Version ↔ Git Commit ↔ Internal Phase ↔ Report 的长期映射。审计基线为
`main` 的 merge commit `1413ba43b4f919f4bfe21e0a1159e2fafb0dab7f`。该提交的第二父提交是
Phase 11 原始提交 `2ac1a20bf337bffd816582d1670af096c9f87dc0`，因此没有发生 squash 或 rebase。

```text
phase_history_preserved=true
```

当前没有 Git tag。下文的 tag 仅为建议，不代表已执行。

## Version ↔ Commit ↔ Phase

| Internal phase | Project version | Commit | Milestone | Report |
|---|---|---|---|---|
| Pre-Phase history | v1.0 | `UNKNOWN` | 基础打靶法、参数延拓、PyQt5 GUI、SymPy 解析 | 未归档；独立代码快照未保留 |
| Course / initial tracked snapshot | v2.0 | `a4ef249dc72fef071ff0cf5b6abe650909298832` | 集成绘图、双语界面、JSON 任务库、辅助输出 | [Historical course report](../reports/v2.0/original-course-report.pdf) |
| Phase 1 | Version not formally assigned (between v2.0 and v2.1) | `1f62eacab8528aa59d3bad7165354094351ed38f` | 自动化基线与严格已知缺陷测试 | 无 |
| Phase 2 | v2.1 | `b9cb298f6b3dc132d39cc99c6a6bd61342f664a0` | 统一结果验收；拒绝错误成功并传播 IVP/延拓失败 | 无 |
| Phase 3 | v2.2 | `fdcc6723453dff194af066ffa3455c3ec7f5a493` | 解析解、制造解、容差敏感性和数值一致性验证 | 无 |
| Phase 4 | v2.3 | `525fac1f22e7850d113bda532935bf1ab6737c2a` | 稳定无 GUI Core API、问题与结果模型 | 无 |
| Phase 5 | v2.4 | `eeee178fed219bd97b0fa8062018231c0b74d3d6` | 不可变 GUI 请求快照、结果溯源和线程生命周期 | 无 |
| Phase 6 | v2.5 | `5f480c3ab8e3feab77e585a044f8747404dbc931` | 解析器和求解器迁入 `bvp_core`，消除对 `main.py` 的反向依赖 | 无 |
| Phase 7 | v2.6 | `e732b4efb0a97bd348d41bd86c42c796f3b72ace` | 受限表达式语言、安全策略和错误边界 | 无 |
| Phase 8 | v2.7 | `b11dc1accc357364d40d461df9a848896ccff2e7` | 显式求解容差和尺度化边界验收 | 无 |
| Phase 9 | v2.8 | `2b1c6d74e14bc7bfe80653f8a5ded805594dcc61` | 结构化运行元数据、统一导出契约和原子写入 | 无 |
| Phase 10 | v2.9 | `7cea5d38f5fe25a169824654968dfc895b4f7b97` | 性能计数、可复现基准和冗余计算清理 | 无 |
| Phase 11 | v2.10 | `2ac1a20bf337bffd816582d1670af096c9f87dc0` | 干净环境、跨平台 CI、GUI 回归和发布准备 | 无 |

`v1.0` 和 `v2.0` 同时出现在最早提交的 `CHANGELOG` 中，但该 Git 历史没有保留一个早于
`a4ef249`、可单独代表 `v1.0` 的代码提交。因此 `v1.0` 必须保持 `git_commit=null`；不能为了
表格完整而把 `a4ef249` 同时绑定给 `v1.0` 和 `v2.0`。

Phase 1 是有价值的工程基线，但 `CHANGELOG` 没有为它分配新的 Project Version。它保持
“Version not formally assigned”，而不是被虚构为 `v2.0.1` 或把 Phase 编号当成版本号。

## Evolution

```text
v1.0 (commit not preserved)
  ↓
v2.0 course / integrated GUI snapshot
  ↓
Phase 1 unversioned automated baseline
  ↓
v2.1–v2.2 numerical correctness and validation
  ↓
v2.3–v2.5 core API, GUI lifecycle, and solver migration
  ↓
v2.6–v2.7 expression security and tolerance semantics
  ↓
v2.8–v2.9 observability, export, and verified performance
  ↓
v2.10 clean environments, CI, and release readiness
```

## Phase 11 integration status

```text
phase_eleven_local_status=PASS
remote_ci_status=PASS
release_readiness_status=PENDING_LICENSE
phase_11_commit=2ac1a20bf337bffd816582d1670af096c9f87dc0
main_merge_commit=1413ba43b4f919f4bfe21e0a1159e2fafb0dab7f
main_ci_run=https://github.com/wuy503797-collab/bvp/actions/runs/31310177928
```

`main` 与 `origin/main` 均指向上述 merge commit；Windows / Python 3.11、Ubuntu / Python 3.11
和 Ubuntu 精确直接依赖参考环境均已成功。仓库仍没有 LICENSE，因此这不是正式 release，
也不表示他人已获得复制、修改或分发项目的许可。

## Recommended report snapshots

以下版本发生了适合独立报告的重大变化。除已经归档的 `v2.0` 外，本次没有生成报告正文。

```text
recommended_report_versions:
- v2.0   # 已有课程报告：初始集成 GUI 快照
- v2.2   # 汇总 v2.1–v2.2：数值正确性与验证体系
- v2.5   # 汇总 v2.3–v2.5：Core API、GUI 生命周期与求解器迁移
- v2.7   # 汇总 v2.6–v2.7：表达式安全与容差语义
- v2.9   # 汇总 v2.8–v2.9：可观测性、导出与性能证据
- v2.10  # 干净环境、跨平台 CI 与发布准备
```

## Recommended tags (not executed)

这些 tag 只绑定已经有明确 commit 的正式版本。`v1.0` 在找到原始代码快照前不应创建 tag。

| Recommended tag | Commit |
|---|---|
| `v2.0` | `a4ef249dc72fef071ff0cf5b6abe650909298832` |
| `v2.1` | `b9cb298f6b3dc132d39cc99c6a6bd61342f664a0` |
| `v2.2` | `fdcc6723453dff194af066ffa3455c3ec7f5a493` |
| `v2.3` | `525fac1f22e7850d113bda532935bf1ab6737c2a` |
| `v2.4` | `eeee178fed219bd97b0fa8062018231c0b74d3d6` |
| `v2.5` | `5f480c3ab8e3feab77e585a044f8747404dbc931` |
| `v2.6` | `e732b4efb0a97bd348d41bd86c42c796f3b72ace` |
| `v2.7` | `b11dc1accc357364d40d461df9a848896ccff2e7` |
| `v2.8` | `2b1c6d74e14bc7bfe80653f8a5ded805594dcc61` |
| `v2.9` | `7cea5d38f5fe25a169824654968dfc895b4f7b97` |
| `v2.10` | `2ac1a20bf337bffd816582d1670af096c9f87dc0` |

本次没有执行 `git tag`、`git push --tags` 或 GitHub Release。

## Phase branch retention

审计时以下 Phase 分支均存在于本地和 `origin`，且其提交全部可从 `main` 到达。

| Branch | Commit | Merged into main | Version mapped | Current classification | Recommendation |
|---|---|---|---|---|---|
| `iteration/phase-1-baseline` | `1f62eac` | yes | no formal version | historical | 在决定是否为基线建立专用 tag 前保留 |
| `iteration/phase-2-correctness` | `b9cb298` | yes | v2.1 | merged | `safe_to_delete_after_tagging` |
| `iteration/phase-3-numerical-validation` | `fdcc672` | yes | v2.2 | merged | `safe_to_delete_after_tagging` |
| `iteration/phase-4-core-api` | `525fac1` | yes | v2.3 | merged | `safe_to_delete_after_tagging` |
| `iteration/phase-5-gui-lifecycle` | `eeee178` | yes | v2.4 | merged | `safe_to_delete_after_tagging` |
| `iteration/phase-6-core-solver-migration` | `5f480c3` | yes | v2.5 | merged | `safe_to_delete_after_tagging` |
| `iteration/phase-7-expression-security` | `e732b4e` | yes | v2.6 | merged | `safe_to_delete_after_tagging` |
| `iteration/phase-8-tolerance-separation` | `b11dc1a` | yes | v2.7 | merged | `safe_to_delete_after_tagging` |
| `iteration/phase-9-observability-exports` | `2b1c6d7` | yes | v2.8 | merged | `safe_to_delete_after_tagging` |
| `iteration/phase-10-performance` | `7cea5d3` | yes | v2.9 | merged | `safe_to_delete_after_tagging` |
| `iteration/phase-11-ci-release-readiness` | `2ac1a20` | yes | v2.10 | merged | `safe_to_delete_after_tagging` |

“Safe after tagging”不是删除授权。至少应先提交并审核本映射、创建并验证对应 tag、确认 tag 和
报告可从干净克隆访问，再由仓库所有者另行决定是否删除分支。本次没有删除任何分支。

名为 `agents/...` 的本地工作树引用不是 Project Version 或 Phase 发布分支，不纳入版本映射，
也不在本次任务中修改。
