# Versioned Technical Reports

本目录是 BVP 项目的报告版本选择器。项目会继续迭代；每份报告只描述其绑定软件版本当时的
设计、实现、测试证据和已知限制。历史报告正文默认冻结，不会因后续代码更新而覆盖。

报告的永久绑定以 Git commit 为核心。Phase 分支只是开发过程的辅助信息；未来即使删除已经
合并的 Phase 分支，commit 或 tag 仍可定位对应代码快照。当前仓库尚未创建任何版本 tag，
因此下表的 Tag 均为空。

不存在 Final Report 或 Latest Final Report。项目未来仍可能继续产生新的 Version Report。

## Version selector

| Version | Period | Main milestone | Report | Code snapshot |
|---|---|---|---|---|
| v1.0 | 2026 | 基础打靶法、参数延拓、PyQt5 GUI、SymPy 解析 | No report snapshot archived | Commit not preserved |
| v2.0 | 2026-05 | 集成绘图、双语界面、JSON 任务库、辅助输出 | [Historical course report](v2.0/original-course-report.pdf) | [`a4ef249`](https://github.com/wuy503797-collab/bvp/tree/a4ef249dc72fef071ff0cf5b6abe650909298832) |
| v2.1 | 2026-08 | 修复错误成功判据与失败传播 | No report snapshot archived | [`b9cb298`](https://github.com/wuy503797-collab/bvp/tree/b9cb298f6b3dc132d39cc99c6a6bd61342f664a0) |
| v2.2 | 2026-08 | 解析解、制造解和数值正确性验证 | No report snapshot archived | [`fdcc672`](https://github.com/wuy503797-collab/bvp/tree/fdcc6723453dff194af066ffa3455c3ec7f5a493) |
| v2.3 | 2026-08 | 稳定的无 GUI Core API 与结果模型 | No report snapshot archived | [`525fac1`](https://github.com/wuy503797-collab/bvp/tree/525fac1f22e7850d113bda532935bf1ab6737c2a) |
| v2.4 | 2026-08 | 不可变 GUI 请求快照与线程生命周期 | No report snapshot archived | [`eeee178`](https://github.com/wuy503797-collab/bvp/tree/eeee178fed219bd97b0fa8062018231c0b74d3d6) |
| v2.5 | 2026-08 | 解析器和求解器迁入 `bvp_core` | No report snapshot archived | [`5f480c3`](https://github.com/wuy503797-collab/bvp/tree/5f480c3ab8e3feab77e585a044f8747404dbc931) |
| v2.6 | 2026-08 | 受限数学表达式语言与安全边界 | No report snapshot archived | [`e732b4e`](https://github.com/wuy503797-collab/bvp/tree/e732b4efb0a97bd348d41bd86c42c796f3b72ace) |
| v2.7 | 2026-08 | 求解容差拆分与尺度化边界验收 | No report snapshot archived | [`b11dc1a`](https://github.com/wuy503797-collab/bvp/tree/b11dc1accc357364d40d461df9a848896ccff2e7) |
| v2.8 | 2026-08 | 结构化运行元数据与统一原子导出 | No report snapshot archived | [`2b1c6d7`](https://github.com/wuy503797-collab/bvp/tree/2b1c6d74e14bc7bfe80653f8a5ded805594dcc61) |
| v2.9 | 2026-08 | 性能计数、基准和无效计算清理 | No report snapshot archived | [`7cea5d3`](https://github.com/wuy503797-collab/bvp/tree/7cea5d38f5fe25a169824654968dfc895b4f7b97) |
| v2.10 | 2026-08 | 干净环境、跨平台 CI、GUI 回归与发布准备 | No report snapshot archived | [`2ac1a20`](https://github.com/wuy503797-collab/bvp/tree/2ac1a20bf337bffd816582d1670af096c9f87dc0) |

完整的版本、提交、内部 Phase 和分支保留策略见
[version-history.md](../evolution/version-history.md)。机器可读映射见 [manifest.json](manifest.json)。

## Historical report policy

- 报告正文、截图、测试数量、架构描述、算法描述和历史限制默认冻结。
- 后续发现的错误应在新版本报告或映射说明中记录，不回写旧报告正文。
- 没有归档报告的版本只记录代码快照；不会创建空 PDF 或虚假报告占位。
- 教材节选和其他参考资料不是项目技术报告，不进入本目录。
- 当前仓库没有 LICENSE；归档报告不构成对复制、修改或分发项目代码的许可。

## v2.0 mapping note

`v2.0` 课程报告的 PDF 元数据日期为 2026-05-25，封面与正文描述集成绘图、双语界面、
JSON 任务库、辅助输出以及当时位于 `main.py` 的 `Dataset`、`SymPyParser`、`BVPSolver`、
`SolverWorker` 和 `IntegratedPlotWidget`。这些事实与同日的最早完整 Git 快照 `a4ef249` 及其
`CHANGELOG` 中的 `v2.0` 记录一致，因此采用该 commit 作为高置信度代码绑定。

`v1.0` 只存在于最早 `CHANGELOG` 的历史说明中；仓库没有保留一个可与 `v2.0` 分离的
`v1.0` commit，所以该字段保持未知，不能把 `a4ef249` 同时伪装成两个版本的快照。
