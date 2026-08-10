# BVP Solver 测试、CI 与仓库验证

[Русская версия](testing.ru.md) · [双语索引](README.md)

## 1. 当前测试基线

Phase 14 开始时当前 `main` 有 307 项 pytest，目标是 0 warnings。测试按 core、numerical、GUI、
integration 与仓库/发布准备分层。历史 v2.10 报告的 285 项是 `2ac1a20` 时期事实，不能当作
当前计数，也不能在历史翻译中改为 307。

## 2. 测试层次

- `tests/core/`：模型、API、求解迁移、表达式安全、容差、请求、元数据、性能和序列化；
- `tests/numerical/`：解析解、制造解、容差敏感性与验证指标；
- `tests/gui/`：离屏生命周期、警告策略、多记录曲线隔离与用户验收；
- `tests/integration/`：`bvp-result-v1` 导出契约；
- 顶层测试：基线、已知失败关闭、结果验收、任务资产布局和发布准备。

负向任务只放在 `tests/fixtures/tasks/`。用户教材示例位于 `examples/tasks/`，测试不能依赖根目录
遗留 JSON。

## 3. 本地 pytest

```bash
python -m pytest -q
```

GUI 测试设置 `QT_QPA_PLATFORM=offscreen`，不得因无显示器而跳过。测试失败必须保持失败；项目
不使用 `continue-on-error` 或全局 warning filter 把问题隐藏成通过。

## 4. 专项验证脚本

```bash
python scripts/run_baseline.py
python scripts/run_numerical_validation.py
python scripts/run_expression_security.py
python scripts/run_tolerance_validation.py
python scripts/run_observability_validation.py
python scripts/run_performance_validation.py --ci
```

基线验证 26.1 和失败传播；其余脚本分别覆盖解析/制造解、表达式策略、容差、元数据/导出以及
确定性性能契约。统一入口 `python scripts/run_ci_validation.py` 顺序执行依赖、导入、编译、pytest、
所有专项脚本与仓库卫生检查。

## 5. GitHub Actions 矩阵

`.github/workflows/ci.yml` 实际覆盖：

1. Windows / Python 3.11 范围依赖；
2. Ubuntu / Python 3.11 范围依赖；
3. Ubuntu / Python 3.11 精确直接依赖参考环境。

每个相关 job 运行 `pip check`、无警告导入、编译、完整 pytest、离屏 GUI 和验证脚本。远端
workflow 的真实 Success 才可称为“CI 已通过”；本地通过不能替代远端运行记录。

## 6. 双语文档检查

```bash
python scripts/check_bilingual_docs.py
```

检查器核对文档对、编号标题、相对路径、公式/代码块数量合理性、历史 SHA、README 语言入口、
双语索引、关键术语与未完成翻译占位符。它只做结构和事实一致性检查，不评价俄语语法质量。

## 7. 仓库卫生

```bash
python scripts/check_repository_hygiene.py
git diff --check
```

卫生检查读取 tracked/visible 路径，检查环境、缓存、临时物、疑似秘密、本机绝对路径、坏 README
链接、大小写、大文件和根目录旧任务资产。Phase 14 的 7 张用户截图保持未跟踪，不移动、不改名、
不纳入文档。`git diff --check` 检查空白错误。

## 8. CI 基础设施维护

GitHub Actions 关于 Node.js 20→24 的注释是低优先级、非阻塞基础设施维护债务，已登记在
[`docs/maintenance.md`](../maintenance.md)。当前状态为 `DEFERRED`，不影响现有 CI PASS；触发
既定条件后再在独立 Phase 升级官方 Action 主版本，不与数值算法或本地化混合。

## 9. 发布边界

仓库没有 `LICENSE`。因此测试和 CI 通过不等于法律发布准备完成，当前仍是
`release_readiness_status=PENDING_LICENSE`。本阶段不选择许可证、不创建 tag/Release、不升级
软件版本，也不开始课程答辩材料。
