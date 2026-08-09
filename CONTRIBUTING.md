# Contributing

本项目当前正式验证 Python 3.11。提交修改前，请从仓库根目录创建独立虚拟环境并安装开发依赖：

```powershell
py -3.11 -m venv .venv-dev
.\.venv-dev\Scripts\python.exe -m pip install --upgrade pip
.\.venv-dev\Scripts\python.exe -m pip install -r requirements-dev.txt
```

Linux 或 macOS 可使用：

```bash
python3.11 -m venv .venv-dev
.venv-dev/bin/python -m pip install --upgrade pip
.venv-dev/bin/python -m pip install -r requirements-dev.txt
```

Windows Git Bash 使用：

```bash
python -m venv .venv-dev
.venv-dev/Scripts/python.exe -m pip install --upgrade pip
.venv-dev/Scripts/python.exe -m pip install -r requirements-dev.txt
```

运行完整测试和与 CI 相同的确定性验证：

```bash
python -m pytest -q
python scripts/run_ci_validation.py
```

建议从最新阶段基线创建用途单一的分支，提交信息说明行为变化和验证证据。不要把无关的数值算法、GUI 和文档重构混入同一个提交。

数值修改必须附带解析解、制造解或可信的交叉验证；不能只为了让测试通过而无依据放宽误差阈值。表达式语言或策略变更必须增加允许/拒绝安全测试。GUI 请求、线程、绘图或关闭行为变更必须增加离屏生命周期测试。

不要提交虚拟环境、缓存、临时导出、真实秘密、本机绝对路径或编辑器状态。提交前运行：

```bash
python scripts/check_repository_hygiene.py
git diff --check
```

本仓库尚未选择开源许可证；在仓库所有者明确决定前，贡献不应假定某种许可证已经适用。
