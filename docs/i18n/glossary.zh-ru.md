# BVP Solver 中俄技术术语表

本表统一当前项目文档中的中文与俄文术语。代码标识符、文件路径、命令、schema 名称和 Git SHA
保持英文原样，不按自然语言翻译。

| 中文 | Русский | 英文/标识 | 使用说明 |
|---|---|---|---|
| 边值问题 | краевая задача | BVP, boundary value problem | 首次出现可写“краевая задача (BVP)” |
| 初值问题 | задача Коши | IVP, initial value problem | 不使用直译“начальная задача” |
| 打靶法 | метод стрельбы | shooting method | 算法名 |
| 参数延拓法 | метод продолжения по параметру | parameter continuation | 不简化为“продолжение”，除非上下文明确 |
| 边界残差 | невязка краевых условий | boundary residual | `Phi` 或 `boundary_residual` |
| 初始猜测 | начальное приближение | initial guess | 与已知初值区分 |
| 未知初始参数 | неизвестные начальные параметры | unknown initial parameters | 参数向量 `p` |
| 同伦 | гомотопия | homotopy | 固定步延拓构造 |
| 阻尼牛顿法 | демпфированный метод Ньютона | damped Newton method | continuation 子问题 |
| 前向有限差分 | прямая конечная разность | forward finite difference | 当前参数 Jacobian |
| 雅可比矩阵 | матрица Якоби | Jacobian | 文中可保留 `Jacobian` 标识 |
| 最小二乘回退 | резервный переход к least_squares | least-squares fallback | 保留 SciPy API 名称 |
| 边界验收 | проверка краевых условий | boundary acceptance | 最终逐分量判据 |
| 尺度 | масштаб компоненты | scale | `boundary_scales` |
| 容差 | допуск | tolerance | 不译成“精度”以免混淆 |
| 解析解 | аналитическое решение | analytical solution | 验证 oracle |
| 制造解 | сконструированное решение | manufactured solution | 不写成“人工解” |
| 数值一致性指标 | показатель численной согласованности | numerical consistency metric | ODE defect 的性质 |
| 后验误差界 | апостериорная оценка погрешности | a posteriori error bound | ODE defect 不是严格误差界 |
| 请求快照 | снимок запроса | request snapshot | 不可变 `SolveRequest` |
| 结果溯源 | происхождение результата | result provenance | 关联请求、问题与运行 |
| 运行元数据 | метаданные запуска | run metadata | `RunMetadata` |
| 性能计数器 | счётчики производительности | performance counters | `SolverCounters` |
| 结构化事件 | структурированное событие | structured event | 标准库 `logging` 记录 |
| 原子替换 | атомарная замена | atomic replacement | 临时文件 + `os.replace` |
| 受限表达式语言 | ограниченный язык математических выражений | restricted expression language | AST 白名单入口 |
| 负向测试 | негативный тест | negative test | 故意无效、奇异或不可解输入 |
| 离屏 GUI 测试 | тест GUI вне экрана | offscreen GUI test | `QT_QPA_PLATFORM=offscreen` |
| 固定步延拓 | продолжение с фиксированным числом шагов | fixed-step continuation | 当前实现无自适应步长 |
| 伪弧长延拓 | продолжение по псевдодуге | pseudo-arclength continuation | 当前未实现 |
| 协作式取消 | кооперативная отмена | cooperative cancellation | 不能即时打断单次 `solve_ivp` |
| 历史快照 | исторический снимок | historical snapshot | 不由后续版本反向改写 |
| 发布准备 | готовность к выпуску | release readiness | 当前受许可证决定阻塞 |

## 固定表达

- `optimizer success != mathematical BVP success`：успешное завершение оптимизатора не означает,
  что краевая задача прошла математическую проверку。
- `software snapshot`：снимок программного обеспечения。
- `report snapshot`：снимок отчёта。
- `license undecided`：лицензия проекта пока не определена。
