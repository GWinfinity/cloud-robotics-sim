# EFE 迁移登记册（data/efe/）

对 [Embodied First Exam](https://arxiv.org/abs/2611.12177)（arXiv:2611.12177，
上游 [mll-lab-nu/embodied-first-exam](https://github.com/mll-lab-nu/embodied-first-exam)，MIT）
全部 **35 套件 / 47 机器人 / 1,156 任务 / 2,288 测试** 的机器可读登记册，
是迁移计划的单一事实来源。梳理与路线见 [`docs/EFE_MIGRATION_PLAN.md`](../../docs/EFE_MIGRATION_PLAN.md)。

## 布局

```
suites/<suite>.yaml   策划记录（人工维护）：efe.* = 论文 Table 2 事实；
                      project.* = 迁移映射（tier/status/target/note）
tasks/<suite>.json    任务清单（生成物）：上游 task.toml 的解析结果
                      （任务名/模式/oracle/预算/机器人），供对账与移植取用
upstream/tasks/...    上游任务目录镜像（手动拉取）：instruction.md /
                      environment/instance.json / tests 判分器——编译输入
compile/              编译层（生成物 + 策划）：<suite>.predicates.yaml（判据表，
                      人工策划）、<suite>.yaml（解析后的知识库）、
                      <suite>.specs.md（判据句摘编，--extract-specs 产物）
robots.yaml           47+4 台机器人资产来源对账（have/compose/menagerie/
                      public/efe_custom/blocked/out_of_paper）
embodiment_gate.yaml  9 个请求套件的本体门决策（migrate/partial/exclude；v2 起
                      quadruped 入 allowed_kinds，Playground 13 任务全迁）
exclusions.yaml       永久排除清单：MyoSuite 整套（Playground 四足原排除，
                      2026-11-20 用户指令覆盖后已全部迁移）
registry.json         合并视图（生成物）：totals/by_tier/by_status/warnings
```

## 编译流水线（P1 起）

```bash
uv run python tools/fetch_upstream_assets.py          # 手动：镜像上游任务目录与模型资产（幂等）
uv run python tools/compile_efe_suite.py --suite metaworldplus --extract-specs
# 人工把 <suite>.specs.md 的判据句转写进 <suite>.predicates.yaml 后：
uv run python tools/compile_efe_suite.py --suite metaworldplus --resolve
uv run python tools/compile_efe_suite.py --suite metaworldplus            # 生成配置
uv run python tools/compile_efe_suite.py --suite metaworldplus --check    # CI 门
```

产物：`configs/tasks/efe/<suite>/*.yaml` + `data/suites/efe_<suite>.yaml`，
由 `tests/efe/test_compiled_configs.py` 做 nop-fail 契约校验（无需仿真器）。

## 门禁（与 embodiments manifest 同一模式）

```bash
uv run python tools/build_efe_registry.py --check
uv run python tools/compile_efe_suite.py --suite metaworldplus --check   # 已编译套件
uv run python tools/check_efe_execution_scope.py --check                 # 排除边界
```

失败条件：结构错误（未知 tier/status/kind、机器人无来源登记）、策划 YAML
与上游任务清单漂移（任务数/机器人集合）、论文总数漂移（35/47/1156/2288/456）。
上游测试目录数多于论文（如 robodojo）只记 warning——以论文为基准定义。
本体门不通过或部分不通过的任务只保留在 `exclusions.yaml` 与论文登记中，
执行范围门禁禁止它们出现在任务配置、套件汇总或知识库可执行区。

## 再生成

```bash
# 1) 手动准备上游快照（稀疏检出即可，只需 task.toml；CI 不做这步）
git clone --depth 1 --filter=blob:none --no-checkout \
  https://github.com/mll-lab-nu/embodied-first-exam /path/to/efe-snapshot
cd /path/to/efe-snapshot
git sparse-checkout init --no-cone
git sparse-checkout set "tasks/**/task.toml" "tasks/**/README.md"
git checkout

# 2) 重新生成任务清单 + 登记册并过门
uv run python tools/fetch_efe_inventory.py --snapshot /path/to/efe-snapshot
uv run python tools/build_efe_registry.py --check
```

刷新上游快照后必须人工复核 diff（上游会持续演进）并同步更新
`suites/*.yaml` 的论文事实块。

## 许可与出处

任务清单与策划事实派生自上游 MIT 仓库的元数据（task.toml/README），
按 MIT 许可使用；上游各基准自身的场景/对象资产保留各自许可，移植时逐
套件核实（见迁移计划 §10）。
