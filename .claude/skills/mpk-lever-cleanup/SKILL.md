---
name: mpk-lever-cleanup
description: >-
  当需要把一批 env-gated（`#ifdef MPK_DSV3_*` / `os.environ` 控制、default-OFF）的 MPK 优化 lever
  收敛成一个「干净的单一路径」版本供开 PR 时调用：把获胜的 lever 全部 hard-wire 成默认、删掉
  legacy 的 `#else` 分支、移除控制新/旧逻辑的 env 变量、revert 掉测得 KILL/NULL/regress 的死
  lever、删掉诊断 probe，然后 commit 一个干净版本。适用于「优化已定型、要合入主线」的收尾阶段。
  不适用于还在探索 lever（那时应保持 env-gated default-OFF）或改 runtime/execution-model 的场景。
tags: ["mpk", "cleanup", "refactor", "pr"]
related_skills: ["mpk-internals", "add-mpk-task"]
version: "1.0.0"
---

# MPK Lever Cleanup — 把 env-gated 优化收敛成干净单一路径

MPK 的性能优化在探索期都是 **env-gated、default-OFF** 的 lever（`#if MPK_DSV3_XXX` +
`#else` 旧路径 + persistent_kernel.py 里的 `os.environ.get(...)` → `-D` 注入）。定型后要
合入主线时，需要把它们收敛成**单一路径**：获胜的 hard-wire 成默认、删掉 legacy、移除控制
变量。这个 skill 是那次收尾 refactor 的**完整流程 + 踩过的坑**。

> 核心心态:这是一次**有意改变 default build** 的 refactor —— 默认路径从「安全的旧逻辑」
> 变成「优化路径」。所以平时的「default build 必须 byte-identical」这条 commit gate 在这里
> 是**被有意豁免**的,那正是这次 commit 的目的。

## 流程(7 步,按顺序)

### 1. 枚举所有 gate
```bash
grep -rhoE "MPK_(DSV3_)?[A-Z0-9_]+" <the megakernel .cuh files> <builder.py> <persistent_kernel.py> \
  | sort -u | grep -vE "MPK_(MAX|PAGE|PROFILING|NUM)"
```
把 `.cuh` 里的 `#ifdef` gate **和** persistent_kernel.py 里的 `os.environ`/`-D` 注入都列出来。

### 2. 分类每个 gate(必须逐个 VERIFY,别假设 default-OFF)
| 类别 | 处理 | 判定依据 |
|---|---|---|
| **WIN** | hard-wire ON:去 gate + 删 `#else` legacy + 移除 env 注入,**保留 geometry guard**(TP8/mbt/workers) | git log 里已 committed 的 lever + experiment_history 里 WIN 行 |
| **DEAD** | 完全 revert(删干净它的所有代码,**回滚它改过的 ABI**) | experiment_history 里 KILL/NULL/REGRESS 行 |
| **DIAGNOSTIC** | 删除(probe/no-op/poison/xor,不是 lever) | 名字带 PROBE/NOOP/POISON/XOR;只用于测量 |
| **ALREADY-ON** | 确认还在,保持无条件默认(别加 gate) | grep persistent_kernel.py 看它是否已无条件 `-D` 注入 |
| **LEAVE-UNTOUCHED** | 不动 | 非本路径的 fallback(如 TP<8 的 ROUTER_GEMV)、inert、或非 DSv3 的通用 flag |

**⚠️ 分类必须核验,不能凭记忆**:实测踩过的坑 —— 以为 default-OFF 要 revert 的
`TOPK_PARALLEL` 其实**每个 decode build 都是开的**(是当前 winning stack 的一部分,应
KEEP);以为 obsolete 的 `ROUTER_GEMV` 其实是 **TP<8 fallback 的 router**(不能删)。
用 `grep -n` 看 persistent_kernel.py 的注入条件 + 看它在生产 geometry 下有没有被调用。

### 3. Codex-vet 分类 + refactor 计划
把 gate 清单 + 分类 + 目标交给 Codex(`mcp__codex__codex`),多轮讨论:验证分类、确认安全
的执行顺序、correctness 风险、ABI-revert 的核验方法、structural-vs-leaf 的区分。Codex 会
抓出分类里的冲突(见第 2 步的坑)。

### 4. 冻结 reference(correctness 对照基线)
在改动前,跑一次**当前 winning stack**(所有 lever ON)的输出:e2e tpot + logit/prose +
（跑两次拿 A/A 非确定性包络）。最终「干净默认 build」对照的是**这个 winning-stack
reference**,不是旧的 safe default。

### 5. 按安全顺序执行(每步 build-check)
1. **先 revert 改过 ABI 的死 lever**(最危险,单独做 + 核验)。若某死 lever 改过
   graph.cc/task_register 的 `(num_in,num_out,TASK_ENUM,variant)` tuple 或加过 tensor,
   ABI 错一个 = runtime「Invalid global read」。若这些改动**从未 commit**,直接
   `git checkout HEAD -- <producer files>` 回到干净 ABI,然后:
   ```bash
   git diff HEAD -- graph.cc task_register.cc tasks.py multigpu.py allreduce.cuh | wc -l   # 应 ≈ 0
   grep -rn "tile_sumsq|<sidecar tokens>|input_ptrs\[N\]" <files>                          # 应 0
   ```
   **同时删掉这个死 lever 的 consumer 半边**(否则 stale-env OOB 地雷)。
2. **删其余死 lever + 所有诊断 probe** → build-check。
3. **hard-wire LEAF wins**(叶子优化,有干净 `#else`):去 gate、删 `#else`、保留 geometry
   guard → build-check。
4. **hard-wire STRUCTURAL wins 最后做**(改 graph shape / 大控制流的 path-selector):删掉
   整条替代路径,保留 TP8/mbt/workers guard → build-check + `--layers 0-3` in-MPK smoke。
   (结构性 gate 比叶子危险 —— 删它 = 删一整条替代代码路径。树变小后再做。)

### 6. 验证 correctness(默认路径的数学变了)
- **不能用 token-identity**(DSv3 TP8 decode 是 FP-非确定的,cross-CTA atomicAdd)。
- 用:**A/A 包络**(winning stack 自比)+ 干净默认 vs winning-stack reference 的
  **per-step logit-cosine 在包络内** + top-k overlap 稳定 + 512-token coherent prose +
  无 NaN/Inf。
- **perf-smoke**:e2e tpot 应 ≈ winning-stack reference(确认没 silently 掉 win)。
- **TP8 JIT-smoke**:`--layers 0-3` 确认 hard-wire 后的 megakernel 真能实例化+运行
  (`#else`-删除后的路径 only instantiates at world_size==8)。
- Qwen3 / TP4 regression smoke 保护没动的 fallback / 非 DSv3 路径。

### 7. Commit 一个干净版本(供 PR)
- **只 stage 源文件**(kernels/.cuh、builder.py、persistent_kernel.py、task_register.cc);
  **排除** `.claude/`、`scratch/`、`experiment_history/`、CSV/outputs/`.pk_compile` 等本地产物。
- 跑 `mpk-commit-reviewer`,**明确告诉它 default-build 改变是有意的**(否则它会按标准
  gate BLOCK);它仍检查 staged-path 卫生、surface、message、correctness 故事。
- Commit message 列全:hard-wired 了哪些(+ 每个的 Δ)、revert 了哪些死 lever、删了哪些
  诊断、ABI 已还原、验证证据、**明确的 pre-merge gate**(若 TP8 runtime 验证被 box 容量
  封死没跑成,写进 message 作为合并前必跑项)。带 `Co-Authored-By`。

## 关键坑(实测)
- **default build 有意不 byte-identical** —— 这是目的,不是 bug;豁免那条 commit gate。
- **ABI-revert 是最危险的一步** —— 单独先做、grep 干净、对照最后一个干净 commit、连
  consumer 半边一起删。
- **structural wins 最后做** —— path-selector 比 leaf-opt 危险(删整条替代路径)。
- **分类必须核验** —— 有的 gate 已经无条件开、有的是 fallback;别凭「default-OFF」假设。
- **孤儿 legacy 函数** —— 删了 `#else` 调用点后,那个 `__device__` 函数定义可能还留着
  (nvcc 会 elide,无害但不干净);要么删,要么在 PR 里标为已知 nit。
- **correctness gate = A/A 包络 + coherence**,不是 token-identity(FP-非确定路径)。
- **TP8 runtime gate 可能被 box 容量封死** —— 对「供 PR review」的 commit,可带着
  documented pre-merge gate 先提交(PR 是 review 不是 auto-merge);别伪造数字。
- **子 agent 会对同一件事说法冲突**(如「lever 被删了」vs「lever hard-wire 了」)——
  用 `grep -c <win 的 body 符号>` 亲自核验 win 的 body 还在(macro 0-refs ≠ win 被删,
  可能只是 gate 移除了、body 变无条件了)。
