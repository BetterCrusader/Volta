# PRD: Volta — High-Performance Multi-Backend ML Runtime (MVP → Production)

## Коротко
Ціль: зробити production-grade runtime для тренування й інференсу невеликих та середніх трансформерів (TinyLM → transformer blocks) з CPU-AVX2 + CUDA бекадами, автоматичним tuning-ом matmul, KV-cache, повноцінним CLI/benchmarks та інтеграцією в автономний агентний цикл (Ralphy + GPT-5.3 Codex). Код мусить бути читаємим, з юніт/інтеграційними тестами, без `panic/unwrap` у runtime, із CI і вимірювальною телеметрією. Реліз — робочий runtime CPU, з опціональним `--features cuda` для GPU scaffold; подальші фазові патчі — повноцінні CUDA kernels та VSCode/IDE інтеграція.

---

## Успіх-метрики (критерії прийняття)
- `cargo test` — всі юніти проходять (≥ 99% coverage на core modules).
- TinyLM inference (cpu) latency/token ≤ baseline (поточний вимір), з KV-cache показник speedup ≥ 1.1x.
- Матмул: автоматичний tuner повертає params які дають ≥5% speedup vs default на тестових CPU-симуляціях (на CI — synthetic microbench).
- AVX2 microkernels — нуль регресій (тести на точність/бітову сумісність).
- Device API: працює за контрактом для CpuDevice і повертає помірний Err для Cuda scaffold без CUDA драйверів.
- Документація: `docs/architecture.md`, `docs/matmul_tuning.md`, `PRD.md` (цей файл) — пройшли рев’ю.

---

## Scope & Phases
### MVP (обовʼязково)
- Структура runtime: Tensor, matmul (blocked, packed), AVX2 microkernels (існуючий код), CpuDevice wrapper.
- Device abstraction + Cuda scaffold (без real kernels) — safe FFI contract (alloc/copy/free/stream).
- Training loop: MSE + CE + simple two-layer trainer (SGD).
- TinyLM inference + KV-cache (CPU).
- Bench harness: per-token latency, op breakdown, bench CLI.
- Matmul tuner: grid search params, thread-safe cache, CLI flag `--tune-matmul`.
- Tests: unit + integration + benchmarks; CI job runs tests, bench smoke.
- Ralphy PRD + `.ralphy/config.yaml` + AGENTS/CLAUDE rules for autonomous agents.

### Phase 1 (near term)
- Multi-head attention (device-aware) + causal mask, incremental per-head KV cache.
- microkernel dispatcher (4x8,8x8,12x8,16x8) polished; AVX2 register live-range tuning for Zen4.
- Persist tuned matmul cache to disk (per CPU signature).
- Improve TinyLM: temperature/top-k sampling, batched generation.

### Phase 2 (mid term)
- CUDA backend real FFI + memory pools + device kernels scaffold.
- Implement GPU matmul kernel (tiling, shared memory, warp layout), softmax kernel, attention fused ops.
- KV-cache layout for GPU (per-head contiguous buffers); incremental multi-head kernel.
- JIT/fusion scaffold (future): op fusion rules, kernel generator hooks.

### Phase 3 (long term)
- VSCode LSP / snippets / extension for model authoring and profiler UI.
- Distributed training (multi-GPU), checkpointing format, export model format.
- Auto-tuning cluster / telemetry dashboard.

## Wave 1 (Quality Fortress Foundation)

Wave 1 establishes enforceable governance and gate foundations before deeper release hardening.

- Add governance docs under `docs/governance/` and validate them with tests.
- Add ownership and PR policy files (`.github/CODEOWNERS`, PR template).
- Add tier-detection and policy-check scripts with unit tests.
- Add PR gate workflow for fmt/clippy/tests/CLI smoke/property-fast checks.
- Keep CLI/API behavior backward compatible while introducing quality enforcement.

---

## Функціональні вимоги (детально)
1. **Tensor API**
   - Rank-checked ops: matmul, relu, softmax (stable), cross_entropy (log-sum-exp), argmax.
   - No panic: повертаємо `Result<_, TensorError>` з людським текстом.
   - `Tensor::matmul` обирає між tuned params або дефолтом.

2. **Matmul**
   - Blocked + packed RHS, панелі з j_block = 8 оптимізацією.
   - Serial + parallel paths (rayon) з контролем `RAYON_NUM_THREADS`.
   - Tuner:
     - Grid search по K_BLOCK ∈ [64,96,128,160,192,256], M_BLOCK ∈ [32,48,64,96,128]
     - Warmup + multiple runs; prealloc matrices; store best params cache
     - CLI: `--tune-matmul --dim N --runs R`
     - Persist cache file per CPU signature (JSON).

3. **AVX2 microkernels**
   - Intrinsics з `#[target_feature(enable = "avx2,fma")]`.
   - Implement 4x8/8x8/12x8/16x8 kernels + dispatcher.
   - Add prefetch schedule adjustable in tuner. No modifications to semantics.
   - Comprehensive tests: numeric parity with scalar fallback; alignment/unaligned loads.

4. **Device abstraction**
   - `Device` trait: `matmul`, `softmax`, `relu`, `alloc`, `free`, `copy_to_device`, `copy_to_host`.
   - `CpuDevice` — uses existing Tensor path; `CudaDevice` — scaffold with safe RAII `DeviceBuffer`.
   - Runtime selection: `DeviceKind::detect()`, env override `VOLTA_DEVICE`.
   - `DeviceHandle` unified proxy.

5. **Transformer & TinyLM**
   - MultiHeadSelfAttentionLayer: Q/K/V projections, head split/merge, optional causal mask.
   - TransformerBlock: attention + residual + FFN + layernorm (or stable variant).
   - TinyLm: forward_step, generate with KV-cache, sampling (temp/top-k), batched gen option.
   - KV-cache: `AttentionKvCache`, `TransformerKvCache` — append per token, no recompute of past K/V.

6. **Training**
   - Two-layer trainer + classification CE pipeline (already present).
   - Device-aware training API: `train_two_layer_classifier_ce_with_device(&dyn Device, ...)`.
   - Strict shape checks and finite checks.

7. **Benchmarks & Profiling**
   - `--bench-infer` CLI: runs tinyLM bench with warmup, runs, tokens.
   - ProfilingDevice proxy counts op calls and timings (matmul/softmax/relu).
   - Output CSV/JSON for visualization.

8. **Autonomous agent integration**
   - Ralphy PRD file (this document), `.ralphy/config.yaml` with strict rules.
   - Task granularity for Ralphy agents (see Tasks section).
   - Engine override example: `ralphy --codex --model gpt-5.3-codex --prd PRD.md`.

---

## Нефункціональні вимоги (quality / security / performance)
- No `unsafe` except intrinsics in well-isolated modules; all `unsafe` must have documented SAFe justification and tests.
- No `panic!` / `unwrap()` in library runtime; `main` CLI may propagate errors with `?`.
- CI: run `cargo check`, `cargo test`, `cargo fmt -- --check`, `clippy -- -D warnings`.
- Bench jobs: run bench smoke (fast) in CI; heavy benches on scheduled runner.
- Coding standards: rustfmt + clippy; PR template requires performance summary & tests.
- Performance targets: pref­er latency/throughput and memory boundedness; avoid allocations in inner loops.

---

## Security & Licensing
- MIT license (project already MIT).
- Do not include proprietary CUDA code; FFI stubs only until proper licensing reviewed.
- No hardcoded credentials; webhook URLs via env vars.

---

## Deliverables (MVP sprint)
1. `PRD.md` (this file) — done.
2. `.ralphy/config.yaml` with rules (example provided).
3. `src/device/mod.rs` + `src/device/cuda.rs` scaffold (no panic path).
4. `src/ir/matmul_tuner.rs` + CLI flag `--tune-matmul`.
5. TinyLM inference + KV cache (CPU) + bench harness (`--bench-infer`).
6. AVX2 microkernels validated with unit tests.
7. `docs/architecture.md`, `docs/matmul_tuning.md`.
8. CI pipeline entries: test, smoke bench, format, clippy.

Acceptance: all above build+test pass on repo.

---

## Tasks (granular checklist for Ralphy agents / devs)
- [ ] Add/verify `PRD.md` in repo root (this file).
- [ ] Add/verify `PRD.md` in repo root (this file).
- [ ] Ensure `.ralphy/config.yaml` exists and matches rules.
- [ ] Run `cargo test` baseline and record metrics (attach to PR).
- [ ] Implement matmul tuner (grid search + cache persist) — tests + CLI.
- [ ] Add persistence for tuning cache (`~/.volta/matmul_tuning/<cpu_sig>.json`).
- [ ] Harden AVX2 intrinsics modules: add numeric parity tests vs scalar fallback.
- [ ] Add `ProfilingDevice` and bench harness for TinyLM.
- [ ] TinyLM: add temperature/top-k and batched generation; tests.
- [ ] Device: implement safe FFI thin wrappers for `cudaMalloc/cudaFree/cudaMemcpy` (feature gated) — RETURN Err if no CUDA driver.
- [ ] Implement incremental per-head KV cache and per-head attention path.
- [ ] Add CLI flags: `--bench-infer`, `--tune-matmul`, `--device`, `--bench-threads`.
- [ ] Add CI: format, clippy, tests, bench smoke; schedule heavy bench nightly.
- [ ] Write docs: architecture, tuning guide, coding standards (CLAUDE.md / .cursorrules excerpt).
- [ ] PRD → create issue + assign tasks; Ralphy agents: run in parallel (3 agents), branch per task, require `cargo test` on each PR.

For each task: include
- owner (human or agent),
- exact commands to run,
- expected test/bench outputs,
- acceptance criteria (pass/fail).

---

## Developer rules to enforce in prompts (for Codex / agents)
1. Always run and attach `cargo test` output and `cargo run --release -- --bench-infer` bench summary in PR description.
2. No code paths that `panic!` in library; use `Result` with clear errors.
3. Microkernel changes require numeric parity tests vs scalar fallback for multiple dims (128,256,512).
4. Device API changes must preserve CpuDevice behavior by default.
5. Performance changes must include microbench harness and op breakdown.
6. All commits must be atomic and focused; one main change per PR.
7. Add unit tests for each new function and integration tests for end-to-end training/inference flows.
8. Document every `unsafe` with short rationale and invariants.

---

## How to run locally (quick)
- Tests: `cargo test`
- Run demo: `cargo run --release`
- Bench infer: `cargo run --release -- --bench-infer --runs 3 --warmup 1 --tokens 8`
- Tune matmul: `cargo run --release -- --tune-matmul --dim 256 --runs 5`
- Ralphy agent (Codex): `ralphy --codex --model gpt-5.3-codex --prd PRD.md`

---

## Final notes (for agent)
- Treat this PRD as authoritative. Do not change acceptance criteria without human review.
- If a proposed change risks AVX2 regressions or breaks `cargo test`, create a draft PR and request human unblock.
- Prioritize stability → correctness → performance in that order.
- Provide short changelog + perf summary in every PR.

---

## Attachment: Minimal PR template (copy into `.github/PULL_REQUEST_TEMPLATE.md`)
