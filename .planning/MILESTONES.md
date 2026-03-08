# Milestones

## v1.0 Volta MVP (Shipped: 2026-03-08)

**Phases completed:** 7 phases, 22 plans, 4 tasks

**Key accomplishments:**
- Adam optimizer досягнув паритету з PyTorch (+25% faster at B≤64); AVX-512 hardcode прибрано, ARM builds розблоковано
- Stable graph fingerprints (SipHasher13), portable gemm_shim (include_bytes!), явна MKL error reporting замість silent dev-path fallback
- MHA full backward pass (7 gradients) — transformer архітектури навчаються коректно
- CPU training fail-fast on non-finite, early-stopping full snapshot restore, deterministic long-loop regression suite (SGD/Adam/AdamW)
- End-to-end parity: compiled ConvNet + honest tiny-transformer проти PyTorch reference з фіксованим seed
- CLI smoke tests, doctor rewrite з capability matrix і MKL/LLVM diagnostics, docs/README sync з реальною поведінкою
- Release pipeline: macOS universal binary via lipo, smoke-check CI job, README install section

**Known gaps (proceeded anyway):**
- UX-V2-01, UX-V2-03, DIST-V2-03: traceability статус "Planned" (код виконаний, traceability не оновлено)
- MODEL-V2-01: AOT training codegen MLP-only — свідомо deferred
- CUDA-V2-01/02/03: out of scope для v1.0

---

