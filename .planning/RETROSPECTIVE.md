# Retrospective: Volta

---

## Milestone: v1.0 — Volta MVP

**Shipped:** 2026-03-08
**Phases:** 7 | **Plans:** 22 | **Commits:** ~75 (2026-03-07 → 2026-03-08)

### What Was Built

- Adam optimizer +25% faster than PyTorch at B≤64 (was 1.9× slower); AVX-512 hardcode removed, ARM builds unblocked
- Stable graph fingerprints (SipHasher13), portable gemm_shim (include_bytes!), actionable MKL error reporting
- MHA full backward pass with 7 gradients — transformer architectures train correctly
- CPU training fail-fast on non-finite, full early-stopping snapshot restore, deterministic long-loop regression suite
- End-to-end PyTorch parity: compiled ConvNet + honest tiny-transformer fixtures with fixed seed
- CLI smoke test scaffold, doctor rewrite with MKL/LLVM detection and capability matrix
- Release pipeline: macOS universal binary via lipo, smoke-check CI, README install section

### What Worked

- **Phase-by-phase approach** — кожна фаза мала чітку мету і Success Criteria; зрозуміло що робити і коли стоп
- **PyTorch parity як oracle** — реальні числа проти reference замість internal-only tests; дало впевненість
- **fail-fast design** — early capability check + non-finite guards усунули silent failure patterns
- **Honesty-first** — Phase 5 explicitly замінив fake confidence реальними fixtures замість закрити формально

### What Was Inefficient

- Traceability статуси (UX-V2-01, UX-V2-03, DIST-V2-03) залишились "Planned" після виконання — не оновлено в REQUIREMENTS.md
- STATE.md marks phases Complete але roadmap checkboxes `[ ]` не оновлювались — розсинхрон між файлами
- Phase 6 не мав CONTEXT.md (discuss-phase не проводився) — планування йшло без explicit design decisions файлу

### Patterns Established

- `include_bytes!` для embedded assets замість path resolution at runtime
- PyTorch oracle fixtures з фіксованим seed як canonical regression cases
- `validate_optimizer()` early failure pattern — capability check перед тривалою операцією
- lipo merge в release job, а не в окремому build step

### Key Lessons

- Traceability потрібно оновлювати одразу при виконанні плану, не після milestone
- SUMMARY.md one_liner поле треба заповнювати явно — gsd-tools не може витягти якщо формат нестандартний
- Roadmap checkboxes `[x]` не оновлюються автоматично — потрібен explicit step або tool support

### Cost Observations

- Sessions: 2 дні активної роботи (2026-03-07 → 2026-03-08)
- Notable: 7 фаз за 2 дні — висока швидкість через чіткий roadmap і GSD workflow

---

## Cross-Milestone Trends

| Milestone | Phases | Plans | Timeline | Key pattern |
|-----------|--------|-------|----------|-------------|
| v1.0 MVP | 7 | 22 | 2 days | Fast execution with clear per-phase Success Criteria |

---

*Retrospective created: 2026-03-08 after v1.0 milestone*
