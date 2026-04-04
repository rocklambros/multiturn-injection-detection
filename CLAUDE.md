# Claude Code Instructions: Multi-Turn Prompt Injection Detection

Read `PRD.md` completely before writing any code. The PRD is the source of truth for all architecture decisions, hyperparameters, file locations, and evaluation requirements.

## Quick Start: Full Build

To build the entire project in one run:

```bash
claude --prompt prompts/00_meta_orchestrator.md --ultrathink
```

The meta orchestrator dispatches subagents for each phase, manages cross-prompt state via goodmem, and checkpoints progress via serena so interrupted sessions resume cleanly.

To run individual steps:

```bash
claude --prompt prompts/01_infrastructure.md --ultrathink
```

## Required Plugins

Every prompt in this project assumes these plugins are active. Do not run without them.

| Plugin | Purpose | Used in |
|--------|---------|---------|
| **superpowers** | File system, git, shell commands | Every prompt |
| **ralph-loop** | Generate → Review → Fix → Confirm cycle | Every code generation step |
| **goodmem** | Persistent memory across prompts (decisions, metrics, paths) | Every prompt writes; downstream prompts read |
| **serena** | Session memory with checkpoints for resumability | Phase boundaries and long tasks |
| **context7** | Third-party API docs (PyTorch, HuggingFace, sklearn) | Prompts with library integration |

## Execution Prompts

| Prompt | Issue(s) | Phase | Key goodmem outputs |
|--------|----------|-------|---------------------|
| `00_meta_orchestrator.md` | All | Orchestration | Dispatches everything |
| `01_infrastructure.md` | #2 | Infrastructure | `foundation.*` |
| `02_data_download.md` | #3 | Data Pipeline | `data.raw_*` |
| `03_data_clean_merge.md` | #4 | Data Pipeline | `data.train_samples`, `data.class_balance_*` |
| `04_tokenizer_vocab.md` | #6 | Data Pipeline | `data.vocab_size`, `data.vocab_path` |
| `05_synthetic_multiturn.md` | #5 | Data Pipeline | `data.multiturn_*`, `data.strategy_distribution` |
| `06_dataset_dataloader.md` | #7 | Data Pipeline | `data.loader_ready` |
| `07_glove_embeddings.md` | #8 | Data Pipeline | `data.glove_coverage`, `data.embedding_dim` |
| `08_eval_framework.md` | #9 | Evaluation | `foundation.eval_ready` |
| `09_training_loop.md` | #10 | Training | `foundation.training_ready` |
| `10_baselines.md` | #11 | Models | `baselines.*` |
| `11_lstm_simple.md` | #12 | Models | `models.iter1_*` |
| `12_lstm_glove.md` | #13 | Models | `models.iter2_*` |
| `13_bilstm_dropout.md` | #14 | Models | `models.iter3_*` |
| `14_gru_comparison.md` | #15 | Models | `models.encoder_decision`, `models.best_single_turn_*` |
| `15_multiturn_classifier.md` | #16 | Models (NOVEL) | `models.core_finding_*` |
| `16_attention.md` | #17 | Models | `models.attention_pattern` |
| `17_threshold_tuning.md` | #18 | Models | `models.iter7_*_threshold` |
| `18_notebook.md` | #19 | Deliverable | `deliverables.notebook_*` |
| `19_report.md` | #20 | Deliverable | `deliverables.report_path` |
| `20_presentation.md` | #21 | Deliverable | `deliverables.presentation_path` |

## Agentic Execution Model

The meta orchestrator (`00`) runs the build as a multi-agent system:

**Phase A (parallel):** Prompts 01, 08, 09 — infrastructure, eval, training loop have no data dependencies. Dispatch as parallel subagents.

**Phase B (sequential with branches):** Prompts 02 → 03 → {04, 05 in parallel} → {06, 07 in parallel after 04}.

**Phase C:** Prompt 10 — baselines.

**Phase D (sequential):** Prompts 11 → 12 → 13 → 14 — each iteration builds on the prior.

**Phase E (sequential):** Prompts 15 → 16 → 17 — multi-turn models (novel contribution).

**Phase F (sequential):** Prompts 18 → 19 → 20 — deliverables.

Each phase ends with a **gate check** (verification commands that must pass) and a **serena checkpoint** (so interrupted builds resume at the last completed phase, not from scratch).

## Goodmem Key Namespaces

All prompts read/write to these namespaces:

- `foundation.*` — Infrastructure readiness flags
- `data.*` — Dataset sizes, vocab, paths, class balance
- `baselines.*` — Baseline F1 scores (performance floor)
- `models.*` — Per-iteration F1, parameter counts, the encoder decision, the core multi-turn finding
- `deliverables.*` — Output file paths, completion flags

Downstream prompts MUST read relevant goodmem keys before generating code. The meta orchestrator injects these as `<prior_decisions>` blocks in subagent context.

## Rules

- Every function must have a docstring explaining inputs, outputs, and side effects
- Every file must start with `from src.utils.seed import set_global_seed` and call `set_global_seed(42)` before any random operations
- Save all results (metrics, plots, model weights) to `results/` and `models/`
- Print shapes at every data transformation step
- Never use Kaggle datasets or APIs
- **All models use PyTorch.** sklearn for baselines only.
- Every code generation step MUST go through ralph-loop: generate → review → fix → confirm. Never commit unreviewed code.
- Markdown cells in the notebook must explain WHY, not just WHAT. Zero credit without reasoning.

## Hardware Constraints

- Target: NVIDIA Jetson Orin AGX (64GB RAM, Ampere GPU)
- Batch sizes: 64 single-turn, 32 multi-turn
- No model exceeds 50M parameters
- Total notebook execution: under 2 hours

## Testing

After each phase, verify via superpowers:

- **Phase B:** `data/processed/` has 3 CSVs. `data/synthetic/` has 3 JSONs. `models/vocab.json` exists.
- **Phase C:** `results/iter0_baseline_*/metrics.json` exist with F1 > 0.8 on single-turn.
- **Phase D:** `results/iter{1,2,3,4}_*/metrics.json` all exist. Encoder decision stored in goodmem.
- **Phase E:** `results/iter{5,6,7}_*/metrics.json` exist. Core finding F1 gap documented.
- **Phase F:** `jupyter nbconvert --execute` passes. Report and deck exist.
