# 00: Meta Orchestrator — Full Project Build

**Description:** Master prompt that executes all 20 sub-prompts in dependency order, managing context, checkpoints, and cross-prompt state throughout the entire build.

**GitHub Issues:** All (#2–#21)

**Prerequisites:** Clean clone of `rocklambros/multiturn-injection-detection` with PRD.md at v2.0.

---

## Role

You are the lead build agent for a PyTorch deep learning project that detects multi-turn distributed prompt injection attacks. You will orchestrate the full build by dispatching subagents for each phase, managing persistent memory for cross-phase decisions, and checkpointing progress so interrupted sessions resume without rework.

## Available Plugins

You have access to and MUST use these plugins throughout execution:

| Plugin | Purpose | When to use |
|--------|---------|-------------|
| **superpowers** | File system, git, environment, shell commands | Every prompt — all file creation, directory ops, git commits |
| **ralph-loop** | Generate → Review → Fix → Confirm cycle | Every code generation step — never commit unreviewed code |
| **goodmem** | Persistent memory across prompts | Store decisions, metrics, schemas, configs that downstream prompts need |
| **serena** | Session memory with checkpoints | Checkpoint after each phase so interrupted sessions resume cleanly |
| **context7** | Third-party API documentation lookup | When writing PyTorch, HuggingFace, sklearn integration code |

## Context Engineering Rules

1. **Before each sub-prompt:** Retrieve all relevant goodmem keys. Inject them into the subagent's context as `<prior_decisions>` block.
2. **After each sub-prompt:** Store outputs (file paths created, metrics, decisions made, architecture choices) to goodmem with namespaced keys: `phase1.skeleton_complete`, `phase2.vocab_size`, `phase3.iter1_f1`, etc.
3. **Between phases:** Create a serena checkpoint with:
   - Phase name and completion status
   - Files created/modified
   - Key metrics or decisions
   - Any blocking issues
4. **On resume:** Read serena checkpoint. Skip completed phases. Resume at last incomplete step.
5. **Context window management:** Do NOT load file contents into context unless the current step requires them. Reference files by path. Let subagents read what they need.

---

## Execution Plan

### Phase A: Foundation (Prompts 01, 08, 09)

These have no data dependencies. Dispatch as parallel subagents.

```
Subagent 1: Execute prompts/01_infrastructure.md → project skeleton
Subagent 2: Execute prompts/08_eval_framework.md → evaluation modules  
Subagent 3: Execute prompts/09_training_loop.md → training infrastructure
```

**Gate A:** All three subagents report success. Verify via superpowers:
```bash
python -c "from src.utils.seed import set_global_seed; set_global_seed(42)"
python -c "from src.utils.config import ITERATIONS; print(len(ITERATIONS))"
python -c "from src.evaluation.metrics import compute_metrics; print('Eval OK')"
python -c "from src.training.train import train_model; print('Train OK')"
```

**Goodmem writes:**
- `foundation.skeleton_complete = true`
- `foundation.eval_ready = true`
- `foundation.training_ready = true`

**Serena checkpoint:** `Phase A complete. Foundation built.`

---

### Phase B: Data Pipeline (Prompts 02, 03, 04, 05, 06, 07)

Sequential chain with one parallel branch.

```
Subagent 4: Execute prompts/02_data_download.md → raw data
  ↓
Subagent 5: Execute prompts/03_data_clean_merge.md → cleaned CSVs
  ↓ (branch)
  ├─ Subagent 6: Execute prompts/04_tokenizer_vocab.md → vocab + tokenizer
  │    ↓ (parallel after vocab)
  │    ├─ Subagent 7: Execute prompts/06_dataset_dataloader.md → DataLoaders
  │    └─ Subagent 8: Execute prompts/07_glove_embeddings.md → embedding matrix
  └─ Subagent 9: Execute prompts/05_synthetic_multiturn.md → synthetic data
```

**Gate B:** Verify all data artifacts exist:
```bash
ls data/processed/single_turn_{train,val,test}.csv
ls data/synthetic/multiturn_{train,val,test}.json
ls models/vocab.json
ls data/embeddings/embedding_matrix.npy
python -c "from src.data.loader import SingleTurnDataset, MultiTurnDataset; print('Loaders OK')"
```

**Goodmem writes:**
- `data.train_samples = <N>`
- `data.val_samples = <N>`  
- `data.test_samples = <N>`
- `data.vocab_size = <N>`
- `data.glove_coverage = <pct>`
- `data.multiturn_train = 5000`
- `data.class_balance = <stats>`

**Serena checkpoint:** `Phase B complete. All data pipelines built and verified.`

---

### Phase C: Baselines (Prompt 10)

```
Subagent 10: Execute prompts/10_baselines.md
```

**Gate C:** `results/iter0_baseline_lr/metrics.json` and `results/iter0_baseline_rf/metrics.json` exist.

**Goodmem writes:**
- `baselines.lr_f1 = <val>`
- `baselines.rf_f1 = <val>`
- `baselines.lr_multiturn_f1 = <val>` (expected to be poor)

**Serena checkpoint:** `Phase C complete. Baselines established.`

---

### Phase D: Single-Turn Models (Prompts 11, 12, 13, 14)

Strictly sequential. Each iteration builds on the prior.

```
Subagent 11: Execute prompts/11_lstm_simple.md → Iteration 1
  ↓
Subagent 12: Execute prompts/12_lstm_glove.md → Iteration 2
  ↓
Subagent 13: Execute prompts/13_bilstm_dropout.md → Iteration 3
  ↓
Subagent 14: Execute prompts/14_gru_comparison.md → Iteration 4 + encoder decision
```

**Gate D:** All four `results/iter{1,2,3,4}_*/metrics.json` exist. Encoder decision stored.

**Goodmem writes:**
- `models.iter1_f1`, `models.iter2_f1`, `models.iter3_f1`, `models.iter4_f1`
- `models.iter3_best_dropout = <0.3 or 0.5>`
- `models.encoder_decision = <LSTM or GRU>`
- `models.encoder_decision_reasoning = <text>`
- `models.best_single_turn_iteration = <N>`
- `models.best_single_turn_path = models/iter<N>_*.pt`

**Serena checkpoint:** `Phase D complete. Single-turn models done. Encoder: <decision>.`

---

### Phase E: Multi-Turn Models (Prompts 15, 16, 17)

Sequential. This is the novel contribution.

```
Subagent 15: Execute prompts/15_multiturn_classifier.md → Iteration 5 (NOVEL)
  ↓
Subagent 16: Execute prompts/16_attention.md → Iteration 6
  ↓
Subagent 17: Execute prompts/17_threshold_tuning.md → Iteration 7
```

**Gate E:** Multi-turn metrics exist. Core finding (F1 gap) documented.

**Goodmem writes:**
- `models.iter5_multiturn_f1 = <val>`
- `models.iter6_attention_f1 = <val>`
- `models.iter7_best_threshold = <val>`
- `models.core_finding_f1_gap = <multiturn F1 minus best single-turn-applied-per-turn F1>`
- `models.attention_pattern = <description of where attention concentrates>`

**Serena checkpoint:** `Phase E complete. Multi-turn models done. Core gap: <val>.`

---

### Phase F: Deliverables (Prompts 18, 19, 20)

Sequential.

```
Subagent 18: Execute prompts/18_notebook.md → Jupyter notebook
  ↓
Subagent 19: Execute prompts/19_report.md → Written report
  ↓
Subagent 20: Execute prompts/20_presentation.md → Slide deck
```

**Gate F:** Notebook passes Restart & Run All. Report and deck exist.

**Goodmem writes:**
- `deliverables.notebook_path = notebooks/Final_Project.ipynb`
- `deliverables.report_path = report/final_report.md`
- `deliverables.presentation_path = report/presentation.*`

**Final serena checkpoint:** `BUILD COMPLETE. All 20 prompts executed. All gates passed.`

---

## Error Handling

- If a subagent fails, read its error output. Attempt a fix using ralph-loop (re-generate the failing piece, review, fix, confirm). If the fix succeeds, continue. If it fails twice, halt and report the blocking issue with full context.
- If a gate check fails, identify which artifact is missing, re-execute only the responsible subagent.
- Never skip a gate. Never proceed to the next phase with a red gate.

## Context Window Management

- Each subagent gets ONLY: its prompt file, the PRD sections it references, and any goodmem keys from prior phases it depends on.
- Do NOT pass the full PRD to every subagent. Pass the relevant sections.
- Do NOT accumulate file contents across phases. Let each subagent read fresh.
- Between phases, summarize in 3-5 lines what was accomplished, then clear working context.

## Final Verification

After Phase F, run this comprehensive check:

```bash
# All source modules import cleanly
python -c "import src.data.download, src.data.clean, src.data.synthetic, src.data.loader"
python -c "import src.utils.tokenizer, src.utils.config, src.utils.seed"
python -c "import src.models.baselines, src.models.single_turn, src.models.multi_turn, src.models.attention"
python -c "import src.training.train"
python -c "import src.evaluation.metrics, src.evaluation.analysis, src.evaluation.visualization"

# All result artifacts exist
ls results/iter0_baseline_lr/metrics.json
ls results/iter1_lstm/metrics.json
ls results/iter5_multiturn/metrics.json
ls results/iter6_attention/metrics.json

# Notebook runs clean
jupyter nbconvert --execute --to html notebooks/Final_Project.ipynb
```

**Execution:** `claude --prompt prompts/00_meta_orchestrator.md --ultrathink`
