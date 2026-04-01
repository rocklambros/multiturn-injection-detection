# Claude Code Instructions: Multi-Turn Prompt Injection Detection

Read `PRD.md` completely before writing any code. The PRD is the source of truth for all architecture decisions, hyperparameters, file locations, and evaluation requirements.

## Execution Prompts

This project uses sequenced prompt files in `/prompts/` for Claude Code execution. Execute them in numeric order. Each prompt specifies its prerequisites, task, and verification steps.

| Prompt | Issue(s) | Phase |
|--------|----------|-------|
| `01_infrastructure.md` | #2 | Infrastructure |
| `02_data_download.md` | #3 | Data Pipeline |
| `03_data_clean_merge.md` | #4 | Data Pipeline |
| `04_tokenizer_vocab.md` | #6 | Data Pipeline |
| `05_synthetic_multiturn.md` | #5 | Data Pipeline |
| `06_dataset_dataloader.md` | #7 | Data Pipeline |
| `07_glove_embeddings.md` | #8 | Data Pipeline |
| `08_eval_framework.md` | #9 | Evaluation |
| `09_training_loop.md` | #10 | Training |
| `10_baselines.md` | #11 | Models |
| `11_lstm_simple.md` | #12 | Models |
| `12_lstm_glove.md` | #13 | Models |
| `13_bilstm_dropout.md` | #14 | Models |
| `14_gru_comparison.md` | #15 | Models |
| `15_multiturn_classifier.md` | #16 | Models (NOVEL) |
| `16_attention.md` | #17 | Models |
| `17_threshold_tuning.md` | #18 | Models |
| `18_notebook.md` | #19 | Deliverable |
| `19_report.md` | #20 | Deliverable |
| `20_presentation.md` | #21 | Deliverable |

## Build Order

Execute prompts in numeric order (01 through 20). Each prompt must produce working, tested code before proceeding to the next.

### Phase 1: Infrastructure
1. Create the directory structure defined in PRD Section 7
2. Write `requirements.txt` (PRD Section 10)
3. Write `src/utils/seed.py` (PRD Section 9)
4. Write `src/utils/config.py` (PRD Section 8)

### Phase 2: Data Pipeline
5. Write `src/data/download.py` (PRD Section 3.1)
6. Write `src/data/clean.py` (PRD Section 3.2)
7. Write `src/utils/tokenizer.py` (PRD Section 3.4)
8. Write `src/data/synthetic.py` (PRD Section 3.3)
9. Write `src/data/loader.py` (PRD Section 3.5)
10. Run the full data pipeline end-to-end. Verify outputs exist and shapes are correct.

### Phase 3: Evaluation Framework
11. Write `src/evaluation/metrics.py` (PRD Section 4.2)
12. Write `src/evaluation/analysis.py` (PRD Section 4.3)
13. Write `src/evaluation/visualization.py` (PRD Section 4.4)

### Phase 4: Models (Iterations 0-7)
14. Write `src/models/baselines.py` (PRD Section 2.4, Iteration 0)
15. Write `src/models/single_turn.py` (PRD Section 2.4, Iterations 1-4)
16. Write `src/models/attention.py` (PRD Section 2.4, Iteration 6)
17. Write `src/models/multi_turn.py` (PRD Section 2.4, Iterations 5-7)

### Phase 5: Training
18. Write `src/training/train.py` (PRD Section 4.1)

### Phase 6: Notebook and Deliverables
19. Build `notebooks/Final_Project.ipynb` following the structure in PRD Section 5.3
20. Run Kernel > Restart & Run All. Fix any failures.
21. Export to HTML.
22. Write `report/final_report.md`
23. Create presentation deck

## Rules

- Every function must have a docstring explaining inputs, outputs, and side effects
- Every file must start with `from src.utils.seed import set_global_seed` and call `set_global_seed(42)` before any random operations
- Save all results (metrics, plots, model weights) to the `results/` and `models/` directories as specified
- Print shapes at every data transformation step
- Never use Kaggle datasets or APIs
- **All models use PyTorch.** sklearn is used for baselines only.
- The notebook must be self-contained: a reviewer who reads only the notebook (not the src code) must understand every decision
- Markdown cells in the notebook must explain WHY each iteration change was made, not just WHAT changed. The rubric gives zero credit without reasoning.

## Hardware Constraints

- Target: NVIDIA Jetson Orin AGX (64GB RAM, Ampere GPU)
- Keep batch sizes at 64 for single-turn, 32 for multi-turn
- No model should exceed 50M parameters
- Total notebook execution time target: under 2 hours

## Testing

After each phase, verify:
- Phase 2: `data/processed/` contains 3 CSV files with expected row counts. `data/synthetic/` contains 3 JSON files with expected sequence counts.
- Phase 3: Calling `evaluate_model()` with dummy data produces a valid metrics.json
- Phase 4: Each model builds without errors and parameter count matches PRD specifications
- Phase 5: Training a model for 1 epoch completes without errors
- Phase 6: Kernel > Restart & Run All completes without errors
