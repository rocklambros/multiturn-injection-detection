# Claude Code Instructions: Multi-Turn Prompt Injection Detection

Read `PRD.md` completely before writing any code. The PRD is the source of truth for all architecture decisions, hyperparameters, file locations, and evaluation requirements.

## Build Order

Execute in this exact order. Do not skip steps. Each step must produce working, tested code before proceeding.

### Phase 1: Infrastructure
1. Create the directory structure defined in PRD Section 2
2. Write `requirements.txt` (PRD Section 11)
3. Write `src/utils/seed.py` (PRD Section 10)
4. Write `src/utils/config.py` (PRD Section 8)

### Phase 2: Data Pipeline
5. Write `src/data/download.py` (PRD Section 3.1)
6. Write `src/data/clean.py` (PRD Section 3.2)
7. Write `src/utils/tokenizer.py` (PRD Section 3.4)
8. Write `src/data/synthetic.py` (PRD Section 3.3)
9. Write `src/data/loader.py` (PRD Section 3.5)
10. Run the full data pipeline end-to-end. Verify outputs exist and shapes are correct.

### Phase 3: Evaluation Framework
11. Write `src/evaluation/metrics.py` (PRD Section 7.1)
12. Write `src/evaluation/analysis.py` (PRD Section 7.2)
13. Write `src/evaluation/visualization.py` (PRD Section 7.3)

### Phase 4: Models
14. Write `src/models/baselines.py` (PRD Section 5.1)
15. Write `src/models/single_turn.py` (PRD Sections 5.2-5.5)
16. Write `src/models/attention.py` (PRD Section 5.7)
17. Write `src/models/multi_turn.py` (PRD Sections 5.6-5.7)

### Phase 5: Training
18. Write `src/training/callbacks.py` (PRD Section 6.2)
19. Write `src/training/train.py` (PRD Section 6.1)

### Phase 6: Notebook
20. Build `notebooks/Final_Project.ipynb` following the structure in PRD Section 9
21. Run Kernel > Restart & Run All. Fix any failures.
22. Export to HTML.

## Rules

- Every function must have a docstring explaining inputs, outputs, and side effects
- Every file must start with `from src.utils.seed import set_global_seed` and call `set_global_seed(42)` before any random operations
- Save all results (metrics, plots, model weights) to the `results/` and `models/` directories as specified
- Print shapes at every data transformation step
- Never use Kaggle datasets or APIs
- All models use Keras/TensorFlow. No PyTorch.
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
- Phase 4: Each model builds without errors and `model.summary()` matches PRD specifications
- Phase 5: Training a model for 1 epoch completes without errors
- Phase 6: Kernel > Restart & Run All completes without errors
