# Contributing

Thank you for your interest in this project. Contributions are welcome for extensions, bug fixes, and documentation improvements.

## Getting Started

1. Fork the repository and clone your fork
2. Create a feature branch: `git checkout -b feature/your-feature`
3. Install dependencies: `pip install -r requirements.txt`
4. Make your changes and add tests

## Code Standards

This project follows strict conventions to ensure reproducibility and consistency:

- **Seed everything.** Every file that performs random operations must import and call `set_global_seed(42)` from `src.utils.seed` before any stochastic code runs. This covers Python's `random`, NumPy, PyTorch, and cuDNN deterministic mode.
- **Docstrings required.** Every function needs a docstring explaining inputs, outputs, and side effects.
- **Print shapes.** Log tensor/array shapes at every data transformation step. This catches dimension mismatches early and documents the data flow.
- **PyTorch for models, sklearn for baselines.** All neural network models use PyTorch. Scikit-learn is used only for TF-IDF baselines (logistic regression, random forest).
- **Save all artifacts.** Metrics go to `results/` as JSON. Model weights go to `models/` as `.pt` files. Plots go to `results/` as PNG.

## Project Layout

| Directory | Contents |
|-----------|----------|
| `data/` | Raw downloads, processed CSVs, synthetic JSONL, GloVe embeddings |
| `src/data/` | Data download, cleaning, synthetic generation, loading |
| `src/models/` | Model architectures and training orchestration |
| `src/evaluation/` | Metrics, analysis, visualization, bootstrap CIs |
| `src/training/` | Training loop with early stopping |
| `src/utils/` | Seed, tokenizer, configuration |
| `scripts/` | Standalone utility scripts (data generation, evaluation, RunPod) |
| `tests/` | Pytest test suite |
| `notebooks/` | Jupyter notebook with full walkthrough |
| `report/` | Academic report (LaTeX + PDF) and presentation |

## Testing

Run the existing test suite before submitting:

```bash
pytest tests/ -v
```

The test suite covers:
- End-to-end pipeline validation (`test_e2e_pipeline.py`)
- Fragment engine correctness (`test_fragment_engine.py`)
- Data partitioning (`test_partition.py`)
- Validation gates (`test_validation_gate.py`)
- Loss function migration (`test_bce_migration.py`)
- Attention masking (`test_mask_fix.py`)

New features should include corresponding tests.

## Pull Requests

- Branch from `main`
- Write a clear description of what changed and why
- Ensure all tests pass
- Do not break existing iteration results (metrics stored in `results/`)
- Keep commits focused and atomic

## Hardware Considerations

The project targets the NVIDIA Jetson Orin AGX (64GB RAM, Ampere GPU). Keep models under 50M parameters and batch sizes at 64 (single-turn) or 32 (multi-turn).
