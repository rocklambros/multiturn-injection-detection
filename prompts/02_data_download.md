# 02: Download Datasets from HuggingFace

**Description:** Download three prompt injection datasets and save as raw parquet with manifest.

**GitHub Issues:** #3 — [Data Pipeline] Download datasets from HuggingFace

**Prerequisites:** Prompt 01 (project skeleton must exist)

**Expected Outputs:**
- `src/data/download.py`
- `data/raw/{source_name}/` directories with parquet files
- `data/raw/manifest.json` with timestamps, row counts, schemas

---

## Prompt

You are a data engineer building the data acquisition pipeline for a prompt injection detection system.

<investigate_before_answering>
1. Read `PRD.md` Section 3.1 (Dataset Acquisition) for exact dataset paths and requirements.
2. Read `src/utils/seed.py` and `src/utils/config.py` to understand project conventions.
3. Check that `data/raw/` directory exists.
</investigate_before_answering>

### Task

Write `src/data/download.py` that downloads three datasets from HuggingFace:
- deepset/prompt-injections
- xTRam1/safe-guard-prompt-injection
- neuralchemy/Prompt-injection-dataset

**Completion criteria:**
- Raw parquet files in `data/raw/{source}/`
- `data/raw/manifest.json` with download timestamps, row counts, column schemas
- Retry logic: 3 attempts with exponential backoff
- Label distributions printed to console
- No Kaggle datasets or APIs anywhere

**Constraints:**
- Use `datasets.load_dataset()` from HuggingFace
- Save via `dataset.to_parquet()`
- Call `set_global_seed(42)` at module top

### Tool Guidance

- **Context7 plugin:** Look up current HuggingFace `datasets` API for `load_dataset()` and `to_parquet()` to verify correct usage.
- **Ralph loops:** Run the download script. Verify manifest.json exists and contains correct row counts. Fix any failures.

### Verification

```bash
python src/data/download.py
cat data/raw/manifest.json
ls data/raw/*/
```

**Execution:** `claude --prompt prompts/02_data_download.md --ultrathink`
