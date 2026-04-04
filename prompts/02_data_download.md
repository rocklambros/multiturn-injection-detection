# 02: Download Datasets from HuggingFace

**Description:** Download three prompt injection datasets and save as raw parquet with manifest.

**GitHub Issues:** #3 — [Data Pipeline] Download datasets from HuggingFace

**Prerequisites:** Prompt 01 complete (project skeleton exists)

**Expected Outputs:**
- `src/data/download.py`
- `data/raw/{source_name}/` directories with parquet files
- `data/raw/manifest.json` with timestamps, row counts, schemas

---

## Role

You are a data engineer building the data acquisition pipeline for a prompt injection detection system.

## Grounding

Use **superpowers** to read:
1. `PRD.md` Section 3.1 (Dataset Acquisition) for exact dataset paths and requirements.
2. `src/utils/seed.py` and `src/utils/config.py` for project conventions.
3. Verify `data/raw/` directory exists.

Use **goodmem** to read `foundation.skeleton_complete` — abort if false.

## Task

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

## Plugin Usage

**context7:** Look up current HuggingFace `datasets` API for `load_dataset()` and `to_parquet()` — verify function signatures before writing integration code.

**superpowers:** Run the download script. Verify manifest.json and parquet files exist.

**ralph-loop:**
1. Generate `src/data/download.py`
2. Execute it: `python src/data/download.py`
3. Review: Does manifest.json have 3 entries with row counts? Do parquet files exist?
4. Fix any download failures or schema issues
5. Confirm all criteria pass

**goodmem:** After completion, persist:
- `data.raw_manifest_path = data/raw/manifest.json`
- `data.deepset_rows = <N>`
- `data.safeguard_rows = <N>`
- `data.neuralchemy_rows = <N>`

## Verification

```bash
python src/data/download.py
cat data/raw/manifest.json
ls data/raw/*/
```

**Execution:** `claude --prompt prompts/02_data_download.md --ultrathink`
