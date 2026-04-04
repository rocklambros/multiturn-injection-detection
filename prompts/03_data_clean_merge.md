# 03: Clean, Merge, Deduplicate, and Split Datasets

**Description:** Merge three raw datasets, normalize, deduplicate, clean, split 70/15/15, generate bias report.

**GitHub Issues:** #4 — [Data Pipeline] Clean, merge, deduplicate, and split datasets

**Prerequisites:** Prompt 02 complete (raw data downloaded)

**Expected Outputs:**
- `src/data/clean.py`
- `data/processed/single_turn_{train,val,test}.csv` with columns: text, label, source
- `data/processed/bias_report.txt`

---

## Role

You are a data engineer building the cleaning and splitting pipeline.

## Grounding

Use **superpowers** to read:
1. `PRD.md` Section 3.2 — all 9 cleaning steps in order.
2. `data/raw/manifest.json` to understand source column names and schemas.

Use **goodmem** to read `data.deepset_rows`, `data.safeguard_rows`, `data.neuralchemy_rows` for expected input sizes.

## Task

Write `src/data/clean.py` implementing all 9 cleaning steps from PRD Section 3.2 in exact order.

**Completion criteria:**
- Labels normalized: 0=benign, 1=injection across all sources
- Removal log printed: `{duplicates: N, near_duplicates: N, empty: N, too_long: N}`
- Split: 70/15/15 stratified on label, seed=42
- CSVs contain columns: text, label, source
- Class distribution printed per split
- `data/processed/bias_report.txt` with all flags from PRD Section 3.2

## Plugin Usage

**superpowers:** Run the cleaning pipeline end-to-end. Verify output files.

**ralph-loop:**
1. Generate `src/data/clean.py`
2. Execute: `python src/data/clean.py`
3. Review: Check output CSV row counts, column names, class balance, bias report content
4. Fix any normalization or splitting issues
5. Confirm all criteria pass

**goodmem:** After completion, persist:
- `data.train_samples = <N>`
- `data.val_samples = <N>`
- `data.test_samples = <N>`
- `data.class_balance_train = <benign_pct / injection_pct>`
- `data.total_after_cleaning = <N>`

**serena:** Checkpoint after this prompt — the cleaned data is the foundation for everything downstream.

## Verification

```bash
python src/data/clean.py
wc -l data/processed/single_turn_*.csv
head -5 data/processed/single_turn_train.csv
cat data/processed/bias_report.txt
```

**Execution:** `claude --prompt prompts/03_data_clean_merge.md --ultrathink`
