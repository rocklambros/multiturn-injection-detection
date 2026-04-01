# 03: Clean, Merge, Deduplicate, and Split Datasets

**Description:** Merge three raw datasets, normalize, deduplicate, clean, split 70/15/15, generate bias report.

**GitHub Issues:** #4 — [Data Pipeline] Clean, merge, deduplicate, and split datasets

**Prerequisites:** Prompt 02 (raw data must be downloaded)

**Expected Outputs:**
- `src/data/clean.py`
- `data/processed/single_turn_{train,val,test}.csv` with columns: text, label, source
- `data/processed/bias_report.txt`

---

## Prompt

You are a data engineer building the cleaning and splitting pipeline.

<investigate_before_answering>
1. Read `PRD.md` Section 3.2 (Data Cleaning and Merging) for all 9 cleaning steps in order.
2. Read `data/raw/manifest.json` to understand source column names and schemas.
3. Read `src/utils/seed.py` for reproducibility setup.
</investigate_before_answering>

### Task

Write `src/data/clean.py` implementing all 9 cleaning steps from PRD Section 3.2 in order.

**Completion criteria:**
- Labels normalized: 0=benign, 1=injection across all sources
- Removal log printed: `{duplicates: N, near_duplicates: N, empty: N, too_long: N}`
- Split: 70/15/15 stratified on label, seed=42
- CSVs contain columns: text, label, source
- Class distribution printed per split
- `data/processed/bias_report.txt` with all flags from PRD Section 3.2

### Tool Guidance

- **Ralph loops:** Run the pipeline, verify output CSVs exist with expected row counts and column names. Check class balance. Fix issues.
- **Goodman plugin:** Persist final dataset sizes and class distributions to agent memory for downstream prompts.

### Verification

```bash
python src/data/clean.py
wc -l data/processed/single_turn_*.csv
head -5 data/processed/single_turn_train.csv
cat data/processed/bias_report.txt
```

**Execution:** `claude --prompt prompts/03_data_clean_merge.md --ultrathink`
