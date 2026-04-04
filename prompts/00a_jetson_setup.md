# 00a: Jetson Environment Setup

**Description:** Install and verify all Python dependencies on NVIDIA Jetson Orin AGX (aarch64). Troubleshoot and fix any build failures automatically.

**Prerequisites:** Python 3.10+, PyTorch, and scikit-learn already installed.

---

## Role

You are a systems engineer configuring an NVIDIA Jetson Orin AGX (aarch64/ARM64) for a PyTorch deep learning project. This is NOT x86. Some packages require ARM-compatible wheels or source builds. Your job is to install every dependency, verify it imports cleanly, and fix whatever breaks.

## Known Constraints

- Architecture: `aarch64` (ARM64). Not x86_64.
- PyTorch is pre-installed (likely via NVIDIA's Jetson PyTorch wheel). Do NOT reinstall or upgrade torch.
- scikit-learn is pre-installed. Do NOT reinstall or upgrade sklearn.
- pip is the package manager. Do NOT use apt for Python packages.
- If a package fails to build from source, try: (1) a different version, (2) conda-forge, (3) system-level build deps.

## Task

Install and verify these packages (from PRD Section 10):

```
numpy>=1.24.0        # likely already present as torch dependency
pandas>=2.0.0
matplotlib>=3.7.0
seaborn>=0.12.0
nltk>=3.8.0
datasets>=2.14.0     # HuggingFace — depends on pyarrow (aarch64 trouble spot)
jupyterlab>=4.0.0
tqdm>=4.65.0
```

## Execution Steps

### Step 1: Check what's already installed

```bash
pip list 2>/dev/null | grep -iE "numpy|pandas|matplotlib|seaborn|nltk|datasets|jupyterlab|tqdm|pyarrow|torch|scikit-learn"
python -c "import platform; print(f'Arch: {platform.machine()}, Python: {platform.python_version()}')"
```

Record what's present. Only install what's missing.

### Step 2: Install the easy ones first

These are pure Python or have reliable aarch64 wheels:

```bash
pip install --no-deps tqdm
pip install pandas matplotlib seaborn nltk jupyterlab
```

If any fail, read the error. Common fixes:
- Missing `libfreetype6-dev` or `libpng-dev` for matplotlib: `sudo apt-get install -y libfreetype6-dev libpng-dev`
- Missing `libhdf5-dev` for pandas: `sudo apt-get install -y libhdf5-dev`

### Step 3: Install pyarrow (the aarch64 pain point)

pyarrow is required by HuggingFace `datasets`. Try in this order:

**Attempt 1:** pip wheel
```bash
pip install pyarrow
```

**Attempt 2:** If that fails, try a specific version known to have aarch64 wheels:
```bash
pip install pyarrow==15.0.0
```

**Attempt 3:** If source build fails, install build deps and retry:
```bash
sudo apt-get install -y cmake libboost-all-dev
pip install pyarrow
```

**Attempt 4:** conda-forge as last resort:
```bash
conda install -c conda-forge pyarrow
```

### Step 4: Install HuggingFace datasets

```bash
pip install datasets
```

If it tries to reinstall pyarrow and fails, pin the working version:
```bash
pip install "datasets>=2.14.0" --no-deps
pip install fsspec aiohttp multiprocess dill xxhash
```

### Step 5: Download NLTK data

```bash
python -c "import nltk; nltk.download('punkt'); nltk.download('punkt_tab')"
```

### Step 6: Full verification

Run this exact script. Every line must print without error:

```python
import sys
import platform
print(f"Python {platform.python_version()} on {platform.machine()}")

import torch
print(f"torch {torch.__version__} | CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"  GPU: {torch.cuda.get_device_name(0)}")
    print(f"  CUDA: {torch.version.cuda}")

import sklearn
print(f"scikit-learn {sklearn.__version__}")

import numpy
print(f"numpy {numpy.__version__}")

import pandas
print(f"pandas {pandas.__version__}")

import matplotlib
print(f"matplotlib {matplotlib.__version__}")
matplotlib.use('Agg')  # headless backend for Jetson

import seaborn
print(f"seaborn {seaborn.__version__}")

import nltk
print(f"nltk {nltk.__version__}")
nltk.data.find('tokenizers/punkt')
print("  punkt tokenizer: OK")

import datasets
print(f"datasets {datasets.__version__}")

import tqdm
print(f"tqdm {tqdm.__version__}")

import jupyterlab
print(f"jupyterlab {jupyterlab.__version__}")

print("\n=== ALL DEPENDENCIES VERIFIED ===")
```

If any import fails, read the traceback, install the missing piece, and re-run. Repeat until all 10 imports pass.

### Step 7: Verify matplotlib backend works headless

The Jetson may not have a display. Confirm matplotlib can render to file:

```python
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
fig, ax = plt.subplots()
ax.plot([1, 2, 3], [1, 4, 9])
fig.savefig('/tmp/test_plot.png')
print(f"Test plot saved: $(ls -la /tmp/test_plot.png)")
```

### Step 8: Report

Print a final summary:

```
Jetson Environment Setup: COMPLETE
Arch: aarch64
Python: <version>
Torch: <version> (CUDA: <yes/no>)
All 10 dependencies verified.
Matplotlib headless rendering: OK
Ready to run: claude --prompt prompts/00_meta_orchestrator.md --ultrathink
```

## Error Handling

- If a package install fails, DO NOT skip it. Read the error, fix it, retry.
- If a build needs system-level deps (apt packages), install them.
- If torch or sklearn get accidentally upgraded, STOP and alert the user. These must stay on the Jetson-specific builds.
- Never run `pip install --upgrade torch` or `pip install --upgrade scikit-learn`.
- If all else fails for a package, report exactly which package, which error, and what you tried.

**Execution:** `claude --prompt prompts/00a_jetson_setup.md --ultrathink`
