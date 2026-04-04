"""Download GloVe 6B 100d and build embedding matrix aligned with project vocabulary.

GloVe source: Stanford NLP (https://nlp.stanford.edu/data/glove.6B.zip)
"""

from src.utils.seed import set_global_seed
set_global_seed(42)

import os
import zipfile
from pathlib import Path
from urllib.request import urlretrieve

import numpy as np

from src.utils.tokenizer import load_vocab


GLOVE_URL = "https://nlp.stanford.edu/data/glove.6B.zip"
GLOVE_DIR = Path("data/embeddings")
GLOVE_FILE = GLOVE_DIR / "glove.6B.100d.txt"
MATRIX_FILE = GLOVE_DIR / "embedding_matrix.npy"
EMBEDDING_DIM = 100

SECURITY_TERMS = ["ignore", "override", "jailbreak", "bypass", "inject", "prompt",
                  "system", "admin", "sudo", "hack", "exploit", "credentials"]


def download_glove():
    """Download and extract GloVe 6B embeddings.

    Side effects:
        Downloads glove.6B.zip to data/embeddings/, extracts 100d file.
    """
    os.makedirs(GLOVE_DIR, exist_ok=True)

    if GLOVE_FILE.exists():
        print(f"GloVe already exists at {GLOVE_FILE}")
        return

    zip_path = GLOVE_DIR / "glove.6B.zip"
    if not zip_path.exists():
        print(f"Downloading GloVe from {GLOVE_URL}...")
        urlretrieve(GLOVE_URL, zip_path)
        print(f"Downloaded to {zip_path}")

    print("Extracting glove.6B.100d.txt...")
    with zipfile.ZipFile(zip_path, "r") as z:
        # Extract only the 100d file
        for name in z.namelist():
            if "100d" in name:
                z.extract(name, GLOVE_DIR)
                extracted = GLOVE_DIR / name
                if extracted != GLOVE_FILE:
                    os.rename(extracted, GLOVE_FILE)
                break

    print(f"Extracted to {GLOVE_FILE}")

    # Clean up zip to save space
    if zip_path.exists():
        os.remove(zip_path)
        print("Removed zip file to save space")


def load_glove_vectors(glove_path):
    """Load GloVe vectors into a dict.

    Args:
        glove_path: Path to glove.6B.100d.txt.

    Returns:
        Dict mapping word -> numpy array of shape (100,).

    Side effects:
        Prints loading progress.
    """
    vectors = {}
    print(f"Loading GloVe vectors from {glove_path}...")
    with open(glove_path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            parts = line.rstrip().split(" ")
            word = parts[0]
            vec = np.array(parts[1:], dtype=np.float32)
            if vec.shape[0] == EMBEDDING_DIM:
                vectors[word] = vec
            if (i + 1) % 100000 == 0:
                print(f"  Loaded {i+1} vectors...")

    print(f"Total GloVe vectors: {len(vectors)}")
    return vectors


def build_embedding_matrix(vocab, glove_vectors):
    """Build embedding matrix aligned with project vocabulary.

    Args:
        vocab: Word-to-index dict from tokenizer.
        glove_vectors: Dict mapping word -> numpy array.

    Returns:
        numpy array of shape (vocab_size, EMBEDDING_DIM).

    Side effects:
        Prints coverage statistics and missing security terms.
    """
    vocab_size = len(vocab)
    matrix = np.zeros((vocab_size, EMBEDDING_DIM), dtype=np.float32)

    # OOV row = mean of all GloVe vectors
    all_vecs = list(glove_vectors.values())
    mean_vec = np.mean(all_vecs, axis=0)
    matrix[1] = mean_vec  # Index 1 = OOV

    # Row 0 (PAD) stays zeros

    found = 0
    missing = 0
    for word, idx in vocab.items():
        if idx < 2:  # Skip PAD and OOV
            continue
        if word in glove_vectors:
            matrix[idx] = glove_vectors[word]
            found += 1
        else:
            matrix[idx] = mean_vec
            missing += 1

    total = found + missing
    coverage = found / total * 100 if total > 0 else 0
    print(f"\nEmbedding matrix shape: {matrix.shape}")
    print(f"Coverage: {found}/{total} = {coverage:.1f}%")
    print(f"Missing: {missing}/{total} = {100-coverage:.1f}%")
    print(f"PAD row sum (should be 0): {matrix[0].sum()}")

    # Check security terms
    print(f"\nSecurity term coverage:")
    missing_security = []
    for term in SECURITY_TERMS:
        in_vocab = term in vocab
        in_glove = term in glove_vectors
        status = "OK" if (in_vocab and in_glove) else ("in vocab but not GloVe" if in_vocab else "not in vocab")
        print(f"  {term}: {status}")
        if in_vocab and not in_glove:
            missing_security.append(term)

    if missing_security:
        print(f"\nMissing security terms in GloVe: {missing_security}")

    return matrix


def run():
    """Download GloVe, load vocab, build and save embedding matrix.

    Side effects:
        Downloads GloVe if not present.
        Saves embedding_matrix.npy to data/embeddings/.
    """
    download_glove()
    glove_vectors = load_glove_vectors(GLOVE_FILE)
    vocab = load_vocab("models/vocab.json")
    matrix = build_embedding_matrix(vocab, glove_vectors)

    np.save(MATRIX_FILE, matrix)
    print(f"\nEmbedding matrix saved to {MATRIX_FILE}")


if __name__ == "__main__":
    run()
