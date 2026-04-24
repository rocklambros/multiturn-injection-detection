"""Generate t-SNE embeddings at 3 GRU encoder stages for manifold visualization."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import torch
import pandas as pd
from sklearn.manifold import TSNE
from src.utils.seed import set_global_seed
set_global_seed(42)

from src.utils.tokenizer import load_vocab, encode_texts
from src.models.single_turn import GRUClassifier

vocab = load_vocab('models/vocab.json')
test_df = pd.read_csv('data/processed/single_turn_test.csv')

n_per_class = 1000
benign = test_df[test_df['label'] == 0].sample(n=n_per_class, random_state=42)
inject = test_df[test_df['label'] == 1].sample(n=n_per_class, random_state=42)
subset = pd.concat([benign, inject]).reset_index(drop=True)
labels = subset['label'].values

token_ids = encode_texts(vocab, subset['text'].tolist(), max_len=256)

model = GRUClassifier(vocab_size=len(vocab), embedding_dim=128, hidden_dim=64,
                      dropout_rate=0.3, dense_dim=32)
model.load_state_dict(torch.load('models/iter4_gru.pt', map_location='cpu', weights_only=True))
model.eval()

with torch.no_grad():
    embedded = model.embedding(token_ids)
    pad_mask = (token_ids != 0).float().unsqueeze(-1)
    rep_embed = (embedded * pad_mask).sum(dim=1) / pad_mask.sum(dim=1).clamp(min=1)

    _, hidden = model.gru(embedded)
    rep_gru = torch.cat((hidden[0], hidden[1]), dim=1)

    rep_dense = torch.relu(model.fc1(rep_gru))

print("t-SNE stage 1/3: embedding layer (128-dim)...")
tsne1 = TSNE(n_components=2, random_state=42, perplexity=30).fit_transform(rep_embed.numpy())
print("t-SNE stage 2/3: GRU hidden state (128-dim)...")
tsne2 = TSNE(n_components=2, random_state=42, perplexity=30).fit_transform(rep_gru.numpy())
print("t-SNE stage 3/3: dense encoding (32-dim)...")
tsne3 = TSNE(n_components=2, random_state=42, perplexity=30).fit_transform(rep_dense.numpy())

np.savez('results/embedding_tsne.npz',
         tsne_embedding=tsne1, tsne_gru=tsne2, tsne_dense=tsne3,
         labels=labels)
print("Saved results/embedding_tsne.npz")
