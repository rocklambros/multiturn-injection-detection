"""Generate 3D loss landscape data for the multi-turn attention model.

Uses the Li et al. (2018) technique: sample two random directions in parameter
space, apply filter-wise normalization, and scan a 2D grid of perturbations.
Turn encoder outputs are pre-computed (frozen), so only the small sequence
LSTM + attention + FC layers are perturbed (~29K params).
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import torch
import json
from src.utils.seed import set_global_seed
set_global_seed(42)

from src.utils.tokenizer import load_vocab, encode_multiturn
from src.models.single_turn import GRUClassifier
from src.models.attention import MultiTurnAttentionClassifier

vocab = load_vocab('models/vocab.json')
with open('data/synthetic/multiturn_test.json') as f:
    test_data = json.load(f)

subset = test_data[:150]
turns_list = [[turn['text'] for turn in seq['turns']] for seq in subset]
labels_list = [seq['label'] for seq in subset]

token_ids, masks = encode_multiturn(vocab, turns_list, max_turns=10, max_len=256)
labels = torch.FloatTensor(labels_list).unsqueeze(1)

turn_encoder = GRUClassifier(vocab_size=len(vocab), embedding_dim=128, hidden_dim=64,
                             dropout_rate=0.3, dense_dim=32)
turn_encoder.load_state_dict(torch.load('models/iter4_gru.pt', map_location='cpu', weights_only=True))
turn_encoder.eval()

model = MultiTurnAttentionClassifier(
    turn_encoder=turn_encoder, turn_encoding_dim=32,
    hidden_dim=64, attention_dim=32, max_turns=10, dropout_rate=0.3
)
model.load_state_dict(torch.load('models/iter6_attention.pt', map_location='cpu', weights_only=True))
model.eval()

print("Pre-computing frozen turn encodings...")
with torch.no_grad():
    turn_encs = []
    for t in range(10):
        enc = turn_encoder.encode(token_ids[:, t, :])
        turn_encs.append(enc)
    turn_encodings = torch.stack(turn_encs, dim=1)

trainable_params = []
for name, param in model.named_parameters():
    if param.requires_grad:
        trainable_params.append(param.data.clone())

theta_star = torch.cat([p.flatten() for p in trainable_params])
total_params = theta_star.numel()
print(f"Trainable parameters: {total_params:,}")


def filter_normalize(direction_flat, params_list):
    result = []
    offset = 0
    for p in params_list:
        n = p.numel()
        d = direction_flat[offset:offset + n].reshape(p.shape)
        if p.dim() >= 2:
            for i in range(p.shape[0]):
                d_norm = d[i].norm()
                if d_norm > 0:
                    d[i] = d[i] * (p[i].norm() / d_norm)
        else:
            d_norm = d.norm()
            if d_norm > 0:
                d = d * (p.norm() / d_norm)
        result.append(d.flatten())
        offset += n
    return torch.cat(result)


torch.manual_seed(42)
delta = filter_normalize(torch.randn(total_params), trainable_params)
eta = filter_normalize(torch.randn(total_params), trainable_params)

criterion = torch.nn.BCELoss()


def set_params(model, theta):
    offset = 0
    for name, param in model.named_parameters():
        if param.requires_grad:
            n = param.numel()
            param.data.copy_(theta[offset:offset + n].reshape(param.shape))
            offset += n


def compute_loss(model, turn_encodings, masks, labels):
    with torch.no_grad():
        lstm_out, _ = model.sequence_lstm(turn_encodings)
        context, _ = model.attention(lstm_out, masks)
        out = torch.relu(model.fc1(context))
        prob = torch.sigmoid(model.fc2(out))
        return criterion(prob, labels).item()


grid_size = 25
alphas = np.linspace(-1.0, 1.0, grid_size)
betas = np.linspace(-1.0, 1.0, grid_size)
loss_grid = np.zeros((grid_size, grid_size))

print(f"Scanning {grid_size}x{grid_size} = {grid_size**2} grid points...")
for i, alpha in enumerate(alphas):
    for j, beta in enumerate(betas):
        theta = theta_star + alpha * delta + beta * eta
        set_params(model, theta)
        loss_grid[i, j] = compute_loss(model, turn_encodings, masks, labels)
    if (i + 1) % 5 == 0:
        print(f"  Row {i + 1}/{grid_size}")

set_params(model, theta_star)

with open('results/iter6_attention/training_history.json') as f:
    history = json.load(f)

np.savez('results/loss_landscape.npz',
         alphas=alphas, betas=betas, loss_grid=loss_grid,
         train_loss=np.array(history['train_loss']),
         val_loss=np.array(history['val_loss']))
print("Saved results/loss_landscape.npz")
