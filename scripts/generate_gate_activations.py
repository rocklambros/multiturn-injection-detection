"""Extract LSTM gate activations per turn for multi-turn conversations.

Manually unrolls the sequence LSTM to capture forget, input, cell, and output
gate values at each conversation turn.
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
from src.models.multi_turn import MultiTurnClassifier

vocab = load_vocab('models/vocab.json')
with open('data/synthetic/multiturn_test.json') as f:
    test_data = json.load(f)

attacks = [s for s in test_data if s['label'] == 1 and s['num_turns'] >= 4][:5]
benign = [s for s in test_data if s['label'] == 0 and s['num_turns'] >= 4][:3]
samples = attacks + benign

turns_list = [[turn['text'] for turn in seq['turns']] for seq in samples]
labels_list = [seq['label'] for seq in samples]
num_turns_list = [seq['num_turns'] for seq in samples]
injection_types = [seq.get('injection_type', 'none') for seq in samples]

token_ids, masks = encode_multiturn(vocab, turns_list, max_turns=10, max_len=256)

turn_encoder = GRUClassifier(vocab_size=len(vocab), embedding_dim=128, hidden_dim=64,
                             dropout_rate=0.3, dense_dim=32)
turn_encoder.load_state_dict(torch.load('models/iter4_gru.pt', map_location='cpu', weights_only=True))
turn_encoder.eval()

model = MultiTurnClassifier(
    turn_encoder=turn_encoder, turn_encoding_dim=32,
    hidden_dim=64, max_turns=10, dropout_rate=0.3
)
model.load_state_dict(torch.load('models/iter5_multiturn.pt', map_location='cpu', weights_only=True))
model.eval()

with torch.no_grad():
    turn_encs = []
    for t in range(10):
        enc = turn_encoder.encode(token_ids[:, t, :])
        turn_encs.append(enc)
    turn_encodings = torch.stack(turn_encs, dim=1)

lstm = model.sequence_lstm
W_ih = lstm.weight_ih_l0
W_hh = lstm.weight_hh_l0
b_ih = lstm.bias_ih_l0
b_hh = lstm.bias_hh_l0
H = 64

N = len(samples)
input_gates = np.zeros((N, 10, H))
forget_gates = np.zeros((N, 10, H))
cell_gates = np.zeros((N, 10, H))
output_gates = np.zeros((N, 10, H))

with torch.no_grad():
    h = torch.zeros(N, H)
    c = torch.zeros(N, H)

    for t in range(10):
        x_t = turn_encodings[:, t, :]
        gates = x_t @ W_ih.T + b_ih + h @ W_hh.T + b_hh

        i_gate = torch.sigmoid(gates[:, 0:H])
        f_gate = torch.sigmoid(gates[:, H:2*H])
        g_gate = torch.tanh(gates[:, 2*H:3*H])
        o_gate = torch.sigmoid(gates[:, 3*H:4*H])

        c = f_gate * c + i_gate * g_gate
        h = o_gate * torch.tanh(c)

        input_gates[:, t, :] = i_gate.numpy()
        forget_gates[:, t, :] = f_gate.numpy()
        cell_gates[:, t, :] = g_gate.numpy()
        output_gates[:, t, :] = o_gate.numpy()

turn_texts = []
for seq in samples:
    turn_texts.append([turn['text'][:120] for turn in seq['turns']])

np.savez('results/gate_activations.npz',
         input_gates=input_gates, forget_gates=forget_gates,
         cell_gates=cell_gates, output_gates=output_gates,
         labels=np.array(labels_list),
         num_turns=np.array(num_turns_list),
         injection_types=np.array(injection_types))

with open('results/gate_activation_texts.json', 'w') as f:
    json.dump(turn_texts, f, indent=2)

print("Saved results/gate_activations.npz and gate_activation_texts.json")
