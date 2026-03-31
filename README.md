# Multi-Turn Distributed Prompt Injection Detection

Detects prompt injection attacks spread across multiple conversation turns where each turn appears benign in isolation.

Single-turn prompt injection detection is solved (ProtectAI DeBERTa, 99%+ accuracy). Multi-turn distributed injection detection has no published solution. This project builds the first.

## Architecture

Two-level LSTM:
- Level 1 (turn encoder): Bidirectional LSTM that encodes each conversation turn into a fixed-length vector
- Level 2 (sequence classifier): LSTM with attention that processes the sequence of turn vectors to classify the full conversation

## Status

PRD complete. Implementation in progress.

See [PRD.md](PRD.md) for full specifications.
