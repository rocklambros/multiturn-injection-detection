"""Strip AI assistant responses from full-dialogue sequences.

Hard and Adversarial tiers include AI responses during generation
(for realism) but they must be removed before model training.
The classifier sees user turns only.
"""


def strip_assistant_responses(sequence):
    """Remove assistant turns from a conversation sequence.

    Args:
        sequence: Dict with "turns" list. Each turn has "role" and "text".

    Returns:
        New dict with only user turns, re-indexed. Original stored
        in "full_dialogue" key for reproducibility.
    """
    full_dialogue = sequence.get("turns", [])
    user_turns = []
    for turn in full_dialogue:
        if turn.get("role", "user") == "user":
            user_turns.append({
                "turn_index": len(user_turns),
                "text": turn["text"],
                "role": "user",
            })

    return {
        **{k: v for k, v in sequence.items() if k != "turns"},
        "turns": user_turns,
        "full_dialogue": full_dialogue,
        "user_turn_count": len(user_turns),
    }


def strip_responses(turns):
    """Strip assistant turns from a raw turn list.

    Args:
        turns: List of turn dicts with "role" and "text".

    Returns:
        List of user-only turn dicts.
    """
    return [t for t in turns if t.get("role", "user") == "user"]


def strip_batch(sequences):
    """Strip assistant responses from a batch of sequences.

    Args:
        sequences: List of sequence dicts.

    Returns:
        List of stripped sequence dicts.
    """
    return [strip_assistant_responses(seq) for seq in sequences]
