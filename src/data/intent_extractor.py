"""Extract attack intents from raw injection texts.

Uses the first sentence of each injection text as a natural-language
intent description. This preserves the diversity of the source corpus
(~26K unique texts -> ~26K unique intents) instead of collapsing to
a handful of regex categories. The LLM receives the intent (not the
raw text), so it generates original dialogue rather than parroting.
"""

import re
from collections import Counter


_SENT_BOUNDARY = re.compile(r'(?<=[.!?])\s+')

_MAX_INTENT_WORDS = 20
_MIN_INTENT_WORDS = 3

FALLBACK_INTENT = "achieve an adversarial goal through prompt manipulation"


def extract_intent(text):
    """Extract a concise intent from an injection text's first sentence.

    Strategy: take the first sentence, truncate to 20 words max.
    If the first sentence is too short (< 3 words), take the first
    two sentences. Falls back to a generic intent only for empty input.

    Args:
        text: Raw injection text string.

    Returns:
        String: natural-language intent description.
    """
    text = text.strip()
    if not text:
        return FALLBACK_INTENT

    sentences = _SENT_BOUNDARY.split(text, maxsplit=2)
    intent = sentences[0].strip()

    if len(intent.split()) < _MIN_INTENT_WORDS and len(sentences) > 1:
        intent = f"{sentences[0].strip()} {sentences[1].strip()}"

    words = intent.split()
    if len(words) > _MAX_INTENT_WORDS:
        intent = " ".join(words[:_MAX_INTENT_WORDS])

    return intent


def extract_intents_batch(texts):
    """Extract intents for a batch of injection texts.

    Args:
        texts: List of injection text strings.

    Returns:
        List of intent strings (same length as input).
    """
    return [extract_intent(t) for t in texts]


def deduplicate_intents(intents):
    """Count unique intents.

    Args:
        intents: List of intent strings.

    Returns:
        List of (intent, count) tuples sorted by frequency descending.
    """
    return Counter(intents).most_common()
