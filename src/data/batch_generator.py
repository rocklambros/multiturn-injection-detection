"""Async LLM batch generation pipeline for intent-based multi-turn data.

Uses Anthropic API (Sonnet 4.6) to generate conversations from
injection intents + strategy descriptions.
"""

import asyncio
import json
import hashlib
import time
from pathlib import Path

import anthropic


STRATEGY_PROMPTS = {
    "fragment_distributed": (
        "Generate a multi-turn conversation where a user gradually works toward "
        "{intent}. Distribute hints of the goal across multiple turns, interleaved "
        "with unrelated benign conversation. No single turn should reveal the full intent."
    ),
    "gradual_escalation": (
        "Generate a conversation where a user starts with innocent questions and "
        "progressively escalates toward {intent}. Early turns establish rapport/context. "
        "Each successive turn pushes slightly closer to the goal. The escalation should "
        "feel natural."
    ),
    "context_priming": (
        "Generate a conversation where a user first establishes a persona or context "
        "(e.g., security researcher, system administrator) in early turns. Middle turns "
        "are fully benign. Final turns leverage the established context to pursue {intent}."
    ),
    "instruction_layering": (
        "Generate a conversation where a user sets constraints and conditions across "
        "multiple turns that cumulatively enable {intent}. Each turn adds a seemingly "
        "reasonable constraint. The cumulative effect of all constraints creates the attack."
    ),
}

DIFFICULTY_MODIFIERS = {
    "easy": {
        "instruction": (
            "The user's language may include security-related vocabulary. "
            "Individual turns can be somewhat suggestive of the intent."
        ),
        "temperature": 0.7,
        "include_assistant": False,
    },
    "medium": {
        "instruction": (
            "The user's language should sound natural and avoid obvious security keywords. "
            "The intent should only become clear when reading multiple turns together."
        ),
        "temperature": 0.7,
        "include_assistant": False,
    },
    "hard": {
        "instruction": (
            "Generate a realistic dialogue between a user and an AI assistant. "
            "Each user message must appear completely innocent in isolation. "
            "The attack intent should only be visible when reading all user messages "
            "in sequence. Include plausible AI assistant responses."
        ),
        "temperature": 0.8,
        "include_assistant": True,
    },
    "adversarial": {
        "instruction": (
            "Generate a dialogue designed to evade a text classifier trained on known "
            "injection patterns. Every user message must read as a normal, innocent "
            "request. Use synonyms, indirect phrasing, and plausible deniability. "
            "Include AI assistant responses. A security-trained human reading any "
            "single message should see nothing suspicious."
        ),
        "temperature": 0.9,
        "include_assistant": True,
    },
}


def build_prompt(intent, strategy, difficulty, num_turns):
    """Build the LLM generation prompt.

    Args:
        intent: Attack intent string (5-15 words).
        strategy: One of the 4 strategy names.
        difficulty: One of easy, medium, hard, adversarial.
        num_turns: Target number of user turns.

    Returns:
        String prompt for the LLM.
    """
    strategy_text = STRATEGY_PROMPTS[strategy].format(intent=intent)
    diff_config = DIFFICULTY_MODIFIERS[difficulty]

    if diff_config["include_assistant"]:
        turn_instruction = f"The conversation should have exactly {num_turns} user messages and {num_turns} assistant responses (alternating)."
    else:
        turn_instruction = f"The conversation should have exactly {num_turns} user messages."

    prompt = f"""You are generating synthetic training data for a prompt injection detection system.

{strategy_text}

{diff_config["instruction"]}

{turn_instruction}

Output the conversation as a JSON array of objects, each with "role" (either "user" or "assistant") and "text" fields.

Attack intent: {intent}

Output ONLY the JSON array, no other text."""

    return prompt


async def generate_one(client, intent, strategy, difficulty, num_turns, model="claude-4-sonnet-20250514"):
    """Generate a single conversation via the Anthropic API.

    Args:
        client: anthropic.AsyncAnthropic instance.
        intent: Attack intent string.
        strategy: Strategy name.
        difficulty: Difficulty tier.
        num_turns: Target user turn count.
        model: Model ID.

    Returns:
        Dict with generated conversation, metadata, and hashes.
    """
    prompt = build_prompt(intent, strategy, difficulty, num_turns)
    diff_config = DIFFICULTY_MODIFIERS[difficulty]

    max_retries = 3
    for attempt in range(max_retries):
        try:
            response = await client.messages.create(
                model=model,
                max_tokens=4096,
                temperature=diff_config["temperature"],
                messages=[{"role": "user", "content": prompt}],
            )

            response_text = response.content[0].text
            turns = json.loads(response_text)

            return {
                "turns": turns,
                "label": 1,
                "intent": intent,
                "strategy": strategy,
                "difficulty": difficulty,
                "generation_method": "llm_intent",
                "model": model,
                "prompt_hash": hashlib.sha256(prompt.encode()).hexdigest(),
                "response_hash": hashlib.sha256(response_text.encode()).hexdigest(),
                "timestamp": time.time(),
                "input_tokens": response.usage.input_tokens,
                "output_tokens": response.usage.output_tokens,
            }
        except (json.JSONDecodeError, IndexError, KeyError) as e:
            return {
                "error": str(e),
                "intent": intent,
                "strategy": strategy,
                "difficulty": difficulty,
            }
        except anthropic.RateLimitError:
            if attempt < max_retries - 1:
                await asyncio.sleep(2 ** attempt * 5)
                continue
            return {"error": "rate_limit_exceeded", "intent": intent,
                    "strategy": strategy, "difficulty": difficulty}
        except anthropic.APIError as e:
            if attempt < max_retries - 1 and e.status_code in (429, 500, 502, 503, 529):
                await asyncio.sleep(2 ** attempt * 5)
                continue
            return {"error": f"api_error_{e.status_code}: {e.message}", "intent": intent,
                    "strategy": strategy, "difficulty": difficulty}
    return {"error": "max_retries_exceeded", "intent": intent,
            "strategy": strategy, "difficulty": difficulty}


async def generate_batch(intents, strategies, difficulty, num_turns_range,
                         output_path, model="claude-4-sonnet-20250514",
                         max_concurrent=50):
    """Generate a batch of conversations asynchronously.

    Args:
        intents: List of intent strings.
        strategies: Dict mapping strategy name to count.
        difficulty: Difficulty tier name.
        num_turns_range: Tuple (min_turns, max_turns).
        output_path: Path to write output JSONL.
        model: Model ID.
        max_concurrent: Max concurrent API requests.

    Returns:
        Dict with generation statistics.
    """
    import random

    client = anthropic.AsyncAnthropic()
    semaphore = asyncio.Semaphore(max_concurrent)
    errors = 0

    async def limited_generate(intent, strategy):
        nonlocal errors
        async with semaphore:
            num_turns = random.randint(*num_turns_range)
            result = await generate_one(client, intent, strategy, difficulty, num_turns, model)
            if "error" in result:
                errors += 1
            return result

    tasks = []
    intent_idx = 0
    for strategy, count in strategies.items():
        for _ in range(count):
            tasks.append(limited_generate(intents[intent_idx % len(intents)], strategy))
            intent_idx += 1

    completed = 0
    total = len(tasks)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w") as f:
        for coro in asyncio.as_completed(tasks):
            result = await coro
            f.write(json.dumps(result) + "\n")
            completed += 1
            if completed % 100 == 0:
                print(f"  [{difficulty}] {completed}/{total} ({errors} errors)")

    stats = {
        "total": total,
        "completed": total - errors,
        "errors": errors,
        "difficulty": difficulty,
        "output_path": str(output_path),
    }
    print(f"  [{difficulty}] Done: {stats['completed']}/{total}, {errors} errors")
    return stats
