"""Async LLM batch generation pipeline for intent-based multi-turn data.

Uses Anthropic API (Sonnet 4.6) to generate conversations from
injection intents + strategy descriptions.
"""

import asyncio
import json
import hashlib
import re
import time
from pathlib import Path

import anthropic


_JSON_FENCE_RE = re.compile(r'```(?:json)?\s*\n?(.*?)\n?\s*```', re.DOTALL)


def _extract_json(text):
    """Extract a JSON array from LLM response, handling markdown fences."""
    text = text.strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    match = _JSON_FENCE_RE.search(text)
    if match:
        return json.loads(match.group(1).strip())

    start = text.find('[')
    end = text.rfind(']')
    if start != -1 and end > start:
        return json.loads(text[start:end + 1])

    raise json.JSONDecodeError("No JSON array found", text, 0)


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
            turns = _extract_json(response_text)

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
            if attempt < max_retries - 1:
                await asyncio.sleep(2 ** attempt)
                continue
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


BENIGN_TOPICS = [
    "planning a weekend hiking trip",
    "choosing a new laptop for college",
    "learning to cook Italian food",
    "training for a half marathon",
    "starting a small vegetable garden",
    "picking a book for a reading club",
    "planning a budget-friendly vacation",
    "learning basic photography techniques",
    "setting up a home office workspace",
    "choosing a programming language to learn",
    "planning a birthday party for a friend",
    "understanding how to invest in index funds",
    "learning to play the guitar",
    "organizing a cluttered garage",
    "adopting a rescue dog",
    "switching to a plant-based diet",
    "building a personal website",
    "preparing for a job interview",
    "learning about home brewing beer",
    "improving public speaking skills",
    "choosing paint colors for a room",
    "understanding nutrition labels on food",
    "planning a road trip across the country",
    "setting up a fish tank",
    "learning basic car maintenance",
    "starting a podcast",
    "finding a good dentist in a new city",
    "understanding different types of tea",
    "helping a kid with math homework",
    "choosing between streaming services",
    "learning to sew and mend clothes",
    "setting up a home security system",
    "planning meals for the week",
    "choosing a new phone plan",
    "understanding how solar panels work",
    "learning about bird watching",
    "writing a best man speech",
    "picking a gym membership",
    "understanding different coffee brewing methods",
    "moving to a new apartment",
    "getting into board games",
    "planning a surprise anniversary dinner",
    "learning basic first aid",
    "choosing a mattress",
    "understanding pet insurance",
    "starting a journal or diary",
    "learning about wine pairing",
    "choosing a daycare for a toddler",
    "fixing a leaky faucet",
    "understanding credit scores",
    "planning a camping trip with kids",
    "learning calligraphy",
    "choosing house plants for low light",
    "preparing for a power outage",
    "understanding different yoga styles",
    "shopping for running shoes",
    "learning to make sourdough bread",
    "understanding health insurance options",
    "setting up parental controls on devices",
    "choosing a bicycle for commuting",
    "learning about composting at home",
    "understanding different types of mortgages",
    "planning a themed movie night",
    "choosing a water filter for the kitchen",
    "getting started with meditation",
    "understanding car insurance coverage options",
    "choosing a gift for a coworker",
    "learning to fold origami",
    "understanding recycling rules in a new area",
    "setting up automatic bill payments",
    "choosing a summer camp for children",
    "learning to tie different knots",
    "understanding airline frequent flyer programs",
    "planning a garage sale",
    "choosing a volunteer opportunity",
    "learning about local history",
    "setting up a rainwater collection system",
    "choosing a type of flooring for a kitchen",
    "understanding how to file taxes",
    "picking a paint-by-numbers kit",
    "learning about different types of mushrooms",
    "choosing a slow cooker recipe",
    "understanding pharmacy generic vs brand medicines",
    "planning a neighborhood block party",
    "choosing a backpack for school",
    "learning basic woodworking",
    "understanding how WiFi mesh systems work",
    "choosing a winter coat",
    "learning about cheese making",
    "planning a small wedding",
    "choosing fonts for a presentation",
    "understanding different laundry detergent types",
    "getting a passport for the first time",
    "learning about stargazing",
    "choosing a desk lamp",
    "understanding protein powder options",
    "planning a picnic",
    "choosing a reliable used car",
    "learning about essential oils safely",
    "understanding different types of rice",
]


BENIGN_DIFFICULTY_MODIFIERS = {
    "easy": {
        "instruction": (
            "The user asks simple, direct questions. "
            "Language is casual and straightforward."
        ),
        "temperature": 0.7,
        "include_assistant": False,
    },
    "medium": {
        "instruction": (
            "The user's language should sound natural and varied. "
            "The conversation should feel authentic with follow-up questions."
        ),
        "temperature": 0.7,
        "include_assistant": False,
    },
    "hard": {
        "instruction": (
            "Generate a realistic dialogue between a user and an AI assistant. "
            "Include plausible AI assistant responses. Both parties should "
            "contribute meaningfully to the conversation."
        ),
        "temperature": 0.8,
        "include_assistant": True,
    },
    "adversarial": {
        "instruction": (
            "Generate a realistic dialogue with sophisticated, varied language. "
            "Include AI assistant responses. The conversation should read as "
            "natural and substantive."
        ),
        "temperature": 0.9,
        "include_assistant": True,
    },
}


def build_benign_prompt(topic, difficulty, num_turns):
    """Build the LLM prompt for generating a benign conversation."""
    diff_config = BENIGN_DIFFICULTY_MODIFIERS[difficulty]

    if diff_config["include_assistant"]:
        turn_instruction = (
            f"The conversation should have exactly {num_turns} user messages "
            f"and {num_turns} assistant responses (alternating)."
        )
    else:
        turn_instruction = (
            f"The conversation should have exactly {num_turns} user messages."
        )

    return f"""You are generating synthetic training data for a prompt injection detection system.

Generate a natural multi-turn conversation where a user seeks help with {topic}. The user should ask follow-up questions that build on previous responses. The conversation should stay on topic and be completely benign.

{diff_config["instruction"]}

{turn_instruction}

Output the conversation as a JSON array of objects, each with "role" (either "user" or "assistant") and "text" fields.

Topic: {topic}

Output ONLY the JSON array, no other text."""


async def generate_benign_one(client, topic, difficulty, num_turns,
                              model="claude-4-sonnet-20250514"):
    """Generate a single benign conversation via the Anthropic API."""
    prompt = build_benign_prompt(topic, difficulty, num_turns)
    diff_config = BENIGN_DIFFICULTY_MODIFIERS[difficulty]

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
            turns = _extract_json(response_text)

            return {
                "turns": turns,
                "label": 0,
                "topic": topic,
                "strategy": "benign",
                "difficulty": difficulty,
                "generation_method": "llm_benign",
                "model": model,
                "prompt_hash": hashlib.sha256(prompt.encode()).hexdigest(),
                "response_hash": hashlib.sha256(response_text.encode()).hexdigest(),
                "timestamp": time.time(),
                "input_tokens": response.usage.input_tokens,
                "output_tokens": response.usage.output_tokens,
            }
        except (json.JSONDecodeError, IndexError, KeyError) as e:
            if attempt < max_retries - 1:
                await asyncio.sleep(2 ** attempt)
                continue
            return {"error": str(e), "topic": topic, "difficulty": difficulty}
        except anthropic.RateLimitError:
            if attempt < max_retries - 1:
                await asyncio.sleep(2 ** attempt * 5)
                continue
            return {"error": "rate_limit_exceeded", "topic": topic,
                    "difficulty": difficulty}
        except anthropic.APIError as e:
            if attempt < max_retries - 1 and e.status_code in (429, 500, 502, 503, 529):
                await asyncio.sleep(2 ** attempt * 5)
                continue
            return {"error": f"api_error_{e.status_code}: {e.message}",
                    "topic": topic, "difficulty": difficulty}
    return {"error": "max_retries_exceeded", "topic": topic,
            "difficulty": difficulty}


async def generate_benign_batch(count, difficulty, num_turns_range,
                                output_path, model="claude-4-sonnet-20250514",
                                max_concurrent=50):
    """Generate a batch of benign conversations asynchronously."""
    import random

    client = anthropic.AsyncAnthropic()
    semaphore = asyncio.Semaphore(max_concurrent)
    errors = 0

    async def limited_generate(topic):
        nonlocal errors
        async with semaphore:
            num_turns = random.randint(*num_turns_range)
            result = await generate_benign_one(
                client, topic, difficulty, num_turns, model,
            )
            if "error" in result:
                errors += 1
            return result

    tasks = []
    for i in range(count):
        topic = BENIGN_TOPICS[i % len(BENIGN_TOPICS)]
        tasks.append(limited_generate(topic))

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
                print(f"  [benign-{difficulty}] {completed}/{total} "
                      f"({errors} errors)")

    stats = {
        "total": total,
        "completed": total - errors,
        "errors": errors,
        "difficulty": difficulty,
        "output_path": str(output_path),
    }
    print(f"  [benign-{difficulty}] Done: {stats['completed']}/{total}, "
          f"{errors} errors")
    return stats


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
