---
title: "Red Team Strategies"
document_id: ""
version: "1"
date: "2026-05-16"
status: "draft"
document_type: ""
content_domain: []
authors: []
organization: "promptfoo.dev"
generation_metadata:
  authored_by: "unknown"
content_hash: "fefaafee99b28e4820e3a48072466a93a2d1ffa64b5a5e0f0294f0fd9043086b"
token_estimate: 1854
recommended_chunk_level: "h2"
abstract_for_rag: "Strategies are attack techniques that systematically probe LLM applications for vulnerabilities. While plugins generate adversarial inputs, strategies determine how these inputs are delivered to maximize attack success rates."
source_url: "https://www.promptfoo.dev/docs/red-team/strategies/"
type: "html"
extracted_via: "trafilatura"
word_count: 722
---

# Red Team Strategies

Strategies are attack techniques that systematically probe LLM applications for vulnerabilities. While [plugins](/docs/red-team/plugins/) generate adversarial inputs, strategies determine how these inputs are delivered to maximize attack success rates.

## Recommended Strategies[](#recommended-strategies)

Most users only need two strategies for comprehensive coverage. These agentic methods provide the highest attack success rates across use cases.

### Meta Agent: Best for Single-Turn[](#meta-agent-best-for-single-turn)

The [Meta Agent](/docs/red-team/strategies/meta/) dynamically builds an attack taxonomy and learns from attack history to optimize bypass attempts. It learns which attack types work best against your specific target.

### Hydra Multi-Turn: Best for Multi-Turn[](#hydra-multi-turn-best-for-multi-turn)

[Hydra](/docs/red-team/strategies/hydra/) runs adaptive multi-turn conversations with persistent scan-wide attacker memory. It can either replay the full transcript to a stateless target or use target-managed session memory for applications that persist prior turns.

### Quick Start[](#quick-start)

For most applications, this configuration provides comprehensive red team coverage:

strategies:

- jailbreak:meta # Single-turn agentic attacks

- jailbreak:hydra # Multi-turn adaptive conversations

## All Strategies[](#all-strategies)

| Category | Strategy | Description | Details | Cost | ASR Increase* |
|---|---|---|---|---|---|
| Static (Single-Turn) |

[Base64](/docs/red-team/strategies/base64/)[Basic](/docs/red-team/strategies/basic/)[camelCase](/docs/red-team/strategies/other-encodings/#camelcase)[Emoji Smuggling](/docs/red-team/strategies/other-encodings/#emoji-encoding)[Hex](/docs/red-team/strategies/hex/)[Homoglyph](/docs/red-team/strategies/homoglyph/)[Image Encoding](/docs/red-team/strategies/image/)[Jailbreak Templates](/docs/red-team/strategies/jailbreak-templates/)[Leetspeak](/docs/red-team/strategies/leetspeak/)[Morse Code](/docs/red-team/strategies/other-encodings/#morse-code)[Pig Latin](/docs/red-team/strategies/other-encodings/#pig-latin)[ROT13](/docs/red-team/strategies/rot13/)[Video Encoding](/docs/red-team/strategies/video/)[Authoritative Markup Injection](/docs/red-team/strategies/authoritative-markup-injection/)[Best-of-N](/docs/red-team/strategies/best-of-n/)[Citation](/docs/red-team/strategies/citation/)[Composite JailbreaksRecommended](/docs/red-team/strategies/composite-jailbreaks/)[GCG](/docs/red-team/strategies/gcg/)[JailbreakRecommended](/docs/red-team/strategies/iterative/)[Likert-based Jailbreaks](/docs/red-team/strategies/likert/)[Math Prompt](/docs/red-team/strategies/math-prompt/)[Meta-Agent JailbreaksRecommended](/docs/red-team/strategies/meta/)[Tree-based](/docs/red-team/strategies/tree/)[Crescendo](/docs/red-team/strategies/multi-turn/)[GOAT](/docs/red-team/strategies/goat/)[Hydra Multi-turn](/docs/red-team/strategies/hydra/)[Mischievous User](/docs/red-team/strategies/mischievous-user/)[Retry](/docs/red-team/strategies/retry/)[Custom Strategies](/docs/red-team/strategies/custom/)[Custom Strategy](/docs/red-team/strategies/custom-strategy/)[Layer](/docs/red-team/strategies/layer/)*🌐 indicates that strategy uses remote inference in Promptfoo Community edition*

## Strategy Categories[](#strategy-categories)

### Static Strategies[](#static-strategies)

Transform inputs using predefined patterns to bypass security controls. These are deterministic transformations that don't require another LLM to act as an attacker. Static strategies are low-resource usage, but they are also easy to detect and often patched in the foundation models. For example, the `base64`

strategy encodes inputs as base64 to bypass guardrails and other content filters. `jailbreak-templates`

wraps the payload in known jailbreak templates like DAN or Skeleton Key.

### Dynamic Strategies[](#dynamic-strategies)

Dynamic strategies use an attacker agent to mutate the original adversarial input through iterative refinement. These strategies make multiple calls to both an attacker model and your target model to determine the most effective attack vector. They have higher success rates than static strategies, but they are also more resource intensive.

By default, dynamic strategies like `jailbreak`

will:

- Make multiple attempts to bypass the target's security controls
- Stop after exhausting the configured token budget
- Stop early if they successfully generate a harmful output
- Track token usage to prevent runaway costs

### Multi-turn Strategies[](#multi-turn-strategies)

Multi-turn strategies use an attacker agent to coerce the target over multiple conversation turns. They are particularly effective against stateful applications where they can convince the target to act against its purpose over time. Multi-turn strategies are more resource intensive than single-turn strategies, but they have the highest success rates.

### Indirect Prompt Injection Strategies[](#indirect-prompt-injection-strategies)

Indirect prompt injection strategies test whether AI agents can be manipulated through malicious instructions embedded in external content they consume. These strategies generate realistic attack surfaces containing hidden payloads to test both data exfiltration and behavior manipulation. Currently available: [ indirect-web-pwn](/docs/red-team/strategies/indirect-web-pwn/) for web browsing agents.

### Regression Strategies[](#regression-strategies)

Regression strategies help maintain security over time by learning from past failures. For example, the `retry`

strategy automatically incorporates previously failed test cases into your test suite, creating a form of regression testing for LLM behaviors.

All single-turn strategies can be applied to multi-turn applications, but multi-turn strategies require a stateful application.

## Configuration[](#configuration)

### Basic Configuration[](#basic-configuration)

strategies:

- jailbreak:meta # string syntax

- id: jailbreak:composite # object syntax

### Plugin Targeting[](#plugin-targeting)

Strategies can be applied to specific plugins or the entire test suite. By default, strategies are applied to all plugins. You can override this by specifying the `plugins`

option in the strategy which will only apply the strategy to the specified plugins.

strategies:

- id: jailbreak:tree

config:

plugins:

- harmful:hate

### Layered Strategies[](#layered-strategies)

Chain strategies in order with the `layer`

strategy. This is useful when you want to apply a transformation first, then another technique:

strategies:

- id: layer

config:

steps:

- base64 # First encode as base64

- rot13 # Then apply ROT13

Notes:

- Each step respects plugin targeting and exclusions.
- Only the final step's outputs are kept.
- Transformations are applied in the order specified.

### Custom Strategies[](#custom-strategies)

For advanced use cases, you can create custom strategies. See [Custom Strategy Development](/docs/red-team/strategies/custom/) for details.

## Related Concepts[](#related-concepts)

[LLM Vulnerabilities](/docs/red-team/llm-vulnerability-types/)- Understand the types of vulnerabilities strategies can test[Red Team Plugins](/docs/red-team/plugins/)- Learn about the plugins that generate the base test cases[Custom Strategies](/docs/red-team/strategies/custom/)- Create your own strategies
