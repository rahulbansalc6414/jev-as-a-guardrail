# Jev vs Haiku: Input Guardrail Benchmark

Benchmarking **TypeSafe Jev** (a lightweight judgment model) against **Claude Haiku 4.5** (a general-purpose LLM) as input safety classifiers for enterprise AI systems.

## What This Tests

Both systems evaluate user inputs for:
- Prompt injection attempts
- Jailbreak attacks
- Harmful intent

The benchmark measures detection rate, false positive rate, latency, and cost across three categories of prompts.

## Datasets

| Category | Dataset | Description |
|----------|---------|-------------|
| **Attack** (800) | [Lshafii/Jailbreak-PromptBank](https://huggingface.co/datasets/Lshafii/Jailbreak-PromptBank) | Curated jailbreak prompts from AdvBench, HarmBench, StrongReject, GPT-Fuzzer, and others |
| **Evasion** (200) | [Mindgard/evaded-prompt-injection-and-jailbreak-samples](https://huggingface.co/datasets/Mindgard/evaded-prompt-injection-and-jailbreak-samples) | Adversarially modified prompts designed to bypass guardrails |
| **Benign** (500) | [furonghuang-lab/PHTest](https://huggingface.co/datasets/furonghuang-lab/PHTest) | Pseudo-harmful prompts that look suspicious but are actually harmless |

Total: 1,500 test cases (scalable with `-n` flag).

## How It Works

**Jev** uses TypeSafe's System One API with typed questions:
- Three `Noul` questions (prompt injection, jailbreak, harmful intent) returning calibrated probabilities
- One `Choice` question (risk level: safe / suspicious / dangerous)
- Flags as UNSAFE if any Noul score > 0.5 or risk level is "dangerous"

**Haiku** uses Anthropic's Messages API with tool use for structured output:
- A `classify_input` tool returning `is_safe`, `risk_level`, `reason`, and `confidence`
- Flags as UNSAFE if `is_safe` is false

## Setup

```bash
# Clone and install
git clone https://github.com/rahulbansalc6414/jev-as-a-guardrail.git
cd jev-as-a-guardrail
uv sync

# Set API keys
cp .env.example .env
# Edit .env with your keys
```

### Required API Keys

| Variable | Source |
|----------|--------|
| `TYPESAFE_API_KEY` | [TypeSafe](https://typesafe.ai) |
| `ANTHROPIC_API_KEY` | [Anthropic](https://console.anthropic.com) |
| `HF_TOKEN` | [Hugging Face](https://huggingface.co/settings/tokens) (needed for gated datasets) |

## Usage

```bash
# Dry run (3 cases, verbose output)
uv run python benchmark.py -n 2

# Quick run
uv run python benchmark.py -n 100

# Medium run
uv run python benchmark.py -n 300

# Full benchmark (1,500 cases)
uv run python benchmark.py
```

The `-n` flag maintains proportional category splits at any scale.

## Output

The benchmark generates three files in `results/`:

- **`raw_results.jsonl`** - Every test case with both systems' raw scores (checkpointed every 50 calls)
- **`summary.json`** - Detection rates, latency percentiles, cost, confusion matrices, head-to-head comparison
- **`detail.csv`** - Flat CSV for further analysis

## Sample Results (n=300)

```
Detection rate:      Jev 91%    vs  Haiku 97%
False positive rate: Jev 11%    vs  Haiku 19%
Median latency:      Jev 333ms  vs  Haiku 2,148ms
Cost per 1K calls:   Jev $0.03  vs  Haiku $1.86
```

## License

MIT
