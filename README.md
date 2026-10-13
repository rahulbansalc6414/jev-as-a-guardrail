# Jev vs Haiku: Input Guardrail Benchmark

Benchmarking **TypeSafe Jev** (a general-purpose judgment engine) against **Claude Haiku 4.5** (a general-purpose LLM) as input safety classifiers for enterprise AI systems.

## What This Tests

Both systems evaluate user inputs for:
- Prompt injection attempts
- Jailbreak attacks
- Harmful intent

The benchmark measures detection rate, false positive rate, latency, and cost across three categories of prompts.

## Datasets

| Category | Dataset | Description |
|----------|---------|-------------|
| **Attack** | [Lshafii/Jailbreak-PromptBank](https://huggingface.co/datasets/Lshafii/Jailbreak-PromptBank) | Curated jailbreak prompts from AdvBench, HarmBench, StrongReject, GPT-Fuzzer, and others |
| **Evasion** | [Mindgard/evaded-prompt-injection-and-jailbreak-samples](https://huggingface.co/datasets/Mindgard/evaded-prompt-injection-and-jailbreak-samples) | Adversarially modified prompts designed to bypass guardrails |
| **Benign** | [furonghuang-lab/PHTest](https://huggingface.co/datasets/furonghuang-lab/PHTest) | Pseudo-harmful prompts that look suspicious but are actually harmless |

The `-n` flag scales each category proportionally (53% attack / 13% evasion / 33% benign). Attack prompts are deduplicated before sampling. Dataset revisions are pinned for reproducibility.

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
# Dry run (verbose output for small runs)
uv run python benchmark.py -n 2

# Quick run
uv run python benchmark.py -n 100

# Default run (1,000 cases)
uv run python benchmark.py

# Full benchmark (all available prompts from each dataset)
uv run python benchmark.py --all
```

The `-n` flag controls how many test cases to run (default: 1,000) and maintains proportional category splits at any scale. Use `--all` to run against every available prompt.

## Output

Each run creates a timestamped directory under `results/` (e.g., `results/1000-run-20261013-162726/`) containing:

- **`summary.json`** -- Detection rates, latency percentiles, cost, confusion matrices, head-to-head comparison
- **`detail.csv`** -- Flat CSV with per-case results for further analysis
- **`raw_results.jsonl`** -- Every test case with both systems' raw scores (checkpointed every 50 calls)

## Results (n=1,000)

> Cost estimates use Jev input pricing ($0.042/1M tokens) and Anthropic Haiku 4.5 pricing ($1.00/1M input, $5.00/1M output) as of October 2026.

| Metric | Jev | Haiku |
|---|---|---|
| Detection rate (overall) | 91.9% | 96.6% |
| Detection (jailbreak) | 95.3% | 97.2% |
| Detection (evasion) | 78.2% | 94.0% |
| False positive rate | 9.6% | 14.1% |
| Accuracy | 91.4% | 93.0% |
| Median latency | 328ms | 2,066ms |
| Cost per 1K calls | $0.03 | $1.87 |

Full analysis: [`results/1000-run-20261013-162726/ANALYSIS.md`](results/1000-run-20261013-162726/ANALYSIS.md)

## License

MIT
