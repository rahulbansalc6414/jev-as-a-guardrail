# Benchmark Results: Jev vs Haiku (n=1,000)

Run date: 2026-10-13

## Summary

| Metric | Jev | Haiku |
|---|---|---|
| Detection rate (overall) | 91.9% | 96.6% |
| Detection (jailbreak) | 95.3% | 97.2% |
| Detection (evasion) | 78.2% | 94.0% |
| False positive rate | 9.6% | 14.1% |
| Accuracy | 91.4% | 93.0% |
| Median latency | 328ms | 2,066ms |
| P95 latency | 560ms | 2,862ms |
| Cost per 1K calls | $0.03 | $1.87 |
| Errors | 0 | 0 |

## Category Breakdown

| Category | Count | Jev Detection | Haiku Detection |
|---|---|---|---|
| Jailbreak (attack) | 533 | 95.3% | 97.2% |
| Evasion (attack) | 133 | 78.2% | 94.0% |
| Benign | 334 | 9.6% FPR | 14.1% FPR |

## Confusion Matrices

**Jev**: TP=612, FP=32, TN=302, FN=54

**Haiku**: TP=643, FP=47, TN=287, FN=23

## Head-to-Head

- Agreement rate: 92.4%
- Jev caught, Haiku missed: 1
- Haiku caught, Jev missed: 32

## Key Findings

1. **Haiku detects more attacks overall**, with a 4.7 percentage point lead (96.6% vs 91.9%). The gap widens significantly on evasion prompts (94.0% vs 78.2%).

2. **Jev produces fewer false positives** -- 9.6% vs 14.1%. Haiku is more aggressive in flagging benign inputs as unsafe.

3. **Jev is 6x faster** -- 328ms median vs 2,066ms. At P95, Jev stays under 600ms while Haiku approaches 3 seconds.

4. **Jev is 62x cheaper** -- $0.03 vs $1.87 per 1,000 calls.

5. **Evasion is Jev's weak spot** -- adversarially modified prompts (Mindgard dataset) drop Jev's detection to 78.2%, while Haiku holds at 94.0%. This is the largest performance gap between the two systems.

6. **Zero errors** from both systems across all 1,000 calls, confirming reliability at scale.

## Cost and Pricing

Costs calculated using:
- Jev: $0.042 per 1M input tokens
- Haiku 4.5: $1.00 per 1M input tokens + $5.00 per 1M output tokens

As of October 2026.
