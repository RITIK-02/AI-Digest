# Interest profile (cold-start seed)

This is the hand-written seed for the interest vector. `embed.py` embeds this
document and uses it as the initial interest vector before any click evidence
exists. Edit this by hand as interests shift — it is not overwritten by the
pipeline.

## Primary

- LLM security: jailbreaks, prompt injection, data/PII extraction,
  red-teaming methodology, guardrail evaluation and bypass, agentic-system
  attack surfaces (tool use, RAG poisoning, memory manipulation).
- Time series foundation models: pretraining objectives for time series,
  zero-shot and few-shot forecasting, cross-domain transfer, benchmarks
  (e.g. long-horizon, multivariate, irregular sampling), architecture
  choices (patch-based transformers, state-space models applied to TS).

## Secondary

- Adversarial robustness more broadly, where it informs LLM security.
- Evaluation methodology: benchmark validity, contamination/leakage,
  LLM-as-judge reliability.
- Efficient inference/serving, when it intersects with the above (e.g.
  cheap red-teaming at scale, efficient TSFM inference).

## Watchlist (named authors)

None yet — add author names here as they become worth tracking; the
`watchlist` section surfaces their work regardless of score.

## Deliberately out of scope

- General LLM capability papers with no security or time-series angle.
- Pure computer vision.
- RLHF/alignment work that isn't adjacent to security or evaluation.
