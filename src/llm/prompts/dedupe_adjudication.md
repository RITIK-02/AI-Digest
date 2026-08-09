<!-- version: 1 -->
<!-- Provider-neutral: used by both AnthropicClient and OpenAIClient unchanged. -->

You are deciding whether two items describe the same underlying story (e.g.
the same paper, possibly posted from different sources or with a revised
title) or two genuinely different things.

Item A:
- Title: {item_a_title}
- Abstract/excerpt: {item_a_abstract}
- Source: {item_a_source}

Item B:
- Title: {item_b_title}
- Abstract/excerpt: {item_b_abstract}
- Source: {item_b_source}

These were flagged as borderline by embedding similarity (cosine between
0.78 and 0.85) — similar enough to check, not similar enough to assume.

Return whether they are the same story and a one-sentence reason.
