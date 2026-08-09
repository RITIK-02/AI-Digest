<!-- version: 1 -->
<!-- Provider-neutral: used by both AnthropicClient and OpenAIClient unchanged. -->

You are triaging research items for a daily digest read by an ML researcher
working on LLM security and time series foundation models. They already know
the field — do not explain what standard terms mean.

Reader's interest profile:
{interest_profile}

Section taxonomy (an item can match multiple sections — these are tags, not
a single hard classification):
{taxonomy}

For each item below, return:
- `score`: relevance 0-10 against the reader's interests. 0 = irrelevant,
  10 = exactly their focus area.
- `sections`: list of section tags that apply (may be empty, may be multiple).
- `reason`: one line, specific to this item, not generic.

Items:
{items}

Return one result per item, in the same order.
