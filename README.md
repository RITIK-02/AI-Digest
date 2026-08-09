# AI Digest

A personal AI research digest. A nightly job ingests arXiv papers, dedupes
them, ranks them against a hand-written interest profile, summarizes the
survivors, writes a daily brief, and publishes a static site plus an email.
Single user, no accounts, everything precomputed. Full architecture and
design rationale live in [CLAUDE.md](CLAUDE.md) — this file is just the
"how do I run it" reference.

## Prerequisites

- Python 3.12+ and [uv](https://docs.astral.sh/uv/)
- Node 20+ and npm (for the site)
- API keys — see [Configuration](#configuration) below

## Setup

```bash
git clone <this repo>
cd AI_Digest

uv sync                 # installs Python deps into .venv, first run also
                         # downloads the local embedding model (~90MB)
npm --prefix site install
```

Copy the env template and fill in your keys:

```bash
cp .env.example .env
```

```
ANTHROPIC_API_KEY=
OPENAI_API_KEY=
OPENROUTER_API_KEY=
RESEND_API_KEY=
RESEND_FROM=
RESEND_TO=
```

You don't need all of these to run the pipeline — see below.

## Configuration

Everything that isn't a secret lives in `config/`:

| File | What it controls |
|---|---|
| `config/interests.md` | The cold-start interest profile. Hand-edit this as your interests shift — it's read fresh every run, never overwritten by the pipeline. |
| `config/sources.yaml` | arXiv categories, rate limit. Only `arxiv` is implemented today. |
| `config/taxonomy.yaml` | The 15 sections and their descriptions. |
| `config/providers.yaml` | Which LLM provider/model handles each pipeline stage, and provider-specific options (reasoning effort, JSON schema mode, prompt caching). |
| `config/limits.yaml` | Hard per-run dollar budget; the run fails loudly if it's exceeded. |

### LLM providers

Three `LLMClient` implementations exist: Anthropic, OpenAI, and OpenRouter
(a proxy that reaches many vendors' models through one account/key — useful
as a stand-in when a direct vendor account is rate- or billing-limited).
`config/providers.yaml`'s `routing` section maps each pipeline stage to a
primary and fallback `{provider, tier}`. You need working keys for whichever
providers appear there — check that file to see the current mapping, since
it's meant to be edited over time (see the `TEMPORARY` note at the top if
present, explaining any short-term substitution).

At minimum you need **one** working provider for the pipeline to produce
anything; two providers get you the documented fallback behavior (if the
primary errors, it retries once on the fallback before giving up).

`RESEND_*` are optional — if unset, `publish.py` logs and skips the email
send rather than failing the run. The site still gets built either way.

## Running locally

```bash
uv run pytest              # unit tests (fixtures only, no network/API calls)
uv run ruff check .        # lint
```

The nightly job is split into two scripts because it's built around each
vendor's async Batch API (~50% cheaper than synchronous calls): `submit.py`
kicks off the LLM batches, `collect.py` resolves them a few hours later.
Locally you can just run them back to back:

```bash
uv run python scripts/submit.py    # ingest -> dedupe -> embed/prefilter -> submit triage batch
uv run python scripts/collect.py   # collect triage -> submit+collect summarize -> brief -> publish + email
```

`submit.py` fetches arXiv live and writes to `data/digest.db` (committed to
the repo — it's the dedup key store and LLM cache, not disposable). If the
triage batch hasn't finished processing yet, `collect.py` says so and exits
cleanly; just run it again once the batch is done (Batch APIs don't
guarantee a turnaround time, though small batches are usually done in
minutes).

To preview the site with whatever's currently in `site/src/data/digest.json`:

```bash
npm --prefix site run dev      # http://localhost:4321
npm --prefix site run build    # static output to site/dist/
```

### Re-running a stage without re-spending

Every LLM stage caches its output keyed by `(content_hash, prompt_version,
model)` in `data/digest.db`. Re-running `submit.py`/`collect.py` on the same
day never re-pays for triage/summarize work already done — only genuinely
new items get sent to an LLM. Bumping a prompt's version string in
`src/llm/prompts/*.md` invalidates that prompt's cache on purpose.

## Deploying

### GitHub Actions (the compute)

Two scheduled workflows do the nightly run — no server, no always-on
process:

- `.github/workflows/submit.yml` — 02:00 UTC
- `.github/workflows/collect.yml` — 05:00 UTC

Add these as **Actions secrets** (repo Settings → Secrets and variables →
Actions) — same names as your `.env`, whichever ones the current
`config/providers.yaml` routing table needs:

```
ANTHROPIC_API_KEY
OPENAI_API_KEY
OPENROUTER_API_KEY
RESEND_API_KEY
RESEND_FROM
RESEND_TO
```

Both workflows commit `data/digest.db` (and `collect.yml` also commits
`site/src/data/digest.json`) back to the repo with `contents: write`
permission — that's already configured, no extra setup needed. You can also
trigger either one manually via the Actions tab (`workflow_dispatch`) to do
a one-off run or backfill.

### Cloudflare Pages (the host)

Deploy is a build artifact push — Pages watches the repo and rebuilds on
every push to `main`, which is why `collect.yml` commits the freshly
generated `digest.json` before anything gets deployed.

Connect the repo in the Cloudflare dashboard (Workers & Pages → Create →
Pages → Connect to Git) with:

| Setting | Value |
|---|---|
| Build command | `npm run build` |
| Build output directory | `dist` |
| Root directory | `site` |

No environment variables or secrets are needed on the Pages side — the site
is fully static and only reads `site/src/data/digest.json`, which is already
baked in by the time Pages builds it.

### First deploy checklist

1. Fill in `config/interests.md` with your real interest profile (the
   version in this repo is a placeholder seed).
2. Add the Actions secrets above.
3. Connect Cloudflare Pages as described.
4. Trigger `submit.yml` then `collect.yml` manually once (via
   `workflow_dispatch`) to confirm the whole chain works before trusting the
   schedule.
5. Watch the first scheduled run for cost — `config/limits.yaml`'s
   `per_run_usd_cap` will fail the run loudly if something regressed, but
   it's worth eyeballing `cost_log` in `data/digest.db` after the first
   few runs regardless.
