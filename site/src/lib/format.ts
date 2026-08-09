// Display-layer helpers. Nothing here changes the data — it only decides how
// a slug, a URL or an ISO date reads to a human at 6am.

/**
 * Section slugs are the pipeline's identifiers (config/taxonomy.yaml); these
 * are their reading names. Deliberately a display-only map — the site never
 * needs the taxonomy's descriptions or grouping, which ship in digest.json.
 */
const SECTION_LABELS: Record<string, string> = {
  "llm-security": "LLM security",
  "adversarial-ml": "Adversarial ML",
  tsfm: "TSFM",
  alignment: "Alignment",
  agents: "Agents",
  efficiency: "Efficiency",
  evaluation: "Evaluation",
  "model-releases": "Model releases",
  policy: "Policy",
  longform: "Longform",
  deadlines: "Deadlines",
  watchlist: "Watchlist",
  rising: "Rising",
  "has-code": "Has code",
  jobs: "Jobs",
};

export function sectionLabel(tag: string): string {
  return SECTION_LABELS[tag] ?? tag.replace(/-/g, " ");
}

const GROUP_LABELS: Record<string, string> = {
  research: "Research",
  ecosystem: "Ecosystem",
  utility: "Utility",
};

export function groupLabel(group: string): string {
  return GROUP_LABELS[group] ?? group;
}

/**
 * What the link actually points at. arXiv gets its identifier rather than a
 * bare hostname — that's the string a researcher recognizes and can paste.
 */
export function sourceLabel(url: string): string {
  try {
    const u = new URL(url);
    const host = u.hostname.replace(/^www\./, "");
    if (host.endsWith("arxiv.org")) {
      const id = u.pathname.match(/\/(?:abs|pdf)\/(.+?)(?:\.pdf)?$/)?.[1];
      if (id) return `arXiv:${id}`;
    }
    if (host === "github.com") {
      const repo = u.pathname.split("/").filter(Boolean).slice(0, 2).join("/");
      if (repo) return repo;
    }
    return host;
  } catch {
    return url;
  }
}

/** "2026-08-09" -> "Sunday, 9 August 2026". Fixed locale so the static build
 * is deterministic regardless of the runner's environment. */
export function longDate(iso: string): string {
  const d = new Date(`${iso}T00:00:00Z`);
  if (Number.isNaN(d.getTime())) return iso;
  return new Intl.DateTimeFormat("en-GB", {
    weekday: "long",
    day: "numeric",
    month: "long",
    year: "numeric",
    timeZone: "UTC",
  }).format(d);
}

/** Non-relevance slots have to explain themselves — see CLAUDE.md. */
const SLOT_LABELS: Record<string, string> = {
  adjacent: "Adjacent",
  wildcard: "Wildcard",
  watchlist: "Watchlist",
  "has-code": "Has code",
};

export function slotLabel(slot: string): string {
  return SLOT_LABELS[slot] ?? slot.replace(/-/g, " ");
}

export function plural(n: number, one: string, many = `${one}s`): string {
  return `${n} ${n === 1 ? one : many}`;
}
