# 07 — UI design

The interface is a conversation, and its design exists to keep it from looking
like one. The mockup at `guides/day-06/mockup/index.html` is the source of
truth for the finished markup; this document is the *why* — the decisions,
the rejected alternatives, and the two self-corrections that survived review.

## Principles

1. **Time is the spine, not a field.** The corpus is 78 discrete amendment
   dates, and presenting that as a pickable day was the old UI's defining lie.
   In a conversation the primary way to time-travel is to *say* "what about in
   2019?" — and every turn is anchored to the date of the law it is about. A
   change of date is a visible event in the record (the rupture), not a
   setting changed off-screen.
2. **Regulation text is the hero.** 17px / 1.65 / 66ch serif. If a decision
   makes the prose harder to read, it loses. Questions use a grotesque
   (Archivo) and metadata uses mono, so the document voice is never diluted.
3. **Declining is designed.** Refusals have named states with copy that says
   both what happened and what to do next. Nothing reads like an error page,
   because nothing here is one. A greeting is a refusal in the database but a
   conversation in the UI — the two must never share a card.
4. **Verification is shown, not hidden.** The ~8s wait streams its phases
   ("Finding provisions in force on 12 Apr 2019 · Reading 8 passages ·
   Drafting · **Verifying 3 citations**"), and the finished answer states
   "2 quotes verified" without anyone opening anything. The guarantee is the
   demo.

## Colour

First pass was cool slate + ink blue + sepia — **thrown out**. Cool grey with
a blue accent is corporate-SaaS by reflex, not a choice made from this
subject. The shipped palette is drawn from the material: archival stock,
seal green, aged amber, oxblood for the one thing that is broken.

| Token | Light | Dark | Role |
|---|---|---|---|
| `--ground` | `#EFF1EE` | `#101311` | Archival stock — off-white with a faint green cast |
| `--surface` | `#FFFFFF` | `#181C19` | Raised blocks |
| `--ink` | `#15181A` | `#E6EAE5` | Body |
| `--muted` | `#5C6560` | `#9AA39D` | Secondary, stamps |
| `--faint` | `#8B948D` | `#6E766F` | Tertiary |
| `--rule` | `#D8DCD7` | `#272C28` | Hairlines, the gutter spine |
| `--signal` | `#0F5132` | `#5FBF8A` | **In force**, interactive, the rupture |
| `--archive` | `#8A5A2B` | `#D2A05C` | **Aged / superseded** |
| `--alert` | `#8C2F1D` | `#DE8E7C` | **Verification failure only** |

Green means valid — in-force text, live interaction, and the as-of marker are
the same idea. Amber means aged. Two rules keep it coherent:

- **`--alert` marks exactly one condition**: an answer discarded because a
  citation or quote failed verification. The moment red also means "no
  results", it stops carrying information. This discipline is why a red card
  in this UI is worth reading.
- **`--archive` appears in exactly two places**, both where aging is literally
  true: the older column of a comparison, and a source badge whose provision
  has since been superseded. It does **not** tint answers rendered at a past
  date. I planned that and cut it: only admissible text is ever retrieved, so
  superseded prose never appears inside an answer. The law in force in 2019
  *was* in force in 2019 — rendering it as aged would assert something false.

## Type

| Role | Face | Why this one |
|---|---|---|
| Questions, UI | **Archivo** | A grotesque with administrative, slightly industrial character. Deliberately not Inter (ubiquitous) or Space Grotesk (the AI-startup default). |
| Regulation prose | **Source Serif 4** | A *text* serif engineered for reading at length. Not Playfair or Fraunces — display faces, part of the look being avoided. |
| Stamps, citations, data | **IBM Plex Mono** | Institutional. Tabular figures so dates and ranks align. |

All three are **vendored locally** (three woff2 files in `web/static/fonts/`),
not CDN-loaded — a deploy that depends on someone else's font host is a
deploy with an extra way to fail. `font-variant-numeric: tabular-nums` is
global: dates and ranks appear in columns everywhere.

## The signature: the temporal gutter and the rupture

Every turn carries a left gutter stamp of the date of the law it is about
(`12 Apr` / `2019`). When the as-of changes between turns, the record ruptures:

```
═══╪══ ⤺ 12 April 2019 · 63 amendments earlier ══════════
```

Time travel stops being a setting adjusted somewhere off-screen and becomes
something that happened, in sequence, in the record. The counting is **strictly
between** the two dates (an amendment on the boundary is the law you are now
reading, not one you skipped), and crossing zero amendments is rendered as a
message — the identical answer the user is about to see is correct, and the UI
says so rather than inviting a bug report. Both directions render: `⤺`
earlier, `⤻` later. The gutter and rupture logic lives in `web/transcript.py`
as pure arithmetic over dates, unit-tested without a browser.

## Space and motion

4px base scale. The transcript is a real grid (gutter column + body column),
not padding. Two animations exist — the phase pulse and the turn fade-in —
and both sit inside a `prefers-reduced-motion` guard.

## Components

| Component | Responsibility |
|---|---|
| `Turn` | Question, as-of stamp, optional `↳ searched:` line, and the answer/refusal body. |
| `Rupture` | The date-change event between turns. `role="separator"` with a spoken label. |
| `Composer` | Input + as-of chip + Ask. The chip is a fallback; the placeholder teaches "or say *what about in 2019?*". |
| `Phases` | `aria-live` progress feed during the turn. |
| `Sources` | One `<details>` per answer; each citation's verified quote is bolded **in the original text**. |
| `Compare across dates` | Offers only the dates the cited provisions actually changed — the day *before* each change, because a change date is the first day of the version already on screen. |
| `How this answer was found` | The trace, de-jargoned: "Found by wording / Found by meaning / Combined", ranks as ordinals, strength as words. Unused rows and "not found" cells stay. |

## Accessibility

- Colour is never the only signal: in-force carries a text label, cited rows
  carry a `●` glyph with an `sr-only` header.
- Phases are announced via `aria-live="polite"` — eight silent seconds is
  indistinguishable from a hang for a screen-reader user.
- The rupture is `role="separator"` with an `aria-label`; the stamp is one
  `aria-label`, not two disconnected spans.
- `:focus-visible` is defined once from `--signal`; contrast ≥ 4.5:1 in both
  themes.

## Deliberately not done

- **No client framework, no build step.** Server-rendered Jinja + vendored
  HTMX. The page is long-form text with one interactive control; a client
  framework would add a toolchain and a hydration story to a problem the page
  doesn't have.
- **No bubbles, avatars, or assistant persona.** A centered column of bubbles
  is the most templated UI in existence, and "another RAG chatbot" is a
  liability when the point is retrieval infrastructure. The conversation
  survives; chat convention does not.
- **No skeleton loaders.** The wait is real and valuable; the phases turn it
  into the demo instead of faking content with shimmer.
- **No aging of past answers.** Scoped out because it is false, not because it
  is ugly — see the `--archive` rule above.
