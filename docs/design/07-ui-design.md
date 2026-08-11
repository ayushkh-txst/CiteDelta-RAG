# 07 — UI design

## Principles

1. **Regulation text is the hero.** Everything else is chrome and recedes.
   Serif, 17px, 1.65 line height, 68ch measure. If a design decision makes the
   regulation harder to read, it loses.
2. **Time is the visual signature.** No screen exists without its as-of
   context visible. This is the one thing CiteDelta does that a normal search
   box doesn't, so it is never more than one glance away.
3. **Uncertainty is shown, not hidden.** Refusals and low confidence are
   designed states with their own layout and colour, not error styling
   borrowed from a 500 page.
4. **The trace is one click away, never in the way.** The default view is
   calm. The machinery is there when you want it, collapsed when you don't.

## Type

| Role | Stack | Size | Why |
|---|---|---|---|
| Regulation prose | Georgia / Iowan Old Style / serif | 17px / 1.65 | Long-form legal text. Serif is measurably easier at length, and it signals "document" rather than "app". |
| Chrome | system-ui | 14–15px | Should disappear. |
| Citation paths | ui-monospace | 12–13px | `8 CFR 214.2(f)(10)(ii)(C)` is a structured identifier, not prose. Mono makes the nesting legible. |

**`font-variant-numeric: tabular-nums` is set globally.** Dates, ranks and RRF
scores appear in columns throughout. Proportional digits make a score table
look subtly broken and nobody can tell you why.

## Colour

Warm paper, near-black ink, one accent. Both themes authored; dark is not a
filter over light.

| Token | Light | Dark | Use |
|---|---|---|---|
| `--bg` | `#FBFAF7` | `#171614` | Page. Warm, not blue-white. |
| `--surface` | `#FFFFFF` | `#201F1C` | Cards. |
| `--ink` | `#1A1917` | `#EDEAE3` | Body text. |
| `--accent` | `#0F5C63` | `#6FBDC2` | Interaction, citation chips, the as-of marker. |
| `--inforce` | `#1F6F4A` | `#6BBF8F` | **In force on the selected date.** |
| `--superseded` | `#8A8580` | `#8E8981` | **Superseded — excluded from retrieval.** |
| `--caution` | `#8A5A00` | `#D3A657` | Refusals. |
| `--danger` | `#A32B21` | `#E2857A` | **Verification failure only.** |

The **in force / superseded** pair carries the whole product. It appears in the
ribbon, the citation badges, and the diff headers, and it always means the same
thing.

**Red is reserved.** It marks exactly one condition: an answer discarded
because a citation could not be verified. It never means "no results" and never
means "your question was unclear" — both of those are amber. If red loses that
specificity it stops being information.

## Space and motion

4px base scale (`--s1` … `--s8`). Generous vertical rhythm around citations so
each reads as a discrete object rather than a paragraph in a list.

Motion is a 120ms opacity fade on HTMX swaps and nothing else. Every state in
this UI is something a screenshot needs to catch.

## Components

| Component | Responsibility |
|---|---|
| `AskBar` | Query + as-of control. The as-of field is the product's defining input and is given equal visual weight to the query, not tucked into an "advanced" drawer. |
| `AnswerCard` | Answer prose, inline citation chips, persistent not-legal-advice footer. |
| `RefusalCard` | States *why*, and what to do next. Amber. |
| `CitationCard` | Mono citation path, in-force badge, snippet, link to real eCFR. |
| `TemporalRibbon` | 2016→2026 axis; one bar per cited provision spanning its effective range; vertical marker at as-of. Server-computed inline SVG. |
| `TraceInspector` | Collapsed by default. Rank, citation, BM25 rank, vector rank, RRF score, effective range — **including retrieved-but-uncited candidates**. |
| `DiffView` | Same question, two dates, side by side, changed text highlighted. |

## Accessibility

- Colour is never the only signal: in-force state is also a text label
  (`in force 2016-04-01 → present`), cited state is also a marker glyph.
- `:focus-visible` outlines use `--accent` at 2px with 1px offset; never removed.
- The ribbon carries an `aria-label` describing what it shows in words.
- Target contrast: 4.5:1 for body text in both themes.

## Deliberately not done

- **No client-side framework.** Server-rendered Jinja plus HTMX. The page is
  mostly text with one interactive control; a framework would add a build step,
  a Node toolchain, and a hydration story to a page that needs none of them.
- **No chart library.** The ribbon is ~40 lines of SVG generated from data the
  trace already has.
- **No skeleton loaders.** Requests take ~1–2s and produce one card. A spinner
  in the button is honest; fake content shimmering into place is not.
