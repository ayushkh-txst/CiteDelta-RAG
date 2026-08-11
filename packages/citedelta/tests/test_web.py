from __future__ import annotations

from citedelta.answer.models import Citation
from citedelta.web.filters import citation_chips


def _citation(chunk_id: int) -> Citation:
    return Citation(
        chunk_id=chunk_id,
        citation_path=f"8 CFR 214.2(f)({chunk_id})",
        effective_from="2016-01-01",
        effective_to=None,
        text="text",
        rrf_score=0.5,
    )


def test_chips_escape_model_text_first() -> None:
    text = "He wrote <script>alert(1)</script> then cited [7]."
    out = str(citation_chips(text, [_citation(7)]))

    assert "<script>" not in out
    assert "&lt;script&gt;" in out
    assert 'class="chip"' in out


def test_chips_map_chunk_id_to_display_ordinal() -> None:
    cits = [_citation(100), _citation(200), _citation(300)]
    out = str(citation_chips("see [200] and [300]", cits))

    assert '<a class="chip" href="#cite-2">2</a>' in out
    assert '<a class="chip" href="#cite-3">3</a>' in out
    assert "[200]" not in out


def test_chips_leave_unmapped_ids_plain() -> None:
    out = str(citation_chips("no such [9999]", [_citation(7)]))

    assert "[9999]" in out
    assert 'class="chip"' not in out
