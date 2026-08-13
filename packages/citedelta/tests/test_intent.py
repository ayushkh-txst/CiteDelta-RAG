from __future__ import annotations

import pytest

from citedelta.answer.intent import Intent, classify


@pytest.mark.parametrize(
    "text",
    [
        "hello",
        "Hi",
        "hey!",
        "hi there",
        "hey there",
        "Good morning",
        "how's it going?",
        "How are you",
        "what's up",
        "thanks",
        "Thank you!",
        "who are you",
        "what can you do",
        "help",
        "bye",
    ],
)
def test_greetings_are_caught(text: str) -> None:
    assert classify(text) is Intent.GREETING


@pytest.mark.parametrize(
    "text",
    [
        "What is the grace period after F-1 program completion?",
        "What is the capital of France?",
        # The ones a substring match would get wrong — and these are the
        # tests that matter, because failing them means answering a real
        # question with a pleasantry.
        "hi there, what's the F-1 grace period?",
        "thanks, and what about STEM OPT?",
        "Hello — can an F-1 student transfer schools?",
        "How long is the grace period?",
        "help me understand practical training",
    ],
)
def test_real_questions_pass_through(text: str) -> None:
    assert classify(text) is Intent.PASSTHROUGH


def test_classification_needs_no_network() -> None:
    """Guard against someone 'improving' this into a model call. The whole
    value is that it is free and instant."""
    import inspect

    from citedelta.answer import intent

    src = inspect.getsource(intent)
    assert "await" not in src
    assert "Completions" not in src
