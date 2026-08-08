"""substrate — infrastructure written by hand for the portfolio sprint.

Everything in here is a standard system-design interview topic implemented as
working, tested code: a durable job queue, backoff with jitter, a circuit
breaker, a content-addressed cache. Nothing in here knows about regulations.
"""

__version__ = "0.1.0"

__all__ = ["__version__"]
