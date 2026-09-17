"""A hard ceiling on the Hugging Face requests one catalog build may spend.

Hugging Face allows 1000 API requests per five minutes. Discovery had no idea
how many it was making: it resolved candidates until every size category was
full, and when one category could not fill -- Medium routinely cannot -- it
swept the whole window instead. At roughly three requests per unmatched
candidate that exhausted the quota, and because a throttled lookup returns
nothing in the same way a missing repository does, the build published a
short catalog as though it were complete.

A budget makes the limit a decision rather than an accident: the build stops
early, says it stopped, and the categories it did fill are still correct.
"""

from __future__ import annotations

from threading import Lock

# Hugging Face's documented allowance for the window a build fits inside.
HF_REQUESTS_PER_WINDOW = 1000

# Leave room for everything else a build and the surrounding session do:
# GGUF metadata range reads, model-page scrapes, the user's own browsing in
# another tab. Discovery is not the only thing spending this quota. A build
# cut short by this ceiling is not wasted -- what it resolved is written
# down, so the next one starts where this one stopped.
DEFAULT_BUDGET = 500

# Reading a repository's serving metadata is several requests, not one: the
# model record, the model page the weight-size table is scraped from, and the
# GGUF header range reads. Charged as a block so the ceiling covers what a
# build actually spends rather than only the matching half of it.
METADATA_REQUEST_COST = 4


class HubBudgetExhausted(RuntimeError):
    """Raised instead of making a request the build cannot afford.

    Distinct from a missing repository: the model was not checked, so it must
    be reported as unchecked rather than dropped as though it had no weights.
    """


class HubLookupIncomplete(RuntimeError):
    """Raised when a search could not be completed, so found nothing proves nothing.

    The matcher probes a canonical repository id and then fans out searches,
    skipping any that error. If every attempt errored -- which is what rate
    limiting looks like from inside those loops -- the empty result is an
    absence of evidence, not evidence of absence. Returning it as a miss let
    throttled runs write "this model has no weights" into the match store and
    be believed for days afterwards.
    """


class HubRequestBudget:
    """Counts Hub requests for one build and refuses to overspend.

    Shared across the resolution thread pool, so every mutation is locked.
    """

    def __init__(self, limit: int = DEFAULT_BUDGET) -> None:
        self._limit = max(0, int(limit))
        self._spent = 0
        self._refused = 0
        self._lock = Lock()

    @property
    def limit(self) -> int:
        return self._limit

    @property
    def spent(self) -> int:
        with self._lock:
            return self._spent

    @property
    def refused(self) -> int:
        """How many requests were declined, i.e. how much was left unchecked."""

        with self._lock:
            return self._refused

    @property
    def exhausted(self) -> bool:
        with self._lock:
            return self._spent >= self._limit

    def spend(self, count: int = 1) -> bool:
        """Claim `count` requests, or refuse and report that nothing is left.

        All-or-nothing: a caller that needs two requests should not be handed
        one and then throttled halfway through.
        """

        with self._lock:
            if self._spent + count > self._limit:
                self._refused += count
                return False
            self._spent += count
            return True


class UnlimitedBudget(HubRequestBudget):
    """For callers outside a catalog build, where no ceiling applies."""

    def __init__(self) -> None:
        super().__init__(limit=0)

    @property
    def exhausted(self) -> bool:
        return False

    def spend(self, count: int = 1) -> bool:
        return True
