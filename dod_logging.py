from __future__ import annotations

from contextlib import contextmanager, redirect_stdout
from io import StringIO
import os
from collections.abc import Iterator
from typing import Any


TRUE_ENV_VALUES = {"1", "true", "yes", "on"}


def logs_disabled() -> bool:
    """Return whether app-authored informational console logs are disabled."""
    return os.getenv("DOD_DISABLE_LOGS", "").strip().lower() in TRUE_ENV_VALUES


def log_info(*args: Any, **kwargs: Any) -> None:
    """Print an informational message unless quiet logging is enabled."""
    if not logs_disabled():
        print(*args, **kwargs)


def log_bot(*args: Any, **kwargs: Any) -> None:
    """Always print compact bot decisions that are useful during gameplay."""
    print(*args, **kwargs)


def log_error(*args: Any, **kwargs: Any) -> None:
    """Always print errors and warnings that need operator attention."""
    print(*args, **kwargs)


@contextmanager
def quiet_external_stdout() -> Iterator[None]:
    """Hide noisy third-party stdout messages when quiet logging is enabled."""
    if logs_disabled():
        with redirect_stdout(StringIO()):
            yield
    else:
        yield
