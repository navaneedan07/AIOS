"""Run an entry-point script with the project's virtualenv interpreter.

Why this exists
---------------
``python`` on PATH is frequently *not* the interpreter this project's
dependencies live in -- on Windows it is often a global install with a
different version and none of the packages. Running a script with it fails
several imports deep, e.g.::

    ModuleNotFoundError: No module named 'cerebrum'

which reads as a broken checkout rather than the wrong interpreter. The
PowerShell/batch launchers already prefer ``.venv``; this gives the Python
entry points the same behaviour so ``python scripts/run_terminal.py`` just
works.

Behaviour
---------
:func:`ensure_project_interpreter` leaves an interpreter that already has the
project's dependencies (a developer's conda env, say) untouched, and re-execs
the current process onto ``.venv`` only when both:

* ``.venv`` exists in the repository root, and
* the running interpreter cannot import the project's key dependencies.

Set ``AIOS_NO_VENV_REEXEC=1`` to disable the re-exec entirely.
"""

from __future__ import annotations

import importlib.util
import os
import sys
from pathlib import Path
from typing import Optional

#: Repository root -- this file lives in ``<root>/scripts``.
ROOT_DIR = Path(__file__).resolve().parent.parent

#: Set this to skip the re-exec (useful when deliberately using another env).
_SKIP_ENV_VAR = "AIOS_NO_VENV_REEXEC"


def _venv_python() -> Optional[Path]:
    """Return the interpreter inside ``.venv``, or ``None`` if there is none."""
    for relative in (Path("Scripts") / "python.exe", Path("bin") / "python"):
        candidate = ROOT_DIR / ".venv" / relative
        if candidate.exists():
            return candidate
    return None


def _has_project_dependencies() -> bool:
    """Whether the running interpreter can import the kernel's dependencies."""
    try:
        return (
            importlib.util.find_spec("litellm") is not None
            and importlib.util.find_spec("cerebrum") is not None
        )
    except (ImportError, ValueError):
        return False


def _configure_utf8_output() -> None:
    """Make this process' output UTF-8, so emoji prints cannot crash it.

    On Windows a redirected cp1252 stream rejects the emoji in the terminal's
    own messages with ``UnicodeEncodeError``. ``errors="replace"`` keeps even an
    oddly-encoded console from turning a status line into a traceback.
    """
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is None:
            continue
        try:
            reconfigure(encoding="utf-8", errors="replace")
        except (ValueError, OSError):
            pass


def ensure_project_interpreter() -> None:
    """Re-exec this script with ``.venv`` when the current one lacks the deps.

    Never returns when a re-exec happens: the process image is replaced, so it
    comes back as the venv interpreter running the same script and arguments.
    """
    _configure_utf8_output()

    if os.environ.get(_SKIP_ENV_VAR):
        return

    venv = _venv_python()
    if venv is None:
        return

    try:
        if Path(sys.executable).resolve() == venv.resolve():
            return
    except OSError:
        pass

    if _has_project_dependencies():
        return

    # UTF-8 mode for the interpreter we are about to become.
    os.environ.setdefault("PYTHONUTF8", "1")
    os.environ.setdefault("PYTHONIOENCODING", "utf-8")

    try:
        os.execv(str(venv), [str(venv), *sys.argv])
    except OSError:
        # Re-exec unavailable (unusual): carry on and let the real import
        # error surface rather than masking it with a bootstrap failure.
        return
