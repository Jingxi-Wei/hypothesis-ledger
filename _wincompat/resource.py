"""No-op stub for the Unix-only stdlib ``resource`` module, so swebench's
harness imports on Windows.

swebench unconditionally does ``import resource`` at the top of
``prepare_images.py`` and ``run_evaluation.py``. The only call site,
``resource.setrlimit(resource.RLIMIT_NOFILE, ...)``, is itself guarded by
``platform.system() == "Linux"`` in run_evaluation, so on Windows these
functions are never actually invoked -- we only need the import to succeed.
The functions below are faithful no-ops in case any unguarded path calls them.

Put this directory on PYTHONPATH (the eval scripts do this automatically) so it
shadows the missing stdlib module. There is no real ``resource`` module on
Windows, so nothing legitimate is being hidden.
"""

RLIMIT_NOFILE = 7
RLIMIT_CORE = 4
RLIMIT_STACK = 3
RLIM_INFINITY = -1


def getrlimit(which):
    return (RLIM_INFINITY, RLIM_INFINITY)


def setrlimit(which, limits):
    return None
