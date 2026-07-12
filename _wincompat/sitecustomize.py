"""Auto-imported at interpreter startup because scripts/_wincompat is on
PYTHONPATH (the eval trigger puts it there). Active on Windows only.

Fixes a Windows-only SWE-bench harness bug: the harness writes ``eval.sh`` and
``patch.diff`` with ``pathlib.Path.write_text()`` in text mode, which on Windows
translates every ``\\n`` into ``\\r\\n``. Those files are copied verbatim into
the Linux evaluation container, where ``/bin/bash`` chokes on the stray ``\\r``:
``conda activate`` fails, ``pytest`` is never on PATH, and ``git apply`` of the
test patch fails because its context lines now end in ``\\r``. The net effect is
that EVERY test errors regardless of the patch -- so nothing ever resolves,
*including SWE-bench's own gold patches*.

Defaulting ``write_text``'s ``newline`` to ``"\\n"`` (when the caller didn't
specify one) suppresses the translation so the bytes that reach the container
are exactly what swebench intended. This only touches text writes in processes
that have _wincompat on PYTHONPATH (the eval subprocess); it does not globally
modify the user's Python.
"""
import os

if os.name == "nt":
    import pathlib

    _orig_write_text = pathlib.Path.write_text

    def _write_text_lf(self, data, encoding=None, errors=None, newline=None):
        if newline is None:
            newline = "\n"
        return _orig_write_text(self, data, encoding=encoding, errors=errors,
                                newline=newline)

    # idempotent: don't wrap twice if imported again
    if getattr(pathlib.Path.write_text, "__name__", "") != "_write_text_lf":
        pathlib.Path.write_text = _write_text_lf

    # ------------------------------------------------------------------------
    # Windows-safe container cleanup. minisweagent's DockerEnvironment.cleanup
    # runs a POSIX shell string -- "(timeout 60 docker stop ID || docker rm -f
    # ID) >/dev/null 2>&1 &" -- through subprocess.Popen(shell=True), i.e. cmd.exe
    # on Windows. cmd.exe can't parse the (...) grouping / || / redirect / &, so
    # every env teardown (__del__) prints "系统找不到指定的路径" to stderr. The
    # container is removed regardless (started with --rm, and run_batch does its
    # own `docker rm -f` sweep), so the original is a no-op-with-noise here.
    # Replace it with a real, QUIET `docker rm -f` (capture_output => no stderr
    # leak; exceptions swallowed so __del__ never raises). rm -f force-removes a
    # running container, so the separate stop step is unnecessary.
    # Wrapped defensively: a failure here must never break interpreter startup.
    try:
        import subprocess as _sp
        from minisweagent.environments.docker import DockerEnvironment as _DE

        def _win_docker_cleanup(self):
            cid = getattr(self, "container_id", None)
            if cid is not None:
                try:
                    _sp.run([self.config.executable, "rm", "-f", cid],
                            capture_output=True, timeout=60)
                except Exception:
                    pass

        if getattr(_DE.cleanup, "__name__", "") != "_win_docker_cleanup":
            _DE.cleanup = _win_docker_cleanup
    except Exception:
        pass  # minisweagent absent / API changed -> keep the (harmless) original
