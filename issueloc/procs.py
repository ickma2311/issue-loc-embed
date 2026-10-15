"""Child processes in their own session, registered in a file so that an outer orchestrator can kill every
session it started (directly or through nested helpers) when a stage times out."""
from __future__ import annotations

import os
import signal
import subprocess

ENV_KEY = "ISSUELOC_CHILD_PIDS"  # path of the registry file; unset = no registration


def _registry() -> str | None:
    return os.environ.get(ENV_KEY)


def popen_session(cmd, env=None, **kw) -> subprocess.Popen:
    """subprocess.Popen in a new session; its pid (= process-group id) is appended to the registry."""
    p = subprocess.Popen(cmd, start_new_session=True, env=env, **kw)
    reg = (env or os.environ).get(ENV_KEY)
    if reg:
        with open(reg, "a") as f:
            f.write(f"{p.pid}\n")
    return p


def kill_session(pid: int) -> None:
    try:
        os.killpg(pid, signal.SIGKILL)
    except (ProcessLookupError, PermissionError):
        pass


def unregister(pid: int, reg: str | None = None) -> None:
    reg = reg or _registry()
    if not reg or not os.path.exists(reg):
        return
    keep = [x for x in open(reg).read().split() if x != str(pid)]
    with open(reg, "w") as f:
        f.write("".join(f"{x}\n" for x in keep))


def kill_registered(reg: str) -> int:
    """Kill every registered session (e.g. vLLM workers started by a timed-out stage); empties the registry."""
    if not os.path.exists(reg):
        return 0
    pids = [int(x) for x in open(reg).read().split()]
    for pid in pids:
        kill_session(pid)
    open(reg, "w").close()
    return len(pids)
