"""A timed-out stage must not leave its own-session children (vLLM workers) running. Harmless processes only."""
import os
import subprocess
import sys
import tempfile
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from issueloc.procs import ENV_KEY, kill_registered, kill_session, popen_session  # noqa: E402

INNER = """
import sys, time
sys.path.insert(0, {root!r})
from issueloc.procs import popen_session
popen_session([sys.executable, "-c", "import time; time.sleep(120)"])
time.sleep(120)
"""


def alive(pid):
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    # a killed child of this test process stays a zombie until reaped; it counts as dead
    st = subprocess.run(["ps", "-o", "stat=", "-p", str(pid)], capture_output=True, text=True).stdout.strip()
    return bool(st) and not st.startswith("Z")


def test_stage_timeout_kills_nested_sessions():
    with tempfile.TemporaryDirectory() as d:
        reg = os.path.join(d, "pids.txt")
        env = dict(os.environ, **{ENV_KEY: reg})
        stage = popen_session([sys.executable, "-c", INNER.format(root=ROOT)], env=env)
        for _ in range(100):  # wait until the stage has started its own-session child
            pids = open(reg).read().split() if os.path.exists(reg) else []
            if len(pids) == 2:
                break
            time.sleep(0.05)
        inner = int(pids[1])
        kill_session(stage.pid)  # what the old stage timeout did
        stage.wait(timeout=10)
        time.sleep(0.2)
        assert alive(inner), "child should still run after killing only the stage's group"
        assert kill_registered(reg) == 2
        for _ in range(50):
            if not alive(inner):
                break
            time.sleep(0.05)
        assert not alive(inner)
