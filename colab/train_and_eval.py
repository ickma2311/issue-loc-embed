"""Colab L4 orchestrator: data conversion, mining, round-1 / ablation / round-2 training and evaluation. Stages run as subprocesses
in their own process group (killed as a group on timeout); every stage writes
"[stage] <name> start|done rc=<n>" to /content/out/run.log. Re-running skips stages whose final marker
file exists. Usage (from the bundle root): python colab/train_and_eval.py"""
import json, os, pickle, signal, subprocess, sys, time
from pathlib import Path

O = Path("/content/out"); O.mkdir(parents=True, exist_ok=True)
DATA = Path("/content/data")
LOG = open(O / "run.log", "a", buffering=1)
ENV = dict(os.environ, EVAL_SET="paper", EMBED_BACKEND="vllm", OUT_DIR=str(O), VLLM_TMP="/content/tmp",
           PYTHONUNBUFFERED="1")
os.makedirs("/content/tmp", exist_ok=True)
sys.path.insert(0, os.getcwd())
from issueloc.procs import ENV_KEY, kill_registered, kill_session, popen_session, unregister  # noqa: E402
CHILD_PIDS = str(O / "child_pids.txt")
ENV[ENV_KEY] = CHILD_PIDS
G = [sys.executable, "gpu_run.py"]


def _watchdog():  # RAM / GPU memory every 60 s -> mem.log; warn in run.log when RAM runs low
    import threading
    def run():
        with open(O / "mem.log", "a", buffering=1) as f:
            while True:
                m = dict(l.split(":")[0:2] for l in open("/proc/meminfo") if l.split(":")[0] in ("MemTotal", "MemAvailable"))
                tot, av = int(m["MemTotal"].split()[0]) // 1024, int(m["MemAvailable"].split()[0]) // 1024
                g = subprocess.run("nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits", shell=True,
                                   capture_output=True, text=True).stdout.strip()
                top = subprocess.run("ps -eo rss,comm --sort=-rss | head -4 | tail -3 | tr '\\n' ' '", shell=True,
                                     capture_output=True, text=True).stdout.strip()
                load = open("/proc/loadavg").read().split()[0]
                f.write(f"{time.strftime('%H:%M:%S')} ram_avail={av}MiB/{tot}MiB gpu_used={g}MiB load={load} top_rss_kB: {top}\n")
                if av < 4000:
                    LOG.write(f"[stage] WARN low RAM: {av} MiB available of {tot}\n")
                time.sleep(10)
    threading.Thread(target=run, daemon=True).start()


_watchdog()
D = ["--data-root", str(DATA)]


def stage(name, cmd, timeout, done_file=None, env=None):
    if done_file and Path(done_file).exists():
        LOG.write(f"[stage] {name} skip (exists)\n"); return 0
    LOG.write(f"[stage] {name} start {time.strftime('%H:%M:%S')}\n")
    p = popen_session(cmd, stdout=LOG, stderr=subprocess.STDOUT, env=env or ENV)
    try:
        rc = p.wait(timeout=timeout)
    except subprocess.TimeoutExpired:
        # the stage and every session it started (vLLM workers run in their own sessions)
        kill_session(p.pid); n = kill_registered(CHILD_PIDS); rc = "timeout"
        LOG.write(f"[stage] {name}: killed the stage and {n} registered sessions\n")
    unregister(p.pid, CHILD_PIDS)
    LOG.write(f"[stage] {name} done rc={rc} {time.strftime('%H:%M:%S')}\n")
    return rc


def need(rc, what):
    if rc != 0:
        LOG.write(f"[stage] ABORT: {what} failed (rc={rc})\n"); LOG.close(); os._exit(1)


def acc10(name):
    return json.loads((O / f"eval_{name}.json").read_text())["func_acc"]["10"]


need(stage("convert", [sys.executable, "-m", "issueloc.paperfmt", "/content/build/data", str(DATA)], 3600,
           DATA / "report_paperfmt.json"), "convert")
LOG.write(f"[stage] paperfmt report {(DATA / 'report_paperfmt.json').read_text()}\n")
(O / "rep.json").write_text(json.dumps({"rep": "R1", "note": "paper protocol: text fixed by the corpus format"}))
need(stage("mine it1", G + ["mine", "--model", "B0", "--rep", "R1", "--name", "it1"] + D, 7200, O / "kept_it1.json"), "mine it1")
need(stage("eval B0_R1", G + ["eval", "--model", "B0", "--rep", "R1", "--name", "B0_R1"] + D, 2400, O / "eval_B0_R1.json"), "B0")
if abs(acc10("B0_R1") - 60.22) > 0.5:  # calibration value on the same protocol and backend
    need(1, f"B0 Acc@10 {acc10('B0_R1'):.2f} differs from calibration 60.22")
stage("eval REF_R1", G + ["eval", "--model", "REF", "--rep", "R1", "--name", "REF_R1"] + D, 2400, O / "eval_REF_R1.json")
need(stage("train M1", G + ["train", "--inst", str(O / "inst_it1.parquet"), "--out-pt", str(O / "M1.pt"), "--gpu", "0",
                            "--steps-file", str(O / "steps_it1.json"), "--role", "leader", "--max-min", "90"] + D,
           7200, O / "M1.pt.json"), "train M1")
rc_a1 = stage("train A1", G + ["train", "--inst", str(O / "inst_it1_nofilter.parquet"), "--out-pt", str(O / "A1.pt"),
                               "--gpu", "0", "--steps-file", str(O / "steps_it1.json"), "--role", "follower"] + D,
              7200, O / "A1.pt.json")
import pyarrow.parquet as pq
pickle.dump(pq.read_table(O / "inst_it1.parquet", columns=["pos"]).column("pos").to_pylist()[:64],
            open("/content/tmp/check.pkl", "wb"))
ok = stage("check_ft M1", [sys.executable, "colab/check_ft.py", str(O / "M1.pt"), "/content/tmp/check.pkl"], 1800) == 0
ENV2, mult = dict(ENV), 1
if not ok:  # fine-tuned models on HF -> B0 must be re-scored on HF too (same backend for every model)
    ENV2["EMBED_BACKEND"], mult = "hf", 3
    os.replace(O / "eval_B0_R1.json", O / "eval_B0_R1_vllm.json")
    need(stage("eval B0_R1 (hf)", G + ["eval", "--model", "B0", "--rep", "R1", "--name", "B0_R1"] + D, 7200, None, ENV2), "B0 hf")
LOG.write(f"[stage] backend for all compared models: {ENV2['EMBED_BACKEND']}\n")
stage("eval M1", G + ["eval", "--model", str(O / "M1.pt"), "--rep", "R1", "--name", "M1"] + D, 3600 * mult, O / "eval_M1.json", ENV2)
if rc_a1 == 0:
    stage("eval A1", G + ["eval", "--model", str(O / "A1.pt"), "--rep", "R1", "--name", "A1"] + D, 3600 * mult, O / "eval_A1.json", ENV2)
else:
    LOG.write(f"[stage] A1 training failed (rc={rc_a1}); the ablation is incomplete\n")
need(stage("mine it2", G + ["mine", "--model", str(O / "M1.pt"), "--rep", "R1", "--name", "it2", "--keep-from", str(O / "kept_it1.json")] + D,
           7200 * mult, O / "mine_it2.json", ENV2), "mine it2")
need(stage("train M2", G + ["train", "--inst", str(O / "inst_it2.parquet"), "--out-pt", str(O / "M2.pt"), "--gpu", "0",
                            "--steps-file", str(O / "steps_it1.json"), "--role", "follower"] + D, 7200, O / "M2.pt.json"), "train M2")
stage("eval M2", G + ["eval", "--model", str(O / "M2.pt"), "--rep", "R1", "--name", "M2"] + D, 3600 * mult, O / "eval_M2.json", ENV2)
stage("compare", [sys.executable, "judge.py", "--dir", str(O)], 600)
LOG.write("[stage] ALL DONE\n")
LOG.close()
os._exit(0)
