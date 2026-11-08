"""Subprocess refusals for predict_guarded.py. CPU only. No dataset reads expected.

(a) state_dict key sig.weight -> exit 2 before any forecast write.
(b) altered tensor whose file sha256 misses PINNED.json -> exit 2 before any forecast write.
"""
import os
import shutil
import subprocess
import sys
from pathlib import Path

os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["HIP_VISIBLE_DEVICES"] = ""

REVIEW = Path(__file__).resolve().parent
REPO = REVIEW.parents[2]
RUN = REVIEW.parent / "server_snapshot" / "runs" / "final_cnn03_s0"
PY = REPO / ".venv" / "bin" / "python"
GUARD = REVIEW / "predict_guarded.py"
TMP = REVIEW / "tmp"
FAILS = []


def check(name, ok, detail=""):
    line = ("PASS" if ok else "FAIL") + " " + name + ((" | " + detail) if detail else "")
    print(line, flush=True)
    if not ok:
        FAILS.append(line)


def run_guard(run_dir, out_dir):
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = ""
    env["HIP_VISIBLE_DEVICES"] = ""
    env["OMP_NUM_THREADS"] = "2"
    proc = subprocess.run(
        [str(PY), str(GUARD), str(run_dir), str(out_dir)],
        cwd=str(REPO),
        env=env,
        capture_output=True,
        text=True,
    )
    return proc


def assert_refused_before_data(name, proc, out_dir, needle):
    err = proc.stderr.strip()
    out = proc.stdout.strip()
    check(name + " exit", proc.returncode == 2, "code=%s" % proc.returncode)
    check(name + " message", needle in err, err.replace("\n", " ")[:400])
    check(name + " no_stdout_write", "wrote" not in out, out[:200])
    check(name + " out_dir_absent", not out_dir.exists(), str(out_dir))


def build_sig_run():
    import torch
    dest = TMP / "guard_sig" / "final_cnn03_s0"
    if dest.exists():
        shutil.rmtree(dest)
    dest.mkdir(parents=True)
    shutil.copy(RUN / "config.json", dest / "config.json")
    blob = torch.load(RUN / "checkpoint.pt", map_location="cpu", weights_only=False)
    state = blob["state"] if isinstance(blob, dict) and "state" in blob else blob
    state["sig.weight"] = torch.zeros(3, 128)
    torch.save(blob, dest / "checkpoint.pt")
    return dest


def build_pin_run():
    import torch
    dest = TMP / "guard_pin" / "final_cnn03_s0"
    if dest.exists():
        shutil.rmtree(dest)
    dest.mkdir(parents=True)
    shutil.copy(RUN / "config.json", dest / "config.json")
    blob = torch.load(RUN / "checkpoint.pt", map_location="cpu", weights_only=False)
    state = blob["state"] if isinstance(blob, dict) and "state" in blob else blob
    key = next(k for k, v in state.items() if hasattr(v, "is_floating_point") and v.is_floating_point())
    state[key] = state[key].clone()
    state[key].view(-1)[0] = state[key].view(-1)[0] + 0.125
    torch.save(blob, dest / "checkpoint.pt")
    return dest, key


def main():
    import torch
    torch.set_num_threads(1)
    sig_dir = build_sig_run()
    pin_dir, pin_key = build_pin_run()
    print("pin altered key", pin_key, flush=True)
    sig_out = TMP / "guard_sig_out"
    pin_out = TMP / "guard_pin_out"
    if sig_out.exists():
        shutil.rmtree(sig_out)
    if pin_out.exists():
        shutil.rmtree(pin_out)
    sig_proc = run_guard(sig_dir, sig_out)
    assert_refused_before_data("sig.weight", sig_proc, sig_out, "sig params")
    pin_proc = run_guard(pin_dir, pin_out)
    assert_refused_before_data("pin_mismatch", pin_proc, pin_out, "checkpoint sha256")
    print("FAILS", len(FAILS))
    return 1 if FAILS else 0


if __name__ == "__main__":
    raise SystemExit(main())
