"""t152 independent extra checks. Usage: python extra_checks.py ENTRY PUBLIC WORKDIR. Never writes into ENTRY."""
import sys, os, shutil, subprocess, csv, math, json, random
from pathlib import Path
entry, public, work = map(lambda p: Path(p).resolve(), sys.argv[1:4])
work.mkdir(parents=True, exist_ok=True)
fails = 0; n = 0
def check(ok, label):
    global fails, n
    n += 1; fails += (not ok); print(("PASS " if ok else "FAIL ") + label, flush=True)
env = dict(os.environ, OMP_NUM_THREADS="2", MKL_NUM_THREADS="2", PYTHONDONTWRITEBYTECODE="1", CUDA_VISIBLE_DEVICES="", HIP_VISIBLE_DEVICES="")
def run(tree, out, data=public, split="test"):
    if out.exists(): out.unlink()
    return subprocess.run([sys.executable, str(tree/"predict.py"), "--data", str(data), "--out", str(out), "--split", split], text=True, capture_output=True, env=env, cwd=str(work))
def tamper(name, fn, needle):
    t = work/("tree_"+name)
    if t.exists(): shutil.rmtree(t)
    shutil.copytree(entry, t, ignore=shutil.ignore_patterns("__pycache__")); fn(t)
    p = run(t, work/(name+".csv"))
    check(p.returncode != 0 and needle in (p.stderr+p.stdout) and not (work/(name+".csv")).exists(), f"refuse {name} rc={p.returncode} needle='{needle}' msg={(p.stderr.strip().splitlines() or [''])[-1][:160]}")
def app(rel, b=b"\n# x\n"):
    return lambda t: open(t/rel, "ab").write(b)
tamper("model_py_edit", app("model.py"), "model.py hash mismatch")
tamper("data_py_edit", app("data.py"), "data.py hash mismatch")
tamper("s2_config_edit", app("components/s2/config.json", b" "), "s2")
tamper("s0_ckpt_truncate", lambda t: open(t/"components/s0/checkpoint.pt","r+b").truncate(1000), "s0 checkpoint hash mismatch")
tamper("missing_s2", lambda t: shutil.rmtree(t/"components/s2"), "unexpected component directory")
tamper("weights_reorder", lambda t: (t/"provenance/weights.json").write_text(json.dumps({**json.load(open(t/"provenance/weights.json")), "weights":[0.6,0.4]})), "weights manifest")
tamper("stray_file", lambda t: (t/"components/s0/x.txt").write_text("x"), "unexpected files in s0")
# wrong data dir
p = run(entry, work/"nodata.csv", data=work/"does_not_exist")
check(p.returncode != 0 and not (work/"nodata.csv").exists(), f"refuse missing data dir rc={p.returncode}")
# unit: probability mean + softmax-consistent fixture
sys.path.insert(0, str(entry)); os.chdir(work)
import predict as P, torch
random.seed(152)
la = torch.tensor([[random.gauss(0,3) for _ in range(5)] for _ in range(50)], dtype=torch.float64)
lb = torch.tensor([[random.gauss(0,3) for _ in range(5)] for _ in range(50)], dtype=torch.float64)
pa, pb = torch.softmax(la,-1), torch.softmax(lb,-1)   # consistent: probability_i = softmax(logit_i)
ma = {str(i): pa[i].tolist() for i in range(50)}; mb = {str(i): pb[i].tolist() for i in range(50)}
out = P.equal_probability_mean([ma, mb])
ref = ((pa+pb)/2); ref = ref/ref.sum(-1, keepdim=True)
err = max(abs(out[str(i)][j]-ref[i,j].item()) for i in range(50) for j in range(5))
check(err <= 1e-15, f"equal_probability_mean == renorm((pa+pb)/2) max_abs={err:.3g}")
lg = torch.softmax((la+lb)/2, -1)
gap = (ref-lg).abs().max().item()
check(gap > 1e-3, f"consistent softmax fixture: prob-mean differs from softmax(mean logits) gap={gap:.6g} (code uses prob-mean)")
check(all(abs(math.fsum(v)-1) < 1e-15 for v in out.values()), "ensemble rows sum to 1")
def expect_exit(fn, label):
    try: fn(); check(False, label+" did not refuse")
    except SystemExit as e: check(e.code != 0, f"{label} refused code={e.code}")
expect_exit(lambda: P.equal_probability_mean([ma, {**mb, "extra":[.2]*5}]), "mismatched ensemble ids")
expect_exit(lambda: P.equal_probability_mean([ma]), "single component")
expect_exit(lambda: P.canonical_renorm([float("nan"),1,1,1,1]), "nan probability")
expect_exit(lambda: P.canonical_renorm([0,0,0,0,0]), "zero mass")
print(f"EXTRA_SUMMARY checks={n} failures={fails}")
sys.exit(1 if fails else 0)
