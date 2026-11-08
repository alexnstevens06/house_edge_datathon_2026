"""Rung2 + final, predeclared: top3/model by rung1 tune_R (+cnnA) -> 30ep seed0 tune-only;
top1/model by rung2 tune_R -> 30ep seeds 0,1,2 phase final. Selection uses tune_R only."""
import json, re, subprocess, os, glob
os.chdir(os.path.expanduser("~/Documents/code/datathon_2026/house-edge")); R = "experiments/mlp_cnn_hpo/runs"
SP = dict(json.load(open(f"{R}/SEARCH_SPACE.json")))
def tune(n):
    m = re.search(r"tune_R': ([0-9.]+)", open(f"{R}/{n}.log").read() if os.path.exists(f"{R}/{n}.log") else "")
    return float(m.group(1)) if m else 9e9
def run(name, args, ep, seed, phase):
    if not os.path.exists(f"{R}/{name}.log") or "RESULT" not in open(f"{R}/{name}.log").read():
        subprocess.run(f".venv/bin/python experiments/mlp_cnn_hpo/source/train.py {args} --epochs {ep} --seed {seed} --phase {phase} --out {R}/{name} > {R}/{name}.log 2>&1", shell=True)
log = open(f"{R}/STAGE2_SELECTION.txt", "w")
best = {}
for m in ("cnn", "mlp"):
    r1 = sorted((tune("r1_" + n), n) for n in SP if n.startswith(m)); print(m, "rung1", r1, file=log, flush=True)
    promo = [n for _, n in r1[:3]] + (["cnnA"] if m == "cnn" and "cnnA" not in [n for _, n in r1[:3]] else [])
    for n in promo: run(f"r2_{n}", SP[n], 30, 0, "tune")
    r2 = sorted((tune("r2_" + n), n) for n in promo); print(m, "rung2", r2, file=log, flush=True); best[m] = r2[0][1]
for m, n in best.items():
    for s in (0, 1, 2): run(f"final_{n}_s{s}", SP[n], 30, s, "final")
print("FINAL", best, file=log, flush=True); open(f"{R}/STAGE2_DONE", "w").write(json.dumps(best))
