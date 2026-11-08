"""Predeclared CNN-only refinement (frozen 19:03 CST before results): 6 more random configs from the SAME frozen CNN space
(rng seed 5678) at 8 epochs tune-only; top2 -> 30 epochs seed0 tune-only; if best rung2 tune_R < cnn03 rung2 (0.3303)
then 3 seeds final, else keep cnn03. Kernel not searched (train.py ignores --kernel)."""
import json, math, random, os, re, subprocess
os.chdir(os.path.expanduser("~/Documents/code/datathon_2026/house-edge")); R = "experiments/mlp_cnn_hpo/runs"
r = random.Random(5678); lu = lambda a, b: round(math.exp(r.uniform(math.log(a), math.log(b))), 5); C = {}
for i in range(6): C["cnnR%d" % i] = f"--model cnn --h {r.choice([64,96,128,160])} --layers {r.choice([3,4,5])} --W {r.choice([32,64,96])} --drop {r.choice([0.1,0.2,0.3])} --lr {lu(3e-4,3e-3)} --wd {lu(1e-3,1e-1)}"
json.dump(C, open(f"{R}/REFINE_SPACE.json", "w"), indent=0)
def tune(n):
    p = f"{R}/{n}.log"; m = re.search(r"tune_R': ([0-9.]+)", open(p).read()) if os.path.exists(p) else None; return float(m.group(1)) if m else 9e9
def run(n, a, ep, s, ph):
    if "RESULT" not in (open(f"{R}/{n}.log").read() if os.path.exists(f"{R}/{n}.log") else ""):
        subprocess.run(f".venv/bin/python experiments/mlp_cnn_hpo/source/train.py {a} --epochs {ep} --seed {s} --phase {ph} --out {R}/{n} > {R}/{n}.log 2>&1", shell=True)
L = open(f"{R}/REFINE_SELECTION.txt", "a")
for n, a in C.items(): run("r1_" + n, a, 8, 0, "tune")
r1 = sorted((tune("r1_" + n), n) for n in C); print("rung1", r1, file=L, flush=True)
for _, n in r1[:2]: run("r2_" + n, C[n], 30, 0, "tune")
r2 = sorted((tune("r2_" + n), n) for _, n in r1[:2]); print("rung2", r2, "vs cnn03 0.3303", file=L, flush=True)
if r2[0][0] < 0.3303:
    for s in (0, 1, 2): run(f"final_{r2[0][1]}_s{s}", C[r2[0][1]], 30, s, "final")
    print("WINNER", r2[0][1], file=L, flush=True)
else: print("KEEP cnn03", file=L, flush=True)
open(f"{R}/REFINE_DONE", "w").write("done")
