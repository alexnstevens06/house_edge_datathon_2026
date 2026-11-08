import json, glob, os, shutil
R = os.path.dirname(os.path.abspath(__file__)) + "/../runs"
rows = []
for f in glob.glob(R + "/*/metrics.json"):
    n = f.split("/")[-2]
    if n == "best": continue
    m = json.load(open(f)); rows.append((m["tune"]["R"], n, m["confirm"]["R"], m["full"]["R"], m["full"]["loss_ratio_by_condition"]))
rows.sort()
for r in rows: print("%-26s tune %.4f confirm %.4f full %.4f %s" % (r[1], r[0], r[2], r[3], {k: round(v, 3) for k, v in r[4].items()}))
best = rows[0][1]; print("SELECTED (by tune):", best)
if os.path.exists(R + "/best"): os.remove(R + "/best")
os.symlink(best, R + "/best")
