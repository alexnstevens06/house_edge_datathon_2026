import csv, math, sys
q = [r["query_id"] for r in csv.DictReader(open(sys.argv[2]))]
r = list(csv.reader(open(sys.argv[1]))); h, rows = r[0], r[1:]
ids = [x[0] for x in rows]; bad = 0; dev = 0
for x in rows:
    p = [float(v) for v in x[1:]]; bad += (len(p) != 5 or not all(math.isfinite(v) and 0 <= v <= 1 for v in p)); dev = max(dev, abs(math.fsum(p)-1))
ok = h == ["query_id","p_0","p_1","p_2","p_5","p_20"] and len(rows) == len(q) and len(set(ids)) == len(ids) and ids == q and bad == 0 and dev < 1e-12
print(f"CSVCHECK {'PASS' if ok else 'FAIL'} rows={len(rows)} unique={len(set(ids))} order_matches_queries={ids==q} bad={bad} max_sum_dev={dev:.3g}")
sys.exit(0 if ok else 1)
