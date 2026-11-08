"""Max abs per probability after kit read_submission normalization.

Usage: compare_csv.py A.csv B.csv
PASS when the largest absolute difference is <= 1e-8. Exit 1 otherwise.
"""
import sys
from pathlib import Path

TOL = 1e-8


def main(argv):
    if len(argv) != 3:
        print("usage: compare_csv.py A.csv B.csv", file=sys.stderr)
        return 2
    repo = Path(__file__).resolve().parents[3]
    sys.path.insert(0, str(repo / "kit"))
    from slot_forecasting.io import read_submission
    from slot_forecasting.schema import PROBABILITY_COLUMNS

    left = read_submission(argv[1])
    right = read_submission(argv[2])
    if set(left) != set(right):
        missing = sorted(set(left) - set(right))[:5]
        extra = sorted(set(right) - set(left))[:5]
        print("id mismatch missing_from_B=%s missing_from_A=%s" % (missing, extra))
        return 2
    per = {col: 0.0 for col in PROBABILITY_COLUMNS}
    max_abs = 0.0
    worst = None
    for query_id in left:
        av = left[query_id]
        bv = right[query_id]
        for idx, col in enumerate(PROBABILITY_COLUMNS):
            diff = abs(av[idx] - bv[idx])
            if diff > per[col]:
                per[col] = diff
            if diff > max_abs:
                max_abs = diff
                worst = (query_id, col, diff)
    for col in PROBABILITY_COLUMNS:
        print("%s max_abs %.12g" % (col, per[col]))
    print("max_abs %.12g" % max_abs)
    if worst is not None:
        print("worst %s %s %.12g" % worst)
    status = "PASS" if max_abs <= TOL else "FAIL"
    print(status)
    return 0 if status == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
