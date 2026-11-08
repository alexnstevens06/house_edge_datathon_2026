"""CPU fp32, 2-thread export from a checkpoint. Usage: predict_cpu.py RUN_DIR OUT_SUFFIX [--audit]

Contamination guard (REVIEW_OPENCODE.md Q4): refuses (exit 2, BEFORE loading any split data) a
run directory containing CONTAMINATED.txt (or any *CONTAMINATED* marker) or a checkpoint config
without no_signal=True (signal-leak mode). With --audit the export proceeds and every output
file is written with an AUDIT_INVALID_ prefix so it can never be mistaken for a submission CSV."""
import os, sys, json, argparse, torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import cnn
from house_edge.data import load_split
from house_edge.scoring import load_official_validation, score_predictions, write_submission_csv
torch.set_num_threads(2)

def contamination_reason(run):
    """Return refusal reason for contaminated/non-compliant run dirs, else None. Loads only the
    checkpoint config — never split data — so the guard fires before any history/queries read."""
    if not os.path.isdir(run): return f"run directory not found: {run}"
    if any("CONTAMINATED" in n for n in os.listdir(run)):
        return "CONTAMINATED.txt / contamination marker present in run directory"
    ck = os.path.join(run, "checkpoint.pt")
    if not os.path.exists(ck): return f"no checkpoint.pt in {run}"
    c = torch.load(ck, map_location="cpu")["config"]
    if not c.get("no_signal", False):
        return "checkpoint config lacks no_signal=True (signal-leak mode, REVIEW_OPENCODE.md Q1)"
    return None

def main(argv=None):
    ap = argparse.ArgumentParser(description="CPU fp32 export from a checkpoint")
    ap.add_argument("run"); ap.add_argument("suffix"); ap.add_argument("--audit", action="store_true",
        help="export a contaminated/non-compliant run anyway; outputs get an AUDIT_INVALID_ prefix")
    a = ap.parse_args(argv)
    run, suf = a.run, a.suffix
    reason = contamination_reason(run)  # guard fires BEFORE any split data is loaded
    if reason and not a.audit:
        print(f"REFUSED {run}: {reason}; pass --audit to export anyway (outputs get AUDIT_INVALID_ prefix)", file=sys.stderr)
        sys.exit(2)
    prefix = "AUDIT_INVALID_" if reason else ""
    ck = torch.load(os.path.join(run, "checkpoint.pt"), map_location="cpu"); c = ck["config"]
    ns = c.get("no_signal", False); cnn.USE_SIGNAL = not ns
    m = cnn.CNN(c["h"], c["layers"], 3, c["drop"], c["W"], use_signal=not ns); m.load_state_dict(ck["state"]); m.eval()
    va = load_split(os.path.join(cnn.PUB, "validation")); te = load_split(os.path.join(cnn.PUB, "test"))
    Qv, ev = cnn.build_query(va, c["W"]); pv = cnn.to_preds(cnn.predict(m, Qv, "cpu"), ev, va)
    Qt, et = cnn.build_query(te, c["W"]); pt = cnn.to_preds(cnn.predict(m, Qt, "cpu"), et, te)
    write_submission_csv(os.path.join(run, f"{prefix}validation_cpu_{suf}.csv"), pv)
    write_submission_csv(os.path.join(run, f"{prefix}test_cpu_{suf}.csv"), pt)
    off = load_official_validation(os.path.join(cnn.PUB, "validation")); sp = json.load(open(os.path.join(cnn.REPO, "runs", "splits.json")))
    met = {k: score_predictions(pv, off, s) for k, s in (("full", None), ("tune", set(sp["tune"])), ("confirm", set(sp["confirm"])))}
    json.dump(met, open(os.path.join(run, f"{prefix}metrics_cpu_{suf}.json"), "w"), indent=1)
    print("CPU", run, prefix or "clean", {k: round(met[k]["R"], 4) for k in met}, {k: round(v, 4) for k, v in met["full"]["loss_ratio_by_condition"].items()})

if __name__ == "__main__": main()
