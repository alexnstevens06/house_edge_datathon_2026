"""Write SPLIT_IDS.json from the frozen split file and TRAIN ids. Not a metric."""
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import soft_moe as S

def main():
    splits = json.load(open(S.SPLITS_PATH))
    train = S.load_split(os.path.join(S.PUB, "train"), with_labels=True)
    query_ids = [row["query_id"] for row in train.queries]
    missing = [qid for qid in query_ids if qid not in train.query_labels]
    payload = {
        "splits_source": "runs/splits.json",
        "tune": list(splits["tune"]),
        "confirm": list(splits["confirm"]),
        "train_episode_ids": [ep.episode_id for ep in train.episodes],
        "n_train_episodes": len(train.episodes),
        "n_train_queries": len(train.queries),
        "n_train_labels": len(train.query_labels),
        "unlabeled_train_queries": missing,
        "labels_cover_all_train_queries": len(missing) == 0 and len(train.query_labels) == len(train.queries),
        "machine_type_map": {"balanced": 0, "jackpot": 1, "steady": 2},
        "payouts": [0, 1, 2, 5, 20],
        "missing": -1,
        "signal_codes": {"-1": 0, "0": 1, "1": 2, "ABSENT": 3},
        "history_rule": "t in [16, cutoff); signal channel ABSENT; labels from history obs only",
        "cutoff_rule": "t == cutoff; TRAIN labels from labels.csv; signal only on these rows",
    }
    out = os.path.join(HERE, "..", "SPLIT_IDS.json")
    json.dump(payload, open(out, "w"), indent=1)
    print(out, "episodes", payload["n_train_episodes"], "labels_ok", payload["labels_cover_all_train_queries"])
    if not payload["labels_cover_all_train_queries"]:
        raise SystemExit(2)

if __name__ == "__main__":
    main()
