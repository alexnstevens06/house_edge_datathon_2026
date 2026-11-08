"""Import a sibling generate.py and write records.jsonl and truth.jsonl.

Read-only toward the family directory. cwd is the output directory the parent passes.
"""
import importlib.util
import json
import sys
from pathlib import Path


def convert(obj):
    if obj is None or isinstance(obj, (str, int, float, bool)):
        return obj
    if isinstance(obj, dict):
        return {str(k): convert(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [convert(v) for v in obj]
    try:
        import numpy as np
        if isinstance(obj, np.generic):
            return obj.item()
        if isinstance(obj, np.ndarray):
            return convert(obj.tolist())
    except ImportError:
        pass
    raise TypeError(f"generate() value of type {type(obj).__name__} is not JSON-safe")


def main():
    gen_path, out_dir, seed, n_episodes, n_ticks, n_machines = sys.argv[1:7]
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    gen_path = str(Path(gen_path).resolve())
    sys.path.insert(0, str(Path(gen_path).parent))
    spec = importlib.util.spec_from_file_location("family_generate", gen_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load generate module from {gen_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    if not hasattr(module, "generate"):
        raise AttributeError(f"{gen_path} has no generate function")
    result = module.generate(
        seed=int(seed), n_episodes=int(n_episodes), n_ticks=int(n_ticks), n_machines=int(n_machines)
    )
    if not isinstance(result, dict):
        raise TypeError(f"generate returned {type(result).__name__}, expected dict")
    records = result.get("records")
    truth = result.get("truth")
    if not isinstance(records, list) or not isinstance(truth, list):
        raise TypeError("generate() dict must contain list fields 'records' and 'truth'")
    for name, rows in (("records.jsonl", records), ("truth.jsonl", truth)):
        path = out / name
        with path.open("w", encoding="utf-8") as handle:
            for row in rows:
                handle.write(json.dumps(convert(row), separators=(",", ":")) + "\n")
    (out / "return_keys.json").write_text(json.dumps(sorted(str(k) for k in result.keys())), encoding="utf-8")
    print(f"WROTE records={len(records)} truth={len(truth)}", flush=True)


if __name__ == "__main__":
    main()
