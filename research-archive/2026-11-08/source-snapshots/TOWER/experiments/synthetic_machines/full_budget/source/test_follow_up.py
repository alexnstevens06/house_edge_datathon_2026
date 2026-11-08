"""No-training checks for the (case, seed) follow-up gate. Does not call main()."""
import ast
from pathlib import Path

from full_compare import follow_up_seeds

HERE = Path(__file__).resolve().parent
CASES = ["tiny", "cnn03", "mix"]
# Frozen server v1 seed 0: mix R is worse, so this is a full loss.
LOSS = [
    {"case": "cnn03", "seed": 0, "tune_R": 0.3291909269650826},
    {"case": "mix", "seed": 0, "tune_R": 0.3968037462541104},
]
TIE = [
    {"case": "cnn03", "seed": 0, "tune_R": 0.33},
    {"case": "mix", "seed": 0, "tune_R": 0.33},
]
WIN = [
    {"case": "cnn03", "seed": 0, "tune_R": 0.40},
    {"case": "mix", "seed": 0, "tune_R": 0.30},
]


def test_loss_and_tie_schedule_nothing():
    assert follow_up_seeds(LOSS, CASES, 0) == ()
    assert follow_up_seeds(TIE, CASES, 0) == ()


def test_seed0_win_schedules_exactly_1_and_2():
    assert follow_up_seeds(WIN, CASES, 0) == (1, 2)


def test_tuple_keys_reject_bare_case_strings():
    by = {(m["case"], m["seed"]): m for m in LOSS}
    assert ("mix" in by) is False and ("cnn03" in by) is False
    assert ("mix", 0) in by and ("cnn03", 0) in by
    assert follow_up_seeds(LOSS, CASES, 0) == ()


def test_source_ast_has_no_string_membership_on_by():
    tree = ast.parse((HERE / "full_compare.py").read_text())
    names = [n.name for n in tree.body if isinstance(n, ast.FunctionDef)]
    assert "follow_up_seeds" in names
    assert "main" in names
    for node in ast.walk(tree):
        if not isinstance(node, ast.Compare) or not any(isinstance(op, ast.In) for op in node.ops):
            continue
        left = node.left
        if isinstance(left, ast.Constant) and left.value in {"mix", "cnn03"}:
            for comp in node.comparators:
                if isinstance(comp, ast.Name) and comp.id == "by":
                    raise AssertionError("string membership against tuple-keyed by")
    fn = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "follow_up_seeds")
    win_return = False
    empty_return = False
    for node in ast.walk(fn):
        if not isinstance(node, ast.Return) or not isinstance(node.value, ast.Tuple):
            continue
        elts = [e.value for e in node.value.elts if isinstance(e, ast.Constant)]
        win_return = win_return or elts == [1, 2]
        empty_return = empty_return or elts == []
    assert win_return and empty_return
    main = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "main")
    calls = [n.func.id for n in ast.walk(main) if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)]
    assert "follow_up_seeds" in calls


if __name__ == "__main__":
    test_loss_and_tie_schedule_nothing()
    test_seed0_win_schedules_exactly_1_and_2()
    test_tuple_keys_reject_bare_case_strings()
    test_source_ast_has_no_string_membership_on_by()
    print("PASS loss=() tie=() win=(1, 2) ast_ok")
