# Synthetic machine review (first pass)

Snapshot: 2026-11-07T19:12:18-06:00. Read-only except this file. No generator, test, utility, or checkpoint files were edited. No GPU, install, git, or network. This pass did not wait for the fleet.

## Verdict

No family is eligible for use, scoring, or training. All 12 `grok_parallel` directories exist and contain only `TASK.txt`, `start_epoch`, and `build.log`. `generate.py`, `test_generator.py`, `manifest.json`, and `README` are absent in every family. `grok_utility` contains only `build.log`.

Absence is not a failed implementation and not a pass. Causality, masks, payout maps, shared state, and oracle-versus-history behavior are **unresolved** until source is on disk and tests run against that source.

Research-only eligibility stays unresolved. `kit/RULES.md` says "Use organizer-released data" and puts test answers and private organizer files off limits. A full-file search for `synthetic`, `organizer-released`, and `private` hits only that rule. This review grants no synthetic exception. Real-only fallback stays in force.

## What was inspected

| Path | sha256 | Role |
| --- | --- | --- |
| `kit/RULES.md` | `fa9d5538a0f13e063927d8a393cfe1d46e2b0defe23cdf0646157b9f6ce7ce44` | Eligibility yardstick. Keyword search of the whole file; lines 25–45 read. |
| `src/house_edge/data.py` | `5cde790750b3b69187f6b92c2da949ca6e11de8d750e6b8a4959131da89c5c7f` | Public observation contract. Lines 1–25 read. |
| `grok_utility/build.log` | `87a805ab81ac8fbee0c4521ebcb26fff1aedc0414365ff9aae0a3660a251c6b7` | Intent text only. 389 bytes. |
| `utility/build.log` | not a code artifact | 29-byte stub (`> build · glm-5.2`). Not the utility runner. |

`start_epoch` is identical in all 12 families: bytes of `1794100137` plus newline, sha256 `d636ab8ae818de34f409e60fecb2d3b658f8fe166f735af6e8c35f86b73f5655`. That is a start timestamp, not a simulator seed file.

Public contract used as the later grading yardstick, from `data.py`: payouts `(0, 1, 2, 5, 20)` map to indexes `0..4`; `MISSING = -1` means unknown, never payout 0; `N_MACHINES = 8`; type index order is `balanced=0`, `jackpot=1`, `steady=2`. The family task text lists types as `steady/balanced/jackpot` without saying that order is the index order. Future generators must not treat task-list order as the slot index.

## Actual tests

Command (stdlib probe, bytecode disabled):

```text
PYTHONDONTWRITEBYTECODE=1 \
  /home/alexn/Documents/code/datathon_2026/house-edge/.venv/bin/python
```

The probe listed 12 family directories at 19:12:18-06:00, required `generate.py`, `test_generator.py`, and `manifest.json`, and found none. `unittest` was not started. A discover run with zero modules would be a vacuous pass, so it was not used as evidence.

Result: exit code 2. `stdlib_tests_run = 0`. No probability, reproducibility, prefix, missing-versus-zero, or oracle test executed.

## Family table

Severity `UNRESOLVED` means no source to judge. It does not mean the family is clean or broken.

| Family | Severity | On disk | Tests | Provenance finding |
| --- | --- | --- | --- | --- |
| abrupt_changepoints | UNRESOLVED | task + log only | not run | no generator source |
| directed_lagged_crossmachine | UNRESOLVED | task + log only | not run | no generator source |
| finite_order_markov | UNRESOLVED | task + log only | not run | no generator source |
| gradual_drift | UNRESOLVED | task + log only | not run | no generator source |
| hidden_regime_switching | UNRESOLVED | task + log only | not run | no generator source |
| long_lag_dependence | UNRESOLVED | task + log only | not run | no generator source |
| machine_type_hierarchical | UNRESOLVED | task + log only | not run | no generator source |
| periodic | UNRESOLVED | task + log only | not run | no generator source |
| rare_heavytail | UNRESOLVED | task + log only | not run | no generator source |
| shared_latent_crossmachine | UNRESOLVED | task + log only | not run | no generator source |
| sparse_gappy | UNRESOLVED | task + log only | not run | no generator source |
| stationary_categorical | UNRESOLVED | task + log only | not run | no generator source |

### Task and log hashes

| Family | TASK.txt | build.log |
| --- | --- | --- |
| abrupt_changepoints | `b9b61acea5e63b60264b415c15d035b70dfa03cae0e70ce0160b8a8fe3cf4d07` | `3f8995811fc72d15d64d498daac6d53b2b7d15f62f5000c9e88ff0e68ffe5e3c` |
| directed_lagged_crossmachine | `ab9f9f0a3f40bcdc5a0cb0b76140e9aaeef23e397de62b7af5ccc8f246c45c62` | `74a5f56910068447d6ce33a66d07177d969450777fdbf9241fe8b81b5441b286` |
| finite_order_markov | `4b301d8d533858f7e29ed3cc8e78b6e884566eedc47d9b9511bfbbfcc046b9f9` | `cda7690a412bdfaacc65876c618a0b749fbcfadeb5cafabe20f16cadd4b9f909` |
| gradual_drift | `edfa9577eddc2947d759a1ee9a1502de569565b505eee034fac00d59d9df01d2` | `66e1180e67ca516b56e8dcda76834c3abc83287734b70968579560961aa64a37` |
| hidden_regime_switching | `956528018e1b975dd2fdc52b1339f92396ab32ebd0e6fcaae8dae8940b2b23a1` | `9b7e9d40b1d557461b7b37297fcb2ce5cf9a4884c748971c6b0d6c604ee9a48e` |
| long_lag_dependence | `5bae72f2bf60795fa2dc9598fdcacfaed5bbac1911f811d6d1b2d5f8cbdc3ca1` | `c07099fc10ba80fa84510296be44c0e395eb1f0c942fab01fefb6b8e714511c2` |
| machine_type_hierarchical | `7f2a8e1ba5bd2394eac829a830dbbb088fa56d57ed5d7a53c9475d83636a0185` | `a10ee2b2ed5a29828737cc1a59f4a776277277a0a803d87d48330a5eda1b95c5` |
| periodic | `b4e131ee02579ee046336e2c274dc5801c9d29a6c3eb4fdcbcea07408d373e63` | `c822daef6fd2778a4e0463f1bbe362fb81218c5e8b4c1354b1d85135515f3d38` |
| rare_heavytail | `e9a4b15de32b46927f45f19772aaff9fcab27e41d10d6aed7ffd139de1793a15` | `c7d285579a6b7c0e7f4599cfdb81c17cdb175732cff64495b41082fcff1cb085` |
| shared_latent_crossmachine | `d0320eae88945f159d693b6e2e07636853bc0bf3a218945f0f5fa030bedba302` | `52d4c975b7d33839e388930262703fb6f144d5307af3475b576f53a5dccf62e0` |
| sparse_gappy | `4ac585944cf462f0db080d0c8084ed5d742bf69dd27d998609ca9b1c63404dda` | `41fab25bf9695bfd9d55809e1aca7903c19dde10ac8d0641e7dcc81deb69972d` |
| stationary_categorical | `14041e221cb5638a6e6bb7c707ea8723b349871e8cc7ddf2fa63d2fef8550014` | `1dc8616b54391a871ffb6990e7670cc7d910d22633ea476d1c04c9a392594c5e` |

Task texts are the same contract with a different family name. They require a self-authored toy generator, separate `records` and `truth`, synchronous updates from ticks `< t`, no contemporaneous cross-machine outcomes, missing unknown rather than 0, and distinct latent-oracle versus observable-history probabilities. Those requirements are unread code, not satisfied checks.

Logs are short intent notes. `rare_heavytail` and `shared_latent_crossmachine` say they will read a public schema and avoid validation and test. This review did not trace process file access, so those sentences are not evidence of what was opened.

No directory contained an organizer simulator, a private seed bundle, or a fitted val/test parameter file. That only describes this snapshot.

## Utility sibling

`experiments/synthetic_machines/grok_utility` has no Python, config, split list, metric table, or checkpoint. Severity for any utility claim: **BLOCKER** (nothing to audit). Each row is unverified, not confirmed safe.

| Check | Result |
| --- | --- |
| Matched steps against a generator | UNVERIFIED. No runner and no generator outputs. |
| Source mix / exposure | UNVERIFIED. No sample accounting. |
| Real TRAIN-only fitting | UNVERIFIED. No fit entry point. |
| Selection on the existing tune split only | UNVERIFIED. No selection record. |
| No confirm set and no test set | UNVERIFIED. No data-path list. |
| Checkpoint quarantine | UNVERIFIED. No checkpoint directory. |

`grok_utility/build.log` says a runner was about to be written and that the first comparison would wait for a family file. Waiting text is not a quarantine design.

## Still required before any later pass can pass a family

A later review has to open the real `generate.py` bytes and re-hash them. It still has to see:

- Prefix invariance: mutating ticks `>= t` leaves probabilities and records at ticks `< t` unchanged, including other machines at the same tick.
- Missing mask: an unobserved cell stays unknown and is not stored or summed as payout 0 or payout index 0.
- Map: machine id `0..7`, payout value to index `(0,1,2,5,20) -> (0,1,2,3,4)`, and type codes aligned with `TYPE_TO_IDX` rather than task-list order.
- No mutable shared state across `generate()` calls or episodes (RNG, caches, running counts).
- Truth probabilities that depend on hidden state are labeled as oracle truth and are not presented as the probability conditional on observable history.
- Manifest `RESEARCH_ONLY`, source hashes matching the reviewed bytes, and a recorded test command whose exit code is from those tests.

Until those files exist, synthetic rows stay out of training and out of model selection.

## Second pass (2026-11-07T19:26:03-06:00)

The first pass is not withdrawn as a snapshot. Generators are now on disk. This pass reran tests and read source. No generator, test, utility, or cloud file was edited. Eligibility is still unresolved. `kit/RULES.md` still has no synthetic exception. No flagship claim and no contest training on these rows.

Public adapter yardstick, unchanged: `TYPE_TO_IDX` is balanced=0, jackpot=1, steady=2, applied by string in `load_split` (`src/house_edge/data.py`). `slot_of` is the index of a machine id inside that episode's sorted id tuple, not "integer id equals type index". Payout index 0 is an observed 0. `MISSING=-1` is unknown. Family task-list order and machine slot are not that map.

### Rerun

`PYTHONDONTWRITEBYTECODE=1 taskset -c 0` and `/home/alexn/Documents/code/datathon_2026/house-edge/.venv/bin/python -m unittest test_generator.py` in each family directory. Hashes are the bytes at the start of that command. A later drift was rerun.

| Family | generate.py | test_generator.py | manifest.json | Result |
| --- | --- | --- | --- | --- |
| abrupt_changepoints | `6f6e29c88e831c72084645d418bd51ba8173b20231e97d17bd70d27850bf2dde` | `42053f14f70e1b8d0ab9476e67f6d3361b0f48315a600fbb8760848b307133ba` | `086d895a3dbaced38a28c781f93e5a9d6038a660a87290949f65c37f6b4c6c1b` | 10 tests, exit 0, 0.568s |
| finite_order_markov | `3683360edb6810ad583b00cb3d329e318b56465b34238041cd981be012ac30ac` | `de2b68a001a77460af0934be304c32d68166ba8d3f84f6d4cde6fe46e212c0df` | `a48ab55c27b9cd2532a8835d81f08f194425b0e4f2541339e4836f2525b041fe` | 12 tests, exit 0, 0.032s |
| gradual_drift | `86e4df42a42bdb7bc17838af6a1466d45f661eedaae2fc0200e55ccfb329eec2` | `cefac7be71a4f1ce8f274f0eb2c16d3d888edaee36009fb5dc2f7081f066f0f0` | `93a0a3bc7ce7f0d180434ad144e5b68108c5c1f1aed21d2a615d3df4a4eb1620` | 9 tests, exit 0, 0.033s |
| hidden_regime_switching | `705bfad75ffe4bb56cdaba9c804f450d622ed9f143406d347613535c9ce75f63` | `6c4aeb2a891a0b1550e31143f7a8fcb7a622fcf80ef2c30ba8eb4b60968599ee` | `6b246c25a7c85a8535de655f7d5e5781124bebffdbd6c4a4caf5f8116be651b0` | 9 tests, exit 0, 0.147s |
| long_lag_dependence | `dc924859645bfcf8ec12703115d98fe1f4744ed770e83ce38b62b2f61f952821` | `86136d20afcc35935a48e72b6086e70b99c0eb6bb7c92e7564cceffa79af6ccc` | `a64c09124cd0696485637b226a5bf949f9ff875a135abd23297e32ae3cf7703e` | 7 tests, exit 0, 0.540s |
| machine_type_hierarchical | `787c12028fa59e7c0ce2b65c2ea06cac89eb57a8b8dbd44ad0793d980f1dcb9e` | `f365bdd53039dfc058ef0993ca7ab426eeab8a6271e0c7a5d495a9c01c4fdb6d` | `6bb6896ab8af258b6822199e3995507a4fa925d1b10930c0b1c14fc57e868e9e` | 16 tests, exit 0, 0.045s |
| periodic | `20be07cd82f857d37772ff6d010368f9baad35dcd8214ddebbec5d7a85710ef6` | `79efcb8fc15226a1340ff042f1a004a6c49e39cd3b93970f9af2bd77d0a718b5` | `5f2fbacf4f4e9a6e98e2ae51c60683af61848afd5687e4f6394a4ecb6ba5517c` | 10 tests, exit 0, 0.235s (manifest appeared after the first suite; this is the rerun) |
| rare_heavytail | `859918026a7a7bbd5218261725fc2f21335e723c6e86433a08f627e3dcb333c9` | `8c65728aa89c16ce136087b4b50f26f353d215fa51abc78eb9cdb036843f0c64` | `7de57398be6a66418902552cbdfe01750ce42844e414f927bebe33f718aead20` | 22 tests, exit 0, 0.109s |
| shared_latent_crossmachine | `2e35812b0c74f58327debbbe478b546d560ec87d1a0d4eefc0fe6c9f16372f3c` | `141ad7b0d595141991fe4e79297535fe5dc1b031be7be223953b611254ff671a` | `f4a4628f99f4c884929b392e87ce4fa616f54b3fc8a54edb17552e8ba94fa731` | 18 tests, exit 0, 0.147s |
| sparse_gappy | `6ea9fb789db6e7e90577c6a80101b659b9849f2898bc65ab132be33a32b32204` | `69d34a5553d4cf8dff362f50d3caa026902ca7137083a6eff3e0126a1ec59386` | `24362b31b690fd33c7ec08e4251afcf13c1a3682d8e8d5c7cb202d1c2a07ba94` | 15 tests, exit 0, 0.097s |
| stationary_categorical | `607a801ca1549373c6e596eba4c69a88efee07c36f64e0885980eebfbd750eaa` | `b8228732d4283c2dbcac3b70395ad33d67a210afb03d5835110d67c32a901cee` | `70fa8690e2b5c8070df8d19cf41170b843cefb920ff75c1bfafb3bd45079656d` | 15 tests, exit 0, 0.132s |
| directed_lagged_crossmachine | `514de62f57914d49b733d30bfc546b7474cdb6e1623513406efd1a4d73b2a785` | missing | missing | not run |

Superseded rare bytes failed before the file changed: generate `e9c0503827b7c94205a0f273a5494079ba2c2f92474c34443ab13e2f78ca6640`, test `744d748cef3ae59225da903e6b9041d5c0f1a505597f4031fa8021f9684fce98`, 22 run, 2 failures (`test_shorter_run_is_the_prefix_of_a_longer_run`, `test_over_cap_and_bad_shape_are_refused`), exit 1. Those bytes are not the passing hash above. Do not cite the failure against `85991802`.

Stationary manifest source hashes match the table. Long-lag manifest source hashes match generate and test; its on-disk manifest hash is the table value `a64c0912...`.

Grep of `generate.py` for `load_split`, `kit/public`, and data-file reads found only output writers and comments that say the toys are not fit. That is not a full syscall trace.

### Actionable blockers

1. **BLOCKER — research-only.** Do not train, select, or export a contest model on these rows. No family manifest grants an exception.

2. **BLOCKER — no utility promotion.** Inspected `grok_utility/ADAPTER_CONTRACT.txt` `c76a17191c138d50af7a288a2008c4c05c1b4e1800881e8a9fd937e685334889`, `PROMOTION_RULE.txt` `f758e664839f3af71b30063df1d6cecffd264a1519f0b98116f104031437dd62`, `source/base_cnn.py` `4d4c44aa748ad2dbaeb2c38d857bf4b3bd65e0ac4a034ec4668a2e8e72c6fd31`. Contract: semantic type lookup, never task-list order or machine slot. Promotion: three matched seeds 0, 1, and 2 against real-only controls before any flagship claim, and every descendant stays research-only. There is no runner, no seed-0/1/2 score, no tune-split selection record, and no checkpoint directory. `tests/` is empty. `base_cnn.py` embeds `typ` with 3 rows and, on the real loader, those rows are `TYPE_TO_IDX[machine_type string]`. It does not contain a synthetic adapter. `USE_SIGNAL` is false. The copied `if __name__` refuses export. Do not point it at confirm or test, and do not call passing generator tests a utility result.

3. **BLOCKER — directed lag, do not consume.** `TYPE_TO_INDEX` is `steady=0, balanced=1, jackpot=2` and `mappings()["type_to_index"]` exports it, with a note that this is task order. A probe of `514de62f` (not a unit test) showed episode 0 tick 0 stable when `n_ticks` goes from 4 to 7, and episode 1 tick 0 types and payouts change, because one `random.Random(seed)` draws types and payouts for every episode. Unobserved rows still store `payout` (example: payout 5 with `feature_payout_index=-1`). No `test_generator.py` yet. Absence of a red test is not a pass.

4. **BLOCKER — cloud types stay unmapped.** `CLOUD_PR_REVIEW.txt` `50bf161b340eb140c4a2fef48cf0ed2d4ccf3fd31874c21e7b321c3e5b4af7d5`. Rerun: `PYTHONPATH=cloud_import` plus the same interpreter, `python -m unittest burst_refractory.tests.test_generators`, 18 tests, exit 0, 0.059s. Generator `95f3c9a1668afff016e5723e8096bbefe4861ae04a8a98ae6a12e9389f0389e8`. Tests `24d92d43b8814be097fa9f423c7ea714f6a8cc746ed90528e804cd1b9b78392c`. A discover run without that `PYTHONPATH` failed to import; that was the invocation, and the package run above is the result. Burst types are spark (machines 0–3) and cascade (4–7). Refractory types are quick, steady, deep. `steady` here is not a license to use public `TYPE_TO_IDX["steady"]`, and slot order is not balanced/jackpot/steady. Do not integrate this tree. The API is not the 12-family `generate()` contract.

5. **BLOCKER — mask before features.** Several generators keep the latent payout on unobserved rows (`payout` filled, `observed` false): abrupt, finite-order, gradual, long-lag, hierarchical, shared-latent, sparse, rare, directed. Stationary and hidden store a null payout. Public windows need `-1`, not 0 and not the hidden spin. `rare_heavytail` `DENSE_SYMBOL_MISSING = 0` means unknown, while an observed 0 is code 1. That 0 is not `MISSING=-1` and not payout index 0. `gradual_drift` sets `slot_index` to the payout index on every row, including gaps. Do not treat `slot_index` as a machine slot or a type index.

6. **HIGH — which vector is the conditional law.** An empirical posterior is not the observable-conditioned law unless the toy says so and the math matches.
   - Stationary, known fixed type: `oracle_probabilities(machine_type)` is already `P(Y_t | history)` because the law does not depend on spins. The uniform-Dirichlet `observable_history_probabilities` is a shrinkage estimator. `test_latent_oracle_differs_from_observable_history` shows estimator versus oracle. It does not show a hidden regime. Score and calibrate against the oracle when the type string is known.
   - Long-lag: oracle conditions on latent lags 24 and 48 even if unobserved. The observable vector drops a missing lag (`None`) instead of mixing the tilt over that latent payout. Manifest calls this `same_formula_with_missing_lags_dropped`. Do not score a model against it as the true conditional.
   - Gradual and abrupt history vectors are pseudo-count or empirical frequencies. Gradual's oracle EWMA updates from the latent spin even when `observed` is false, using a pre-tick snapshot so the same tick does not leak. That oracle is not the filter of phase given masks. The count vector is not that filter either.
   - Hierarchical oracle shrinks toward a hidden machine base and adds a jackpot heat term from the last observed 5 or 20. The history reader uses `TYPE_PROTOTYPE` only. Neither row is the posterior over the house and machine latents.
   - Finite-order belief update, hidden-regime emission mixture, and shared-latent belief are written as filters of their own toys (marginalize a gap; do not code it as 0). They can be the observable-conditioned law of that toy when the type string is known. They are still research-only.

7. **HIGH — do not import task-order index dicts.** Safe index dicts on the reviewed bytes: stationary, periodic, hidden, long-lag (`balanced=0, jackpot=1, steady=2`), and periodic writes `machine_type_index` from the string after `MACHINE_TYPE_BY_SLOT` (slot 0 is the string `steady`, index 2, not 0). Unsafe if used as embedding ids: abrupt `TYPE_TO_INDEX` (`steady=0, balanced=1, jackpot=2`, defined and not referenced in that file), directed export above, and any `enumerate` of `MACHINE_TYPES = ("steady", "balanced", "jackpot")` in gradual, finite-order, hierarchical, and shared-latent. Those files emit the type string. The adapter has to look the string up. It must not use `machine_id` or `slot`.

8. **MEDIUM — gradual query cutoff.** `cutoff = n_ticks - 1`, so the last generated tick is the label. `history_before` keeps `tick < cutoff`. A loader that treats every record as history leaks that label.

Passing tests on the ten green families are mostly non-vacuous (prefix mutation, missing versus zero, kernel or replay checks), and they still do not authorize promotion. Directed has no such test. Cloud's 18 tests do not cover a public type map or an arbitrary future-history mutation, which `CLOUD_PR_REVIEW.txt` already records.
