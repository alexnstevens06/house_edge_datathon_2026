# stationary_categorical (RESEARCH_ONLY)

Synthetic slot panel where each machine's payout is i.i.d. from a fixed categorical distribution determined by `machine_type`. Invented toy weights. Not fit on contest logs.

Eligibility is **unresolved**. The rules say to use organizer-released data and do not state a verified synthetic exception. Competition training and submissions stay on the **real-only fallback** (organizer-released train, and later the released adaptation train). Do not train a submission on these files.

## Law

- Payouts, in probability-index order: `0, 1, 2, 5, 20` → columns `p_0, p_1, p_2, p_5, p_20`.
- Type index order matches `src/house_edge/data.py`: `balanced=0`, `jackpot=1`, `steady=2`.
- Unnormalized toy weights `(0, 1, 2, 5, 20)`:
  - steady `(72, 18, 6, 3, 1)`
  - balanced `(40, 25, 18, 12, 5)`
  - jackpot `(50, 15, 10, 10, 15)`
- Stored oracle probabilities divide by the weight sum and put the remainder on the last bin so the five numbers sum to 1.
- Types are shuffled from that cycle per episode, from the seed, before tick 0. Outcomes do not change types.
- Each cell uses its own RNG stream `(seed, episode, tick, machine)`. A tick is drawn for every machine from that pre-tick state, then committed. No machine reads another machine's same-tick payout. The state is not updated after the tick.
- Each cell is observed with probability `0.7` (MCAR). `observation_mask` is `1` or `0`. Unobserved rows store `payout: null`, `payout_index: null`, and `dense_symbol: -1` (the house-edge `MISSING` code). That is not payout `0` (`dense_symbol` 0, mask 1). A 6-way token uses index `5` for missing.
- `slot_index` is `machine_id` (`0 .. n_machines-1`, default `0..7`).
- History ticks are `0 .. n_ticks-1`. The prediction cutoff is `n_ticks` (the next spin). No row is at or after the cutoff.
- No query signal is invented. Organizer `signal` is `-1`, `0`, or `1` only on a real query at the cutoff. Oracle and history probabilities ignore `signal`.

## Two probability objects

- **Latent oracle** (`truth[].probabilities`): the type categorical. Same at every tick. Not a function of spins.
- **Observable history** (`observable_history_probabilities`): Dirichlet-multinomial posterior with prior α = `(1,1,1,1,1)` on that machine's **observed** payouts at ticks **strictly `< t`**. Missing rows are skipped even if a payout number was filled in. Other machines are ignored. This is not the oracle.

## Interface

```python
from generate import generate
panel = generate(seed=0, n_episodes=8, n_ticks=96, n_machines=8)
# panel["records"], panel["truth"]
```

Default size is `8 * 96 * 8 = 6144` records, at or under the `8192` cap.

```bash
/home/alexn/Documents/code/datathon_2026/house-edge/.venv/bin/python generate.py \
  --seed 0 --episodes 8 --ticks 96 --out out
```

Writes `records.jsonl`, `truth.jsonl`, and `manifest.json` (`RESEARCH_ONLY`, seed, settings, sha256 of sources and outputs).

## Tests

```bash
/home/alexn/Documents/code/datathon_2026/house-edge/.venv/bin/python -m unittest test_generator.py -v
```

Run from this directory. The command, result, and hashes are recorded in `manifest.json`.
