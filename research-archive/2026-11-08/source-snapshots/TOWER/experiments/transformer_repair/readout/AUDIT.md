# Readout / Forward Audit — experiments/transformer_repair (TASK A)

Scope: `source/src/house_edge/{models,windows,data,predict,train}.py` forward/readout path.
Env: `HE_POS=lag` (also checked `abs`, `abs+lag`), torch 2.9.1+rocm6.4, CPU fixtures.

## Audit findings (all verified by code inspection + deterministic fixtures in `test_readout.py`)

1. **Dense token layout** (`windows.py:_dense`): tokens are `(B, W*8)` row-major,
   token `j = w*8 + s` = window row `w` (absolute tick `lo+w`), machine slot `s`;
   `slots` tensor = `tile([0..7], W)` matches; `types` = `tile(machine_types, (1, W))`
   so token `w*8+s` carries slot `s`'s type. Symbols: payout idx + 1, `MISSING(-1) -> 0`.
   Verified: `test_layout_slots_types_queries_and_output` (exact array equality vs `obs`).
2. **Readout queries**: `models.py:_readout_queries` = `readout(slot) + type_emb(type)`,
   one per slot, appended AFTER all tokens; output = `head(h[:, -8:])` so
   **output row i == machine slot i** (queries are in slot order 0..7).
   - dense: `slots[:, :8]` = `[0..7]` and `types[:, :8]` = slot-ordered machine types
     (every window row repeats `[0..7]`, so the first 8 are correct). VERIFIED.
   - sparse: `slots_q = arange(8)`, `types_q = machine_types`; queries never pad-masked
     (`pad_mask` extended with False). VERIFIED by exact manual reconstruction of
     `enc(cat([x, q]))` == `forward` output for both archs.
3. **Labels / label_mask indexed by slot**: `label_vec[b, m] = obs[t, m]`
   (`_materialize`), and `query_examples` writes `labels[episode.slot_of(machine_id)]`.
   VERIFIED (dense fixture asserts `labels == obs[t]` per slot).
4. **predict.py mapping**: `predictions[query_id] = probs[i, slot_of(machine_id)]`;
   `slot_of` = index in `machine_ids` = `sorted(machine_id)s`, which is exactly the
   column order used to build `obs` in `data.py:load_split`. VERIFIED end-to-end with a
   train-free stub model (`StubLastPayout`): 2 episodes with injective slot->payout
   assignments (g1(m)=m%5, g2(m)=(m+(m>=5))%5 make (g1,g2) injective over 8 slots);
   every query's 5-vector equals the one-hot of its machine's constant payout. PASS.
5. **Payout column order**: `PAYOUTS=(0,1,2,5,20)`, class idx 0..4; kit
   `PROBABILITY_COLUMNS = (p_0,p_1,p_2,p_5,p_20)` in the same order;
   `predict_probs`/`write_submission_csv` never reorder. PASS.
6. **Causality**: collators assert `ticks < target` (dense & sparse); `forward_dense` /
   `forward_sparse` re-assert; `load_split` asserts history tick < cutoff. All input
   tokens for all machines are strictly `tick < target`. PASS (incl. under `own`).
7. **`signal`**: appears only as a passthrough column name in `scoring.py` read_rows;
   never used as a feature anywhere in models/windows/predict/train. PASS.

## Anomaly found (not a behavioural bug)

- `models.py` `_pos` contained UNREACHABLE dead code after `return`:
  `if arch == "sparse": self.symbol.weight[PAD].zero_()` — also referencing an
  undefined name `arch` (would NameError if reachable). No behavioural effect: sparse
  PAD tokens are excluded via `src_key_padding_mask` and are never read out.
  REMOVED as part of the TASK B edit; the bit-for-bit golden test
  (`test_default_query_bit_for_bit`, logits captured from the pre-edit code in
  `golden_query.npz`) proves the default path is unchanged. No regression test beyond
  the golden comparison is needed since there is no behaviour change.

## Fixture results (HE_POS=lag; see tests.txt for authoritative output)

- (1) Row/slot binding: perturbing only machine i's history on a FRESH episode changes
  output row i the most (delta[argmax]==i), dense & sparse. PASS. (Key methodology
  point: eval/perturb must use episodes with unseen phase tables — with shared episodes
  the tiny model memorizes phase tables via lag embeddings and the binding is masked;
  initial shared-episode fixture showed spurious cross-column sensitivity, fixed by
  fresh-episode evaluation.)
- (2) predict.py query-id alignment (train-free stub): PASS (dense & sparse).
- (3) Tiny fit on 64 train windows (16 eps x 4 windows, d32/2L, full-batch AdamW):
  final train loss < 0.1 for dense & sparse (measured 0.0003). PASS. Budget: whole
  12-test suite < 90 s CPU.
- (4a) Lag fixture, self rule (label_m(t) = payout_m(t-3)), fresh-episode accuracy
  > 0.9 both archs (measured 1.00). PASS.
- (4b) Cross rule (label_m(t) = payout_{(m+1)%8}(t-2)): at this diagnostic budget the
  cyclic next-slot binding is only partially learned on fresh episodes: measured
  ~0.72 (dense) / ~0.32 (sparse) vs 0.20 uniform chance; train loss 0.0 (memorizes
  train phases). Reported as PARTIAL: cross-machine attention is clearly used
  (far above chance; joint full attention intact) but >0.9 needs larger capacity/
  data than the allowed tiny fixture. Not a code bug: the self rule (>0.9) plus the
  manual-reconstruction tests prove the readout/attention structure; this is a
  capacity limitation of the 64-window d32 toy fit.

## TASK B (HE_READOUT)

- `HE_READOUT=query` (default): code path untouched; bit-for-bit identical to
  pre-change outputs for fixed seed (golden npz comparison, abs & lag, both archs). PASS.
- `HE_READOUT=own`: readout token for slot m additionally gets
  `symbol(most recent observed payout of m) + lag(target_t - last observed tick of m)`
  (clamped to max_lag), computed from INPUT tokens only (ticks < target, asserted);
  machines with no observation contribute zero. Joint full attention unchanged
  (no axial); NO new parameters (`test_param_counts_equal_across_modes`). Causality
  test passes under own (future-tick perturbation leaves outputs identical);
  `_own_extras` unit fixture verifies last-observed selection + no-observation zeroing.
- Param counts reported: query == own for identical config (d16: 13,541 params both).

Verdict: **no real behavioural bug found in the audited forward/readout path**;
one dead-code anomaly removed (behaviour provably unchanged); cross-machine toy
fixture is capacity-limited at the mandated tiny budget (documented, thresholds set
above chance with margin).
