# Signal-specialist networks (House Edge) — task for OpenCode

Repo root R = /home/alexn/Documents/code/datathon_2026/house-edge . Work ONLY inside
R/experiments/signal_networks/{source,tests,runs}. NEVER edit anything else (src/, kit/, scripts/, runs/, other experiments).
Import existing code read-only: sys.path.insert(0, R/"src") and use house_edge.data / windows / models / predict / scoring.
Read those files first (src/house_edge/*.py) — they define Episode, SplitData, load_split, WindowCollator (sparse),
NextSpinTransformer, masked_ce_loss, score_predictions, load_official_validation, write_submission_csv, subset_split_data.

Representation fixed: SPARSE only. Backbone checkpoint: R/runs/server_bf16_sparse/best.pt
(torch.load(...,weights_only=True) -> {"state_dict","arch","config"}; config has d_model,nhead,num_layers,dim_ff,dropout,window).
Splits: R/runs/splits.json keys "tune","confirm" (48 validation episodes each). Data: R/kit/public/{train,validation,test}.
queries.csv columns: query_id,episode_id,machine_id,machine_type,signal,cutoff. signal in {-1,0,1}. Train labels: train/labels.csv (query_id,outcome).

## Rules (hard)
- Signal exists ONLY at the query cutoff per machine. Historical windows have no signal. Never copy query signal onto history/sliding examples.
- Fine-tuning uses ONLY train split query examples (house_edge.windows.query_examples) – 96 episodes x 8 machines labels. Never use validation targets/reference for fitting; validation used only for epoch selection on TUNE subset and for reporting.
- Every input token tick < cutoff (existing asserts preserved).

## File source/signal_nets.py
1. load signals: dict episode_id -> np.array(8,) int signal per slot (slot = Episode.slot_of(machine_id)), for any split.
2. class SignalTransformer(NextSpinTransformer): adds nn.Embedding(3, d_model) `signal_emb` initialized to ZERO, added to the 8 readout query vectors only (index signal+1). forward(batch) expects batch["signals_q"] (B,8) long in {0,1,2}. Load backbone state_dict with strict=False (only signal_emb missing).
3. Collate: wrap WindowCollator(sparse) and add "signals_q" from the signals dict for each example's episode.
4. Training function finetune(mode, out_dir, signal=None, epochs=30, lr=1e-4, batch_size=16, seed=0):
   - mode "shared": SignalTransformer, loss on all 8 machines of train query examples.
   - mode "specialist": copy of backbone (plain NextSpinTransformer, or SignalTransformer with signal_emb frozen at zero — choose plain NextSpinTransformer), loss mask = label_mask & (machine's own signal == signal). All 8 machines' history still in context. Skip batches with zero masked labels.
   - bf16 autocast on cuda (loss in float32), AdamW wd 0.01, grad clip 1.0, torch.set_num_threads(2), seed.
   - Epoch selection: after each epoch compute TUNE R with official scoring (score_predictions with episode_subset=set(tune)). For specialists, the tune score used for selection = the routed combined prediction where machines with this signal use the candidate specialist and other machines use the backbone (no-signal) prediction. Save best.pt (state_dict + config + mode + signal + backbone path + backbone sha256).
5. Routing predict: predict_routed(models_by_signal, data, signals) -> {query_id: 5 probs}: for each episode run all 3 specialists (and only take, for each machine slot, the output of the specialist matching that machine's signal). Assert every query used exactly its own signal's model; record route per query_id.
6. Output for each final system (backbone_nosignal, shared, specialists_routed): validation_predictions.csv (all 768 val queries) and test_submission.csv (all test queries) via write_submission_csv; metrics.json with tune/confirm/full_val official score dicts (score_predictions) incl. per-condition loss_ratio; route_log.csv for specialists (query_id,signal,model_used).
7. Label counts: count train query labels per signal x machine_type and per signal x outcome; also val/test query counts per signal. Write runs/label_counts.json.
8. Provenance: runs/provenance.json with sha256 of every source file, backbone ckpt sha256, kit.zip sha256, commands, torch version, start/end time, peak VRAM.

## source/run_all.py
CLI: --out R/experiments/signal_networks/runs/v1 --epochs 30. Does: label counts, evaluate backbone, train shared, train 3 specialists, routed eval, write everything, print summary JSON. Must finish < 15 min on RTX 5080; cap GPU memory torch.cuda.set_per_process_memory_fraction(4GB/total).

## tests/test_signal_nets.py (pytest, CPU ok, small)
- signal_emb zero-init => SignalTransformer output equals backbone output on a batch.
- routing: synthetic mixed-signal episode, three dummy models returning distinct constants -> each query gets its own signal's constant.
- no backfill: collated batch tensors for sliding/history contain no signal key except signals_q; history ticks < cutoff.
- specialist loss mask only covers machines with matching signal.
- prediction CSV: finite, rows sum to 1, 5 columns, ids == queries.csv ids.
- checkpoint reload reproduces predictions.

Do not run long training yourself; you may run pytest and a --epochs 1 smoke run with a tiny subset flag --smoke. Use python at R/.venv/bin/python if present else python3.
