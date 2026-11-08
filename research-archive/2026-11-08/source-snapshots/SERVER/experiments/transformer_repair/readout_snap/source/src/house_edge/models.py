"""Matched joint full-attention transformers for dense and sparse tokenization.

Both architectures share identical hyperparameters and (by construction)
identical parameter counts: 6-row symbol embedding (dense: {MISSING, payouts};
sparse: {payouts, PAD}), absolute-tick embedding, machine-slot embedding,
machine-type embedding, 8 learned readout queries, a bidirectional
TransformerEncoder over ALL tokens of all 8 machines, and a 5-way softmax head.

Causality: every input token has tick < target tick (asserted in forward).
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .data import MAX_TICK, MISSING, N_MACHINES, N_PAYOUTS

PAD = 5  # sparse pad symbol index (zero-initialized embedding row)


class NextSpinTransformer(nn.Module):
    def __init__(self, arch: str, d_model: int = 64, nhead: int = 4, num_layers: int = 2,
                 dim_ff: int = 128, dropout: float = 0.1, max_tick: int = MAX_TICK,
                 n_types: int = 3, n_slots: int = N_MACHINES):
        super().__init__()
        assert arch in ("dense", "sparse")
        self.arch = arch
        self.d_model = d_model
        self.symbol = nn.Embedding(6, d_model)          # dense: 0=MISSING,1..5 payouts / sparse: 0..4 payouts,5=PAD
        self.tick = nn.Embedding(max_tick + 1, d_model)
        import os as _os
        self.pos_mode = _os.environ.get("HE_POS", "abs")  # abs | abs+lag | lag
        self.readout_mode = _os.environ.get("HE_READOUT", "query")  # query | own
        assert self.readout_mode in ("query", "own"), f"bad HE_READOUT={self.readout_mode!r}"
        self.max_lag = 300
        self.lag = nn.Embedding(self.max_lag + 1, d_model)
        self.slot = nn.Embedding(n_slots, d_model)
        self.type_emb = nn.Embedding(n_types, d_model)
        self.readout = nn.Embedding(n_slots, d_model)   # one learned query per machine slot
        layer = nn.TransformerEncoderLayer(
            d_model, nhead, dim_feedforward=dim_ff, dropout=dropout,
            activation="gelu", batch_first=True, norm_first=True)
        self.enc = nn.TransformerEncoder(layer, num_layers=num_layers, enable_nested_tensor=False)
        self.head = nn.Linear(d_model, N_PAYOUTS)

    def _pos(self, ticks, target_t):
        lag = (target_t[:, None] - ticks).clamp(0, self.max_lag)
        if self.pos_mode == "abs":
            return self.tick(ticks)
        if self.pos_mode == "lag":
            return self.lag(lag)
        return self.tick(ticks) + self.lag(lag)

    def _readout_queries(self, slots_q: torch.Tensor, types_q: torch.Tensor) -> torch.Tensor:
        return self.readout(slots_q) + self.type_emb(types_q)

    def _own_extras(self, symbols, ticks, slots, valid, target_t):
        """HE_READOUT=own: per (batch, machine slot) extras for the readout token:
        embedding of the machine's most recent observed payout symbol plus the
        lag embedding of (target_t - last observed tick), clamped. Computed from
        INPUT tokens only (all ticks < target_t, asserted by callers); machines
        with no observed token in the window contribute a zero vector.
        symbols/ticks/slots: (B, N); valid: (B, N) bool = observed input token."""
        B = symbols.shape[0]
        last_sym = torch.zeros(B, N_MACHINES, dtype=torch.long, device=symbols.device)
        last_tick = torch.zeros(B, N_MACHINES, dtype=torch.long, device=symbols.device)
        has = torch.zeros(B, N_MACHINES, dtype=torch.bool, device=symbols.device)
        for m in range(N_MACHINES):
            match = valid & (slots == m)
            masked = torch.where(match, ticks, torch.full_like(ticks, -1))
            last = masked.max(dim=1).values
            has[:, m] = last >= 0
            idx = masked.argmax(dim=1)  # valid only where has
            last_sym[:, m] = symbols.gather(1, idx.unsqueeze(1)).squeeze(1)
            last_tick[:, m] = torch.where(has[:, m], last, torch.zeros_like(last))
        lag = (target_t[:, None] - last_tick).clamp(0, self.max_lag)
        extras = self.symbol(last_sym) + self.lag(lag)
        return extras * has.unsqueeze(-1)

    def forward_dense(self, symbols, ticks, slots, types, target_t):
        """symbols/ticks/slots/types: (B, W*8); target_t: (B,) -> logits (B, 8, 5)."""
        assert (ticks < target_t[:, None]).all(), "dense causality violated"
        B = symbols.shape[0]
        x = self.symbol(symbols) + self._pos(ticks, target_t) + self.slot(slots) + self.type_emb(types)
        q = self._readout_queries(slots[:, :N_MACHINES], types[:, :N_MACHINES])
        if self.readout_mode == "own":
            q = q + self._own_extras(symbols, ticks, slots, symbols > 0, target_t)
        x = torch.cat([x, q], dim=1)
        h = self.enc(x)
        return self.head(h[:, -N_MACHINES:])

    def forward_sparse(self, symbols, ticks, slots, types, pad_mask, slots_q, types_q, target_t):
        """symbols/ticks/slots/types: (B, N); pad_mask True=pad; -> logits (B, 8, 5)."""
        assert (ticks < target_t[:, None]).all(), "sparse causality violated"
        x = self.symbol(symbols) + self._pos(ticks, target_t) + self.slot(slots) + self.type_emb(types)
        q = self._readout_queries(slots_q, types_q)
        if self.readout_mode == "own":
            q = q + self._own_extras(symbols, ticks, slots, ~pad_mask, target_t)
        x = torch.cat([x, q], dim=1)
        mask = torch.cat([pad_mask, pad_mask.new_zeros(pad_mask.shape[0], N_MACHINES)], dim=1)
        h = self.enc(x, src_key_padding_mask=mask)
        return self.head(h[:, -N_MACHINES:])

    def forward(self, batch: dict) -> torch.Tensor:
        if self.arch == "dense":
            return self.forward_dense(batch["symbols"], batch["ticks"], batch["slots"],
                                      batch["types"], batch["target_t"])
        return self.forward_sparse(batch["symbols"], batch["ticks"], batch["slots"],
                                   batch["types"], batch["pad_mask"], batch["slots_q"],
                                   batch["types_q"], batch["target_t"])

    def predict_probs(self, batch: dict) -> torch.Tensor:
        return F.softmax(self.forward(batch), dim=-1)


def masked_ce_loss(logits: torch.Tensor, labels: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Cross-entropy over observed next-tick outcomes only; (B,8,5),(B,8),(B,8)bool."""
    B = logits.shape[0]
    flat = F.cross_entropy(logits.reshape(-1, N_PAYOUTS), labels.reshape(-1).clamp(min=0), reduction="none")
    flat = flat.reshape(B, N_MACHINES)
    total = (flat * mask).sum()
    count = mask.sum()
    assert count > 0, "batch has no observed target labels"
    return total / count


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
