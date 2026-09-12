# Surgery runbook — bf16mtp checkpoint (OPTIONAL Profile A variant)

Profile B does NOT need this: it serves halt95's checkpoint as-is.
Profile A's default does NOT need it either: the original checkpoint serves as-is
on 8 GPUs (measured 2026-09-12: boots TP4×PP2+EP+MTP-4 at util 0.92, KV pool
1,333,383 cold / 1,325,073 warm, P1 89.0 tok/s on prose probes and 104.2 on
technical prompts, 4-stream aggregate 267 same-prompt / 192-253 distinct,
acceptance 1.96 per draft, 4/4 oracle).

A same-meter A/B (2026-09-12, identical prompts, back-to-back boots) measured this
surgery's output head-to-head against the original checkpoint and found it NOT
faster: P1 89.3 vs 89.0, 4-stream 185-186 vs 267 (same-prompt) / 186 vs 192-253
(distinct), acceptance 1.92 vs 1.96 — and the BF16 draft costs ~106k of KV pool
(1,219,309-1,213,265 tokens = 4.63-4.65x vs 1,325,073). Run it ONLY to reproduce
the series' bf16mtp lane verbatim (series: P1 94.6, 4-stream 202.2, acceptance
~3.5 under its own prompts — a meter the original checkpoint was never run under).
It converts halt95's packed INT4 MTP draft-head experts to BF16 so the draft loads
through the fork's unquantized-draft path. Plain copies, no hardlinks.
~30 min on CPU, needs roughly 2x the checkpoint size in free disk
(in ~116 GiB, out ~119 GiB).

The scripts are byte-identical to the ones that produced and verified the
published measurements — they use module-level constants instead of CLI flags,
so you edit three paths instead of passing arguments.

## 1. Edit the constants

`surgery/convert_merlin_draft.py`:
```
SRC = ...   # halt95's checkpoint dir you downloaded (contains config.json + shards)
DST = ...   # where the bf16mtp copy goes (a NEW dir on a disk with ~119 GiB free)
```
`surgery/verify_merlin_draft.py`: same two constants
(`SRC` = halt95's original — the verifier regenerates independently from it;
`DST` = your output).

## 2. Run

```bash
python3 -m venv .venv && .venv/bin/pip install torch safetensors numpy  # CPU torch is fine
.venv/bin/python surgery/convert_merlin_draft.py 2>&1 | tee convert.log
.venv/bin/python surgery/verify_merlin_draft.py   # fail-closed: any mismatch -> nonzero exit
```

## 3. What success looks like

- convert.log pass 2: `model_extra_tensors.safetensors` written at 4.86 GiB
  (29 existing + 1,536 new BF16 draft-expert tensors, all fp32 round-trip
  requant checks passed); pass 3: 35 files copied bit-perfect.
- verify: index<->disk key exactness both directions, all 1,536 new tensors BF16
  with shape == source weight_shape, bitwise-identical to independent regeneration.

## 4. Serve

Point Profile A's `MODEL` at the DST dir (instead of halt95's original) — the
compose is unchanged. (The config ignore list is collapsed to `re:^mtp\..*` by the
script — that is the load path the series measurements used.)
