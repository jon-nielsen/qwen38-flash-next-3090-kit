# Surgery runbook — bf16mtp checkpoint (needed for Profile A only)

Profile B does NOT need this: it serves halt95's checkpoint as-is.
This converts halt95's packed INT4 MTP draft-head experts to BF16 so the draft loads
through the fork's unquantized-draft path. Plain copies, no hardlinks. ~30 min on
CPU, needs roughly 2x the checkpoint size in free disk (in ~116 GiB, out ~119 GiB).

The scripts are byte-identical to the ones that produced and verified the
measurements in the README — they use module-level constants instead of CLI flags,
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

Point Profile A's `MODEL` at the DST dir. (The config ignore list is collapsed to
`re:^mtp\..*` by the script — that is the load path the measurements used.)
