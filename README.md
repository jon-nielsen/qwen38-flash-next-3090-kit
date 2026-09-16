# Qwen3.8-Flash-Next: full 262k context with MTP-4 on 4x RTX 3090 (24 GB) — serving kit

Two measured launch profiles for serving **halt95/Qwen3.8-Flash-Next-W4A16-Merlin**
(W4A16 quant of Qwen3.8-Flash-Next) on RTX 3090-class hardware, plus the runtime
image recipe. The headline result: the
4-GPU profile serves the model at its full 262,144-token context with MTP-4.
Everything here was measured on a real rig — no estimates. Both profiles run on
halt95's checkpoint exactly as downloaded.

**v2 (2026-09-16):** runtime rebased onto the fork's v0.13.0 line and re-measured
end to end. Profile B gains ~+12% single-stream / ~+40% 4-stream aggregate over
the v1 kit numbers. The v1 release (fork 3bec27573, v0.11.3 line) is preserved
verbatim under the [`v1` tag](../../tree/v1).

## Quick start

Serving is front-doored through the HF kit — it carries the composes and the
consumer walk-through (requirements, checkpoint download, kit download,
`docker compose up`, success lines):

https://huggingface.co/Jon-Nielsen/Qwen3.8-Flash-Next-Merlin-3090-Serving-Kit

This repo is the build and provenance side: `Dockerfile` + `tree/` reproduce
the runtime image `ghcr.io/jon-nielsen/vllm-backport-flashnext-sm86:2b21fbe-bfe237ee`
(see "Image provenance and rebuild" below).

## The two profiles, as measured (v2)

| | Profile B (4 GPUs) — recommended | Profile A (8 GPUs) |
|---|---|---|
| Shape | TP2×PP2 + EP + MTP-4 | TP4×PP2 + EP + MTP-4 |
| Context | 262,144 | 262,144 |
| KV cache | FP8 E4M3 + calibrated sidecar | BF16 |
| Draft head | INT4 packed (halt95's original) | INT4 packed (halt95's original) |
| KV pool | 438,539 tokens (1.67x) | 1,317,519 tokens (5.03x) |
| Single-stream | 105.6 tok/s | 131.0 tok/s (warm) |
| 4-stream aggregate | ~267 median, ceiling 279 | 238.7 median (223.5-265.3) |
| Concurrent streams | 4 (max_num_seqs) | 32 (headroom unmeasured past 4) |
| Checkpoint | halt95's, as-is | halt95's, as-is |

Profile B is the default for the typical 2-4 concurrent-stream load: full context,
best aggregate throughput per GPU, and it sustained a ~4h15m real-traffic shift
with no restarts. Profile A is the capacity play: 3x the KV pool for deep-context
or many parallel sessions, plus the best single-stream latency. Numbers are
decode-only (768-token generations, temp 0, 12-pass pooled meter).

## What changed in v2 (and how it was validated)

The runtime moved from the fork's v0.11.3 line to its v0.13.0 line (base image
`lazmio/vllm-backport@sha256:bfe237ee...`, fork tag v0.13.0 = a350766628), with
our FP8-KV/QSA port rebased onto the v0.13 kernels and a merge of fork master
(cde54e8ed3) that brings in Hauck's PP-rank and mamba-state fixes (#76/#77) and
lazmio's `VLLM_DETERMINISTIC_MOE_ALIGN=0` default. The merge's single conflict
(ngram_embedding.py) resolved keep-ours: our fused FP8 UVA lookup kernel stays.

A same-day 5-arm ladder on identical hardware isolated each change (TP2×PP2,
262k, pool 438,539 on every arm; oracle 4/4 on every arm):

| Arm | P1 (2 passes) | 4-stream agg | Verdict |
|---|---|---|---|
| control (v0.13 rebase, DET=1) | 90.9 / 96.5 | ~199 flat | baseline |
| det0 (DET=0) | 103.2 / 110.1 | ~273, ceiling 282 | DET=0 is the big win |
| sync (+ fork-master merge) | 108.4 / 112.2 | ~265 median, ceiling 281 | merged tree holds |
| syncmoe (+ fused_moe aux overlap) | flat | ~211 | REJECTED: -20% at seqs=4 |
| bakeval (baked image, no mounts) | 105.6 / 105.1 | ~267, ceiling 279 | published image |

Profile A (TP4×PP2) was then validated end-to-end on the baked image (2026-09-16):
pool 1,317,519, P1 131.0 warm, 4-stream median 238.7, oracle 4/4 semantically
correct. The published tag is the bakeval arm — the bytes that were measured are
the bytes that ship.

DET=0 note: under DET=0 the 4-GPU lane alternates fast/slow decode passes
(~215 floor / ~270 ceiling band structure, real and reproducible). Medians pool
both bands; single-pass numbers are not comparable across arms.

## Rig constraints that are baked into these profiles (do not remove)

- **The PP trio** (`NCCL_P2P_DISABLE=1`, `VLLM_SKIP_P2P_CHECK=1`,
  `--disable-custom-all-reduce`): without it, PP hangs at NCCL communicator setup on
  multi-GPU PCIe rigs where NVML reports P2P OK but the VBIOS/BAR1 blocks it. Common
  on 3090 boards. Missing the first flag cost a full hang once; all three stay.
- **No `expandable_segments`**: its tax scales with memory pressure (measured +13-37%
  multi-stream on starved lanes, 0 single-stream, bitwise-stable). Removed from both.
- `NCCL_PROTO=LL128`: +6.2% on PCIe/SHM transports.
- **x16 vs x8 slots: myth.** With P2P disabled, NCCL falls back to SHM; an x8 quartet
  measured bitwise-identical outputs and within-noise speed vs the x16 quartet.
- `--enable-expert-parallel` is forced: 512-expert/128-group shapes never divide
  evenly under plain TP on this quant.
- **Do not add the fused_moe aux-stream overlap** (upstream's shared-experts
  overlap): measured -20% at max_num_seqs=4 — a large-batch design on a small-batch
  lane. The syncmoe arm exists to document the rejection.

## Honest limits of the measurements

- Profile B caps at `max_num_seqs=4` concurrent streams (by design; 4-stream numbers
  are at the cap). Its only stability data beyond the meter is one ~4h15m
  real-traffic shift with no restarts (error rates not instrumented during it).
  Profile A allows 32 streams; headroom past 4 was never measured.
- FP8-KV acceptance is ~3.0 under load vs ~3.5 BF16 — inside the band halt95 reports
  for the same design (2.43-2.78). Inherent FP8 numerics, not a port defect; it is
  the price of the 1.67x pool at 262k.
- KV pool varies ~0.6% boot-to-boot (memory-profiler variance).
- Deep prefill tested to 68k tokens; the meter is decode-only by construction
  (~20-token prompts), so prefill/TTFT is not characterized — deep-prefill
  behavior at 262k is untested.
- Quality gating: a 4-prompt greedy oracle, 4/4 correct on every v2 arm.
  Bitwise output identity held across five TP2×PP2 arms (same TP shape); the
  TP4×PP2 Profile A diverges from them at the reduction-order level, as expected
  physics — compare output hashes only within the same profile shape.
- Cross-day comparisons carry ~±10% day drift; same-day numbers above were
  measured back-to-back.
- `restart: unless-stopped` is the only lifecycle change vs the measured configs;
  engine arguments are untouched (the shipped composes — in the HF serving kit —
  differ from the measured lanes only in image/container/port/naming and the
  `:?` key guard).

## The stack and who made it

| Part | Source | Author |
|---|---|---|
| Base model + BF16 parts | Qwen3.8-Flash-Next via Intel's release | Qwen / Intel |
| W4A16 experts (INT4 g128) | Intel/Qwen3.8-Flash-Next-W4A16-AutoRound | Intel (AutoRound) |
| FP8 PLE n-gram table | RadixArk/Qwen3.8-Flash-Next-NVFP4 | RadixArk |
| MTP draft INT4 pack, GDN INT8, KV sidecar, assembly | Qwen3.8-Flash-Next-W4A16-Merlin | halt95 (packer adapted from DominikBucko/qwen38-flash-next-2x3090) |
| vLLM fork (sm_86 backport, v0.13.0) | wtdcode/vllm-backport, fork tag v0.13.0 (a350766628); image lazymio/vllm-backport | wtdcode / lazmio |
| Flash-Next arch (qwen4_exp) + PLE CPU-offload | in the fork; upstream PRs #53896 (merged), #53899 (open) | huanghaoyan.hhy (Alibaba) |
| PP-with-MTP draft fix | fork commit (v1 lineage, carried in v0.13) | Matri Ning |
| PP-rank raw-input fix; mamba block-size-on-read | fork PRs #76/#77 (in the merged tree) | Hauck |
| DET=0 default; pinned-PLE byte-move; PLE ignore-entry fix | fork master cde54e8ed3 (in the merged tree) | lazmio |
| Draft-unquantized block (mtp.py) | fork commit | lazmio |
| FP8 E4M3 QSA KV reader (Triton integer-decode, sm_86) | halt95 patches/0001, ported to v0.13 in the merged tree | halt95 (Apache-2.0) |
| v0.13 rebase of the FP8-KV/QSA port + PP-aware sidecar loader fixes, PP KV-alloc cross-rank parity fix, INT4-draft probe gate, FP8-PLE embedding gate, merged-tree bake | this kit (tree/) | Jon-Nielsen |
| Draft INT4->BF16 surgery + fail-closed verifier (optional variant) | this kit (tag v1, surgery/) | Jon-Nielsen |
| Profiles, knob measurements, rig findings, 5-arm ladder, 8-GPU v2 validation, live-traffic shift | this kit (composes in the HF serving kit) | Jon-Nielsen |

## Image provenance and rebuild

The v2 image = base `lazmio/vllm-backport@sha256:bfe237ee...` (the fork's v0.13.0
sm_86 build) COPY-merged with the 11 files in `tree/` (10 vLLM runtime files +
halt95's KV-scales sidecar to /opt). Those 10 files are exactly what the image
changes vs its base, so the full delta is reproducible from public materials:
clone `wtdcode/vllm-backport` at tag `v0.13.0` (a350766628) and diff against
`tree/vllm/` (e.g. `diff -ru <fork-clone>/vllm tree/vllm`).
`Dockerfile` is the build; the OCI labels carry the full provenance.
Published: `ghcr.io/jon-nielsen/vllm-backport-flashnext-sm86:2b21fbe-bfe237ee`,
digest `sha256:e88c57b4485ce2c577b283ec5de5ad02329953d0e62a94106980ede4a9e2fd45`.

Rebuild procedure (what was actually done for the published tag):

1. Stage: COPY the 11 files from the validated worktree (byte-verified with cmp
   against the tree commit — published bytes = validated bytes).
2. Build: `docker build` from this repo root; tag as
   `<tree-sha7>-<base-digest8>` (here: `2b21fbe-bfe237ee`). Never move a
   published tag — publish a new one and re-validate.
3. Verify in-image: md5 the 11 files against the worktree; import vllm; check
   the DET default.
4. Boot + meter + oracle on the ACTUAL image (the bakeval arm) before pushing.
5. Push to GHCR; verify with `docker buildx imagetools inspect` (digest recorded
   in the README of the HF serving kit).

Two PP-awareness fixes were required that halt95's PP=1 lane never hit:
sibling-stage sidecar entries must be ignored, not fatal, and the validation
loop must skip non-local layer names. The v0.13 rebase additionally fixed
KV-cache group allocation under PP (empty groups must be kept for cross-rank
index parity and skipped in the tensor builder).

## v1 and the optional surgery (previous release)

The v1 kit (fork 3bec27573, v0.11.3 line; image
`ghcr.io/jon-nielsen/vllm-backport-flashnext-sm86:3bec275-bb1f7777`, still
published and pinned) lives verbatim under the [`v1` git tag](../../tree/v1) —
overlays, patches, Dockerfile, composes, surgery and README as originally
released. v1 measured: Profile B P1 94.2 / 4-stream 189.3 / pool 416,490;
Profile A pool 1,333,383.

Also under the tag: the optional bf16mtp draft-head surgery (`surgery/` there)
— an INT4→BF16 dequant + fail-closed verifier, kept only to reproduce the v1
series lane. The v1-era same-meter A/B (2026-09-12) found it NOT faster than
halt95's original checkpoint: equal single-stream, slower multi-stream, and
~106k less KV pool. Neither v2 profile needs it.

## License

Code in this repo: Apache-2.0 (includes the FP8-KV port, derived from halt95's
Apache-2.0 patch set and from Apache-2.0 vLLM code). No model weights are hosted
here — the checkpoint is halt95's, under the Qwen Community License 1.0.
