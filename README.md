# Qwen3.8-Flash-Next: full 262k context with MTP-4 on 4x RTX 3090 (24 GB) — serving kit

Two measured launch profiles for serving **halt95/Qwen3.8-Flash-Next-W4A16-Merlin**
(W4A16 quant of Qwen3.8-Flash-Next) on RTX 3090-class hardware, plus the runtime
image and the checkpoint surgery tooling they rest on. The headline result: the
4-GPU profile serves the model at its full 262,144-token context with MTP-4.
Everything here was measured on a real rig — no estimates.

## Quick start

Requirements: NVIDIA GPUs (24 GB, sm_86), Docker + nvidia-container-toolkit with CDI
(`nvidia-ctk cdi generate`), ~48 GiB free host RAM for the CPU-offloaded FP8 PLE
table, disk for the checkpoint (~116 GiB download + workspace).

```bash
# 1. checkpoint (the only large download)
hf download halt95/Qwen3.8-Flash-Next-W4A16-Merlin --local-dir /path/to/merlin

# 2. this kit
git clone https://github.com/jon-nielsen/qwen38-flash-next-3090-kit
cd qwen38-flash-next-3090-kit/composes

# 3. Profile B — 4x RTX 3090, full 262,144-token context (no surgery needed)
MODEL=/path/to/merlin VLLM_API_KEY=yourkey \
  docker compose -f profile-b-4gpu-tp2pp2-mtp4-262k-fp8kv.yml up -d
docker logs -f flashnext-3090-4gpu-262k   # ~5-6 min cold; success line: "GPU KV cache size: 416,490 tokens"

# Profile A — 8x RTX 3090, max capacity/speed (needs the one-time surgery, see surgery/RUNBOOK.md)
MODEL=/path/to/bf16mtp-output VLLM_API_KEY=yourkey \
  docker compose -f profile-a-8gpu-tp4pp2-mtp4-bf16kv.yml up -d
```

Both profiles expect the image at
`ghcr.io/jon-nielsen/vllm-backport-flashnext-sm86:3bec275-bb1f7777` (compose pulls it).
The tag is the runtime fork commit + the base-image digest all measurements ran on.

## The two profiles, as measured

| | Profile A (8 GPUs) | Profile B (4 GPUs) |
|---|---|---|
| Shape | TP4×PP2 + EP + MTP-4 | TP2×PP2 + EP + MTP-4 |
| Context | 262,144 | 262,144 |
| KV cache | BF16 | FP8 E4M3 + calibrated sidecar |
| Draft head | BF16 (dequantized via surgery) | INT4 packed (halt95's original) |
| KV pool | 1,213,265 tokens (4.63x) | 416,490 tokens (1.59x) |
| Single-stream | 94.6 tok/s | 94.2 tok/s |
| 4-stream aggregate | 202.2 tok/s | 189.3 tok/s |
| Checkpoint | surgery output (bf16mtp) | halt95's, as-is |

Profile B is the interesting one: full-context serving on four 24 GB cards, at
single-stream parity with the 8-GPU lane. At C=1 the lane is communication-latency
bound and TP2×PP2's 24 two-rank allreduces per stage beat TP4's 48 four-rank ones —
the GPU count is not the limit, the shape is.

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

## Honest limits of the measurements

- Profile B caps at `max_num_seqs=4` concurrent streams (by design; 4-stream numbers
  are at the cap). Profile A allows 32.
- FP8-KV acceptance is ~3.0 under load vs ~3.5 BF16 — inside the band halt95 reports
  for the same design (2.43-2.78). Inherent FP8 numerics, not a port defect; it is
  the price of the 1.59x pool at 262k.
- Deep prefill tested to 68k tokens; 131k-class depth untested on Profile B.
- Output correctness was gated with a 4-prompt oracle (bitwise where numerics allow,
  semantic otherwise): 4/4 correct on both profiles, and the FP8 lanes match the
  BF16 lane bitwise on the most stable prompts.
- `restart: unless-stopped` is the only lifecycle change vs the measured configs;
  engine arguments are untouched.

## The stack and who made it

| Part | Source | Author |
|---|---|---|
| Base model + BF16 parts | Qwen3.8-Flash-Next via Intel's release | Qwen / Intel |
| W4A16 experts (INT4 g128) | Intel/Qwen3.8-Flash-Next-W4A16-AutoRound | Intel (AutoRound) |
| FP8 PLE n-gram table | RadixArk/Qwen3.8-Flash-Next-NVFP4 | RadixArk |
| MTP draft INT4 pack, GDN INT8, KV sidecar, assembly | Qwen3.8-Flash-Next-W4A16-Merlin | halt95 (packer adapted from DominikBucko/qwen38-flash-next-2x3090) |
| vLLM fork (sm_86 backport, v0.11.3) | wtdcode/vllm-backport @ 3bec27573, image lazymio/vllm-backport | wtdcode / lazymio |
| Flash-Next arch + PLE CPU-offload | in the fork; upstream PRs #53896 (merged), #53899 (open) | huanghaoyan.hhy (Alibaba) |
| PP-with-MTP draft fix | fork commit | Matri Ning |
| Draft-unquantized block (mtp.py) | fork commit | lazymio |
| FP8 E4M3 QSA KV reader (Triton integer-decode, sm_86) | patches/0001, ported to this fork | halt95 (Apache-2.0) |
| PP-aware sidecar loader fixes (2), INT4-draft probe gate, FP8-PLE-in-W4A16 embedding gate, 0.11.3 WeightsMapper port | this kit (patches/, overlay-fp8/) | Jon-Nielsen |
| Draft INT4→BF16 surgery + fail-closed verifier | this kit (surgery/) | Jon-Nielsen |
| Profiles, knob measurements, rig findings | this kit (composes/) | Jon-Nielsen |

The port ships as 6 files baked into the image (see `patches/` for the exact diffs
vs the fork tree at 3bec27573). Two PP-awareness fixes were required that halt95's
PP=1 lane never hit: sibling-stage sidecar entries must be ignored, not fatal, and
the validation loop must skip non-local layer names.

## License

Code in this repo: Apache-2.0 (includes the FP8-KV port, derived from halt95's
Apache-2.0 patch set and from Apache-2.0 vLLM code). No model weights are hosted
here — the checkpoint is halt95's, under the Qwen Community License 1.0.
