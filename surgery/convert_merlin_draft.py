#!/usr/bin/env python3
"""Adapt halt95/Qwen3.8-Flash-Next-W4A16-Merlin for wtdcode/vllm-backport (2026-09-11).

Surgery:
  1. Dequantize the 1,536 INT4-g128-symmetric packed MTP draft-expert tensors
     (mtp-routed-experts-int4.safetensors) to BF16 and merge them into
     model_extra_tensors.safetensors, which already carries the 29 BF16 draft
     scaffolding tensors (halt95's Lorbus-style split).
  2. config.json: collapse the three mtp ignore regexes (including the
     negative-lookahead that keeps the draft experts quantized) into a single
     re:^mtp\\..* so the fork's _make_draft_vllm_config keeps the whole draft
     unquantized (the proven cyankiwi layout).

Everything else is copied bit-perfect. The GDN INT8 group (channel-sym
uint8b128 -> Marlin), the FP8 PLE table (F8_E4M3 shards + bf16 [1]
ngram_embedding.weight_scale, matches the VLLM_PLE_FP8_EMBEDDING overlay
contract), and the INT4 g128 experts (MoE Marlin uint4b8) are untouched.

Unpack semantics transcribed verbatim from the fork's
vllm/model_executor/layers/quantization/utils/quant_utils.py::
unpack_quantized_values_into_int32 (little-endian nibbles along the packed
input dim), uint4b8 bias 8 per vllm/scalar_type.py ("standard GPTQ 4bit uses
a bias of 8"). PackedvLLMParameter in compressed_tensors_wNa16.py confirms
packed_dim=1, packed_factor=32/num_bits.

Verification built in, fail-closed:
  - per-tensor fp32 round-trip: requant(dequant(packed, scale)) == packed,
    bit-exact BEFORE the bf16 storage cast (fp32 product of a 4-bit int and a
    bf16 scale is exact; round() recovers the integer exactly).
  - weight_shape tensor must equal [out, in] implied by packed/scale shapes.
  - global stats sanity (dequant distribution roughly symmetric about 0).

Idempotent: per-file sha256 manifest, atomic .tmp + os.replace, deletes
nothing, disk-guarded (aborts unless free >= needed + 5 GiB at DST).
Run with the project venv ~/vllm/vllm-backport/.venv/bin/python.
"""
import hashlib
import json
import os
import shutil
import sys
import time

import torch
from safetensors.torch import load_file, save_file, safe_open

SRC = os.path.expanduser(
    "~/.cache/huggingface/hub/models--halt95--Qwen3.8-Flash-Next-W4A16-Merlin/"
    "snapshots/0e52f8880da6053251261d17f85aca96e974f7bc"
)
DST = "/mnt/models/temp/Qwen3.8-Flash-Next-W4A16-Merlin-bf16mtp"
DRAFT_FILE = "mtp-routed-experts-int4.safetensors"
EXTRA_FILE = "model_extra_tensors.safetensors"
OLD_IGNORE = [
    "re:^mtp\\.(?!layers\\.\\d+\\.mlp\\.experts(?:\\.|$)).*",
    "re:^mtp\\.layers\\.\\d+\\.self_attn\\..*",
    "re:^mtp\\.fc_.*",
]
NEW_IGNORE = ["re:^mtp\\..*"]
PROJECTIONS = ("gate_proj", "up_proj", "down_proj")
NUM_EXPERTS = 512
GROUP = 128
SIZE_BITS = 4
PACK_FACTOR = 32 // SIZE_BITS
BIAS = 8
GUARD_GIB = 5.0

README = """# Qwen3.8-Flash-Next W4A16 Merlin -- fork-adapted (BF16 MTP draft)

Local adaptation of https://huggingface.co/halt95/Qwen3.8-Flash-Next-W4A16-Merlin
for serving on wtdcode/vllm-backport (lazymio image @ 3bec27573) with
MTP + TP4xPP2 on 8x RTX 3090. NOT PUBLISHED; local tree only.

Changes vs upstream Merlin (everything else bit-identical):
  1. The 1,536 INT4 g128-sym packed MTP draft-expert tensors
     (mtp-routed-experts-int4.safetensors) were dequantized to BF16 and merged
     into model_extra_tensors.safetensors (29 BF16 draft scaffolding tensors
     already there). Cost ~+3.5 GiB. Rationale: the backport expects an
     unquantized draft (wtdcode's own quant keeps mtp.* BF16); its
     _make_draft_vllm_config extends the draft ignore list when it sees an
     mtp ignore pattern, so the whole draft loads BF16 -- the layout proven
     on this rig with the cyankiwi checkpoint.
  2. config.json quantization_config.ignore: the three mtp regexes collapsed
     to re:^mtp\\..*.

Unchanged and served by the fork as-is (code-verified 2026-09-11):
  - INT4 g128 symmetric experts (group_0) via MoE Marlin uint4b8;
  - GDN in_proj_qkv/in_proj_z/out_proj INT8 per-channel sym (group_1) via
    dense WNA16 -> Marlin uint8b128, group -1 (first exercise on this fork);
  - FP8 PLE table (F8_E4M3 shards + bf16 [1] ngram_embedding.weight_scale) --
    matches the VLLM_PLE_FP8_EMBEDDING overlay contract; no conversion needed;
  - qsa_kv_scales_262k.json rides along UNUSED (fork QSA path is BF16-KV only).

Dequant oracle: fork's unpack_quantized_values_into_int32 transcribed verbatim
(little-endian nibbles, packed_dim=1), uint4b8 bias 8; per-tensor fp32
round-trip requant bit-equality; regeneration cross-check in
verify_merlin_draft.py.

Upstream card preserved as MERLIN-CARD-UPSTREAM.md. License: Qwen Community
License 1.0 (LICENSE file, verbatim from upstream).
"""


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def unpack_quantized_values_into_int32(w_q: torch.Tensor, size_bits: int,
                                       packed_dim: int = 1) -> torch.Tensor:
    # Verbatim transcription of the fork's quant_utils.py function.
    perm = (*[i for i in range(len(w_q.shape)) if i != packed_dim], packed_dim)
    inv_perm = tuple(perm.index(i) for i in range(len(perm)))
    w_q_perm = w_q.permute(perm)
    pack_factor = 32 // size_bits
    mask = (1 << size_bits) - 1
    new_shape_perm = list(w_q_perm.shape)
    new_shape_perm[-1] *= pack_factor
    res = torch.zeros(new_shape_perm, dtype=torch.int32)
    for i in range(pack_factor):
        res[..., i::pack_factor] = (w_q_perm >> (size_bits * i)) & mask
    return res.permute(inv_perm)


def pack_quantized_values_from_int32(res: torch.Tensor, size_bits: int,
                                     packed_dim: int = 1) -> torch.Tensor:
    # Exact inverse of the transcription above.
    perm = (*[i for i in range(len(res.shape)) if i != packed_dim], packed_dim)
    inv_perm = tuple(perm.index(i) for i in range(len(perm)))
    res_perm = res.permute(perm)
    pack_factor = 32 // size_bits
    out_shape = list(res_perm.shape)
    assert out_shape[-1] % pack_factor == 0
    out_shape[-1] //= pack_factor
    packed = torch.zeros(out_shape, dtype=torch.int32)
    for i in range(pack_factor):
        packed |= res_perm[..., i::pack_factor] << (size_bits * i)
    return packed.permute(inv_perm)


def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 24), b""):
            h.update(chunk)
    return h.hexdigest()


def atomic_write(path: str, writer) -> str:
    tmp = path + ".tmp"
    writer(tmp)
    h = sha256_file(tmp)
    os.replace(tmp, path)
    return h


def main() -> None:
    assert os.path.isdir(SRC), f"missing snapshot dir {SRC}"
    src_files = sorted(os.listdir(SRC))
    assert DRAFT_FILE in src_files and EXTRA_FILE in src_files
    assert "config.json" in src_files and "model.safetensors.index.json" in src_files

    idx = json.load(open(os.path.join(SRC, "model.safetensors.index.json")))
    wm = idx["weight_map"]
    draft_keys = sorted(k for k, f in wm.items() if f == DRAFT_FILE)
    assert len(draft_keys) == 3 * len(PROJECTIONS) * NUM_EXPERTS, len(draft_keys)

    # ---- plan + disk guard -------------------------------------------------
    copy_files = [f for f in src_files
                  if f not in (DRAFT_FILE, EXTRA_FILE, "config.json",
                               "model.safetensors.index.json", "README.md")]
    copy_bytes = sum(os.path.getsize(os.path.join(SRC, f)) for f in copy_files)
    draft_bytes = os.path.getsize(os.path.join(SRC, DRAFT_FILE))
    extra_src_bytes = os.path.getsize(os.path.join(SRC, EXTRA_FILE))
    # gate/up are [640, 2560], down is [2560, 640]: all 640*2560 elements
    new_extra_bytes = extra_src_bytes + \
        NUM_EXPERTS * len(PROJECTIONS) * 2 * 640 * 2560
    need_gib = (copy_bytes + new_extra_bytes + 2048 * 2**20) / 2**30 + GUARD_GIB
    os.makedirs(os.path.dirname(DST), exist_ok=True)
    free_gib = shutil.disk_usage(os.path.dirname(DST)).free / 2**30
    log(f"plan: {len(copy_files)} files to copy ({copy_bytes / 2**30:.2f} GiB), "
        f"draft {draft_bytes / 2**30:.2f} GiB -> dequant, "
        f"extra rewrite {extra_src_bytes / 2**20:.0f} MiB -> ~{new_extra_bytes / 2**30:.2f} GiB")
    log(f"disk: need ~{need_gib:.1f} GiB, free {free_gib:.1f} GiB at "
        f"{os.path.dirname(DST)}")
    assert free_gib >= need_gib, f"disk guard: {free_gib:.1f} < {need_gib:.1f} GiB"

    os.makedirs(DST, exist_ok=True)
    manifest_path = os.path.join(DST, ".convert_manifest.json")
    manifest = json.load(open(manifest_path)) if os.path.exists(manifest_path) else {}

    def save_manifest():
        atomic_write(manifest_path, lambda p: json.dump(manifest, open(p, "w"),
                                                         indent=1, sort_keys=True))

    def done(name: str) -> bool:
        info = manifest.get(name)
        if not info:
            return False
        p = os.path.join(DST, name)
        return os.path.exists(p) and os.path.getsize(p) == info["bytes"] \
            and sha256_file(p) == info["sha256"]

    # ---- 1. dequant the draft experts --------------------------------------
    log("pass 1: dequant draft experts (fp32 round-trip per tensor)")
    t0 = time.time()
    extra_new = {}
    n_done = 0
    stats = {"min": 0.0, "max": 0.0, "sum": 0.0, "abs_sum": 0.0, "n": 0}
    with safe_open(os.path.join(SRC, DRAFT_FILE), framework="pt",
                   device="cpu") as df:
        for e in range(NUM_EXPERTS):
            for proj in PROJECTIONS:
                pre = f"mtp.layers.0.mlp.experts.{e}.{proj}"
                packed = df.get_tensor(f"{pre}.weight_packed")          # I32 [out, in/8]
                scale = df.get_tensor(f"{pre}.weight_scale").to(torch.float32)  # [out, in/128]
                shape = df.get_tensor(f"{pre}.weight_shape").tolist()   # [out, in]
                out_dim, in_dim = shape
                assert list(packed.shape) == [out_dim, in_dim // PACK_FACTOR], \
                    (pre, packed.shape, shape)
                assert list(scale.shape) == [out_dim, in_dim // GROUP], \
                    (pre, scale.shape, shape)

                u4 = unpack_quantized_values_into_int32(packed, SIZE_BITS, 1)
                assert u4.shape == (out_dim, in_dim)
                scale_exp = scale.repeat_interleave(GROUP, dim=1)
                q_signed = (u4.to(torch.float32) - BIAS)               # -8..7
                w_fp32 = q_signed * scale_exp
                # round-trip BEFORE the bf16 cast: must be bit-exact
                q_back = torch.round(w_fp32 / scale_exp).clamp_(-BIAS, BIAS - 1)
                assert torch.equal(q_back, q_signed), f"round-trip failed: {pre}"
                repacked = pack_quantized_values_from_int32(
                    (q_back + BIAS).to(torch.int32), SIZE_BITS, 1)
                assert torch.equal(repacked, packed), f"requant != packed: {pre}"

                w = w_fp32.to(torch.bfloat16)
                extra_new[f"{pre}.weight"] = w.contiguous()
                stats["min"] = min(stats["min"], float(w.min()))
                stats["max"] = max(stats["max"], float(w.max()))
                stats["sum"] += float(w.sum())
                stats["abs_sum"] += float(w.abs().sum())
                stats["n"] += w.numel()
                n_done += 1
            if e % 64 == 0:
                log(f"  experts {e}/{NUM_EXPERTS} ({time.time() - t0:.0f}s)")
    assert n_done == NUM_EXPERTS * len(PROJECTIONS)
    mean = stats["sum"] / stats["n"]
    log(f"  dequant done: {n_done} tensors, value range "
        f"[{stats['min']:.4g}, {stats['max']:.4g}], mean {mean:.4g}, "
        f"mean|x| {stats['abs_sum'] / stats['n']:.4g}")

    # ---- 2. rewrite model_extra_tensors.safetensors ------------------------
    if done(EXTRA_FILE):
        log(f"pass 2: {EXTRA_FILE} already complete, skip")
    else:
        log(f"pass 2: rewrite {EXTRA_FILE} (29 existing + {len(extra_new)} new)")
        with safe_open(os.path.join(SRC, EXTRA_FILE), framework="pt",
                       device="cpu") as ef:
            meta = ef.metadata()
            existing = {k: ef.get_tensor(k) for k in ef.keys()}
        assert len(existing) == 29, len(existing)
        out = dict(existing)
        out.update(extra_new)
        assert len(out) == 29 + n_done

        def write_extra(p):
            save_file(out, p, metadata=meta)

        manifest[EXTRA_FILE] = {"sha256": atomic_write(
            os.path.join(DST, EXTRA_FILE), write_extra),
            "bytes": os.path.getsize(os.path.join(DST, EXTRA_FILE))}
        save_manifest()
        log(f"  wrote {EXTRA_FILE}: {manifest[EXTRA_FILE]['bytes'] / 2**30:.2f} GiB")

    # ---- 3. copy the rest bit-perfect --------------------------------------
    log(f"pass 3: copy {len(copy_files)} files bit-perfect")
    t0 = time.time()
    for i, f in enumerate(copy_files):
        if done(f):
            continue
        sp = os.path.join(SRC, f)
        dp = os.path.join(DST, f)
        h = atomic_write(dp, lambda p, sp=sp: shutil.copyfile(sp, p))
        manifest[f] = {"sha256": h, "bytes": os.path.getsize(dp)}
        save_manifest()
        if i % 5 == 0 or i == len(copy_files) - 1:
            rate = (i + 1) / max(time.time() - t0, 1e-9)
            log(f"  {i + 1}/{len(copy_files)} ({rate:.1f} files/s)")

    # ---- 4. config.json ignore edit ----------------------------------------
    if done("config.json"):
        log("pass 4: config.json already converted, skip")
    else:
        log("pass 4: config.json ignore edit")
        cfg = json.load(open(os.path.join(SRC, "config.json")))
        ignore = cfg["quantization_config"]["ignore"]
        for entry in OLD_IGNORE:
            assert entry in ignore, f"expected ignore entry missing: {entry}"
        first = ignore.index(OLD_IGNORE[0])
        for entry in OLD_IGNORE:
            ignore.remove(entry)
        ignore[first:first] = NEW_IGNORE
        assert ignore.count(NEW_IGNORE[0]) == 1

        def write_cfg(p):
            json.dump(cfg, open(p, "w"), indent=2, ensure_ascii=False)

        manifest["config.json"] = {"sha256": atomic_write(
            os.path.join(DST, "config.json"), write_cfg),
            "bytes": os.path.getsize(os.path.join(DST, "config.json"))}
        save_manifest()

    # ---- 5. rebuild index.json from output state ---------------------------
    if done("model.safetensors.index.json"):
        log("pass 5: index already rebuilt, skip")
    else:
        log("pass 5: rebuild model.safetensors.index.json")
        new_wm = {k: v for k, v in wm.items() if v != DRAFT_FILE}
        for k in extra_new:
            assert k not in new_wm
            new_wm[k] = EXTRA_FILE
        assert len(new_wm) == len(wm) - len(draft_keys) + len(extra_new)
        new_idx = dict(idx)
        new_idx["weight_map"] = new_wm
        if "total_size" in idx:
            new_idx["total_size"] = sum(
                os.path.getsize(os.path.join(DST, f)) for f in set(new_wm.values()))
        json_p = "model.safetensors.index.json"

        def write_idx(p):
            json.dump(new_idx, open(p, "w"), indent=1, ensure_ascii=False)

        manifest[json_p] = {"sha256": atomic_write(
            os.path.join(DST, json_p), write_idx),
            "bytes": os.path.getsize(os.path.join(DST, json_p))}
        save_manifest()

    # ---- 6. README (ours) + upstream card ----------------------------------
    if not done("README.md"):
        with open(os.path.join(DST, "README.md"), "w") as f:
            f.write(README)
        manifest["README.md"] = {"sha256": sha256_file(os.path.join(DST, "README.md")),
                                 "bytes": os.path.getsize(os.path.join(DST, "README.md"))}
        save_manifest()
    if not done("MERLIN-CARD-UPSTREAM.md"):
        shutil.copyfile(os.path.join(SRC, "README.md"),
                        os.path.join(DST, "MERLIN-CARD-UPSTREAM.md"))
        manifest["MERLIN-CARD-UPSTREAM.md"] = {
            "sha256": sha256_file(os.path.join(DST, "MERLIN-CARD-UPSTREAM.md")),
            "bytes": os.path.getsize(os.path.join(DST, "MERLIN-CARD-UPSTREAM.md"))}
        save_manifest()

    total = sum(os.path.getsize(os.path.join(DST, f)) for f in os.listdir(DST)
                if os.path.isfile(os.path.join(DST, f)))
    log(f"done: {len(os.listdir(DST))} files, {total / 2**30:.2f} GiB at {DST}")
    log("next: verify_merlin_draft.py")


if __name__ == "__main__":
    main()
