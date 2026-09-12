#!/usr/bin/env python3
"""Independent verification of the Merlin fork-adaptation output.

Checks (fail-closed, exit 1 on any failure):
  1. index <-> disk key exactness, both directions.
  2. The 1,536 new draft-expert tensors: BF16 dtype, shape == source
     weight_shape, and BITWISE equality with an independent regeneration
     (unpack + dequant + bf16 cast, recomputed here from the SOURCE draft
     file -- catches any write/corruption issue and any nondeterminism).
  3. The 29 existing draft scaffolding tensors: bitwise equal to source.
  4. Every bit-copied safetensors file: sha256 == source sha256.
  5. config.json: exactly one mtp ignore entry re:^mtp\\..*, the three
     upstream mtp regexes gone, group_0/group_1 config unchanged.
  6. DRAFT_FILE absent from DST and from the new index.
  7. PLE contract intact: 128 F8_E4M3 shard keys + bf16 [1]
     ngram_embedding.weight_scale present in the DST index.
  8. Size accounting vs expectation (~118.7 GiB).
"""
import hashlib
import json
import os
import sys

import torch
from safetensors.torch import safe_open

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from convert_merlin_draft import (  # noqa: E402
    DST, DRAFT_FILE, EXTRA_FILE, GROUP, NUM_EXPERTS, NEW_IGNORE, OLD_IGNORE,
    PROJECTIONS, SRC, BIAS, SIZE_BITS, unpack_quantized_values_into_int32,
)

FAILS = []


def check(name: str, ok: bool, detail: str = "") -> None:
    print(f"{'PASS' if ok else 'FAIL'}  {name}" + (f"  ({detail})" if detail else ""))
    if not ok:
        FAILS.append(name)


def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 24), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> None:
    src_idx = json.load(open(os.path.join(SRC, "model.safetensors.index.json")))
    dst_idx = json.load(open(os.path.join(DST, "model.safetensors.index.json")))
    dst_wm = dst_idx["weight_map"]
    src_wm = src_idx["weight_map"]

    # 1. index <-> disk exactness
    disk = {f for f in os.listdir(DST) if f.endswith(".safetensors")}
    idx_files = set(dst_wm.values())
    check("index files == disk safetensors", idx_files == disk,
          f"idx-only={sorted(idx_files - disk)} disk-only={sorted(disk - idx_files)}")
    missing = [k for k, f in dst_wm.items()
               if not os.path.exists(os.path.join(DST, f))]
    check("all index keys resolve on disk", not missing, f"{len(missing)} missing")
    headers = {}
    for f in sorted(idx_files):
        with safe_open(os.path.join(DST, f), framework="pt", device="cpu") as sf:
            headers[f] = set(sf.keys())
    on_disk_keys = set()
    for f, ks in headers.items():
        on_disk_keys |= ks
    check("index keys == header keys (both ways)",
          set(dst_wm) == on_disk_keys,
          f"idx-only={len(set(dst_wm) - on_disk_keys)} "
          f"hdr-only={len(on_disk_keys - set(dst_wm))}")

    # 2. new draft tensors: dtype, shape, bitwise regeneration
    n_new = 0
    bad_shape, bad_dtype, bad_bits = [], [], []
    with safe_open(os.path.join(SRC, DRAFT_FILE), framework="pt",
                   device="cpu") as df, \
         safe_open(os.path.join(DST, EXTRA_FILE), framework="pt",
                   device="cpu") as ef:
        for e in range(NUM_EXPERTS):
            for proj in PROJECTIONS:
                pre = f"mtp.layers.0.mlp.experts.{e}.{proj}"
                key = f"{pre}.weight"
                if key not in ef.keys():
                    bad_shape.append(key)
                    continue
                n_new += 1
                w = ef.get_tensor(key)
                shape = df.get_tensor(f"{pre}.weight_shape").tolist()
                if list(w.shape) != shape:
                    bad_shape.append(f"{key} {tuple(w.shape)} != {shape}")
                if w.dtype != torch.bfloat16:
                    bad_dtype.append(f"{key} {w.dtype}")
                packed = df.get_tensor(f"{pre}.weight_packed")
                scale = df.get_tensor(f"{pre}.weight_scale").to(torch.float32)
                u4 = unpack_quantized_values_into_int32(packed, SIZE_BITS, 1)
                exp = ((u4.to(torch.float32) - BIAS)
                       * scale.repeat_interleave(GROUP, dim=1)).to(torch.bfloat16)
                if not torch.equal(w, exp):
                    bad_bits.append(key)
    check("1536 new draft tensors present", n_new == NUM_EXPERTS * len(PROJECTIONS),
          f"{n_new}")
    check("new draft tensor shapes == weight_shape", not bad_shape,
          f"{len(bad_shape)} bad, e.g. {bad_shape[:2]}")
    check("new draft tensors BF16", not bad_dtype, f"{len(bad_dtype)} bad")
    check("new draft tensors bitwise == regeneration", not bad_bits,
          f"{len(bad_bits)} bad, e.g. {bad_bits[:2]}")

    # 3. scaffolding 29 bitwise
    bad_scaf = []
    with safe_open(os.path.join(SRC, EXTRA_FILE), framework="pt", device="cpu") as se, \
         safe_open(os.path.join(DST, EXTRA_FILE), framework="pt", device="cpu") as ee:
        src_keys = set(se.keys())
        for k in src_keys:
            if k not in ee.keys():
                bad_scaf.append(f"{k} missing")
            elif not torch.equal(se.get_tensor(k), ee.get_tensor(k)):
                bad_scaf.append(f"{k} differs")
    check("29 scaffolding tensors bitwise identical", not bad_scaf and len(src_keys) == 29,
          f"{len(src_keys)} src keys, {len(bad_scaf)} bad")

    # 4. copied safetensors sha256
    bad_hash = []
    copied = sorted(disk - {EXTRA_FILE})
    for f in copied:
        if sha256_file(os.path.join(SRC, f)) != sha256_file(os.path.join(DST, f)):
            bad_hash.append(f)
    check(f"{len(copied)} copied safetensors sha256 == source", not bad_hash,
          f"bad: {bad_hash[:3]}")

    # 5. config.json
    cfg = json.load(open(os.path.join(DST, "config.json")))
    q = cfg["quantization_config"]
    ignore = q["ignore"]
    check("config: exactly one mtp ignore entry",
          sum(1 for i in ignore if "mtp" in i) == 1
          and NEW_IGNORE[0] in ignore,
          f"{[i for i in ignore if 'mtp' in i]}")
    check("config: old mtp regexes gone",
          not any(i in ignore for i in OLD_IGNORE))
    check("config: group_0/group_1 unchanged",
          set(q["config_groups"]) == {"group_0", "group_1"}
          and q["config_groups"]["group_1"]["targets"]
          == ["re:.*\\.linear_attn\\.(in_proj_qkvz|in_proj_qkv|in_proj_z|out_proj)$"])
    check("config: ple_embedding_dtype preserved",
          cfg["text_config"].get("ple_embedding_dtype") is None)

    # 6. draft file gone
    check("draft int4 file absent from DST", DRAFT_FILE not in disk)
    check("draft file absent from index", DRAFT_FILE not in idx_files)

    # 7. PLE contract
    ple_shards = [k for k in dst_wm if ".ple." in k and "shard_" in k]
    wsc = "model.language_model.layers.1.ple.ple_embedding.ngram_embedding.weight_scale"
    with safe_open(os.path.join(DST, dst_wm[wsc]), framework="pt",
                   device="cpu") as sf:
        t = sf.get_tensor(wsc)
    check("PLE: 128 shard keys + bf16 [1] weight_scale",
          len(ple_shards) == 128 and t.dtype == torch.bfloat16
          and list(t.shape) == [1])

    # 8. size accounting
    total = sum(os.path.getsize(os.path.join(DST, f)) for f in os.listdir(DST)
                if os.path.isfile(os.path.join(DST, f)))
    expect_lo, expect_hi = 117.5, 120.5
    got = total / 2**30
    check(f"total size in [{expect_lo}, {expect_hi}] GiB",
          expect_lo <= got <= expect_hi, f"{got:.2f} GiB")

    print()
    if FAILS:
        print(f"VERIFY FAILED: {len(FAILS)} check(s): {FAILS}")
        sys.exit(1)
    print(f"VERIFY PASSED: all checks green. {got:.2f} GiB at {DST}")


if __name__ == "__main__":
    main()
