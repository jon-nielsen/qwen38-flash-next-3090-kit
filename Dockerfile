# Runtime for serving Qwen3.8-Flash-Next (halt95 Merlin W4A16) on 24 GB Ampere (sm_86).
# v2 KIT IMAGE: base = lazymio/vllm-backport v0.13.0 build (fork tag v0.13.0, a350766628);
# content = merged tree jon-nielsen kit-v2-small-sync-cde54e8 @ 2b21fbe1f8
# (kit-v2-small branch + wtdcode/vllm-backport origin/master cde54e8ed3, single conflict
# in ngram_embedding.py resolved keep-ours: fused FP8 UVA kernel kept).
# Tag maps to tree commit + base digest. Do not "update" the tag without re-validating.
FROM lazymio/vllm-backport@sha256:bfe237ee0886d9b4b2357b84bfbe09805678cc80b2182dd2a87cefb89497342d

LABEL org.opencontainers.image.title="vllm-backport-flashnext-sm86" \
      org.opencontainers.image.description="v2: wtdcode/vllm-backport v0.13.0 (lazmio sm86 image) + Flash-Next sm86 kit merged tree 2b21fbe1f8: FP8 E4M3 QSA KV reader (halt95 patches/0001 port, Apache-2.0) + PP-aware sidecar loader fixes, INT4-draft probe gate, PP KV-alloc cross-rank parity fix, fused FP8 PLE UVA lookup kernel, upstream hybrid-state fixes #76/#77. VLLM_DETERMINISTIC_MOE_ALIGN default flipped to 0 (upstream master); measured +37% aggregate on TP2xPP2+EP+MTP-4 @262k vs flag=1." \
      org.opencontainers.image.source="https://github.com/jon-nielsen/qwen38-flash-next-3090-kit" \
      org.opencontainers.image.base.name="lazmio/vllm-backport:latest-sm86" \
      org.opencontainers.image.base.digest="sha256:bfe237ee0886d9b4b2357b84bfbe09805678cc80b2182dd2a87cefb89497342d" \
      org.opencontainers.image.revision="2b21fbe1f87f116ea1b78758d9174388a14816bc" \
      org.opencontainers.image.licenses="Apache-2.0"

# --- merged-tree runtime files (COPY dir-merge overlays only the files present) ---
COPY tree/vllm/ /usr/local/lib/python3.12/dist-packages/vllm/
# --- halt95 FP8-KV calibration sidecar (12 layers, 2026-09-08) ---
COPY tree/opt/qsa_kv_scales_262k.json /opt/qsa_kv_scales_262k.json
