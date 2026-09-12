# Runtime for serving Qwen3.8-Flash-Next (halt95 Merlin W4A16) on 24 GB Ampere (sm_86).
# Base image is PINNED to the exact digest all published measurements were taken on.
# Do not "update" the tag without re-validating:  ghcr.io/jon-nielsen/... tags map to
# fork commit + base digest.
FROM lazymio/vllm-backport@sha256:bb1f7777193ed34871b0e278bcba66c65cdd15d6811eef7e6b128c280ccbbe29

LABEL org.opencontainers.image.title="vllm-backport-flashnext-sm86" \
      org.opencontainers.image.description="wtdcode/vllm-backport @3bec27573 (v0.11.3) + Flash-Next FP8-KV/PLE-offload overlays: FP8 E4M3 QSA KV reader ported from halt95/qwen38-flash-next-3090s patches/0001 (Apache-2.0), PP-aware sidecar loader fixes + INT4-draft probe gate by Jon-Nielsen" \
      org.opencontainers.image.source="https://github.com/jon-nielsen/qwen38-flash-next-3090-kit" \
      org.opencontainers.image.base.name="lazymio/vllm-backport:latest-sm86" \
      org.opencontainers.image.base.digest="sha256:bb1f7777193ed34871b0e278bcba66c65cdd15d6811eef7e6b128c280ccbbe29" \
      org.opencontainers.image.licenses="Apache-2.0"

# PLE CPU-offload layer (huanghaoyan.hhy, vLLM PR #53899 lineage)
COPY overlay/ple_layer.py /usr/local/lib/python3.12/dist-packages/vllm/models/qwen4_exp/nvidia/ple_layer.py
# FP8 E4M3 QSA KV: halt95 patches/0001 port + PP-aware fixes + probe gate
COPY overlay-fp8/qsa.py     /usr/local/lib/python3.12/dist-packages/vllm/models/qwen4_exp/nvidia/qsa.py
COPY overlay-fp8/ops_qsa.py /usr/local/lib/python3.12/dist-packages/vllm/models/qwen4_exp/nvidia/ops/qsa.py
COPY overlay-fp8/model.py   /usr/local/lib/python3.12/dist-packages/vllm/models/qwen4_exp/nvidia/model.py
COPY overlay-fp8/mtp.py     /usr/local/lib/python3.12/dist-packages/vllm/models/qwen4_exp/nvidia/mtp.py
COPY overlay-fp8/qsa_kv_scales_262k.json /opt/qsa_kv_scales_262k.json
