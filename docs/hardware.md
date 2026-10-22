# Hardware & model serving

This guide maps SentientAI's two model tiers onto the reference machine: two
**Radeon Pro V340** cards for heavy reasoning and a **GTX 1660 Ti** for fast
routing, reflection and embeddings. It also corrects two assumptions in the original
plan. Measure everything on your own box with `sentientai bench`.

## What the hardware actually is

| Card | GPUs exposed | VRAM | Architecture | Software path |
|---|---|---|---|---|
| Radeon Pro V340 (x2) | 2 per card, **4 total** | 16 GB HBM2 per GPU, **64 GB total** | Vega 10 / GCN5 (`gfx900`) | llama.cpp Vulkan (or HIP built for gfx900) |
| GTX 1660 Ti | 1 | 6 GB GDDR6 | Turing (sm_75), CUDA | llama.cpp CUDA, Ollama |

Sources: the V340 carries two Vega GPUs with 16 GB of ECC HBM2 each, 32 GB per card
([Phoronix](https://www.phoronix.com/news/AMD-Radeon-Pro-V340),
[ServeTheHome](https://www.servethehome.com/amd-radeon-pro-v340-dual-vega-32gb-vdi-solution-launched/)).
AMD lists `gfx900` as **unsupported** in current ROCm releases and dropped it from default
library builds ([ROCm system requirements](https://rocm.docs.amd.com/projects/install-on-linux/en/latest/reference/system-requirements.html),
[ArchWiki: MI25](https://wiki.archlinux.org/title/AMD_Radeon_Instinct_MI25)).

### Two corrections to the original plan

1. **No vLLM tensor parallelism on the V340s.** vLLM's ROCm backend targets newer
   Instinct/RDNA parts, and gfx900 is outside current ROCm support. Use **llama.cpp**
   and split the model's layers across the four Vega GPUs (`--split-mode layer
   --tensor-split 1,1,1,1`). This is pipeline-style splitting: capacity scales with
   GPU count, but generation speed is bounded by one GPU's memory bandwidth at a time.
2. **There is more VRAM than assumed.** 64 GB across four GPUs comfortably holds a
   **70B model at ~4-bit** (about 40-43 GB of weights) with room for a 16k-token KV
   cache, or a 32B model at 8-bit with long context.

## Recommended layout

| Tier | Where | Model class (GGUF) | Notes |
|---|---|---|---|
| heavy | 4x Vega (Vulkan) | 70B Q4_K_M, or 32B Q6_K/Q8_0 | Llama 3.x 70B, Qwen 2.5/3 32B, DeepSeek-R1 distills. Expect low single-digit tokens/s for 70B; 32B is a good speed/quality balance. |
| fast | 1660 Ti (CUDA) | 3-4B at Q5/Q6, or 7-8B at Q4_K_M with ~4k context | 8B Q4 weights are about 5 GB, so the KV cache must stay small on 6 GB. KV-cache quantization (`--cache-type-k q8_0`) helps. |
| embedding | 1660 Ti or CPU | nomic-embed-text v1.5 (768-d), bge-m3 (1024-d) | Tiny. If VRAM is tight with an 8B fast model, run embeddings on CPU. |

Sizes are approximate. Confirm with `rocm-smi` / `vulkaninfo` / `nvidia-smi` and
`sentientai bench`.

Set the matching task prefixes for asymmetric embedding models (nomic uses
`search_query: ` / `search_document: `) under `models.embedding`. Changing the
embedding model or dimensions requires `sentientai memory reindex`.

## Building and running llama.cpp

```bash
git clone https://github.com/ggml-org/llama.cpp && cd llama.cpp

# Heavy tier: Vulkan backend works on gfx900 without ROCm
cmake -B build-vulkan -DGGML_VULKAN=ON && cmake --build build-vulkan -j --config Release

# Alternative: HIP backend compiled explicitly for gfx900 (needs a ROCm that still builds it)
# HIPCXX="$(hipconfig -l)/clang" cmake -B build-hip -DGGML_HIP=ON -DAMDGPU_TARGETS=gfx900

# Fast/embedding tier: CUDA for the 1660 Ti
cmake -B build-cuda -DGGML_CUDA=ON && cmake --build build-cuda -j --config Release
```

```bash
# heavy: all four Vega GPUs, layers split evenly
GGML_VK_VISIBLE_DEVICES=0,1,2,3 ./build-vulkan/bin/llama-server \
  -m /models/Llama-3.3-70B-Instruct-Q4_K_M.gguf --host 127.0.0.1 --port 8000 \
  --n-gpu-layers 999 --split-mode layer --tensor-split 1,1,1,1 --ctx-size 16384 --jinja

# fast: 1660 Ti
CUDA_VISIBLE_DEVICES=0 ./build-cuda/bin/llama-server \
  -m /models/Qwen3-8B-Q4_K_M.gguf --host 127.0.0.1 --port 8002 \
  --n-gpu-layers 999 --ctx-size 4096 --cache-type-k q8_0 --jinja

# embeddings
CUDA_VISIBLE_DEVICES=0 ./build-cuda/bin/llama-server \
  -m /models/nomic-embed-text-v1.5.Q8_0.gguf --host 127.0.0.1 --port 8001 --embedding
```

`--jinja` enables the model's own chat template, which llama.cpp needs for native
tool calling. If a model or template can't do native tool calls, set
`supports_tools: false` for that tier. SentientAI then switches to a prompted JSON
tool protocol automatically.

Systemd units for these servers are in `deploy/systemd/llama-server@.service`. The
compose file has a `models` profile with the official llama.cpp server images.

## Verify

```bash
sentientai -c sentient.yaml doctor                         # every subsystem, including both model tiers
sentientai -c sentient.yaml bench --requests 5             # p50/p95 latency and tokens/s per tier
sentientai -c sentient.yaml eval run --tag grounding       # behavioural check on real models
```

## Storage tiers

| Tier | Holds | Config |
|---|---|---|
| NVMe | Postgres, Redis AOF, vector index, data dir, audit log | `storage.data_dir`, compose `SSD_PATH` |
| SSD/HDD | Episodic history, structured logs (in Postgres) | Postgres data directory |
| Mass HDD | Raw ingested documents, archived by content hash | `storage.archive_dir`, compose `HDD_PATH` |
