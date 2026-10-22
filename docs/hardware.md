# Hardware & model serving

This guide maps Fuzzy's model tiers (and image generation) onto the reference machine:
an **RTX 2070 (8 GB)** and a **GTX 1660 Ti (6 GB)**. The Radeon Pro V340s that earlier
versions of this guide planned around are gone, which simplifies things: both GPUs are
NVIDIA Turing, so **one llama.cpp CUDA build serves everything**. There is no Vulkan,
ROCm or mixed-vendor setup to maintain.

Everything below is a plan to verify, not a benchmark. Measure on your own box with
`nvidia-smi` while Fuzzy is working and with `sentientai bench`.

## What you have

| Card | VRAM | Notes |
|---|---|---|
| RTX 2070 | 8 GB | Turing (sm_75), tensor cores, fast FP16. The only card that can hold a decent 8B model with context. |
| GTX 1660 Ti | 6 GB | Turing (sm_75) without tensor cores. Fine for small LLMs and embeddings. **Image generation is awkward**: see below. |

The two cards do not pool memory. You have two separate budgets (8 GB and 6 GB), not 14 GB.
(llama.cpp can split one model's layers across both with `--split-mode layer
--tensor-split 8,6`, but that uses both cards for one model and leaves nothing for the fast tier.)

## Recommended layout: one tier per card

| Tier | Card | Model (GGUF) | Weights | Approx. total VRAM |
|---|---|---|---|---|
| heavy (chat, reasoning, deliberation) | RTX 2070 | Qwen3-8B at Q5_K_M, 8k context | ~5.9 GB | ~7 to 7.5 GB |
| fast (reflection, appraisal, grounding checks, summaries) | GTX 1660 Ti | Qwen3-4B at Q4_K_M, 8k context shared by two slots | ~2.5 GB | ~3.4 GB |
| embeddings | CPU | nomic-embed-text v1.5 (768-d) at Q8_0 | ~0.15 GB | none |
| images | GTX 1660 Ti (leftover) | SD 1.5-class model, low-VRAM mode | ~2 GB+ | see "Images" |

Sources for the sizes: third-party mirrors list Qwen3-8B Q5_K_M at about 5.85 GB and
Qwen3-4B Q4_K_M at about 2.50 GB ([8B listing](https://app.alphaneural.io/assets/qtum/Qwen3-8B-GGUF),
[4B listing](https://huggingface.co/6block/Qwen3-4B-GGUF)). Confirm file sizes on the repository you download from.

The KV-cache part of the totals is an **estimate** from the architecture (36 layers, 8 KV
heads, 128-dim heads, so roughly 144 KB per token at FP16): about 1.2 GB for 8k tokens, or about
0.6 GB with `q8_0` cache quantization, plus a few hundred MB of compute buffers. If the 2070 also
drives a monitor, expect desktop VRAM use on top: run it headless or lower `--ctx-size`.

Why this split: most of Fuzzy's model calls are small, frequent and background (appraising a
conversation, checking a reply against memory, choosing what to do next, summarising). Those go to the fast tier
so the heavy model is free for conversations. Fuzzy's background life is never allowed to starve a person who is waiting:
when someone talks to it, its own activities pause.

### Other models

Anything with a GGUF and a chat template works; 7 to 9B instruct models are the sweet spot for the 2070.
Check each model's licence before running it on a public site. A 14B model at Q4 is roughly 9 GB, which
does not fit the 2070 alone and would need the two-card split above (so no always-on fast tier). Not recommended.

## Building and running llama.cpp

```bash
git clone https://github.com/ggml-org/llama.cpp && cd llama.cpp
cmake -B build -DGGML_CUDA=ON && cmake --build build -j --config Release
nvidia-smi -L        # note which index is the 2070 and which is the 1660 Ti
```

Pin devices by PCI order so the indices don't change between boots:

```bash
export CUDA_DEVICE_ORDER=PCI_BUS_ID

# heavy: RTX 2070 (replace 0 with its index)
CUDA_VISIBLE_DEVICES=0 ./build/bin/llama-server \
  -m /models/Qwen3-8B-Q5_K_M.gguf --host 127.0.0.1 --port 8000 \
  --n-gpu-layers 999 --ctx-size 8192 --flash-attn on \
  --cache-type-k q8_0 --cache-type-v q8_0 --jinja --parallel 1

# fast: GTX 1660 Ti (replace 1 with its index). Two slots share the 8192 context, so each gets 4096.
CUDA_VISIBLE_DEVICES=1 ./build/bin/llama-server \
  -m /models/Qwen3-4B-Q4_K_M.gguf --host 127.0.0.1 --port 8002 \
  --n-gpu-layers 999 --ctx-size 8192 --flash-attn on \
  --cache-type-k q8_0 --cache-type-v q8_0 --jinja --parallel 2

# embeddings: CPU, so no VRAM is spent
CUDA_VISIBLE_DEVICES="" ./build/bin/llama-server \
  -m /models/nomic-embed-text-v1.5.Q8_0.gguf --host 127.0.0.1 --port 8001 \
  --embedding --n-gpu-layers 0
```

Notes:

* `--flash-attn` takes `on`, `off` or `auto` in current builds ([server docs](https://manpages.debian.org/unstable/llama.cpp-tools/llama-server.1.en.html));
  older builds use a bare `--flash-attn`. Quantizing the V cache generally needs flash attention enabled.
* `--jinja` uses the model's own chat template, which llama.cpp needs for native tool calling. If a
  model's template can't do tools, set `supports_tools: false` for that tier and Fuzzy switches to a
  prompted JSON tool protocol automatically.
* With `--parallel 2`, `--ctx-size` is divided between slots, so the config's `models.fast.context_window`
  is the per-slot figure (4096 here).
* Systemd units are in `deploy/systemd/llama-server@.service`; the compose file has a `models` profile.

Set the matching task prefixes for asymmetric embedding models (nomic uses `search_query: ` /
`search_document: `) under `models.embedding`. Changing the embedding model or dimensions requires
`sentientai memory reindex`.

## Images

Image generation runs on a separate server that Fuzzy calls over HTTP. Two kinds are supported
(`images.backend`): `automatic1111` (Stable Diffusion WebUI or Forge, started with its `--api` option)
and `openai` (any OpenAI-compatible `/v1/images/generations` server).

**The 1660 Ti is the awkward part of this machine.** Two separate problems:

1. *Black or NaN images.* GTX 16-series cards are widely reported to produce black images with
   half-precision Stable Diffusion ([example write-up](https://techtactician.com/stable-diffusion-black-image-nans-tensor-fix/)).
   The reported fix is full precision (`--no-half --precision full` in the WebUI), **which uses more
   VRAM**. Other flags (such as `--upcast-sampling` or `--no-half-vae`) are often suggested; check
   the WebUI's own documentation for your version before relying on them.
2. *VRAM.* With the 4B fast model resident (~3.4 GB), about 2.5 GB of the 1660 Ti is left. That will
   not hold a full-precision Stable Diffusion model plus activations.

Your realistic options, best first:

| Option | How | Trade-off |
|---|---|---|
| **A. Swap on demand** | Keep the image server stopped; start it (and stop the fast server) only when images are wanted, for example with a model-swapping proxy or a small script. | Reflection and background thinking pause while an image is made. Works with full precision. |
| **B. Low-VRAM mode alongside the fast model** | Run the WebUI with its low-VRAM options on the 1660 Ti next to the 4B model. | May run out of memory with full precision; slow. Test with `nvidia-smi` during a generation. |
| **C. Put images on the 2070** | Drop heavy to a smaller model/shorter context to free ~3 GB, or swap as in A. | The 2070 handles FP16 properly. Costs chat quality. |
| **D. Disable** | `images.backend: disabled` (the default). | Fuzzy can't make pictures; everything else works. |

Image generation is serialized inside Fuzzy (one at a time, `images.timeout_s` default 240 s) and
rate-limited per person (`accounts.daily_images`). SDXL-class models need considerably more VRAM than
a 6 GB card shared with an LLM and are not recommended here.

## Verify

```bash
sentientai -c sentient.yaml doctor                         # every subsystem, including both model tiers
sentientai -c sentient.yaml bench --requests 5             # p50/p95 latency and tokens/s per tier
nvidia-smi -l 2                                            # watch VRAM while chatting and while an image is made
```

If a card runs out of memory: lower `--ctx-size`, use `q8_0` cache types, choose a smaller quantization,
or move a tier to the other card.

## Storage tiers

| Tier | Holds | Config |
|---|---|---|
| NVMe | Postgres, Redis AOF, vector index, data dir, audit log, generated images | `storage.data_dir`, compose `SSD_PATH` |
| SSD/HDD | Episodic history, structured logs (in Postgres) | Postgres data directory |
| Mass HDD | Raw ingested documents, archived by content hash | `storage.archive_dir`, compose `HDD_PATH` |
