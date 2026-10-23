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
| heavy (chat, reasoning, deliberation) | RTX 2070 | Qwen3-8B at Q5_K_M, 16k context | ~5.9 GB | ~7.3 to 7.8 GB |
| fast (reflection, appraisal, grounding checks, summaries) | GTX 1660 Ti | Qwen3-4B at Q4_K_M, 16k context shared by two slots (8k each) | ~2.5 GB | ~4 GB |
| embeddings | CPU | nomic-embed-text v1.5 (768-d) at Q8_0 | ~0.15 GB | none |
| images | GTX 1660 Ti (leftover) | SD 1.5-class model, low-VRAM mode | ~2 GB+ | see "Images" |

Sources for the sizes: third-party mirrors list Qwen3-8B Q5_K_M at about 5.85 GB and
Qwen3-4B Q4_K_M at about 2.50 GB ([8B listing](https://app.alphaneural.io/assets/qtum/Qwen3-8B-GGUF),
[4B listing](https://huggingface.co/6block/Qwen3-4B-GGUF)). Confirm file sizes on the repository you download from.

The KV-cache part of the totals is an **estimate** from the architecture (36 layers, 8 KV
heads, 128-dim heads, so roughly 144 KB per token at FP16): about 1.2 GB for 8k tokens, or about
0.6 GB per 8k with `q8_0` cache quantization (1.2 GB for the 16k used here), plus a few hundred MB of compute buffers. If the 2070 also
drives a monitor, expect desktop VRAM use on top: run it headless or lower `--ctx-size`.

Why this split: most of Fuzzy's model calls are small, frequent and background (appraising a
conversation, checking a reply against memory, choosing what to do next, summarising). Those go to the fast tier
so the heavy model is free for conversations. Fuzzy's background life is never allowed to starve a person who is waiting:
when someone talks to it, its own activities pause.

### Going bigger: one 14B model across both cards

Two small servers are the fast, safe layout. If you want a noticeably more capable mind and can accept
having just one model, a **14B at Q4_K_M** (about 9 GB of weights; see the
[bartowski quant of mlabonne's abliterated Qwen3-14B](https://huggingface.co/bartowski/mlabonne_Qwen3-14B-abliterated-GGUF),
which lists Q4_K_M at 9.00 GB) can be layer-split over both cards (8 + 6 = 14 GB of VRAM). That leaves
room for a 16k to 32k context with the `q8_0` cache (roughly 80 KB per token for a 40-layer, 8-KV-head
model, so about 1.3 GB per 16k tokens: an estimate, check with `nvidia-smi`).

Trade-offs, plainly: both of Fuzzy's tiers share one model, so a conversation and its background
thinking queue behind each other (two slots keep them from blocking completely); generation is slower than
an 8B on one card because the layers are split across PCIe; and nothing is left on the 1660 Ti for images.
In exchange the replies, the self-reflection and the choices it makes about its own life come from a
model about twice the size.

```bash
huggingface-cli download bartowski/mlabonne_Qwen3-14B-abliterated-GGUF \
  --include "*Q4_K_M*" --local-dir /mnt/nvme/models
# deploy/.env:  BIG_MODEL=<that file's name>   HEAVY_URL=http://llm-big:8000/v1   FAST_URL=http://llm-big:8000/v1
#               BIG_CTX_SIZE=24576  HEAVY_CTX=12288  FAST_CTX=12288  BIG_SPLIT=6,8
# BIG_SPLIT is the share per card in `nvidia-smi -L` order (the 6 GB card gets the smaller share).
# If a card runs out of memory while loading (cudaMalloc failed), lower BIG_CTX_SIZE (and the two
# *_CTX values to half of it) or move more to the other card, e.g. 5,9.
docker compose -f deploy/docker-compose.yml --profile models down      # stop the two small servers
docker compose -f deploy/docker-compose.yml --profile bigmodel up -d --build
```

To go back, `--profile bigmodel down`, clear `HEAVY_URL`/`FAST_URL`, and bring `--profile models` up again.
Anything with a GGUF and a chat template works; check each model's licence before running it on a public site.

### Simplest and coolest: one 8B on the 2070 only

Compose profile `single` runs just the embedding server and one model on the RTX 2070, and Fuzzy uses it for both
tiers (two slots, so a conversation and its background work don't block each other). The 1660 Ti stays idle: less
heat, no cross-card traffic, and none of the 16-series half-precision quirks. A 5 to 6 GB model such as a Llama 3.1 8B
Q5_K_M leaves room for 16k of context. See `deploy/.env.example` (Alternative 1) for the settings.

If the output ever turns to gibberish (a stream of repeated digits, for example, or an empty reply to a long prompt),
run `sentientai probe`. If short prompts are fine and long ones are junk, suspect flash attention or the quantized
cache on your GPU/driver combination: try a different llama.cpp build first (`LLAMA_CUDA_IMAGE=ghcr.io/ggml-org/llama.cpp:server-cuda12`, the CUDA 12 image, since the default image may be built for CUDA 13), then `FLASH_ATTN=off` and `KV_TYPE=f16` (the f16 cache is twice the size, so
halve the context: `HEAVY_CTX_SIZE=8192 HEAVY_PARALLEL=1 HEAVY_CTX=8192 FAST_CTX=8192`), and try the model on the other card to rule hardware in or out. Watch temperatures on the Brain
page's System tab; a card that sits above roughly 83 C will throttle. Limiting board power usually costs little speed
and lowers heat a lot: `sudo nvidia-smi -i <index> -pl 140` (the default for the 2070 is 175 W; it resets on reboot).

### Thinking models and latency

Qwen3 models "think" before answering. That helps the quality of what Fuzzy decides, and it costs seconds.
If replies feel slow you can turn thinking off for a tier with `models.<tier>.extra_body`
(see `sentient.example.yaml`); the trade is shallower reasoning. Fuzzy's own background beats are fine with
thinking on. Fuzzy never cuts a thought off for length: each background step has a generous output budget
(up to 8000 tokens for building, 4000 for most others), and a thought that doesn't come out in the expected
JSON shape is kept as written instead of being dropped.

### Context and time limits

"No limit" isn't possible: a model has a fixed context window and a GPU has fixed memory. What Fuzzy does
instead is the following. Conversations use the full window the server offers (the server's `--ctx-size`
divided by its slots, matched by `models.<tier>.context_window`); when a conversation outgrows that, the older
part is summarised, and the verbatim text stays in memory, searchable, so nothing is lost. The default time
limits are very generous (`models.*.timeout_s` 900 s fast / 3600 s heavy, a chat reply may take up to
`accounts.chat_timeout_s` = 2 hours, autonomous tasks up to `limits.max_wall_time_s` = 2 hours and
1,000,000 tokens). The loop breakers that stop a model repeating the same tool call forever are kept: they
exist to stop a stuck loop, not to limit thinking.

## Seeing the machine

Administrators get a **Server** panel (Brain → System, and Admin → Server) with each GPU's temperature, load,
VRAM, power and fan, the CPU model and load, memory, any CPU/board temperature sensors the kernel exposes, and disk
space. It reads `nvidia-smi` and `/proc`, `/sys` from inside the container; the compose file gives the
`sentientai` container the NVIDIA `utility` capability for this (stats only, no compute). Containers often can't
see CPU package temperatures; then the panel says so rather than guessing. For temperatures on the host itself:
`watch -n2 nvidia-smi` and `sensors` (from `lm-sensors`).

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
  --n-gpu-layers 999 --ctx-size 16384 --flash-attn on \
  --cache-type-k q8_0 --cache-type-v q8_0 --jinja --parallel 1

# fast: GTX 1660 Ti (replace 1 with its index). Two slots share the 16384 context, so each gets 8192.
CUDA_VISIBLE_DEVICES=1 ./build/bin/llama-server \
  -m /models/Qwen3-4B-Q4_K_M.gguf --host 127.0.0.1 --port 8002 \
  --n-gpu-layers 999 --ctx-size 16384 --flash-attn on \
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
  is the per-slot figure (8192 here).
* Systemd units are in `deploy/systemd/llama-server@.service`; the compose file has a `models` profile.

Set the matching task prefixes for asymmetric embedding models (nomic uses `search_query: ` /
`search_document: `) under `models.embedding`. Changing the embedding model or dimensions requires
`sentientai memory reindex`.

## Images

Fuzzy makes pictures through an image server it calls over HTTP (`images.backend`): the bundled one below,
`automatic1111` (Stable Diffusion WebUI/Forge with `--api`) or any OpenAI-compatible `/v1/images/generations`
server. Pictures made in a conversation appear in the chat (private to that person); pictures Fuzzy makes on its
own go in the public gallery. Prompts asking for sexual images are refused, and the bundled server also runs every
finished picture through the Stable Diffusion safety checker (on the CPU) and discards explicit ones.

### The bundled server, on demand on the 1660 Ti (compose profile `images`)

Both cards hold a language model, so there is no spare GPU memory for an image model. The bundled server
(`deploy/imagegen`) solves this by borrowing the 1660 Ti only while it draws:

1. Fuzzy asks for a picture. The app stops the model server on that card (`IMAGE_SWAP_CONTAINER`, normally
   `sentientai-llm-fast-1`) through the Docker API. Its background work falls back to the 2070 meanwhile.
2. A short-lived worker loads the image model, draws in **full precision** (GTX 16-series cards make black images in
   half precision) and exits, so the card is completely free again.
3. A minute after the last picture (`images.swap_grace_s`), the model server is started again. Several pictures in a
   row only pause it once. If anything is interrupted, the app starts it again at boot.

The server holds no GPU memory while idle. On first start it downloads the model (about 5 GB, into
`MODELS_PATH/imagegen`) and reads it once so the operating system keeps it in RAM; each picture then loads it in
seconds. If the card still runs out of memory it retries with model offloading, then on the CPU (slow, but it
works). `IMAGE_DEVICE=cpu` keeps it off the GPUs entirely. `IMAGE_MODEL` takes any SD 1.5-class diffusers repo
(default `Lykon/dreamshaper-8`).

```bash
cat >> deploy/.env <<'ENV'
SENTIENT_IMAGE_BACKEND=openai
SENTIENT_IMAGE_URL=http://imagegen:7861
IMAGE_SWAP_CONTAINER=sentientai-llm-fast-1
ENV
docker compose -f deploy/docker-compose.yml --profile models --profile images up -d --build
docker compose -f deploy/docker-compose.yml logs -f imagegen     # first start: downloading the model
```

Expect roughly 20 to 60 seconds per 512x512 picture on the 1660 Ti, plus about a minute before the background model
is back. Image generation is serialized (one at a time) and rate-limited per person (`accounts.daily_images`).

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
