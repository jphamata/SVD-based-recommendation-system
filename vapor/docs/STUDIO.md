# The studio — media graphs that check themselves

`Vapor.Studio` is an editor of typed-node graphs for image, sound, video,
3D, diffusion and reinforcement learning, in the spirit of ComfyUI. It has three
things that ComfyUI does not have:

1. **Exact content-addressed cache.** A node's key is the SHA-256 of its type,
   version, parameters and the keys of whatever feeds it (a Merkle DAG). If
   you change a parameter, only what depends on it runs again. The key is not
   a "looks the same" heuristic: deterministic nodes give the same bits.
2. **Receipts.** Each output carries the digest of its exact bits, and each
   run ends in a **Merkle root**. `Studio.verify/3` re-executes the
   graph without the cache and compares the roots. Anyone with the graph can check
   that a result came from it.
3. **Complete refusal.** A badly typed graph, or a ComfyUI workflow with a
   node that has no translation, is refused before running. All the problems appear
   together, each one with its repair. Nothing runs halfway.

| where | what |
|---|---|
| `Vapor.Studio` | registry, validation (types, cycles), execution, cache, receipts, `verify` |
| `Vapor.Studio.Nodes.*` | the nodes: Core, Image, Audio, Video, Vision, Diffusion, RL, Geom |
| `Vapor.Studio.Comfy` | import of ComfyUI workflows (API format) |
| `Vapor.Studio.Templates` | six starter graphs, which also serve as examples for agents |
| `Vapor.MCP.Server` / `mix vapor.mcp` | the studio and the search with proofs as MCP tools |
| console → *Studio* | the node screen ([CONSOLE.md](CONSOLE.md)) |

![The studio in the console](img/console-estudio.png)

## 1. The nodes

There are 64 types in all; the live list comes from `Studio.catalogue/0` or from the
MCP tool `studio_catalogue`. Parameters are typed and have a range and a
default. The values that travel along the wires are `image`, `mask`, `audio`,
`video`, `mesh`, `text`, `number`, `tensor`, `latent` and `json`, and
`Studio.Value` defines the digest of each one.

| category | nodes | checked fidelity |
|---|---|---|
| image | load (PNG, JPEG, GIF, PPM), scene, solid, resize, scale_by, crop, pad, flip, rotate, blur, sharpen, invert, grayscale, levels, blend, composite; masks (threshold, rect, invert, feather) | resizing = `torch.nn.functional.interpolate` (nearest-exact, bilinear, bicubic a = −0.75, area) and Pillow's Lanczos, within 2.2·10⁻⁶; native program = the BEAM's sparse evaluator, bit for bit (`studio_media_test.exs`) |
| sound | load (WAV), tone, noise, gain, mix, concat, trim, fade, normalize, reverse, resample, spectrogram | resampling 48 → 16 kHz: 69.6 dB SNR against the ideal tone (misaligned decimation: 17.7 dB); 7 kHz → 8 kHz does not alias (< −40 dB) |
| video | load (Y4M, MJPEG-AVI, GIF), still, camera (Ken Burns), frame, map (subgraph per frame), resize, trim, reverse, concat, crossfade, fps, upscale | AVI = libjpeg frame by frame; Y4M planes = ffmpeg; GIF read by Pillow with our pixels |
| vision | upscale (×2/×4), OCR, handwritten digit | the consistent upscaler (§3) |
| 3D | shape (sphere, torus, box, gyroid, SDF blobs), heightmap, render, turntable | closed mesh, volume and area within 0.5% of the analytic sphere; GLB/OBJ/PLY read by trimesh |
| RL | episode (CartPole, FrozenLake, two-joint arm; trained, random or expert policy) | dynamics = gymnasium within 10⁻¹²; FrozenLake table = gymnasium's (`rl_test.exs`) |
| diffusion | checkpoint, text, empty_latent, sample, decode, encode, generate | = diffusers pipelines within ~10⁻⁶ (§2) |

All nodes follow the same **determinism contract**: the same inputs
and parameters give the same bits on any machine. Only IEEE operations
(+ − × ÷ √) and the correctly rounded functions of
`Vapor.CR` (sin, cos, exp, log, pow) go into the BEAM. The rest runs as a certified program
in the worker.

The codecs were written here:

- **GIF**: *median cut* palette, optional Floyd–Steinberg, LZW identical
  to Go's `compress/lzw`.
- **Y4M**: BT.601 in limited range.
- **MJPEG-AVI**: reading of the `movi` chunks, with the Huffman tables of Annex
  K inserted when the DHT is missing.

H.264, MP4 and WebM were left out (§7).

## 2. Stable Diffusion

`Vapor.Diffusion.Pipeline` opens a directory in the diffusers layout and
admits each network through the airlock:

| part | adapter | check |
|---|---|---|
| `text_encoder/` (CLIPTextModel) | `Encoder` (clip_text) | already checked against `transformers` (0.6) |
| `unet/` (UNet2DConditionModel) | `UNet` (new) | SD 1.x: 1.17·10⁻⁶; SD 2.x (linear projections, heads of 64): 1.03·10⁻⁶ |
| `vae/` (AutoencoderKL) | `VAE`, encoder **and** decoder | encoder 1.14·10⁻⁶ (diffusers' asymmetric {0,1,0,1} *pads*) |
| `scheduler/` | `Vapor.Diffusion.Scheduler` | DDIM, Euler, DPM-Solver++ 2M × leading/linspace/trailing spacing × 10/25 steps: the same *timesteps* and the same final latents (≤ 6·10⁻⁷, diffusers' float32) |
| `tokenizer/` | `Vapor.Tokenizer` | `tokenizer.json` or the slow files (vocab.json + merges.txt), assembled into the tokenizer that `transformers` would write; same ids |

The networks run as float32 programs compiled once per shape. The
sampler runs in binary64 on the BEAM: the *schedule*, classifier-free
guidance (CFG), the step and inpainting's latent blending. The initial
noise comes from `Vapor.Modal.Rng`. **An image is a function of (weights, prompt,
seed, steps, guidance, sampler).**

The whole pipeline was compared with diffusers itself on a tiny
random checkpoint, included in `priv/quality/sd_tiny`, with the same ids and
the same noise:

| mode | diffusers | largest per-pixel difference |
|---|---|---|
| text → image, DDIM | `StableDiffusionPipeline` | 9.5·10⁻⁷ |
| text → image, Euler | same | 9.2·10⁻⁷ |
| text → image, DPM++ 2M | same | 9.5·10⁻⁷ |
| image → image (strength 0.6) | `StableDiffusionImg2ImgPipeline` | 1.1·10⁻⁶ |
| inpainting (4-channel U-Net) | `StableDiffusionInpaintPipeline` | 1.4·10⁻⁶ |
| **control**: DPM++ against the DDIM reference | — | 0.096 |

There are two deliberate differences from diffusers, both declared:

- The VAE encoder's latent is the **mean** of the distribution. diffusers
  samples, which is a second, hidden source of randomness; in the
  comparison, diffusers was forced to the mean.
- Inpainting is done by latent blending with a 4-channel U-Net. The
  9-channel inpainting U-Net is refused by the airlock.

In the studio, the `diffusion.*` nodes have ComfyUI's shape: checkpoint → text
→ empty latent → sampler → decoding. ComfyUI's txt2img workflow
is imported, runs and gives **the same bits** as `Pipeline.generate`
(`diffusion_pipeline_test.exs`).

The checkpoint must be a **diffusers directory** inside the
studio's directory or `VAPOR_MODELS`. A single-file `.safetensors` is refused
with the conversion to perform:
`StableDiffusionPipeline.from_single_file(f).save_pretrained(dir)`.

## 3. The consistent upscaler

`Vapor.Vision.Upscale` (`image.upscale`, `video.upscale`,
`mix vapor.upscale`) upscales ×2 or ×4 with a guarantee: **downscaling the
result gives back the input**. In symbols, D(y) = x, where D is the 2×2
block mean, and this holds by construction, up to binary64 rounding.

The model works in three steps:

1. A *patch* MLP (49 → 64 → 64 → 4) predicts a correction on top of
   Lanczos ×2.
2. An exact projection imposes D(y) = x, redistributing the difference when there is
   saturation.
3. In color images, luma goes through the network and chroma through Lanczos; everything
   is projected at the end.

Training was done here: 30 thousand steps on public images from scikit-image
and text rendered in free fonts. It is reproducible bit for bit, and the
training receipt is in `priv/upscale/config.json`.

PSNR on held-out images (never seen in training):

| image | vapor | Lanczos + the same projection |
|---|---:|---:|
| text 0 | **24.58** | 21.17 |
| text 1 | **22.91** | 20.16 |
| Shepp–Logan phantom | **34.04** | 28.23 |
| photograph (camera) | 30.49 | 30.42 |
| photograph (chelsea) | 34.46 | **34.69** |
| text 2 | **21.36** | 19.81 |
| grass | 23.94 | **24.03** |
| color wheel (smooth gradient) | 53.73 | **55.15** |

- **Inconsistency** |D(y) − x|: ~10⁻¹⁶ in vapor, against 0.05–0.17 for
  Lanczos and bicubic.
- **Downstream OCR**: the CER drops by half relative to the small image.
- **Finding**: the projection alone already improves Lanczos by 0.2–1 dB.

The honest claim is this:

- on text and line graphics, the upscaler beats the best baseline with the same
  guarantee by +1.5 to +5.8 dB;
- on photographs, it ties (−0.2 to +0.1 dB);
- on a smooth gradient, it loses 1.4 dB, with both above 53 dB.

It **does not invent detail**, and that is on purpose: a result that
contradicts the input is not consistent. A GAN that hallucinates texture is another
tool, and it is not here.

## 4. Reinforcement learning, 3D, games

Reinforcement learning (`Vapor.RL`):

- **Environments**: CartPole (gymnasium's dynamics), FrozenLake (gymnasium's P
  table) and a two-joint arm that reaches targets.
- **Algorithms**: value iteration, Q-learning, REINFORCE (policy
  gradient as a program, via `Vapor.Autodiff`) and behavior
  cloning.
- **Results of the included policies**:
  - CartPole: return 472.9/500 on never-seen episodes; untrained, 18.
  - Arm: 77% of the held-out targets; the expert hits 100%, the random
    policy 1%.
  - FrozenLake: Q-learning finds the optimal policy, with 74.7% success;
    "always left" gives 0%.
- **Replay**: an episode is the seed plus the actions. The `rl.episode` node
  returns the video, the return and the actions, and the replay redraws the same
  frames.

3D (`Vapor.Geom`):

- **Meshes**: marching tetrahedra over SDFs, with vertex welding and
  outward orientation; relief from images.
- **Export**: OBJ, PLY and GLB.
- **Visualization**: rasterizer with a deterministic z-buffer and turntable
  video.

## 5. For agents: the MCP server

```sh
mix vapor.mcp --dir /caminho/do/trabalho
```

This starts a Model Context Protocol server (stdio, JSON-RPC 2.0) with six
tools. It was tested with the **official client of the MCP Python SDK**
(`mcp_server_test.exs`). Each tool attacks a pain of people who use agents:

| tool | pain |
|---|---|
| `studio_catalogue` | the agent guesses node and parameter names; here it reads them, with types and ranges |
| `studio_validate` | a wrong wire discovered after minutes of computation; here before, with the node and the repair |
| `studio_run` | on every new attempt, everything is recomputed; here the cache **lives between calls**, and an agent that edits a node recomputes only what depends on it (measured: 2 of 5 nodes) |
| `studio_verify` | "did the tool really produce this?"; the Merkle root is verifiable by anyone |
| `comfy_import` | workflows shared as ComfyUI JSON |
| `context_search` | citations the agent may invent; here each chunk comes with its Merkle proof against the corpus root |

Behavior details:

- Outputs are written as files with the digest in the name, and small
  images also come back embedded in the response.
- A tool failure comes back as a result with `isError` and the rejection
  (what was expected, how to repair it), so that the model reads it and corrects.
- A node that raises an exception becomes a tool error; the server stays
  up.
- Paths outside the working directory are refused.

## 6. The Hugging Face courses: coverage map

The request was "cover everything in the courses" (translated). Covering everything is not a verifiable
goal. What can be verified is this map: what each course teaches, what
exists here, checked, and what is missing, with the reason.

| course | checked here | missing, and why |
|---|---|---|
| LLM | decoders through the airlock (Llama, Qwen2/3, Mistral, Mixtral, Phi-3, Granite, DeepSeek-V3…), BPE/SentencePiece tokenizers, engine with continuous batching, speculation, structured output | Unigram/WordPiece (T5/BERT) |
| smol (fine-tuning small models) | LoRA by distillation as a recurrent program (`Vapor.Train`), model merging (`Vapor.Merge`), checkpoint judge | SFT with instruction data, DPO (training only distills today) |
| MCP / context | MCP server (this document), MCP client for agents, RAG with Merkle proofs | signed context manifests |
| Agents | agent runtime with journal, *replay* and effect classes; MCP tools | — |
| Deep RL | Q-learning, value iteration, REINFORCE, cloning; CartPole/FrozenLake = gymnasium | PPO/DQN, Atari, MuJoCo (no simulators here) |
| Robotics (LeRobot) | behavior cloning on a simulated arm | LeRobot dataset format, ACT/Diffusion Policy policies, hardware |
| Computer vision | OCR with tables, ViT/CLIP checked, consistent upscaler, digit classifier | detection/segmentation with trained weights |
| Audio | Whisper checked (encoder + decoder), speech → digit, sound nodes, spectrogram | log-mel front end equal to `transformers`'s, TTS |
| Cookbook | RAG, structured output, agents, merging — recipes with tests | — |
| ML for games | RL episodes as video, seeded procedural scenes | game environments (Unity ML-Agents, Godot) |
| Diffusion | own DDPM/DDIM (digits), **SD 1.x/2.x = diffusers**, DDIM/Euler/DPM++ 2M, img2img, inpainting | ControlNet, diffusion LoRA, SDXL (two text towers), text-conditioned DiTs (SD3, Flux) |
| ML for 3D | SDF meshes, relief, GLB/OBJ/PLY, rendering | NeRF, Gaussian splatting, image → 3D with trained weights |

## 7. Limits

This document claims nothing beyond what is below:

- **Stable Diffusion with real weights has not run here.** This machine does not
  download weights; parity is with tiny random weights. The speed of
  a 512×512 SD on the CPU was not measured either. Each step is a compiled
  program, and the CPU will be slow.
- **ComfyUI pixels**: a translated KSampler uses diffusers'
  samplers over the checkpoint's *schedule*, not ComfyUI's sigmas; each
  translation warns about this. ComfyUI's code was not reproduced: the translations
  were written from the nodes' documented semantics.
- **Video**: there is no H.264, MP4 or WebM. They are huge formats, and an
  ffmpeg via *shell* violates the product's rule, which does not execute foreign
  processes. Reading covers Y4M, MJPEG-AVI and GIF; writing, GIF and Y4M.
- **Studio performance**: the camera over a 320×192 image takes about
  0.25 s per frame on 2 cores (the Lanczos weights go through the correctly
  rounded sine). Frames are computed in parallel, and a fully cached
  run comes back in about 1 s with the previews.
- **Cache**: the checkpoint enters the key by its path. If the weights are
  swapped at the same path, the cache does not notice; use a new directory.

## How to challenge this

```sh
mix test test/vapor/diffusion_pipeline_test.exs   # schedulers, U-Net, VAE, pipelines and ComfyUI × diffusers
mix test test/vapor/studio_test.exs test/vapor/studio_media_test.exs test/vapor/geom_test.exs \
         test/vapor/rl_test.exs test/vapor/upscale_test.exs test/vapor/mcp_server_test.exs test/vapor/console_test.exs
python3 test/python/diffusers_pipeline.py /tmp/sd  # the tiny checkpoint and the diffusers references
mix vapor.mcp --dir /tmp/trabalho                  # and any MCP client
mix vapor.quality                                  # §5e of docs/bench/QUALITY.md
```
