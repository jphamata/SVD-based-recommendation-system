# Spatial, latent and audio: convolution, VAE, DiT, cross-attention, Whisper

The round's second attachment asked what vapor would still lack to reach
image and video generators (Sora, Midjourney, world models). This page's answer
is the part that belongs to a runtime: the operators and the topologies
that those models use, **without a single new kernel**, checked against the
reference implementations (torch, diffusers, transformers). The part that
does not belong — data, training, compute — is stated at the end.

## 1. Convolution without a convolution kernel

An image of `H×W` pixels and `C` channels is the table `f32[H·W, Cp]` (pixels in
rows, channels in columns, `Cp` = `C` rounded up to 16, extra columns
exactly zero). Then

    conv(x) = reshape(sel(pad, ½, 0, gather_row(x, idx)), [n, k·C])·Wᵀ + b

- `idx` lists, for each output pixel and each kernel *tap*, the input row
  it reads (the convolution's im2col, a constant);
- `pad` is 1 on the *taps* that fall in the *padding*: the **selection** returns `+0`
  there — selection, not multiplication, so exactly `+0` even if the row
  read contains `NaN`;
- `reshape` (a byte copy) joins a pixel's *taps* into one row;
- a `linear` does the contraction: each output is a canonical product over `k·C`,
  **the same bits on every substrate**.

`Vapor.Spatial.conv2d/7` and `conv3d/7` (video, volumes) accept *stride*,
*padding* and dilation. Group normalization (`group_norm/9`) uses
contractions by exact selector matrices (0/1); `upsample_nearest/5` is a
`gather_row`; `pixel_attention/9` is the encoder's attention (horizon at the end).
All checked against torch to ≤ 4·10⁻⁷.

What is **not** here: grouped/*depthwise* convolutions (the
im2col layout interleaves *taps* and channels) and transposed ones (*upsampling* is *nearest*
+ convolution, as the diffusers decoders do). Mamba solves its
causal *depthwise* convolution another way — as state (FRONTIER §4).

## 2. Continuous VAE (`Vapor.Lock.Adapters.VAE`)

The decoder of diffusers' `AutoencoderKL` — the VAE of Stable Diffusion 1.x/2.x,
SDXL and, with 16 latent channels, the shape of Flux and SD3:

    z → post_quant_conv → conv_in → middle: resnet, attention over pixels, resnet
      → up blocks (resnets, nearest ×2 + conv) → GroupNorm → SiLU → conv_out

`:map` contract: latents in rows → pixels in rows. An instructive bug
showed up here: the program as a **tree** (without names) grew
exponentially with depth; every intermediate became a
named *let-binding* (`Spatial.name/3`) and the program is a DAG. Checked
against diffusers: 8.7·10⁻⁷ and 7.2·10⁻⁷ (two sizes).

## 3. DiT (`Vapor.Lock.Adapters.DiT`)

The `DiTTransformer2DModel` (Peebles & Xie): *patchify* (strided convolution =
`linear`), sinusoidal 2-D position table (**bit for bit** diffusers'),
sinusoidal *timestep embedding* with correctly rounded exponentials and sines,
class embedding, blocks with **adaLN-Zero** (scale,
shift and gate per block computed from the conditioning), and the
final layer with *unpatchify*. 3.1·10⁻⁷ against diffusers.

## 4. Cross-attention: fusion in latent space

Multimodal fusion "without a text pivot" — audio attending to video
*patches*, a DiT attending to a prompt — is the **existing attention**: the
other stream's projections are the table of keys and values, and the horizon
of every query is the last row of that table. No new operator;
≤ 10⁻⁶ against `torch.nn.MultiheadAttention` (`cross_attention_test.exs`).

## 5. Whisper (`Vapor.Lock.Adapters.EncoderDecoder`)

Speech recognition as **two programs** of the same model — the airlock's
contract gained declared *parts* (`spec.parts`):

- **encoder** (`part: :encoder`, `:encoder` contract): log-mel frames in
  rows → `conv1d → gelu → conv1d(stride 2) → gelu` (the convolutions of §1, with
  height 1) → + stored positions → bidirectional pre-LN layers →
  LayerNorm; and, once per audio clip, the **cross keys and values** of each
  decoder layer — a decoding step never recomputes them;
- **decoder** (default, `:causal_lm` contract): tokens + learned
  positions → causal self-attention (KV cache) → **cross-attention** (§4) →
  GELU MLP → LayerNorm → logits (head tied to the embeddings).

The query is scaled by `dh^−½` before the product, in `transformers`' order.
Checked against `WhisperForConditionalGeneration`: encoder states
4.8·10⁻⁷, logits 4.2·10⁻⁷, identical greedy decoding — in the oracle and on the
native worker (equal bits) — with two controls that must fail: the
decoder fed with the encoder of **another** audio clip, and the frames
**reversed** in time. The `transformers` log-mel front-end is not
reproduced (the adapter receives the *features*); vapor's is
`Vapor.Modal.Speech`.

At whisper-tiny's widths (d 384, 4 + 4 layers), 30 s of audio
(3,000 frames → 1,500 positions) go through the encoder in ≈ 1.6 s on this
2-core VM; a decoder step costs ≈ 8 ms
([bench/FRONTIER.md](bench/FRONTIER.md)).

## 6. What separates this from Sora and Midjourney

The operators and topologies above are what a runtime needs to offer;
they are here, checked. An image or video generator is, beyond that:

- **trained weights** on hundreds of millions of pairs — they are not in this
  environment, and without them no claim of visual quality is possible (which
  is why there is none);
- a **U-Net** with ControlNet (a copy of the encoder + zero convolutions) for
  spatial conditioning — the blocks exist, the adapter does not;
- **long memory** for video: of the three forms measured (ring window,
  latent cache, SSM state), none was trained for video;
- **compute**: a video DiT is a GPU (or many) for minutes — and
  resident sessions on Vulkan have existed since 0.8, but have only been measured on
  lavapipe.

vapor does not reach Sora; it reaches the point where a trained Sora could
run **with the same bits on every machine** and with a certificate.
