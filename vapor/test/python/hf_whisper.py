"""Reference checkpoint for the Whisper adapter (torch tier).

usage: hf_whisper.py OUT_DIR SEED

A WhisperForConditionalGeneration built and saved by Hugging Face
transformers (float32, random weights), reduced (d_model 64, 2 + 2 layers,
80 mel bins, 24 encoder positions = 48 frames), and its outputs:

    features : float32[80, 48]   the log-mel input (random, as features are)
    hidden   : float32[24, 64]   encoder last_hidden_state
    prompt   : int32[T]          decoder input ids
    logits   : float32[T, V]     decoder logits over the prompt
    greedy   : int32[T + N]      argmax continuation, recomputed from scratch
                                 at every step (no cache: the plain definition)
"""
import sys, math
import torch
from safetensors.torch import save_file
from transformers import WhisperConfig, WhisperForConditionalGeneration

out, seed = sys.argv[1], int(sys.argv[2])
torch.manual_seed(seed)
cfg = WhisperConfig(vocab_size=96, num_mel_bins=80, d_model=64, encoder_layers=2, decoder_layers=2,
                    encoder_attention_heads=4, decoder_attention_heads=4, encoder_ffn_dim=128, decoder_ffn_dim=128,
                    max_source_positions=24, max_target_positions=32, pad_token_id=0, bos_token_id=1, eos_token_id=2,
                    decoder_start_token_id=1, suppress_tokens=None, begin_suppress_tokens=None)
cfg._attn_implementation = "eager"
model = WhisperForConditionalGeneration(cfg)
model.eval()
with torch.no_grad():
    for name, p in model.named_parameters():
        if "embed_positions" in name and "encoder" in name:
            continue                      # the sinusoids transformers writes
        if "layer_norm" in name and name.endswith("weight"):
            p.copy_(1.0 + 0.2 * torch.randn_like(p))
        elif name.endswith("bias"):
            p.copy_(0.1 * torch.randn_like(p))
        elif "embed" in name:
            p.copy_(0.5 * torch.randn_like(p))
        elif p.dim() >= 2:
            p.copy_(torch.randn_like(p) / math.sqrt(p[0].numel()))
model.save_pretrained(out)

g = torch.Generator().manual_seed(seed + 1)
features = torch.randn(1, 80, 48, generator=g)
prompt = torch.tensor([[1] + torch.randint(3, 96, (5,), generator=g).tolist()])
with torch.no_grad():
    hidden = model.model.encoder(features).last_hidden_state[0]
    logits = model(input_features=features, decoder_input_ids=prompt).logits[0]
    ids = prompt
    for _ in range(10):
        nxt = model(input_features=features, decoder_input_ids=ids).logits[0, -1].argmax().view(1, 1)
        ids = torch.cat([ids, nxt], dim=1)
save_file({"features": features[0].contiguous(), "hidden": hidden.contiguous(), "prompt": prompt[0].to(torch.int32),
           "logits": logits.contiguous(), "greedy": ids[0].to(torch.int32)}, f"{out}/reference.safetensors")
