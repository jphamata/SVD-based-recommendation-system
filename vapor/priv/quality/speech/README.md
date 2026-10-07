# Held-out speech for the quality suite

100 recordings (digits 0–9, takes 0–9) of the speaker `theo` from the
**Free Spoken Digit Dataset** (https://github.com/Jakobovski/free-spoken-digit-dataset),
licensed CC BY-SA 4.0, copied unchanged (8 kHz, 16-bit mono WAV).

The speech reader shipped in `priv/speech` was trained on the five other
speakers (george, jackson, lucas, nicolas, yweweler) and never heard this
voice: `mix vapor.quality` measures it here, speaker-independently. The
held-out speaker was chosen before training.
