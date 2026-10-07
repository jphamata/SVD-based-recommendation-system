"""Sampling with diffusers' schedulers under a stand-in model — the reference
for `Vapor.Diffusion.Scheduler`.

usage: diffusers_scheduler.py KIND SPACING STEPS
  KIND     ddim | euler | dpmpp_2m
  SPACING  leading | linspace | trailing
Prints JSON {timesteps, init_sigma, x}: the initial latents are 16 fixed
numbers, the "model" predicts eps = 0.3·x + t/4000 (x the scaled input), and
x is the result after all steps (float64 throughout where diffusers allows).
"""
import json, sys, torch
from diffusers import DDIMScheduler, EulerDiscreteScheduler, DPMSolverMultistepScheduler

kind, spacing, steps = sys.argv[1], sys.argv[2], int(sys.argv[3])
cfg = dict(num_train_timesteps=1000, beta_start=0.00085, beta_end=0.012, beta_schedule="scaled_linear",
           timestep_spacing=spacing, steps_offset=1)
if kind == "ddim":
    s = DDIMScheduler(**cfg, set_alpha_to_one=False, clip_sample=False)
elif kind == "euler":
    s = EulerDiscreteScheduler(**cfg)
else:
    s = DPMSolverMultistepScheduler(**cfg, algorithm_type="dpmsolver++", solver_order=2)
s.set_timesteps(steps)
x = torch.tensor([((i * 37) % 17) / 8.0 - 1.0 for i in range(16)], dtype=torch.float64) * s.init_noise_sigma
for t in s.timesteps:
    xi = s.scale_model_input(x, t)
    eps = 0.3 * xi + float(t) / 4000.0
    x = s.step(eps, t, x).prev_sample.to(torch.float64)
print(json.dumps({"timesteps": [float(t) for t in s.timesteps], "init_sigma": float(s.init_noise_sigma), "x": x.tolist()}))
