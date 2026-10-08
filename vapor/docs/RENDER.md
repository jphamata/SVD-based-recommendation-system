# Render — physically based light, on the viewer's GPU, with a reference that checks itself

> Request (0.12), translated: "the scene and creation and editing in the studio much more
> flexible and customisable and aiming at the possibility of photo-realism".
> Scrutiny: [DIRECTIVE.md §15](DIRECTIVE.md).

The photo-realism of film renderers comes from a single equation —
light transport — solved by Monte Carlo sampling
(path tracing). Here there are two tracers of the **same scene
format and the same materials**:

- `Vapor.Render` (Elixir): the **reference** — deterministic per
  (seed, pixel, sample), rows spread across all the
  schedulers, PNG with ACES mapping and sRGB;
- `priv/console/gpu_tracer.js` (WebGL2): **progressive on the browser's
  GPU** — one sample per pixel per frame accumulated in a float
  texture, the image converging as you watch.

Console *Make → Render* · MCP `render_scene`.

## 1. The scene

```
camera pos=0,1.2,4.5 look=0,0.8,0 fov=45 [aperture=… focus=…]
sky top=0.55,0.7,1.0 bottom=1,1,1          # or color=
sun dir=0.4,1,0.3 color=1,0.95,0.85 power=2.5
plane y=0 mat=diffuse albedo=0.75,0.75,0.75 checker=0.5
sphere c=0,0.8,0 r=0.8 mat=glass ior=1.5
sphere c=-1.7,0.6,-0.5 r=0.6 mat=metal albedo=0.95,0.75,0.4 rough=0.08
box min=-0.4,0,-2 max=0.4,1.2,-1.4 mat=diffuse albedo=0.3,0.5,0.8
sphere c=0,4,0 r=0.5 mat=emit color=1,0.9,0.8 power=12
exposure value=1.2
```

Materials: Lambertian diffuse (cosine sampling), metal (mirror
with roughness), glass (Fresnel–Schlick, Snell, total internal reflection),
emissive (area lights). Lights: emissive objects, sky (uniform or
vertical gradient) and a directional sun **sampled explicitly**
(next-event estimation). Russian roulette terminates paths without bias.

In the console, the text is the source of truth and editing is live: each
keystroke rebuilds the scene and restarts convergence; **dragging the image
orbits the camera and the wheel zooms — and the text's `camera` line is
rewritten**, so that what you see is always reproducible from the text.

## 2. How it is checked

The way renderer authors check theirs
(`render_test.exs`, §5h):

- **White furnace**: a sphere of albedo a in a uniform environment of
  radiance 1 has to show exactly a at every pixel — energy
  conservation. Maximum error 0 (cosine sampling makes each sample
  exact).
- **Gradient furnace**: under the sky L(ω) = (1 + ω_y)/2 a convex
  Lambertian surface shows a·(½ + n_y/3). This checks the *distribution*
  of the estimator, which the uniform furnace does not see: mean error 6·10⁻⁴; the
  biased estimator (uniform directions treated as cosine — the
  control) gives a·(½ + n_y/4) and is caught (−0.04).
- **N^−½ convergence**: the RMS error against a 4096-sample reference
  falls with slope −0.5 ± 0.12 between 8 and 512 samples.
- **Determinism**: same seed, same image, bit for bit.
- **The GPU against the reference** (headless Chromium, WebGL2 on
  SwiftShader): the gradient furnace passes on the GPU and the mean radiance
  of a scene with glass, metal and sun agrees with Elixir's within 3%
  (two independent Monte Carlo estimates); in the console, the white
  furnace agrees within 0.00%.

## 3. Honest limits

- No multiple importance sampling (MIS): small lights
  reached only through the BRDF — the sun seen through glass (caustics),
  small emitters — have heavy-tailed variance ("fireflies").
  That is why the N^−½ check uses a diffuse scene.
- No acceleration structure: up to 48 objects on the GPU and 200 on the server;
  no triangle meshes, image textures, volumes or subsurface.
- The server limits width × height × samples to 6 million per request;
  the browser's GPU has no such limit.
