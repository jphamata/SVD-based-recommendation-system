# Living scene, sketch → drawing, save and export (0.11)

> Request, translated: "from a sketch, move on to models to draw something with
> AI (photorealistic, architectural or even engineering) […] animations
> (offline) or infinite loops with entropy and interaction of scenes from
> drawings […] from a complex image, turn it into an interactive infinite
> loop with NPCs, 3D, depth, free navigation through the scene,
> skeletons and actions for entities, effects (light, gravity and beyond) […]
> with direction and interactive adjustments via prompt, and saving and exporting
> of the results (this applies to everything!)". Scrutiny:
> [DIRECTIVE.md §14](DIRECTIVE.md). Tests: `scene_test.exs`,
> `sketch_test.exs`, `archive_test.exs`, `console_test.exs`. Console:
> *Make → Living scene*, *Make → Sketch*, *Trust → Archives*.

## 1. What is possible here, and what is not

Taken literally, the request is an AI world generator: image →
learned depth → 3D mesh → characters with a learned skeleton →
video synthesis. Each of these pieces, in the state of the art, is a **trained
network** (monocular depth estimation, segmentation, *pose
estimation*, video diffusion) — and this machine has no GPU and cannot
download weights. Faking those pieces would be delivering noise dressed up as magic.

What can be done from first principles — and is what was done — is
a **2.5D** scene: the image separated into layers at plausible
depths, what lies behind each one reconstructed, a ground one
can walk on, and an engine that brings all of this to life with simple physics, light and
weather, **directed by words** and **reproducible from the seed**. Where a
trained network would come in (depth, segmentation, generation), the interface
is the same: the layers and their depths are data, editable by hand, and
replaceable by a model when there is one.

## 2. The analysis (`Vapor.Scene.analyze/2`)

| step | method | what it guarantees |
|---|---|---|
| regions | SLIC superpixels (Achanta et al. 2012) in CIELAB, merged over the adjacency graph by colour (small regions first) | regions that respect edges |
| sky | a smooth, bright (or bluish) region touching the top of the image, **grown downward** through smooth regions of neighbouring colour (a sky gradient is several regions) | horizon where the sky ends |
| depth | the **ground plane**: an object is where its lowest pixels are, and on a ground seen from eye height the depth of a point on line *y* below the horizon is ∝ 1/(y − y_h); the sky at infinity | depth order consistent with perspective — **a heuristic, stated as such**, editable layer by layer |
| layers | regions grouped by log-depth (at most 6) | few layers, well separated |
| what is behind | each layer keeps its pixels and fills the band hidden behind the nearer layers by **push-pull** (Gortler et al. 1996) with bilinear pull; the back layer fills everything | moving the camera reveals a plausible continuation, not holes |
| walkable ground | cells of the ground regions (those touching the bottom edge) below the horizon | paths for A* |
| light | centroid and colour of the brightest 1% of pixels | the hearth of a room, the sun of a landscape |

Measured on images drawn by a script with known geometry
(`priv/quality/scene`): landscape horizon at 0.44 (truth 0.417), sky
at infinity, ground last and walkable; in the guild hall, the light at the
hearth (x = 0.20, warm). A 640×420 landscape is analysed in ~5 s on the
BEAM.

## 3. The engine (`SceneEngine`, in the console and in the exported HTML)

Canvas 2D, no libraries. Each layer is a **plane** at a depth
*D*, seen by a camera (x, y, z) with perspective projection: moving the
camera shifts each layer by 1/(D − z) — correct parallax for planes.
The ground is drawn in **strips**, each at the depth of its line,
so that it recedes like a real ground.

- **Inhabitants** walk the ground by **A\*** on the walkable grid; the height
  of someone on line *y* comes from the geometry of a photo taken at eye
  height: **the head stays on the horizon** (camera at 1.6 m, person at 1.7 m).
  Destinations by word ("to the door" = the back of the ground, "left",
  "front", "the light"…).
- **Weather and effects**: rain (drops in a slice of depths, with
  a splash where the drop meets the ground at its depth), snow,
  fog by depth (1 − e^{−D/6}), storm with lightning,
  gusty wind that sways the vegetation (green layers drawn in
  shifted strips), torches and candles with flicker, embers with buoyancy and
  turbulence, smoke, fireflies, birds, butterflies, leaves.
- **Time of day**: each layer is "graded" once per change —
  multiplied by the colour of the light (dawn, day, dusk, night) and
  fogged by depth, with alpha preserved —, and the day cycle
  regrades every half second.
- **Entropy**: a single generator (mulberry32) with the scene's seed and a
  fixed 30 Hz step; entropy controls gusts, the inhabitants' hesitation,
  flight noise, flicker. **Same scene, same operations,
  same seed → the same loop**, on any machine.

## 4. Drawings that move (`Vapor.Scene.rig/2`)

Dark strokes on light paper → thinning (Zhang & Suen 1984) → skeleton
graph (nodes by *crossing number*, so that a pixel staircase does not
become a junction; closed loops cut at one pixel) → each chain becomes bones
(RDP), oriented away from the junction nearest the centre → a mesh
covers the ink and each vertex is bound to the two nearest bones. The
engine animates by **forward kinematics and linear skinning**, drawing each
mesh triangle with its own affine transform: *wave* (the chain whose
tip is highest), *walk* (downward chains swing alternately,
the others in opposition), *dance*, *breathe*, *sway*.

Measured: the test stick figure has its **four** extremities where
they were drawn (within 30 px). **The motion comes from the topology of the
skeleton, not from knowing what the drawing shows** — an "arm" is a chain
that ends at a free point.

## 5. Sketch → technical drawing and floor plan → 3D (`Vapor.Sketch`)

**Vectorisation with constraints.** Each skeleton chain is fitted by
a line (total least squares), a circle or arc (Kåsa's algebraic
fit) or broken into lines; then **beautified**: orientation snapped to 0°,
45°, 90° when within 4°; near-parallels made parallel;
collinear horizontals and verticals aligned; corners welded at the
least-squares point of their lines. The constraints found are listed
— what the drawing *meant to say*, declared. Outputs: **SVG** and **DXF R12**.

Measured (`priv/quality/sketch`, drawn by script with hand tremor):
a rectangle drawn **2.2° askew** comes back straight and closed (the control —
the same fit without constraints — stays askew); the circle comes back with
centre and radius within 1 px; the hypotenuse at nearly 45° goes to 45°; a
free line at 29.5° stays where it is.

**Floor plan → 3D.** The lines are walls; a gap between collinear walls is
a **door** (its width corrected for the stroke thickness, which
thinning shortens by half at each end); the **rooms** are the faces
of the planar graph of the walls (doors closed), found by walking the
half-edges; the scale comes from the longest wall (8 m by default,
adjustable) or from `scale:`. The walls are extruded (2.7 m, 15 cm
thick, lintels over the doors at 2.1 m) into a `Vapor.Geom.Mesh`
exported as **glTF (GLB)**; the console has its own 3D viewer.
Measured: rooms of **11.98 and 19.80 m²** (truth 12 and 20), doors of
**0.90 and 1.00 m** (truth 0.9 and 1.0), the GLB opened by trimesh with 2.7 m
height and the plan as its base.

**The photorealistic** (sketch → render): vapor's Stable Diffusion pipeline
(img2img, inpainting) takes the sketch as the initial image — with a
checkpoint that the user loads. None ships with it; none of this was measured
here.

## 6. Direction by prompt (`Vapor.Scene.direct/1`)

A vocabulary in Portuguese and English becomes **operations** — weather and
intensity ("heavy rain", "light snow"), time, wind, lights, inhabitants
and quantities ("three people", "many birds"), camera ("orbit",
"zoom in"), entropy ("calm", "chaotic"), animation of the drawing, destinations
("to the door"), negations ("no rain"). **Every word not understood is
reported, never guessed.** The operations form the scene's **script**,
saved with it. A loaded language model could emit the same
operation schema under the JSON Schema constrained decoding that
vapor already has — not wired in this round (TODO).

### 6.1 Named inhabitants, in time (0.12)

Since 0.12 direction reaches **one inhabitant at a time and one instant**.
The prompt is split into clauses (`,` `;` `then` `depois` `e então`); a
clause that names an inhabitant — created in the prompt itself ("um cavaleiro
chamado Artur", "a guard named Ana") or already in the scene — becomes an operation on
it: go to a place, say a line in quotes (speech balloon), wave,
dance, sit, jump, run, stop, patrol, flee, follow another;
a **pronoun** ("ele", "she") refers to the last one named; an unnamed
role with a verb ("a guard patrols") creates an inhabitant with the role's
name; a clause that begins with a time ("aos 3 s", "after 5 seconds",
or a time alone followed by a comma) becomes a **keyframe** that the
engine plays at that instant.

```
a knight named Arthur walks to the door, then at 3s he says "hello" and waves; a guard patrols
→ spawn Arthur · Arthur goto door · (3 s) Arthur say "hello" · (3 s) Arthur wave · Guard patrol
```

In the console, the **inhabitants** inspector lists each one (name, colour,
behaviour), lets you rename, change speed, size and colour, make them
speak, send them to a place and **trace a route** by clicking on the ground; a
click on the figure selects it; the **timeline** shows the
keyframes. **Exact-frame GIF**: the engine has a fixed step and a
seed, so the frames are computed outside real time (not
recorded from the screen) and encoded on the server. Checked in
`scene_test.exs` (target, time, pronoun; the control: a sentence with no names
targets no one) and in headless Chromium (`console_desks.mjs`).

## 7. Save and export — everything (`Vapor.Archive`)

Every result in the console has **Save**: a zip with `manifest.json`
(kind, version, semantics, **recipe**, SHA-256 of each file) and the
files; the **identity** is the SHA-256 of the canonical manifest. **Trust →
Archives** checks an archive byte by byte and, if the kind is deterministic
(sorting networks, matrix multiplication, geometry, homology,
science, the direction of a scene — words → operations…), **recomputes the
recipe and compares**; the rest (a living scene saved whole, with its
layers) are checked byte by byte against the manifest — which catches
corruption and careless editing, not a coherent lie (the manifest is not
signed; signing archives with the operator key from 0.10 is on the
TODO). An archive is untrusted bytes: at most 512 entries and
256 MB uncompressed, counted **while** decompressing (a 1 MB zip
bomb that claims 300 MB is refused without being expanded — tested), and
every recipe parameter is bounded before running (a 32-wire sorting
network, 10⁶ self-play games: refused). An archive
names a *kind*, never a function: opening an archive runs nothing that
it chooses. Measured: the intact archive redoes `{:ok, :same}`; the same with
one byte of the result changed and re-zipped is refused as
`{:tampered, ["result.json"]}`; a coherent lie (manifest redone)
passes the integrity check and is caught by the recomputation.

The living scene also leaves as **a single HTML file** that plays offline (engine,
layers, script and seed embedded; ~130 kB for the guild hall) and
as **video** recorded from the canvas (WebM, the browser's own
encoder). The sketch leaves as SVG, DXF and GLB.

## 8. Limits

- **Heuristic** depth: excellent for scenes with a ground (landscapes,
  rooms, streets); a photo of a face, a ceiling, an aerial view have no ground
  — the layers exist, the order may be wrong (the sliders
  exist for that).
- No semantic segmentation: a person in the photo does not become an
  animatable inhabitant unless the user brings them in as a drawing; the inhabitants are
  engine figures, coloured with the scene's palette.
- 2.5D, not 3D: the camera moves within a small window (what lies behind is
  plausible, not true); free navigation "inside" the scene would require
  geometry that a single image does not have.
- A drawing's skeleton comes from the strokes: a filled drawing (a
  silhouette) gives a skeleton along the medial axis, not always the anatomical one.
- Video: the browser records in real time (WebM); since 0.12, exact-frame
  GIF (up to 240). Exact-frame MP4 remains on the TODO.
- Direction understands a grammar of clauses, not free language: "Ana
  e Bento dançam" directs the first name found; subordinate clauses are not
  parsed.

## 9. Free scenes (0.14): a document edited by operations

Scenes no longer depend on predetermined actions. A scene is a JSON document; it changes
through **text operations** (`Vapor.Scene.Ops`), which anyone can write — the person, a program
or a model:

```
add circle sol { x: 0.7, y: 0.22, r: 0.06, color: "#E8B04A" }
add particles chuva { count: 200, x: fract(u + 0.05*t), y: fract(0.3*t + u*7) }
set sol.r = 0.06 + 0.01*sin(2*t)
at 4: remove chuva
set world.weather = rain
```

Types: `circle`, `ring`, `rect`, `line`, `text`, `glow`, `particles`, `trail`; `at SECONDS:` schedules an operation and `set world.…` changes the weather, the time, the wind, the camera. Any field
accepts an expression from Alembic's numeric subset in `t` (time), `i`/`n`/`u` (index,
total and fraction in particles) and `aspect`; the expression becomes a tree interpreted in the browser
([ALEMBIC.md §3](ALEMBIC.md)) — never code. `direct "words"` asks the model for the operations and
they go through the same reader; whatever is not a valid operation comes back as a problem, with the line.

```
vapor scene new --w 960 --h 600 > s.json
vapor scene edit s.json "add glow sol { x: 0.7, y: 0.2 }" > s2.json
vapor scene direct s2.json "uma chuva fina caindo na diagonal" > s3.json   # with VAPOR_MIND
vapor scene export s3.json > cena.html                                    # self-contained HTML
```

