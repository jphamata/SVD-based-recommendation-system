# Sketch → drawing, floor plan → 3D, and archives for everything

> Since 0.11 (until 0.17 in docs/SCENE.md, with the living scene, which was removed:
> [DIRECTIVE.md §21](DIRECTIVE.md)). Code: `lib/vapor/sketch.ex`, `lib/vapor/raster.ex`
> (fitting, thinning, skeletons, simplification), `lib/vapor/archive.ex`. Tests:
> `sketch_test.exs`, `archive_test.exs`, `console_test.exs`. Console: *Make → Sketch*,
> *Trust → Archives*.

## 1. Sketch → technical drawing (`Vapor.Sketch.vectorize/2`)

Dark strokes on light paper are thinned (Zhang & Suen 1984, `Vapor.Raster.thin/1`) to a
one-pixel skeleton, whose chains are simplified (Ramer–Douglas–Peucker) and fitted:

- **a line** by total least squares, **a circle or arc** by Kåsa's algebraic fit, or the
  chain broken into lines when neither fits;
- then **beautified**: orientations snapped to 0°, 45°, 90° when within 4°; near-parallels
  made parallel; collinear horizontals and verticals aligned; corners welded at the
  least-squares point of their lines.

The constraints found are listed: what the drawing *meant to say*, declared. Outputs:
**SVG** (`svg/1`) and **DXF R12** (`dxf/2`).

Measured (`priv/quality/sketch`, drawn by script with hand tremor): a rectangle drawn
**2.2° askew** comes back straight and closed, and the control (the same fit without
constraints, `snap: false`) stays askew; the circle comes back with centre and radius
within 1 px; the hypotenuse at nearly 45° goes to 45°; a free line at 29.5° stays where it is.

## 2. Floor plan → 3D (`Vapor.Sketch.plan/2`)

The lines are walls; a gap between collinear walls is a **door** (its width corrected for
the stroke thickness, which thinning shortens by half at each end); the **rooms** are the
faces of the planar graph of the walls (doors closed), found by walking the half-edges;
the scale comes from the longest wall (8 m by default) or from `scale:`. The walls are
extruded (2.7 m high, 15 cm thick, lintels over the doors at 2.1 m) into a
`Vapor.Geom.Mesh` exported as **glTF (GLB)**; the console has its own 3D viewer.

Measured: rooms of **11.98 and 19.80 m²** (truth 12 and 20), doors of **0.90 and 1.00 m**
(truth 0.9 and 1.0), the GLB opened by trimesh with 2.7 m height and the plan as its base.

Sketch → picture is not done here: the scene language of [RENDER.md](RENDER.md) draws what
is written (physically, or in ink), and vapor's diffusion pipeline takes a sketch as the
initial image for img2img with a checkpoint the user brings. None ships with vapor, and
none of that was measured here.

## 3. Save and export — everything (`Vapor.Archive`)

Every result in the console has **Save**: a zip with `manifest.json` (kind, version,
semantics, **recipe**, the SHA-256 of each file) and the files; the archive's **identity**
is the SHA-256 of the canonical manifest. **Trust → Archives** checks an archive byte by
byte and, when its kind is deterministic (geometry, homology, science, the finance desk's
tasks: `Vapor.Archive.replayable/0`), **recomputes the recipe and compares**.

- Integrity catches corruption and careless editing, not a coherent lie (whoever rewrites
  the result can rewrite the manifest). For that there are **signatures** (0.13):
  `sign/2` adds Ed25519 over the manifest with the operator key of `mix vapor.audit
  keygen`, and `verify(zip, trusted: keys)` refuses an archive unsigned, signed by an
  unknown key, or whose signature does not hold.
- An archive is untrusted bytes: at most 512 entries and 256 MB uncompressed, counted
  **while** inflating (a 1 MB zip bomb that claims 300 MB is refused without being
  expanded), and every recipe parameter is bounded before running.
- An archive names a *kind*, never a function: opening one runs nothing it chose.

Measured: the intact archive replays to `{:ok, :same}`; the same with one byte of the
result changed and re-zipped is refused as `{:tampered, ["result.json"]}`; a coherent lie
(manifest redone) passes integrity and is caught by the recomputation. The sketch leaves
as SVG, DXF and GLB.
