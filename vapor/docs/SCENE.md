# The living scene — removed in 0.17

From 0.11 to 0.16 vapor had a "living scene": a picture split into 2.5D layers by
heuristic depth, inhabitants walking on A\*, weather, fireflies, drawings animated by
skeleton, directed by sentences and, from 0.14, by text operations. In 0.17 it was asked
to aim at fine control, abstraction, photorealism or stylisation — or to go. It went.
The reasons, the measurements and the alternatives weighed are in
[DIRECTIVE.md §21](DIRECTIVE.md); in short:

- it was a toy in the sense that matters here: nothing in it could be checked against
  anything but itself (a heuristic depth "stated as such", a grammar of clauses, a 2D
  canvas engine), in a project whose every other result carries its evidence;
- its two serious ingredients stay: **sketch → drawing and floor plan → 3D**, with the
  archives, are in [SKETCH.md](SKETCH.md); the raster tools it shared (fitting,
  thinning, skeletons) are `Vapor.Raster`;
- what replaces it is the renderer that was already checked: [RENDER.md](RENDER.md),
  physically based light (furnaces, N^−½, GPU against the reference), and since 0.17 the
  **same scene stylised in ink** (`Vapor.Render.ink/2`): flat bands, hard shadows,
  outlines on silhouettes and folds, deterministic — fine control is the scene text itself.

Removed with it: the console's *Living scene* tab and the HTML export, the `vapor scene`
verb (now `vapor render`), the MCP tool `scene_ops`, the mind layer's `direct/3`, the archive
kind `scene`, and the portable numeric tree that ran a scene's motion in the browser.
