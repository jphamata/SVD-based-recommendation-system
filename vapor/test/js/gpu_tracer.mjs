// The console's GPU path tracer (priv/console/gpu_tracer.js), run in headless
// Chromium (WebGL2 on SwiftShader): the gradient furnace — a Lambertian sphere
// of albedo a under the sky L(ω) = (1 + ω_y)/2 must show a·(1/2 + n_y/3) —
// and a scene's mean radiance, for the Elixir reference to compare.
// usage: node gpu_tracer.mjs TRACER_JS SCENE_FILE FRAMES  → JSON {furnace, scene_mean, w, h}
import { readFileSync } from "node:fs";
import { createRequire } from "node:module";
const require = createRequire(import.meta.url);
let playwright;
try { playwright = require("playwright"); } catch { playwright = require("/opt/node22/lib/node_modules/playwright"); }
const [tracerPath, scenePath, framesArg] = process.argv.slice(2);
const tracer = readFileSync(tracerPath, "utf8");
const sceneText = readFileSync(scenePath, "utf8");
const frames = Number(framesArg || 256);

const browser = await playwright.chromium.launch({ args: ["--use-angle=swiftshader", "--enable-unsafe-swiftshader", "--ignore-gpu-blocklist"] });
const page = await browser.newPage();
await page.setContent("<canvas id=c width=32 height=32></canvas><canvas id=d width=48 height=30></canvas>");
const out = await page.evaluate(({ tracer, sceneText, frames }) => {
  const GPUTracer = new Function(tracer + "\nreturn GPUTracer;")();
  // the gradient furnace
  const f = GPUTracer.parseScene("camera pos=0,0,3 look=0,0,0 fov=40\nsky top=1,1,1 bottom=0,0,0\nsphere c=0,0,0 r=1 mat=diffuse albedo=0.8,0.8,0.8");
  const t = GPUTracer.create(document.getElementById("c"));
  t.setScene(f.scene); t.render(frames);
  const lin = t.readLinear();
  const th = Math.tan(40 * Math.PI / 360);
  let errs = [];
  for (let y = 0; y < 32; y++) for (let x = 0; x < 32; x++) {
    // readPixels rows go bottom-up, like gl_FragCoord
    const u = ((x + 0.5) / 32 * 2 - 1) * th, v = ((y + 0.5) / 32 * 2 - 1) * th;
    const n = Math.hypot(u, v, 1); const d = [u / n, v / n, -1 / n];
    const b = 3 * d[2]; const disc = b * b - 8;
    if (disc <= 0.3) continue;
    const tt = -b - Math.sqrt(disc); const ny = d[1] * tt;
    if (ny <= 0.5) continue;
    errs.push(lin.data[(y * 32 + x) * 4] - 0.8 * (0.5 + ny / 3));
  }
  // a scene's mean linear radiance
  const s = GPUTracer.parseScene(sceneText);
  const g = GPUTracer.create(document.getElementById("d"));
  g.setScene(s.scene); g.render(frames);
  const L = g.readLinear();
  let sum = 0; for (let i = 0; i < L.w * L.h; i++) sum += 0.2126 * L.data[i * 4] + 0.7152 * L.data[i * 4 + 1] + 0.0722 * L.data[i * 4 + 2];
  return { furnace: { mean_error: errs.reduce((a, b) => a + b, 0) / errs.length, pixels: errs.length }, scene_mean: sum / (L.w * L.h), w: L.w, h: L.h, errors: s.errors };
}, { tracer, sceneText, frames });
await browser.close();
console.log(JSON.stringify(out));
