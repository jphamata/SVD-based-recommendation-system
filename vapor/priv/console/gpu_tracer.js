/* gpu-tracer:start */
/* The console's progressive path tracer (docs/RENDER.md): the scene format
   and the materials of Vapor.Render, on the viewer's GPU through WebGL2 —
   one sample per pixel per frame, accumulated in a float texture, so the
   picture converges while one watches. The Elixir tracer is the reference:
   test/js/gpu_tracer.mjs renders the same furnace in headless Chromium and
   compares the statistics. */
const GPUTracer = (() => {
  const MAX = 48;

  function parseScene(text) {
    const sc = { camera: { pos: [0, 1, 4], look: [0, 0.5, 0], fov: 45, aperture: 0, focus: 0 }, sky: { top: [0.5, 0.7, 1], bottom: [1, 1, 1] }, sun: null, objects: [], exposure: 1 };
    const errors = [];
    text.split("\n").forEach((raw, i) => {
      const l = raw.split("#")[0].trim();
      if (!l) return;
      const [kind, ...kvs] = l.split(/\s+/);
      const o = {};
      for (const kv of kvs) {
        const [k, v] = kv.split("=");
        if (v === undefined) { errors.push(`line ${i + 1}: expected key=value: ${kv}`); continue; }
        const nums = v.split(",").map(Number);
        o[k] = nums.every((x) => !Number.isNaN(x)) ? (nums.length === 1 ? nums[0] : nums) : v;
      }
      const v3 = (x, d) => (x === undefined ? d : typeof x === "number" ? [x, x, x] : x);
      const mat = () => ({ kind: o.mat || "diffuse", albedo: v3(o.albedo, [0.8, 0.8, 0.8]), rough: o.rough || 0, ior: o.ior || 1.5,
        emit: (o.mat === "emit") ? v3(o.color, [1, 1, 1]).map((c) => c * (o.power ?? 1)) : [0, 0, 0], checker: o.checker || 0 });
      switch (kind) {
        case "camera": Object.assign(sc.camera, { pos: v3(o.pos, sc.camera.pos), look: v3(o.look, sc.camera.look), fov: o.fov ?? sc.camera.fov, aperture: o.aperture || 0, focus: o.focus || 0 }); break;
        case "sky": sc.sky = { top: v3(o.top ?? o.color, sc.sky.top), bottom: v3(o.bottom ?? o.color, sc.sky.bottom) }; break;
        case "sun": { const d = v3(o.dir, [0.3, 1, 0.2]); const n = Math.hypot(...d); sc.sun = { dir: d.map((x) => x / n), color: v3(o.color, [1, 1, 1]).map((c) => c * (o.power ?? 2)), size: o.size ?? 0.03 }; break; }
        case "exposure": sc.exposure = o.value ?? 1; break;
        case "sphere": sc.objects.push({ type: 0, c: v3(o.c, [0, 0, 0]), r: o.r ?? 1, m: mat() }); break;
        case "plane": sc.objects.push({ type: 1, y: o.y ?? 0, m: mat() }); break;
        case "box": { const a = v3(o.min, [-0.5, -0.5, -0.5]), b = v3(o.max, [0.5, 0.5, 0.5]);
          sc.objects.push({ type: 2, min: a.map((x, k) => Math.min(x, b[k])), max: a.map((x, k) => Math.max(x, b[k])), m: mat() }); break; }
        default: errors.push(`line ${i + 1}: unknown statement ${kind}`);
      }
    });
    if (sc.objects.length > MAX) errors.push(`at most ${MAX} objects on the GPU`);
    return { scene: sc, errors };
  }

  const VS = `#version 300 es
in vec2 p; void main(){ gl_Position = vec4(p, 0.0, 1.0); }`;

  const FS = `#version 300 es
precision highp float;
precision highp int;
uniform vec2 uRes; uniform int uFrame; uniform int uSeed; uniform sampler2D uAccum;
uniform vec3 uCamPos; uniform vec3 uCamLook; uniform float uFov; uniform float uAperture; uniform float uFocus;
uniform vec3 uSkyTop; uniform vec3 uSkyBot; uniform vec3 uSunDir; uniform vec3 uSunCol; uniform float uSunSize; uniform int uHasSun;
uniform int uN; uniform int uMaxDepth;
uniform vec4 uA[${MAX}]; uniform vec4 uB[${MAX}]; uniform vec4 uM1[${MAX}]; uniform vec4 uM2[${MAX}]; uniform vec4 uM3[${MAX}];
out vec4 frag;
uint st;
uint pcg(){ st = st * 747796405u + 2891336453u; uint w = ((st >> ((st >> 28u) + 4u)) ^ st) * 277803737u; return (w >> 22u) ^ w; }
float rnd(){ return float(pcg()) * (1.0 / 4294967296.0); }
const float PI = 3.14159265358979;
bool isect(int i, vec3 o, vec3 d, out float t, out vec3 n){
  vec4 a = uA[i]; vec4 b = uB[i]; int ty = int(a.x);
  if (ty == 0) { vec3 oc = o - a.yzw; float bb = dot(oc, d); float q = dot(oc, oc) - b.x*b.x; float disc = bb*bb - q;
    if (disc < 0.0) return false; float s = sqrt(disc); t = -bb - s; if (t <= 1e-4) t = -bb + s; if (t <= 1e-4) return false;
    n = (o + d*t - a.yzw) / b.x; return true; }
  if (ty == 1) { if (abs(d.y) < 1e-12) return false; t = (a.y - o.y) / d.y; if (t <= 1e-4) return false; n = vec3(0.0, d.y < 0.0 ? 1.0 : -1.0, 0.0); return true; }
  vec3 mn = a.yzw, mx = b.xyz; vec3 inv = 1.0 / d; vec3 t0 = (mn - o) * inv, t1 = (mx - o) * inv;
  vec3 lo = min(t0, t1), hi = max(t0, t1); float tn = max(max(lo.x, lo.y), lo.z), tf = min(min(hi.x, hi.y), hi.z);
  if (tn > tf || tf < 1e-4) return false; t = tn > 1e-4 ? tn : tf; vec3 p = o + d*t;
  vec3 dm = abs(p - mn), dx = abs(p - mx); float m = 1e30; n = vec3(0.0);
  if (dm.x < m) { m = dm.x; n = vec3(-1,0,0); } if (dx.x < m) { m = dx.x; n = vec3(1,0,0); }
  if (dm.y < m) { m = dm.y; n = vec3(0,-1,0); } if (dx.y < m) { m = dx.y; n = vec3(0,1,0); }
  if (dm.z < m) { m = dm.z; n = vec3(0,0,-1); } if (dx.z < m) { m = dx.z; n = vec3(0,0,1); }
  return true;
}
bool hit(vec3 o, vec3 d, out float t, out vec3 n, out int id){
  t = 1e30; id = -1; for (int i = 0; i < ${MAX}; i++) { if (i >= uN) break; float ti; vec3 ni; if (isect(i, o, d, ti, ni) && ti < t) { t = ti; n = ni; id = i; } }
  return id >= 0;
}
vec3 sky(vec3 d){ float k = 0.5 * (d.y + 1.0); return mix(uSkyBot, uSkyTop, k); }
void onb(vec3 n, out vec3 t, out vec3 b){ vec3 a = abs(n.x) > 0.9 ? vec3(0,1,0) : vec3(1,0,0); t = normalize(cross(a, n)); b = cross(n, t); }
vec3 cosdir(vec3 n){ float u1 = rnd(), u2 = rnd(); float r = sqrt(u1), ph = 2.0*PI*u2; vec3 t, b; onb(n, t, b); return normalize(t*r*cos(ph) + b*r*sin(ph) + n*sqrt(max(0.0, 1.0-u1))); }
vec3 albedo(int i, vec3 p){ vec3 a = uM1[i].xyz; float s = uM3[i].y; if (s <= 0.0) return a; float c = floor(p.x/s) + floor(p.z/s); return mod(c, 2.0) == 0.0 ? a : a*0.35; }
vec3 trace(vec3 o, vec3 d){
  vec3 thr = vec3(1.0), acc = vec3(0.0); bool spec = true;
  for (int depth = 0; depth < 32; depth++) {
    if (depth >= uMaxDepth) break;
    float t; vec3 n; int id;
    if (!hit(o, d, t, n, id)) {
      vec3 sun = vec3(0.0);
      if (uHasSun == 1 && spec && dot(d, uSunDir) > cos(uSunSize)) sun = uSunCol / (2.0*PI*(1.0 - cos(uSunSize)));
      return acc + thr * (sky(d) + sun);
    }
    vec3 p = o + d*t; int kind = int(uM1[id].w);
    acc += thr * uM2[id].xyz;
    if (kind == 3) return acc;
    if (depth >= 3) { float q = clamp(dot(thr, vec3(0.2126, 0.7152, 0.0722)), 0.05, 0.95); if (rnd() >= q) return acc; thr /= q; }
    if (kind == 0) {
      vec3 nn = dot(n, d) > 0.0 ? -n : n; vec3 a = albedo(id, p);
      if (uHasSun == 1) { vec3 tt, bb; onb(uSunDir, tt, bb); float u1 = rnd(), u2 = rnd(); float r = sqrt(u1), ph = 2.0*PI*u2;
        vec3 l = normalize(uSunDir + (tt*r*cos(ph) + bb*r*sin(ph) + uSunDir*sqrt(max(0.0,1.0-u1))) * uSunSize);
        float c = dot(nn, l); float ts; vec3 ns; int is;
        if (c > 0.0 && !hit(p + nn*1e-4, l, ts, ns, is)) acc += thr * a * uSunCol * c / PI; }
      thr *= a; o = p + nn*1e-4; d = cosdir(nn); spec = false;
    } else if (kind == 1) {
      vec3 nn = dot(n, d) > 0.0 ? -n : n; vec3 r = reflect(d, nn); float rough = uM2[id].w;
      if (rough > 0.0) r = normalize(r + cosdir(nn) * rough);
      if (dot(r, nn) <= 0.0) return acc;
      thr *= uM1[id].xyz; o = p + nn*1e-4; d = r; spec = rough < 0.2;
    } else {
      float ior = uM3[id].x; vec3 nn; float eta, cosi;
      if (dot(d, n) < 0.0) { nn = n; eta = 1.0/ior; cosi = -dot(d, n); } else { nn = -n; eta = ior; cosi = dot(d, n); }
      float k2 = 1.0 - eta*eta*(1.0 - cosi*cosi); float f0 = pow((1.0-ior)/(1.0+ior), 2.0); float fr = f0 + (1.0-f0)*pow(1.0-cosi, 5.0);
      if (k2 < 0.0 || rnd() < fr) { o = p + nn*1e-4; d = reflect(d, nn); }
      else { d = normalize(d*eta + nn*(eta*cosi - sqrt(k2))); o = p - nn*1e-4; thr *= uM1[id].xyz; }
      spec = true;
    }
  }
  return acc;
}
void main(){
  ivec2 px = ivec2(gl_FragCoord.xy);
  st = uint(px.x) * 1973u + uint(px.y) * 9277u + uint(uFrame) * 26699u + uint(uSeed) * 104729u; pcg();
  vec3 fwd = normalize(uCamLook - uCamPos); vec3 right = normalize(cross(fwd, vec3(0,1,0))); vec3 up = cross(right, fwd);
  float th = tan(uFov * PI / 360.0); float asp = uRes.x / uRes.y;
  vec2 j = vec2(rnd(), rnd());
  float u = ((gl_FragCoord.x - 0.5 + j.x) / uRes.x * 2.0 - 1.0) * th * asp;
  float v = ((gl_FragCoord.y - 0.5 + j.y) / uRes.y * 2.0 - 1.0) * th;
  vec3 d = normalize(fwd + right*u + up*v); vec3 o = uCamPos;
  if (uAperture > 0.0) { float f = uFocus > 0.0 ? uFocus : length(uCamLook - uCamPos); vec3 tg = o + d*f;
    float r = uAperture * sqrt(rnd()), a = 2.0*PI*rnd(); o = o + right*r*cos(a) + up*r*sin(a); d = normalize(tg - o); }
  vec3 c = trace(o, d);
  vec4 prev = texelFetch(uAccum, px, 0);
  frag = vec4((prev.rgb * float(uFrame) + c) / float(uFrame + 1), 1.0);
}`;

  const SHOW = `#version 300 es
precision highp float; uniform sampler2D uAccum; uniform float uExposure; out vec4 frag;
vec3 aces(vec3 x){ return clamp(x*(2.51*x+0.03)/(x*(2.43*x+0.59)+0.14), 0.0, 1.0); }
vec3 srgb(vec3 c){ return mix(12.92*c, 1.055*pow(c, vec3(1.0/2.4)) - 0.055, step(0.0031308, c)); }
void main(){ vec3 c = texelFetch(uAccum, ivec2(gl_FragCoord.xy), 0).rgb; frag = vec4(srgb(aces(c * uExposure)), 1.0); }`;

  function program(gl, vs, fs) {
    const mk = (type, src) => { const s = gl.createShader(type); gl.shaderSource(s, src); gl.compileShader(s);
      if (!gl.getShaderParameter(s, gl.COMPILE_STATUS)) throw new Error(gl.getShaderInfoLog(s)); return s; };
    const p = gl.createProgram(); gl.attachShader(p, mk(gl.VERTEX_SHADER, vs)); gl.attachShader(p, mk(gl.FRAGMENT_SHADER, fs)); gl.linkProgram(p);
    if (!gl.getProgramParameter(p, gl.LINK_STATUS)) throw new Error(gl.getProgramInfoLog(p));
    return p;
  }

  function create(canvas, opts = {}) {
    const gl = canvas.getContext("webgl2", { antialias: false, preserveDrawingBuffer: true });
    if (!gl) throw new Error("WebGL2 is not available in this browser");
    if (!gl.getExtension("EXT_color_buffer_float")) throw new Error("float render targets (EXT_color_buffer_float) are not available");
    const trace = program(gl, VS, FS), show = program(gl, VS, SHOW);
    const buf = gl.createBuffer(); gl.bindBuffer(gl.ARRAY_BUFFER, buf);
    gl.bufferData(gl.ARRAY_BUFFER, new Float32Array([-1, -1, 1, -1, -1, 1, 1, 1]), gl.STATIC_DRAW);
    let W = 0, H = 0, tex = [], fbo = [], ping = 0, frame = 0, scene = null, seed = opts.seed || 1, raf = 0, running = false, maxDepth = opts.maxDepth || 12;
    function targets(w, h) {
      W = w; H = h; tex.forEach((t) => gl.deleteTexture(t)); fbo.forEach((f) => gl.deleteFramebuffer(f)); tex = []; fbo = [];
      for (let i = 0; i < 2; i++) {
        const t = gl.createTexture(); gl.bindTexture(gl.TEXTURE_2D, t);
        gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA32F, w, h, 0, gl.RGBA, gl.FLOAT, null);
        gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.NEAREST); gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.NEAREST);
        const f = gl.createFramebuffer(); gl.bindFramebuffer(gl.FRAMEBUFFER, f); gl.framebufferTexture2D(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT0, gl.TEXTURE_2D, t, 0);
        tex.push(t); fbo.push(f);
      }
      frame = 0;
    }
    function quad(p) { const loc = gl.getAttribLocation(p, "p"); gl.bindBuffer(gl.ARRAY_BUFFER, buf); gl.enableVertexAttribArray(loc); gl.vertexAttribPointer(loc, 2, gl.FLOAT, false, 0, 0); gl.drawArrays(gl.TRIANGLE_STRIP, 0, 4); }
    function upload() {
      gl.useProgram(trace);
      const u = (n) => gl.getUniformLocation(trace, n);
      const c = scene.camera;
      gl.uniform3fv(u("uCamPos"), c.pos); gl.uniform3fv(u("uCamLook"), c.look); gl.uniform1f(u("uFov"), c.fov); gl.uniform1f(u("uAperture"), c.aperture); gl.uniform1f(u("uFocus"), c.focus);
      gl.uniform3fv(u("uSkyTop"), scene.sky.top); gl.uniform3fv(u("uSkyBot"), scene.sky.bottom);
      gl.uniform1i(u("uHasSun"), scene.sun ? 1 : 0);
      if (scene.sun) { gl.uniform3fv(u("uSunDir"), scene.sun.dir); gl.uniform3fv(u("uSunCol"), scene.sun.color); gl.uniform1f(u("uSunSize"), scene.sun.size); }
      const A = new Float32Array(MAX * 4), B = new Float32Array(MAX * 4), M1 = new Float32Array(MAX * 4), M2 = new Float32Array(MAX * 4), M3 = new Float32Array(MAX * 4);
      const kinds = { diffuse: 0, metal: 1, glass: 2, emit: 3 };
      scene.objects.slice(0, MAX).forEach((o, i) => {
        if (o.type === 0) { A.set([0, ...o.c], i * 4); B.set([o.r, 0, 0, 0], i * 4); }
        else if (o.type === 1) { A.set([1, o.y, 0, 0], i * 4); }
        else { A.set([2, ...o.min], i * 4); B.set([...o.max, 0], i * 4); }
        M1.set([...o.m.albedo, kinds[o.m.kind] ?? 0], i * 4); M2.set([...o.m.emit, o.m.rough], i * 4); M3.set([o.m.ior, o.m.checker, 0, 0], i * 4);
      });
      gl.uniform4fv(u("uA"), A); gl.uniform4fv(u("uB"), B); gl.uniform4fv(u("uM1"), M1); gl.uniform4fv(u("uM2"), M2); gl.uniform4fv(u("uM3"), M3);
      gl.uniform1i(u("uN"), Math.min(scene.objects.length, MAX)); gl.uniform1i(u("uMaxDepth"), maxDepth);
    }
    function step() {
      gl.useProgram(trace);
      gl.bindFramebuffer(gl.FRAMEBUFFER, fbo[1 - ping]); gl.viewport(0, 0, W, H);
      gl.activeTexture(gl.TEXTURE0); gl.bindTexture(gl.TEXTURE_2D, tex[ping]);
      gl.uniform1i(gl.getUniformLocation(trace, "uAccum"), 0); gl.uniform2f(gl.getUniformLocation(trace, "uRes"), W, H);
      gl.uniform1i(gl.getUniformLocation(trace, "uFrame"), frame); gl.uniform1i(gl.getUniformLocation(trace, "uSeed"), seed);
      quad(trace); ping = 1 - ping; frame++;
    }
    function present() {
      gl.useProgram(show); gl.bindFramebuffer(gl.FRAMEBUFFER, null); gl.viewport(0, 0, canvas.width, canvas.height);
      gl.activeTexture(gl.TEXTURE0); gl.bindTexture(gl.TEXTURE_2D, tex[ping]);
      gl.uniform1i(gl.getUniformLocation(show, "uAccum"), 0); gl.uniform1f(gl.getUniformLocation(show, "uExposure"), scene ? scene.exposure : 1);
      quad(show);
    }
    const api = {
      setScene(sc) { scene = sc; if (canvas.width !== W || canvas.height !== H || !tex.length) targets(canvas.width, canvas.height); frame = 0; upload(); },
      reset() { frame = 0; },
      render(n = 1) { for (let i = 0; i < n; i++) step(); present(); return frame; },
      play(onFrame) { if (running) return; running = true; const loop = () => { if (!running) return; api.render(1); onFrame && onFrame(frame); raf = requestAnimationFrame(loop); }; loop(); },
      pause() { running = false; cancelAnimationFrame(raf); },
      get frames() { return frame; },
      get playing() { return running; },
      readLinear() { gl.bindFramebuffer(gl.FRAMEBUFFER, fbo[ping]); const out = new Float32Array(W * H * 4); gl.readPixels(0, 0, W, H, gl.RGBA, gl.FLOAT, out); return { w: W, h: H, data: out }; },
      png() { return canvas.toDataURL("image/png"); },
      destroy() { api.pause(); tex.forEach((t) => gl.deleteTexture(t)); fbo.forEach((f) => gl.deleteFramebuffer(f)); },
    };
    return api;
  }
  return { parseScene, create, MAX };
})();
/* gpu-tracer:end */
