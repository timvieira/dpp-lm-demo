// 3D view: quality-scaled embeddings as points, the selected set as arrows from the
// origin, and (for K ≤ 3) the parallelepiped whose squared volume the DPP scores.
import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';

// The x, y, z axes through the origin, from −1 to 1 (the length of every embedding). Coloured per
// vertex so the near end is darker than the far end (see shadeDepth).
function axesLines(scale) {
  // Drawn first and without depth writes, so points, vectors and the volume always sit on top.
  const g = new THREE.Group(), mat = new THREE.LineBasicMaterial({ vertexColors: true, depthWrite: false });
  const line = (pts, ink) => {
    const geo = new THREE.BufferGeometry().setFromPoints(pts);
    geo.setAttribute('color', new THREE.BufferAttribute(new Float32Array(pts.length * 3), 3));
    const l = new THREE.Line(geo, mat);
    l.renderOrder = -1; l.userData.ink = ink;   // how dark the line is, near side (shadeDepth)
    return l;
  };
  for (const d of [[1, 0, 0], [0, 1, 0], [0, 0, 1]]) {
    const pts = []; for (let t = -1; t <= 1.0001; t += 0.05) pts.push(new THREE.Vector3(d[0] * t, d[1] * t, d[2] * t));
    g.add(line(pts, 0.6));
  }
  g.scale.setScalar(scale);
  return g;
}

// The floor: a grid in the x–z plane through the origin, cells of 1/8, that fades out with distance
// from the origin (gone by 6 units), so it reads as endless and no edge ever comes into view.
function endlessFloor(scale, bg, ink) {
  const R = 6, step = 1 / 8, seg = 1 / 16, pos = [], col = [], c = new THREE.Color();
  const shade = (x, z) => { const t = Math.max(0, 1 - Math.hypot(x, z) / R); c.copy(bg).lerp(ink, 0.3 * t); return [c.r, c.g, c.b]; };
  for (let k = -R; k <= R + 1e-9; k += step) {
    for (let u = -R; u < R - 1e-9; u += seg) {
      for (const [x0, z0, x1, z1] of [[k, u, k, u + seg], [u, k, u + seg, k]]) {
        pos.push(x0, 0, z0, x1, 0, z1);
        col.push(...shade(x0, z0), ...shade(x1, z1));
      }
    }
  }
  const geo = new THREE.BufferGeometry();
  geo.setAttribute('position', new THREE.Float32BufferAttribute(pos, 3));
  geo.setAttribute('color', new THREE.Float32BufferAttribute(col, 3));
  const floor = new THREE.LineSegments(geo, new THREE.LineBasicMaterial({ vertexColors: true, depthWrite: false }));
  floor.renderOrder = -2;
  floor.scale.setScalar(scale);
  return floor;
}

export class Plot3D {
  constructor(el, { onHover, onTap, safeRect, onFrame }) {
    this.onFrame = onFrame;
    this.el = el;
    // The open part of the plot, in canvas pixels, not covered by panels floating over it.
    this.safeRect = safeRect ?? (() => ({ L: 8, T: 8, R: (el.clientWidth || 600) - 8, B: (el.clientHeight || 450) - 8 }));
    this.onHover = onHover;
    this.onTap = onTap;
    const W = el.clientWidth || 600, H = el.clientHeight || 450;
    this.renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true });
    this.renderer.setPixelRatio(Math.min(2, window.devicePixelRatio || 1));
    this.renderer.setSize(W, H, false);
    el.appendChild(this.renderer.domElement);
    this.scene = new THREE.Scene();
    // Orthographic: no perspective, so every point is drawn the same size at any depth and
    // lengths (quality) are drawn to one scale. The view holds the unit ball on any aspect.
    const a = W / H, h = Math.max(0.78, 0.9 / a);
    this.camera = new THREE.OrthographicCamera(-h * a, h * a, h, -h, -50, 50);
    this.camera.position.set(1.6, 1.2, 2.2);
    this.controls = new OrbitControls(this.camera, this.renderer.domElement);
    this.controls.enableDamping = true;
    // No continuous rotation: one 30° turn when the view first appears (none if the visitor asks for
    // reduced motion), stopped by any touch. Panning slides the picture on screen, so the view still
    // turns about the origin: Shift-drag or right-drag with a mouse, two fingers on a touch screen.
    // Double-click returns to the starting view.
    this.controls.enablePan = false;
    this.turnLeft = 0;
    this.view = { cx: 0, cy: 0, h: 1 };
    // Where the origin sits on screen, as a fraction of the open area (0..1 across, 0..1 down). Fixed
    // until the visitor moves the picture, so the view visibly turns about the origin.
    this.anchor = { ax: 0.5, ay: 0.5 };
    this.home = { ax: 0.5, ay: 0.5 };
    this.installPan();
    this.group = new THREE.Group();
    this.scene.add(this.group);
    this.scene.add(new THREE.AmbientLight(0xffffff, 0.8));
    const dl = new THREE.DirectionalLight(0xffffff, 0.8);
    dl.position.set(2, 3, 4);
    this.scene.add(dl);
    this.ray = new THREE.Raycaster();
    this.mouse = new THREE.Vector2();
    this.hovered = -1;
    this.targets = null;
    this.current = null;
    // Hover tooltips for mice; a tap (pointer down and up without dragging) inspects
    // the nearest point on screen within a finger-sized radius.
    const cv = this.renderer.domElement;
    cv.addEventListener('pointermove', (e) => { if (e.pointerType === 'mouse') this.pick(e); });
    cv.addEventListener('pointerleave', () => { this.hovered = -1; this.onHover(-1); });
    let down = null;
    cv.addEventListener('pointerdown', (e) => (down = [e.clientX, e.clientY]));
    cv.addEventListener('pointerup', (e) => {
      if (!down || Math.hypot(e.clientX - down[0], e.clientY - down[1]) > 6) return;
      this.onTap?.(this.nearest(e.clientX, e.clientY, e.pointerType === 'mouse' ? 16 : 32));
    });
    this.alive = true;
    // When the plot area changes size (window resize, the docked controls measuring themselves),
    // resize the canvas and re-frame, so the drawing is never stretched or cut off.
    this.ro = new ResizeObserver(() => {
      const W = el.clientWidth, H = el.clientHeight;
      if (!W || !H) return;
      this.renderer.setSize(W, H, false);
      if (this.o && this.camSet) this.setView(this.view);   // new aspect; the fit follows in tick()
    });
    this.ro.observe(el);
    const loop = () => {
      if (!this.alive) return;
      requestAnimationFrame(loop);
      this.tick();
      this.controls.update();
      this.shadeDepth();
      this.renderer.render(this.scene, this.camera);
      this.onFrame?.();
    };
    loop();
  }

  update(o) {
    this.o = o;
    const N = o.coords.length;
    // The scene's scale, set once when the points are first drawn and then left alone: later changes of
    // the selection, the mode, T or K never rescale it (the visitor's view is theirs).
    if (this.scale == null) {
      let m = 0;
      for (const c of o.coords) m = Math.max(m, Math.hypot(...c));
      this.scale = 1 / (m || 1);
    }
    const s = this.scale;
    // Placed and framed once, when the points are first drawn: looking at the origin, with the drawing
    // centred in the open area and filling it at the angle where the opening turn ends. After that, only
    // the visitor moves the view.
    if (!this.camSet) {
      this.camSet = true;
      this.startView();
      this.turnLeft = matchMedia('(prefers-reduced-motion: reduce)').matches ? 0 : (30 * Math.PI) / 180;
      // Place the origin for the angle where the opening turn ends, then start the turn from before it.
      const Y = new THREE.Vector3(0, 1, 0), upd = () => { this.camera.lookAt(0, 0, 0); this.camera.updateMatrixWorld(); };
      this.camera.position.applyAxisAngle(Y, this.turnLeft); upd();
      this.placeOrigin(o, s);
      this.fitOnce();   // also sets the size dots and rods are drawn at for this frame
      this.initial = { view: { ...this.view }, pos: this.camera.position.clone() };
      this.camera.position.applyAxisAngle(Y, -this.turnLeft); upd();
    }
    this.targets = o.coords.map((c, i) => {
      const f = o.c[i] * s;
      return new THREE.Vector3(c[0] * f, c[1] * f, c[2] * f);
    });
    if (!this.current || this.current.length !== N) this.current = this.targets.map((v) => v.clone());
    this.rebuildStatic();
  }

  rebuildStatic() {
    const o = this.o;
    const N = o.coords.length;
    for (const ch of [...this.group.children]) { this.group.remove(ch); ch.geometry?.dispose(); ch.material?.dispose(); }
    // The x, y, z axes through the origin, which the view turns about, with arrowheads at their positive
    // ends; and a faint floor grid in the x–z plane through the origin (as in the paper's Fig. 1).
    this.globe = axesLines(this.scale);
    this.globeInk = new THREE.Color(o.text ?? '#555555');
    this.group.add(this.globe);
    const z = this.frameH ?? this.camera.top;
    this.heads = [];
    const headMat = new THREE.MeshBasicMaterial({ color: new THREE.Color(o.bg ?? '#fbfaf8').lerp(this.globeInk, 0.55) });
    for (const d of [[1, 0, 0], [0, 1, 0], [0, 0, 1]]) {
      const dir = new THREE.Vector3(...d), head = new THREE.Mesh(new THREE.ConeGeometry(0.014 * z, 0.045 * z, 14), headMat);
      head.quaternion.setFromUnitVectors(new THREE.Vector3(0, 1, 0), dir);
      head.position.copy(dir).multiplyScalar(this.scale);   // the tip sits just past the end of the axis
      head.position.addScaledVector(dir, 0.0225 * z);
      this.heads.push(head);
      this.group.add(head);
    }
    this.group.add(endlessFloor(this.scale, new THREE.Color(o.bg ?? '#fbfaf8'), new THREE.Color(o.text ?? '#555555')));
    // Points.
    // Radius tied to the zoom: about 1.5 px on a 600-px-tall plot, the same as the 2-D dots.
    const sphere = new THREE.SphereGeometry(0.0055 * (this.frameH ?? this.camera.top), 10, 8);
    this.mesh = new THREE.InstancedMesh(sphere, new THREE.MeshLambertMaterial(), N);
    const sel = new Set(o.sel);
    // Base colours; shadeDepth() fades each toward the background by its distance from the viewer.
    this.base = Array.from({ length: N }, (_, i) => new THREE.Color(sel.has(i) ? o.selColor(i) : o.color(i)));
    this.isSel = Array.from({ length: N }, (_, i) => sel.has(i));
    this.bgCol = new THREE.Color(o.bg ?? '#fbfaf8');
    for (let i = 0; i < N; i++) this.mesh.setColorAt(i, this.base[i]);
    this.group.add(this.mesh);
    // Arrows + parallelepiped edges are rebuilt every tick from current positions.
    this.dyn = new THREE.Group();
    this.group.add(this.dyn);
    this.layout();
  }

  layout() {
    if (!this.mesh) return;
    const o = this.o;
    const m = new THREE.Matrix4();
    const sel = new Set(o.sel);
    const sc = this.frameH ? this.view.h / this.frameH : 1;   // constant size on screen as the view zooms
    this.laidH = this.view.h;
    for (const hd of this.heads ?? []) hd.scale.setScalar(sc);
    this.current.forEach((p, i) => {
      m.makeScale(sc, sc, sc).setPosition(p);   // all points the same size; colour marks the selected ones
      this.mesh.setMatrixAt(i, m);
    });
    this.mesh.instanceMatrix.needsUpdate = true;
    for (const ch of [...this.dyn.children]) { this.dyn.remove(ch); ch.geometry?.dispose(); ch.material?.dispose?.(); }
    const few = o.sel.length <= 3;
    const O = new THREE.Vector3();
    // An arrow from the origin to each selected point, whatever K (one colour per item).
    for (const i of o.sel) {
      const p = this.current[i];
      const len = p.length();
      if (len < 1e-4) continue;
      const color = new THREE.Color(o.selColor(i));
      const z = this.view?.h ?? this.frameH ?? 1;   // widths follow the zoom, so they look the same at any frame
      this.dyn.add(new THREE.ArrowHelper(p.clone().normalize(), O, len, color, Math.min(0.06 * z, len * 0.3), Math.min(0.03 * z, len * 0.15)));
    }
    if (o.sel.length >= 2 && few) {   // the box the selected vectors span, whenever there are 2 or 3 of them
      const v = o.sel.map((i) => this.current[i]);
      const n = v.length;
      const corner = (mask) => v.reduce((c, x, k) => (mask & (1 << k) ? c.add(x) : c), new THREE.Vector3());
      const grey = new THREE.Color(o.text);
      // Translucent faces.
      const tri = [];
      for (let a = 0; a < n; a++) for (let b = a + 1; b < n; b++) {
        const others = [...Array(n).keys()].filter((k) => k !== a && k !== b);
        for (const off of others.length ? [0, 1 << others[0]] : [0]) {
          const q = [off, off | (1 << a), off | (1 << a) | (1 << b), off | (1 << b)].map(corner);
          tri.push(q[0], q[1], q[2], q[0], q[2], q[3]);
        }
      }
      const fg = new THREE.BufferGeometry().setFromPoints(tri);
      this.dyn.add(new THREE.Mesh(fg, new THREE.MeshBasicMaterial({ color: grey, transparent: true, opacity: 0.08, side: THREE.DoubleSide, depthWrite: false })));
      // Dark edges, except the ones from the origin (drawn above in colour).
      const seg = [];
      for (let a = 0; a < 1 << n; a++) for (let k = 0; k < n; k++) {
        const b = a | (1 << k);
        if (b !== a && a !== 0) seg.push(corner(a), corner(b));
      }
      this.dyn.add(new THREE.LineSegments(new THREE.BufferGeometry().setFromPoints(seg), new THREE.LineBasicMaterial({ color: grey, transparent: true, opacity: 0.75 })));
    }
  }

  // Depth shading: a point's colour moves toward the background the farther it is from the viewer
  // (up to 60% for grey points, 35% for the selected ones), so the cloud's depth reads as it turns.
  shadeDepth() {
    if (!this.mesh || !this.current || !this.base) return;
    const fwd = new THREE.Vector3(); this.camera.getWorldDirection(fwd);
    const ds = this.current.map((p) => p.dot(fwd));
    let lo = Infinity, hi = -Infinity; for (const d of ds) { if (d < lo) lo = d; if (d > hi) hi = d; }
    const span = Math.max(hi - lo, 1e-6), c = new THREE.Color();
    ds.forEach((d, i) => {
      const t = (d - lo) / span;   // 0 nearest, 1 farthest
      c.copy(this.base[i]).lerp(this.bgCol, (this.isSel[i] ? 0.35 : 0.6) * t);
      this.mesh.setColorAt(i, c);
    });
    this.mesh.instanceColor.needsUpdate = true;
    // The axes: near end `ink` of the way from the background to the ink colour, far side 5%.
    if (this.globe) {
      for (const line of this.globe.children) {
        const pos = line.geometry.attributes.position, col = line.geometry.attributes.color;
        for (let k = 0; k < pos.count; k++) {
          const t = (1 - (pos.getX(k) * fwd.x + pos.getY(k) * fwd.y + pos.getZ(k) * fwd.z)) / 2;   // 1 nearest
          c.copy(this.bgCol).lerp(this.globeInk, 0.05 + (line.userData.ink - 0.05) * t);
          col.setXYZ(k, c.r, c.g, c.b);
        }
        col.needsUpdate = true;
      }
    }
  }

  // Look down from 35°: a vector lying near the floor is foreshortened at most to sin 35° ≈ 0.57 of its
  // length as the view turns, so no selected vector can collapse onto the origin.
  startView() {
    const el = (35 * Math.PI) / 180, az = (40 * Math.PI) / 180;
    this.controls.target.set(0, 0, 0);
    this.camera.position.set(3 * Math.cos(el) * Math.sin(az), 3 * Math.sin(el), 3 * Math.cos(el) * Math.cos(az));
    this.camera.zoom = 1;
    this.camera.lookAt(0, 0, 0);
    this.camera.updateMatrixWorld();
  }
  // The rectangle on screen that holds everything drawn (the points and, for K ≤ 3, every corner of the
  // spanned volume) as the view turns by up to `turn` about the vertical axis from where it is now.
  frameFor(o, s, turn) {
    const pts = o.coords.map((c, i) => c.map((x) => x * o.c[i] * s));
    pts.push([0, 0, 0]);
    const sel = o.sel.length >= 2 && o.sel.length <= 3 ? o.sel : [];
    for (let mask = 3; mask < 1 << sel.length; mask++) {
      if ((mask & (mask - 1)) === 0) continue;   // single vectors are points already
      pts.push([0, 1, 2].map((d) => sel.reduce((a, i, k) => a + (mask & (1 << k) ? pts[i][d] : 0), 0)));
    }
    let x0 = Infinity, x1 = -Infinity, y0 = Infinity, y1 = -Infinity;
    const Y = new THREE.Vector3(0, 1, 0), steps = turn > 0 ? 12 : 0;
    for (let k = 0; k <= steps; k++) {
      const q = new THREE.Quaternion().setFromAxisAngle(Y, steps ? (turn * k) / steps : 0).multiply(this.camera.quaternion);
      const right = new THREE.Vector3(1, 0, 0).applyQuaternion(q), up = new THREE.Vector3(0, 1, 0).applyQuaternion(q);
      for (const p of pts) {
        const v = new THREE.Vector3(...p), x = v.dot(right), y = v.dot(up);
        if (x < x0) x0 = x; if (x > x1) x1 = x; if (y < y0) y0 = y; if (y > y1) y1 = y;
      }
    }
    const W = this.el.clientWidth || 600, H = this.el.clientHeight || 450;
    return { cx: (x0 + x1) / 2, cy: (y0 + y1) / 2, h: Math.max(0.05, 1.08 * Math.max((y1 - y0) / 2, ((x1 - x0) / 2) * H / W)) };
  }
  // The camera's view rectangle: centred at (cx, cy) in screen coordinates, half-height h.
  setView(v) {
    this.view = { cx: v.cx, cy: v.cy, h: v.h };
    const W = this.el.clientWidth || 600, H = this.el.clientHeight || 450, a = W / H;
    Object.assign(this.camera, { left: v.cx - v.h * a, right: v.cx + v.h * a, top: v.cy + v.h, bottom: v.cy - v.h });
    this.camera.updateProjectionMatrix();
  }
  installPan() {
    const cv = this.renderer.domElement, touches = new Map();
    let drag = null;
    const perPx = () => (2 * this.view.h) / (this.camera.zoom * (this.el.clientHeight || 450));
    const shift = (dx, dy) => this.setView({ cx: this.view.cx - dx * perPx(), cy: this.view.cy + dy * perPx(), h: this.view.h });
    const stopTurn = () => (this.turnLeft = 0);
    cv.addEventListener('pointerdown', (e) => {
      stopTurn();
      if (e.pointerType === 'touch') { touches.set(e.pointerId, [e.clientX, e.clientY]); return; }
      if (e.button === 2 || (e.button === 0 && e.shiftKey)) drag = [e.clientX, e.clientY];
    });
    window.addEventListener('pointermove', (e) => {
      if (drag && e.pointerType !== 'touch') { shift(e.clientX - drag[0], e.clientY - drag[1]); drag = [e.clientX, e.clientY]; return; }
      if (!touches.has(e.pointerId)) return;
      if (touches.size === 2) {
        const [a, b] = [...touches.values()], mid0 = [(a[0] + b[0]) / 2, (a[1] + b[1]) / 2];
        touches.set(e.pointerId, [e.clientX, e.clientY]);
        const [c, d] = [...touches.values()], mid1 = [(c[0] + d[0]) / 2, (c[1] + d[1]) / 2];
        shift(mid1[0] - mid0[0], mid1[1] - mid0[1]);
      } else touches.set(e.pointerId, [e.clientX, e.clientY]);
    });
    const up = (e) => { if (e.pointerType === 'touch') touches.delete(e.pointerId); else drag = null; };
    window.addEventListener('pointerup', up);
    window.addEventListener('pointercancel', up);
    cv.addEventListener('contextmenu', (e) => e.preventDefault());
    cv.addEventListener('dblclick', () => {
      if (!this.o) return;
      // Back to the view as it was first drawn.
      stopTurn();
      if (!this.initial) return;
      this.camera.position.copy(this.initial.pos); this.camera.zoom = 1;
      this.controls.target.set(0, 0, 0); this.camera.lookAt(0, 0, 0);
      this.setView(this.initial.view);
    });
  }

  // Put the origin where the drawing, seen from the current view, is centred in the open area.
  placeOrigin(o, s) {
    let x0 = 0, x1 = 0, y0 = 0, y1 = 0;
    for (const [x, y] of this.projected(o, s)) { x0 = Math.min(x0, x); x1 = Math.max(x1, x); y0 = Math.min(y0, y); y1 = Math.max(y1, y); }
    const r = this.safeRect(), k = Math.min((r.R - r.L) / Math.max(x1 - x0, 1e-6), (r.B - r.T) / Math.max(y1 - y0, 1e-6));
    const ox = (r.L + r.R) / 2 - k * (x0 + x1) / 2, oy = (r.T + r.B) / 2 + k * (y0 + y1) / 2;
    this.home = { ax: (ox - r.L) / (r.R - r.L), ay: (oy - r.T) / (r.B - r.T) };
    this.anchor = { ...this.home };
  }
  // The initial framing (called once, when the points are first drawn): the largest zoom that keeps
  // every point and every corner of the volume inside the open area, with the origin where placeOrigin
  // put it. After this the view belongs to the visitor.
  fitOnce(now = true) {
    if (!this.o || !this.camSet) return;
    const W = this.el.clientWidth || 600, H = this.el.clientHeight || 450, r = this.safeRect(), m = 1.08;
    // The zoom: the largest that keeps everything inside the open area with the origin at its default
    // place (so the visitor's panning never changes it); then the origin goes where the visitor put it.
    const hx = r.L + this.home.ax * (r.R - r.L), hy = r.T + this.home.ay * (r.B - r.T);
    let k = Infinity;   // pixels per unit
    for (const [x, y] of this.projected(this.o, this.scale)) {
      if (x > 1e-9) k = Math.min(k, (r.R - hx) / x); else if (x < -1e-9) k = Math.min(k, (hx - r.L) / -x);
      if (y > 1e-9) k = Math.min(k, (hy - r.T) / y); else if (y < -1e-9) k = Math.min(k, (r.B - hy) / -y);
    }
    const ox = r.L + this.anchor.ax * (r.R - r.L), oy = r.T + this.anchor.ay * (r.B - r.T);
    if (!isFinite(k) || k <= 0) k = H / 2;
    k /= m;
    const h = H / (2 * k), g = { cx: (W / 2 - ox) / k, cy: (oy - H / 2) / k, h };
    const v = this.view, t = now ? 1 : 0.35;
    if (now || Math.abs(g.cx - v.cx) + Math.abs(g.cy - v.cy) + Math.abs(g.h - v.h) > 1e-5 * v.h) {
      this.setView({ cx: v.cx + (g.cx - v.cx) * t, cy: v.cy + (g.cy - v.cy) * t, h: v.h + (g.h - v.h) * t });
      if (now) this.frameH = this.view.h;
      if (Math.abs(this.view.h - (this.laidH ?? 0)) > 0.01 * this.view.h) this.layout();
    }
  }
  // Everything drawn (points, and for K ≤ 3 every corner of the volume), in screen coordinates
  // relative to the origin, for the current view.
  projected(o, s) {
    const pts = o.coords.map((c, i) => c.map((x) => x * o.c[i] * s));
    const sel = o.sel.length >= 2 && o.sel.length <= 3 ? o.sel : [];
    for (let mask = 3; mask < 1 << sel.length; mask++) {
      if ((mask & (mask - 1)) === 0) continue;
      pts.push([0, 1, 2].map((d) => sel.reduce((acc, i, k) => acc + (mask & (1 << k) ? pts[i][d] : 0), 0)));
    }
    const right = new THREE.Vector3(1, 0, 0).applyQuaternion(this.camera.quaternion), up = new THREE.Vector3(0, 1, 0).applyQuaternion(this.camera.quaternion);
    return pts.map(([x, y, z]) => [x * right.x + y * right.y + z * right.z, x * up.x + y * up.y + z * up.z]);
  }

  tick() {
    const tickStart = performance.now();
    if (this.turnLeft > 0) {
      const now = performance.now(), dt = Math.min(0.1, (now - (this.lastTick ?? now)) / 1000);
      // ease out over about 2 s, at any frame rate
      const step = Math.min(this.turnLeft, Math.max(0.02 * dt, this.turnLeft * (1 - Math.exp(-2.5 * dt))));
      this.camera.position.applyAxisAngle(new THREE.Vector3(0, 1, 0), step);
      this.turnLeft -= step;
    }
    this.lastTick = tickStart;
    if (!this.targets || !this.current) return;
    let moving = false;
    this.current.forEach((p, i) => {
      const t = this.targets[i];
      if (p.distanceToSquared(t) > 1e-10) { p.lerp(t, 0.18); moving = true; }
    });
    if (moving) this.layout();
  }

  pick(e) {
    if (!this.mesh) return;
    const r = this.renderer.domElement.getBoundingClientRect();
    this.mouse.set(((e.clientX - r.left) / r.width) * 2 - 1, -((e.clientY - r.top) / r.height) * 2 + 1);
    // The nearest point on screen within 10 px (the dots are too small to hit exactly).
    const i = this.nearest(e.clientX, e.clientY, 10);
    if (i !== this.hovered) { this.hovered = i; this.layout(); }
    this.onHover(i, e.clientX, e.clientY);
  }

  // Index of the point drawn nearest to a screen position, or -1 if none within r px.
  nearest(x, y, r) {
    if (!this.current) return -1;
    const rect = this.renderer.domElement.getBoundingClientRect();
    let best = -1, bd = r * r, bz = Infinity;
    const v = new THREE.Vector3();
    this.current.forEach((p, i) => {
      v.copy(p).project(this.camera);
      const sx = rect.left + ((v.x + 1) / 2) * rect.width, sy = rect.top + ((1 - v.y) / 2) * rect.height;
      const d = (sx - x) ** 2 + (sy - y) ** 2;
      if (d < bd || (d === bd && v.z < bz)) { bd = d; bz = v.z; best = i; }
    });
    return best;
  }

  // Position of point i in CSS pixels relative to the canvas.
  screenXY(i) {
    if (!this.current?.[i]) return null;
    const v = this.current[i].clone().project(this.camera);
    const w = this.renderer.domElement.clientWidth, h = this.renderer.domElement.clientHeight;
    return [((v.x + 1) / 2) * w, ((1 - v.y) / 2) * h];
  }

  highlight(i) { this.hovered = i; this.layout(); }

  dispose() {
    this.alive = false;
    this.ro?.disconnect();
    this.controls.dispose();
    this.renderer.dispose();
    this.renderer.domElement.remove();
  }
}
