// flock_core.js: reference simulation core (generalised Vicsek model with triadic alignment, arXiv 2608.10844), plain JS, no DOM.
// Used by the reference page (inlined) and by node tests. rule "P" = the authors' published code (Zenodo 10.5281/zenodo.20793454):
//   m_i = mean heading of {i} U {j : |xi-xj| < R};  h_i = mean over MUTUAL TRIANGLES (i,j,k pairwise < R, i excluded from j,k) of 0.5*[(vk.vi) vj + (vj.vi) vk];
//   u = alpha*m + (1-alpha)*h; theta' = atan2(u) + eta*(U-1/2) (u ~ 0: keep heading); then x += v0*(cos,sin)(theta') with the NEW heading.
function mulberry32(a) { return function () { a |= 0; a = a + 0x6D2B79F5 | 0; let t = Math.imul(a ^ a >>> 15, 1 | a); t = t + Math.imul(t ^ t >>> 7, 61 | t) ^ t; return ((t ^ t >>> 14) >>> 0) / 4294967296; }; }
class Flock {
  constructor(N, rho, alpha, seed, start) {
    this.N = N; this.alpha = alpha; this.v0 = 0.03; this.L = Math.sqrt(N / rho); this.rng = mulberry32(seed);
    const n = N; this.x = new Float64Array(n); this.y = new Float64Array(n); this.c = new Float64Array(n); this.s = new Float64Array(n);
    for (let i = 0; i < n; i++) { this.x[i] = this.rng() * this.L; this.y[i] = this.rng() * this.L; const th = start === 'ordered' ? 0 : (this.rng() * 2 - 1) * Math.PI; this.c[i] = Math.cos(th); this.s[i] = Math.sin(th); }
    this.cells = Math.max(1, Math.floor(this.L)); this.cs = this.L / this.cells; this.head = new Int32Array(this.cells * this.cells); this.next = new Int32Array(n);
    this.nb = new Int32Array(n * 128); this.nn = new Int32Array(n); this.mx = new Float64Array(n); this.my = new Float64Array(n); this.hx = new Float64Array(n); this.hy = new Float64Array(n); this.hc = new Int32Array(n);
  }
  order() { let a = 0, b = 0; for (let i = 0; i < this.N; i++) { a += this.c[i]; b += this.s[i]; } return Math.hypot(a, b) / this.N; }
  neighbours() {
    const { N, L, cells, cs, head, next, x, y, nb, nn } = this; head.fill(-1); nn.fill(0);
    for (let i = 0; i < N; i++) { let cx = Math.min(cells - 1, Math.floor(x[i] / cs)), cy = Math.min(cells - 1, Math.floor(y[i] / cs)); const k = cy * cells + cx; next[i] = head[k]; head[k] = i; }
    for (let i = 0; i < N; i++) {
      const cx = Math.min(cells - 1, Math.floor(x[i] / cs)), cy = Math.min(cells - 1, Math.floor(y[i] / cs));
      const seen = []; for (let dx = -1; dx <= 1; dx++) for (let dy = -1; dy <= 1; dy++) { const k = ((cy + dy + cells) % cells) * cells + ((cx + dx + cells) % cells); if (seen.indexOf(k) >= 0) continue; seen.push(k);
        for (let j = head[k]; j >= 0; j = next[j]) { if (j <= i) continue; let ddx = x[i] - x[j], ddy = y[i] - y[j]; ddx -= L * Math.round(ddx / L); ddy -= L * Math.round(ddy / L);
          if (ddx * ddx + ddy * ddy < 1) { if (nn[i] < 127 && nn[j] < 127) { nb[i * 128 + nn[i]++] = j; nb[j * 128 + nn[j]++] = i; } } } }
    }
  }
  step(eta) {
    const { N, L, alpha, c, s, x, y, nb, nn, mx, my, hx, hy, hc } = this; this.neighbours();
    for (let i = 0; i < N; i++) { let sx = c[i], sy = s[i]; for (let a = 0; a < nn[i]; a++) { const j = nb[i * 128 + a]; sx += c[j]; sy += s[j]; } mx[i] = sx / (nn[i] + 1); my[i] = sy / (nn[i] + 1); hx[i] = 0; hy[i] = 0; hc[i] = 0; }
    if (alpha < 1) {
      for (let i = 0; i < N; i++) { const n = nn[i]; for (let a = 0; a < n; a++) { const j = nb[i * 128 + a]; if (j < i) continue; for (let b = a + 1; b < n; b++) { const k = nb[i * 128 + b]; if (k < i) continue;
        let dx = x[j] - x[k], dy = y[j] - y[k]; dx -= L * Math.round(dx / L); dy -= L * Math.round(dy / L); if (dx * dx + dy * dy >= 1) continue;
        const sij = c[i] * c[j] + s[i] * s[j], sik = c[i] * c[k] + s[i] * s[k], sjk = c[j] * c[k] + s[j] * s[k];
        hx[i] += 0.5 * (sik * c[j] + sij * c[k]); hy[i] += 0.5 * (sik * s[j] + sij * s[k]); hc[i]++;
        hx[j] += 0.5 * (sij * c[k] + sjk * c[i]); hy[j] += 0.5 * (sij * s[k] + sjk * s[i]); hc[j]++;
        hx[k] += 0.5 * (sik * c[j] + sjk * c[i]); hy[k] += 0.5 * (sik * s[j] + sjk * s[i]); hc[k]++; } } }
      for (let i = 0; i < N; i++) if (hc[i] > 0) { hx[i] /= hc[i]; hy[i] /= hc[i]; }
    }
    for (let i = 0; i < N; i++) { const ux = alpha * mx[i] + (1 - alpha) * hx[i], uy = alpha * my[i] + (1 - alpha) * hy[i];
      let th = (Math.abs(ux) + Math.abs(uy) < 1e-12) ? Math.atan2(s[i], c[i]) : Math.atan2(uy, ux); th += eta * (this.rng() - 0.5); c[i] = Math.cos(th); s[i] = Math.sin(th);
      x[i] += this.v0 * c[i]; y[i] += this.v0 * s[i]; x[i] -= L * Math.floor(x[i] / L); y[i] -= L * Math.floor(y[i] / L); }
  }
}
if (typeof module !== 'undefined') module.exports = { Flock, mulberry32 };
