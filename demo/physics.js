// Motion of two thin shells between crossings, in the Schwarzschild time t of the middle region (the paper's clock).
// State y = [R_A, dR_A/dtau, R_B, dR_B/dtau]; region masses 0 (inside), Mmid, Mout. Same equations as the paper's
// solver: dR/dt = v * k, dv/dt = -V'(R)/2 * k, with k = f(R; Mmid) / sqrt(v^2 + f(R; Mmid)) for either shell.
function fM(R, M) { return 1 - (2 * M) / R + R * R; }
function dV(R, lo, hi, m, w) {
  const u = m * Math.pow(R, -2 * w);
  return 2 * R + (hi + lo) / (R * R) - (4 * w * (hi - lo) ** 2) / (R * u * u) + ((2 + 4 * w) * u * u) / (4 * R ** 3);
}
function rhs(y, Mmid, Mout, aInside, m, w, out) {
  for (let s = 0; s < 2; s++) {
    const R = y[2 * s], v = y[2 * s + 1];
    const inner = (s === 0) === aInside;
    const lo = inner ? 0 : Mmid, hi = inner ? Mmid : Mout;
    const f = fM(R, Mmid), k = f / Math.sqrt(v * v + f);
    out[2 * s] = v * k;
    out[2 * s + 1] = -0.5 * dV(R, lo, hi, m, w) * k;
  }
}
// Integrate one flight with classical RK4 and return sampled radii; h is the target step in t.
function flight(fl, duration, Mout, m, w, h) {
  const n = Math.max(1, Math.ceil(duration / h)), dt = duration / n;
  const Mmid = fl[5], aIn = fl[6] === 1;
  let y = [fl[1], fl[3], fl[2], fl[4]];
  const RA = new Float64Array(n + 1), RB = new Float64Array(n + 1);
  RA[0] = y[0]; RB[0] = y[2];
  const k1 = [0, 0, 0, 0], k2 = [0, 0, 0, 0], k3 = [0, 0, 0, 0], k4 = [0, 0, 0, 0], tmp = [0, 0, 0, 0];
  for (let i = 1; i <= n; i++) {
    rhs(y, Mmid, Mout, aIn, m, w, k1);
    for (let j = 0; j < 4; j++) tmp[j] = y[j] + 0.5 * dt * k1[j];
    rhs(tmp, Mmid, Mout, aIn, m, w, k2);
    for (let j = 0; j < 4; j++) tmp[j] = y[j] + 0.5 * dt * k2[j];
    rhs(tmp, Mmid, Mout, aIn, m, w, k3);
    for (let j = 0; j < 4; j++) tmp[j] = y[j] + dt * k3[j];
    rhs(tmp, Mmid, Mout, aIn, m, w, k4);
    for (let j = 0; j < 4; j++) y[j] += (dt / 6) * (k1[j] + 2 * k2[j] + 2 * k3[j] + k4[j]);
    RA[i] = y[0]; RB[i] = y[2];
  }
  return { RA, RB, dt, n };
}
if (typeof module !== "undefined") module.exports = { flight, fM, dV, rhs };
