/**
 * wristlift.js - JavaScript port of algorithm.py for Zepp OS (Amazfit Active Edge, API_LEVEL 3.x).
 * Pure JS, no dependencies. Input: uniformly sampled accel (m/s^2) + optional gyro (any unit).
 *
 *   const det = new WorkoutDetector({ fs: 25 });
 *   det.pushSample(ax, ay, az, gx, gy, gz);   // call once per sample, returns array of events
 *   det.flush();                              // at end of workout
 *
 * Events: {type:'set_start',t} {type:'reps',t,count} {type:'set_end',start,end,reps,restStarts}
 *         {type:'set_discarded',t}
 */

export const G = 9.80665

export const DEFAULTS = {
  fs: 25,
  winS: 4, hopS: 1, startWindows: 3, endIdleS: 7, maxSetS: 600,
  fLo: 0.2, fHi: 1.3, minEnergy: 0.7, minConc: 0.35, minAc: 0.35,
  repLpHz: 6, minRepS: 0.8, maxRepS: 6, promFrac: 0.25,
  minReps: 4, intervalTol: 0.4, intervalFrac: 0.7,
  recountEveryS: 2 // live rep recount period while in a set (saves battery)
}

// ------------------------------------------------------------------ numeric helpers
function percentile(arr, q) {
  const s = Float64Array.from(arr).sort()
  const idx = (q / 100) * (s.length - 1)
  const lo = Math.floor(idx), hi = Math.ceil(idx)
  return s[lo] + (s[hi] - s[lo]) * (idx - lo)
}

function median(arr) {
  const s = Float64Array.from(arr).sort()
  const m = s.length >> 1
  return s.length % 2 ? s[m] : (s[m - 1] + s[m]) / 2
}

function reflectIdx(i, n) {
  const p = 2 * n
  i = ((i % p) + p) % p
  return i < n ? i : p - 1 - i
}

/** Gaussian smoothing, scipy gaussian_filter1d semantics (mode 'reflect', truncate 4). */
function gauss(x, sigma) {
  const n = x.length
  const r = Math.floor(4 * sigma + 0.5)
  const k = new Float64Array(2 * r + 1)
  let sum = 0
  for (let j = -r; j <= r; j++) { k[j + r] = Math.exp(-0.5 * (j * j) / (sigma * sigma)); sum += k[j + r] }
  for (let j = 0; j < k.length; j++) k[j] /= sum
  const out = new Float64Array(n)
  for (let i = 0; i < n; i++) {
    let s = 0
    for (let j = -r; j <= r; j++) s += k[j + r] * x[reflectIdx(i + j, n)]
    out[i] = s
  }
  return out
}

/** Zero-phase 2nd-order Butterworth low-pass (forward-backward, odd-extension padding). */
function lowpass(x, fc, fs) {
  const n = x.length
  if (n < 40 || fc >= fs / 2) return Float64Array.from(x)
  const w0 = 2 * Math.PI * fc / fs, cw = Math.cos(w0), alpha = Math.sin(w0) / (2 * Math.SQRT1_2)
  const a0 = 1 + alpha
  const b0 = ((1 - cw) / 2) / a0, b1 = (1 - cw) / a0, b2 = b0
  const a1 = (-2 * cw) / a0, a2 = (1 - alpha) / a0
  const pad = Math.min(30, n - 1)
  const p = new Float64Array(n + 2 * pad)
  for (let i = 0; i < pad; i++) p[i] = 2 * x[0] - x[pad - i]
  for (let i = 0; i < n; i++) p[pad + i] = x[i]
  for (let k = 0; k < pad; k++) p[pad + n + k] = 2 * x[n - 1] - x[n - 2 - k]
  const run = (a) => {
    // initialise filter state to the steady state of the first sample (limits the transient)
    const z2i = a[0] * (b2 - a2)
    let z2 = z2i, z1 = a[0] * (b1 - a1) + z2i
    const o = new Float64Array(a.length)
    for (let i = 0; i < a.length; i++) {
      const v = b0 * a[i] + z1
      z1 = b1 * a[i] - a1 * v + z2
      z2 = b2 * a[i] - a2 * v
      o[i] = v
    }
    return o
  }
  let y = run(p)
  y.reverse()
  y = run(y)
  y.reverse()
  return y.slice(pad, pad + n)
}

/** First principal component of a 3-channel signal -> 1-D Float64Array. */
function pca1(ch) {
  const n = ch[0].length
  const c = ch.map((a) => {
    let m = 0
    for (let i = 0; i < n; i++) m += a[i]
    m /= n
    const o = new Float64Array(n)
    for (let i = 0; i < n; i++) o[i] = a[i] - m
    return o
  })
  const cov = [[0, 0, 0], [0, 0, 0], [0, 0, 0]]
  for (let a = 0; a < 3; a++) for (let b = a; b < 3; b++) {
    let s = 0
    for (let i = 0; i < n; i++) s += c[a][i] * c[b][i]
    cov[a][b] = cov[b][a] = s
  }
  let v = [0.577, 0.577, 0.577]
  for (let it = 0; it < 40; it++) {
    const w = [0, 1, 2].map((a) => cov[a][0] * v[0] + cov[a][1] * v[1] + cov[a][2] * v[2])
    const nr = Math.hypot(w[0], w[1], w[2]) || 1
    v = w.map((q) => q / nr)
  }
  const out = new Float64Array(n)
  for (let i = 0; i < n; i++) out[i] = c[0][i] * v[0] + c[1][i] * v[1] + c[2][i] * v[2]
  return out
}

/** Autocorrelation for lags [lo, hi): normalised by lag 0, with mild bias correction. */
function autocorrRange(x, lo, hi) {
  const n = x.length
  let ac0 = 0
  for (let i = 0; i < n; i++) ac0 += x[i] * x[i]
  ac0 += 1e-12
  const out = new Float64Array(Math.max(0, hi - lo))
  for (let l = lo; l < hi; l++) {
    let s = 0
    for (let i = 0; i + l < n; i++) s += x[i] * x[i + l]
    out[l - lo] = (s / ac0) / Math.max(1 - l / n, 0.3)
  }
  return out
}

/** scipy.signal.find_peaks subset: distance (applied first) + prominence, or height. */
function findPeaks(x, { distance = 0, prominence = null, height = null } = {}) {
  const n = x.length
  let pk = []
  for (let i = 1; i < n - 1; i++) {
    if (x[i] > x[i - 1]) {
      let j = i
      while (j < n - 1 && x[j + 1] === x[i]) j++
      if (j < n - 1 && x[j + 1] < x[i]) pk.push((i + j) >> 1)
      i = j
    }
  }
  if (height !== null) pk = pk.filter((p) => x[p] >= height)
  if (distance > 1 && pk.length > 1) {
    const keep = new Array(pk.length).fill(true)
    const order = pk.map((_, i) => i).sort((a, b) => x[pk[a]] - x[pk[b]])
    for (let o = order.length - 1; o >= 0; o--) {
      const i = order[o]
      if (!keep[i]) continue
      for (let k = i - 1; k >= 0 && pk[i] - pk[k] < distance; k--) keep[k] = false
      for (let k = i + 1; k < pk.length && pk[k] - pk[i] < distance; k++) keep[k] = false
    }
    pk = pk.filter((_, i) => keep[i])
  }
  if (prominence !== null) {
    pk = pk.filter((p) => {
      let lm = x[p], i = p
      while (i >= 0 && x[i] <= x[p]) { if (x[i] < lm) lm = x[i]; i-- }
      let rm = x[p]
      i = p
      while (i < n && x[i] <= x[p]) { if (x[i] < rm) rm = x[i]; i++ }
      return x[p] - Math.max(lm, rm) >= prominence
    })
  }
  return pk
}

// ------------------------------------------------------------------ stage 2: window features
export function windowFeatures(acc, gyro, fs, cfg) {
  const x = acc.map((c) => lowpass(c, 6, fs))
  const n = x[0].length
  let energy = 0
  for (const c of x) {
    let m = 0
    for (let i = 0; i < n; i++) m += c[i]
    m /= n
    let v = 0
    for (let i = 0; i < n; i++) v += (c[i] - m) * (c[i] - m)
    energy += v / n
  }
  energy = Math.sqrt(energy)
  const sig = pca1(x)

  // Hann-windowed DFT on a 0.05 Hz grid, 0.05..5 Hz
  const w = new Float64Array(n)
  for (let i = 0; i < n; i++) w[i] = 0.5 - 0.5 * Math.cos((2 * Math.PI * i) / (n - 1))
  const freqs = [], pw = []
  for (let f = 0.05; f <= 5.0001; f += 0.05) {
    let re = 0, im = 0
    const k = (2 * Math.PI * f) / fs
    for (let i = 0; i < n; i++) { const v = sig[i] * w[i]; re += v * Math.cos(k * i); im -= v * Math.sin(k * i) }
    freqs.push(f); pw.push(re * re + im * im)
  }
  let dom = 0, best = -1, tot = 1e-12
  for (let i = 0; i < freqs.length; i++) {
    if (freqs[i] >= 0.1) tot += pw[i]
    if (freqs[i] >= 0.1 && freqs[i] <= 4.0 && pw[i] > best) { best = pw[i]; dom = freqs[i] }
  }
  let num = 0
  for (let i = 0; i < freqs.length; i++) if (freqs[i] >= dom - 0.4 && freqs[i] <= dom + 0.4) num += pw[i]
  const conc = num / tot

  const lo = Math.floor(0.5 * fs), hi = n >> 1
  let acpk = 0
  if (hi > lo + 2) {
    const ac = autocorrRange(sig, lo, hi)
    for (const p of findPeaks(ac)) if (ac[p] > acpk) acpk = ac[p]
  }

  let gr = 0
  if (gyro) {
    for (const c of gyro) {
      let m = 0
      for (let i = 0; i < n; i++) m += c[i]
      m /= n
      let v = 0
      for (let i = 0; i < n; i++) v += (c[i] - m) * (c[i] - m)
      gr += v / n
    }
    gr = Math.sqrt(gr)
  }
  return { energy, dom, conc, acpk, gr }
}

export function ruleActivity(f, cfg) {
  return f.energy >= cfg.minEnergy && f.dom >= cfg.fLo && f.dom <= cfg.fHi &&
    (f.conc >= cfg.minConc || f.acpk >= cfg.minAc)
}

// ------------------------------------------------------------------ stage 3: rep counting
export function countReps(acc, gyro, fs, cfg) {
  const none = { count: 0, confirmed: false, peakTimes: [], period: 0, strength: 0, tStart: 0, tEnd: 0 }
  const n = acc[0].length
  if (n < Math.floor(4 * cfg.minRepS * fs)) return none

  const cands = [pca1(acc.map((c) => lowpass(c, cfg.repLpHz, fs)))]
  if (gyro && gyro.some((c) => c.some((v) => v !== 0))) cands.push(pca1(gyro.map((c) => lowpass(c, cfg.repLpHz, fs))))

  let best = null
  for (const sig of cands) {
    const lo = Math.floor(cfg.minRepS * fs), hi = Math.min(Math.floor(cfg.maxRepS * fs), n >> 1)
    if (hi <= lo + 2) continue
    const seg = autocorrRange(sig, lo, hi)
    const pk = findPeaks(seg, { height: cfg.minAc })
    if (!pk.length) continue
    let hmax = -1
    for (const p of pk) if (seg[p] > hmax) hmax = seg[p]
    const k = pk.find((p) => seg[p] >= 0.75 * hmax)
    const st = seg[k]
    if (!best || st > best.st) best = { sig, lag: lo + k, st }
  }
  if (!best) return none

  const P = best.lag, sig = best.sig
  const sm = gauss(sig, P / 8)

  const fast = gauss(sig, 0.1 * fs)
  const grad = new Float64Array(n)
  for (let i = 0; i < n; i++) {
    const a = fast[Math.max(0, i - 1)], b = fast[Math.min(n - 1, i + 1)]
    grad[i] = Math.abs((b - a) / (i === 0 || i === n - 1 ? 1 : 2))
  }
  const env = gauss(grad, 0.15 * fs)
  const thr = 0.15 * percentile(env, 95)
  let ta = -1, tb = -1
  for (let i = 0; i < n; i++) if (env[i] > thr) { if (ta < 0) ta = i; tb = i }
  if (ta < 0) return none
  const nEst = (tb - ta) / P

  let bestPk = [], bestErr = 1e9
  for (const pol of [1, -1]) {
    const x = sm.map((v) => v * pol)
    const amp = percentile(x, 90) - percentile(x, 10)
    const pk = findPeaks(x, { distance: Math.max(1, Math.floor(0.6 * P)), prominence: cfg.promFrac * amp })
    const inner = pk.filter((p) => p > ta && p < tb)
    const err = Math.abs(inner.length - nEst)
    if (err < bestErr - 1e-9 || (Math.abs(err - bestErr) < 1e-9 && inner.length > bestPk.length)) {
      bestPk = inner; bestErr = err
    }
  }

  let ok = false
  if (bestPk.length >= 2) {
    const iv = []
    for (let i = 1; i < bestPk.length; i++) iv.push((bestPk[i] - bestPk[i - 1]) / fs)
    const med = median(iv)
    const frac = iv.filter((v) => Math.abs(v - med) <= cfg.intervalTol * med).length / iv.length
    ok = bestPk.length >= cfg.minReps && frac >= cfg.intervalFrac
  }
  return {
    count: bestPk.length, confirmed: ok, peakTimes: bestPk.map((p) => p / fs),
    period: P / fs, strength: best.st, tStart: ta / fs, tEnd: tb / fs
  }
}

// ------------------------------------------------------------------ streaming detector
export class WorkoutDetector {
  constructor(opts = {}) {
    this.cfg = Object.assign({}, DEFAULTS, opts)
    const c = this.cfg
    this.winN = Math.round(c.winS * c.fs)
    this.hopN = Math.round(c.hopS * c.fs)
    this.acc = [[], [], []]
    this.gyro = [[], [], []]
    this.hasGyro = null
    this.dropped = 0
    this.total = 0
    this.nextEnd = this.winN
    this.inSet = false
    this.actRun = 0
    this.idleS = 0
    this.setStart = 0
    this.lastActiveEnd = 0
    this.lastReps = 0
    this.stepsInSet = 0
  }

  /** accel in m/s^2. gyro optional (pass undefined/null to disable). Returns events array. */
  pushSample(ax, ay, az, gx, gy, gz) {
    if (this.hasGyro === null) this.hasGyro = gx !== undefined && gx !== null
    this.acc[0].push(ax); this.acc[1].push(ay); this.acc[2].push(az)
    if (this.hasGyro) { this.gyro[0].push(gx); this.gyro[1].push(gy); this.gyro[2].push(gz) }
    this.total++
    const events = []
    while (this.nextEnd <= this.total) {
      for (const e of this._step(this.nextEnd)) events.push(e)
      this.nextEnd += this.hopN
    }
    this._trim()
    return events
  }

  flush() { return this.inSet ? this._finalize() : [] }

  _slice(a, b) {
    const s = a - this.dropped, e = b - this.dropped
    const acc = this.acc.map((c) => Float64Array.from(c.slice(s, e)))
    const gyro = this.hasGyro ? this.gyro.map((c) => Float64Array.from(c.slice(s, e))) : null
    return [acc, gyro]
  }

  _step(end) {
    const c = this.cfg, fs = c.fs, ev = []
    const [a, g] = this._slice(end - this.winN, end)
    const active = ruleActivity(windowFeatures(a, g, fs, c), c)

    if (!this.inSet) {
      this.actRun = active ? this.actRun + 1 : 0
      if (this.actRun >= c.startWindows) {
        this.inSet = true
        this.setStart = Math.max(this.dropped, end - this.winN - (c.startWindows - 1) * this.hopN)
        this.lastActiveEnd = end
        this.idleS = 0
        this.lastReps = 0
        this.stepsInSet = 0
        ev.push({ type: 'set_start', t: this.setStart / fs })
      }
      return ev
    }

    if (active) { this.lastActiveEnd = end; this.idleS = 0 } else this.idleS += c.hopS
    this.stepsInSet++

    if (this.stepsInSet % Math.max(1, Math.round(c.recountEveryS / c.hopS)) === 0) {
      const [sa, sg] = this._slice(this.setStart, end)
      const r = countReps(sa, sg, fs, c)
      if (r.confirmed && r.count !== this.lastReps) {
        this.lastReps = r.count
        ev.push({ type: 'reps', t: end / fs, count: r.count })
      }
    }
    if (this.idleS >= c.endIdleS || (end - this.setStart) / fs > c.maxSetS) {
      for (const e of this._finalize()) ev.push(e)
    }
    return ev
  }

  _finalize() {
    const c = this.cfg, fs = c.fs
    const s0 = this.setStart, e0 = this.lastActiveEnd
    this.inSet = false; this.actRun = 0; this.idleS = 0
    const [a, g] = this._slice(s0, e0)
    const r = countReps(a, g, fs, c)
    if (!r.confirmed) return [{ type: 'set_discarded', t: e0 / fs }]
    const start = s0 / fs + r.tStart, end = s0 / fs + r.tEnd
    return [{ type: 'set_end', start: +start.toFixed(2), end: +end.toFixed(2), reps: r.count, restStarts: +end.toFixed(2) }]
  }

  _trim() {
    const c = this.cfg
    let keep = this.inSet ? this.setStart : this.nextEnd - this.winN - c.startWindows * this.hopN
    keep = Math.max(this.dropped, keep)
    const cut = keep - this.dropped
    if (cut > 0) {
      for (let k = 0; k < 3; k++) { this.acc[k].splice(0, cut); if (this.hasGyro) this.gyro[k].splice(0, cut) }
      this.dropped = keep
    }
  }
}

// ------------------------------------------------------------------ irregular -> uniform resampler
/**
 * Zepp sensor callbacks fire at an unspecified, possibly uneven rate. Feed (timeMs, [6 values])
 * here and get uniformly spaced samples at fs via linear interpolation.
 */
export class Resampler {
  constructor(fs, onSample) {
    this.dt = 1000 / fs
    this.onSample = onSample
    this.prev = null
    this.nextT = null
  }

  push(tMs, v) {
    if (this.prev === null) { this.prev = { t: tMs, v }; this.nextT = tMs; return }
    if (tMs <= this.prev.t) return
    while (this.nextT <= tMs) {
      const a = (this.nextT - this.prev.t) / (tMs - this.prev.t)
      this.onSample(v.map((x, i) => this.prev.v[i] + a * (x - this.prev.v[i])))
      this.nextT += this.dt
    }
    this.prev = { t: tMs, v }
  }
}
