/**
 * Zepp OS page: reads accelerometer + gyroscope, resamples to a fixed 25 Hz, runs WorkoutDetector,
 * shows live reps / set count / rest timer. Written against Zepp OS API_LEVEL 3.x (Active Edge = 3.5).
 *
 * NOT TESTED ON A WATCH. The algorithm in ../lib/wristlift.js is tested in Node; this file is the
 * glue. Check each API call against docs.zepp.com if something fails (marked CHECK below).
 */
import * as hmUI from '@zos/ui'
import { Accelerometer, Gyroscope, FREQ_MODE_HIGH } from '@zos/sensor'
import { pauseDropWristScreenOff, resetDropWristScreenOff } from '@zos/display'
import { WorkoutDetector, Resampler } from '../lib/wristlift'

const FS = 25 // Hz fed to the algorithm (sensor callbacks are resampled to this)

Page({
  state: { sets: [], reps: 0, inSet: false, restFrom: 0, rawCount: 0, lastHzT: 0, hz: 0 },

  build() {
    // 360x360 round screen (Active Edge)
    this.txtReps = hmUI.createWidget(hmUI.widget.TEXT, {
      x: 0, y: 70, w: 360, h: 120, text: '0', text_size: 96, color: 0xffffff,
      align_h: hmUI.align.CENTER_H, align_v: hmUI.align.CENTER_V
    })
    this.txtStatus = hmUI.createWidget(hmUI.widget.TEXT, {
      x: 0, y: 190, w: 360, h: 40, text: 'ready', text_size: 30, color: 0x9be564,
      align_h: hmUI.align.CENTER_H, align_v: hmUI.align.CENTER_V
    })
    this.txtSets = hmUI.createWidget(hmUI.widget.TEXT, {
      x: 0, y: 235, w: 360, h: 40, text: 'sets 0', text_size: 28, color: 0xcccccc,
      align_h: hmUI.align.CENTER_H, align_v: hmUI.align.CENTER_V
    })
    // diagnostics: measured sensor callback rate - tells you the real sampling rate on this watch
    this.txtDiag = hmUI.createWidget(hmUI.widget.TEXT, {
      x: 0, y: 285, w: 360, h: 30, text: '', text_size: 20, color: 0x777777,
      align_h: hmUI.align.CENTER_H, align_v: hmUI.align.CENTER_V
    })
    this.startTracking()
  },

  startTracking() {
    // CHECK: keep the screen awake so the foreground app keeps receiving sensor data.
    try { pauseDropWristScreenOff({ duration: 0 }) } catch (e) { /* ignore */ }

    this.det = new WorkoutDetector({ fs: FS })
    this.resampler = new Resampler(FS, (v) => this.onSample(v))
    this.lastGyro = [0, 0, 0]

    this.accel = new Accelerometer()
    this.gyro = new Gyroscope()
    try { this.accel.setFreqMode(FREQ_MODE_HIGH); this.gyro.setFreqMode(FREQ_MODE_HIGH) } catch (e) { /* ignore */ }

    this.gyroCb = () => {
      const g = this.gyro.getCurrent()
      this.lastGyro = [g.x, g.y, g.z] // deg/s, only used for periodicity, unit irrelevant
    }
    this.accelCb = () => {
      const a = this.accel.getCurrent() // cm/s^2 -> m/s^2
      const t = Date.now()
      this.state.rawCount++
      if (t - this.state.lastHzT >= 2000) {
        this.state.hz = Math.round((this.state.rawCount * 1000) / Math.max(1, t - this.state.lastHzT))
        this.state.rawCount = 0
        this.state.lastHzT = t
        this.txtDiag.setProperty(hmUI.prop.TEXT, 'sensor ' + this.state.hz + ' Hz')
      }
      this.resampler.push(t, [a.x / 100, a.y / 100, a.z / 100, this.lastGyro[0], this.lastGyro[1], this.lastGyro[2]])
    }
    this.gyro.onChange(this.gyroCb)
    this.accel.onChange(this.accelCb)
    this.gyro.start()
    this.accel.start()

    this.timer = setInterval(() => this.tick(), 1000)
  },

  onSample(v) {
    const events = this.det.pushSample(v[0], v[1], v[2], v[3], v[4], v[5])
    for (const e of events) this.handle(e)
  },

  handle(e) {
    const s = this.state
    if (e.type === 'set_start') {
      s.inSet = true
      s.reps = 0
      this.txtStatus.setProperty(hmUI.prop.TEXT, 'set ' + (s.sets.length + 1))
    } else if (e.type === 'reps') {
      s.reps = e.count // shown only once >= 4 reps are confirmed, then back-filled
      this.txtReps.setProperty(hmUI.prop.TEXT, String(s.reps))
    } else if (e.type === 'set_end') {
      s.inSet = false
      s.sets.push(e.reps)
      s.reps = e.reps
      s.restFrom = Date.now()
      this.txtReps.setProperty(hmUI.prop.TEXT, String(e.reps))
      this.txtSets.setProperty(hmUI.prop.TEXT, 'sets ' + s.sets.length + '  last ' + e.reps)
    } else if (e.type === 'set_discarded') {
      s.inSet = false
      this.txtReps.setProperty(hmUI.prop.TEXT, '0')
      this.txtStatus.setProperty(hmUI.prop.TEXT, 'ready')
    }
  },

  tick() {
    const s = this.state
    if (!s.inSet && s.restFrom) {
      const r = Math.floor((Date.now() - s.restFrom) / 1000)
      this.txtStatus.setProperty(hmUI.prop.TEXT, 'rest ' + Math.floor(r / 60) + ':' + ('0' + (r % 60)).slice(-2))
    }
  },

  onDestroy() {
    try {
      clearInterval(this.timer)
      this.accel.offChange(this.accelCb)
      this.gyro.offChange(this.gyroCb)
      this.accel.stop()
      this.gyro.stop()
      resetDropWristScreenOff()
    } catch (e) { /* ignore */ }
  }
})
