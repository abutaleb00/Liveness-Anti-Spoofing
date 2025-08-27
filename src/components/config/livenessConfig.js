// src/config/livenessConfig.js
// Central config with ENV overrides. Supports Vite's import.meta.env and Node's process.env.
const env = (k, d) => {
  const v = (typeof import.meta !== 'undefined' && import.meta.env && import.meta.env[k]) ||
            (typeof process !== 'undefined' && process.env && process.env[k]) ||
            d
  return v
}
const num = (k, d) => {
  const v = env(k, d)
  const n = Number(v)
  return Number.isFinite(n) ? n : d
}

export const CFG = {
  // Eye thresholds
  EAR_OPEN: num('VITE_EAR_OPEN', 0.21),
  EAR_CLOSED: num('VITE_EAR_CLOSED', 0.17),

  // Smile thresholds
  MAR_SMILE: num('VITE_MAR_SMILE', 0.28),
  SMILE_MAR_DELTA: num('VITE_SMILE_MAR_DELTA', 0.05),
  SMILE_W_DELTA: num('VITE_SMILE_W_DELTA', 0.035),
  SMILE_CURVE_DELTA: num('VITE_SMILE_CURVE_DELTA', 0.012),
  SMILE_MAR_DELTA2: num('VITE_SMILE_MAR_DELTA2', 0.06),

  // Yaw orientation / thresholds
  YAW_FLIP: env('VITE_YAW_FLIP', 'true') !== 'false',
  YAW_RAD: num('VITE_YAW_RAD', 0.10),
  YAW_MARGIN: num('VITE_YAW_MARGIN', 0.02),

  // Sec-based holds
  HOLD_SEC_LEFT_RIGHT: num('VITE_HOLD_SEC_LEFT_RIGHT', 1.2),
  HOLD_SEC_SMILE: num('VITE_HOLD_SEC_SMILE', 0.9),

  // Frame-based fallback (60fps baseline)
  HOLD_FRAMES: num('VITE_HOLD_FRAMES', 18),
  HOLD_FRAMES_TURN: num('VITE_HOLD_FRAMES_TURN', 24),

  // Blink
  BLINK_TARGET_MIN: num('VITE_BLINK_TARGET_MIN', 3),
  BLINK_TARGET_MAX: num('VITE_BLINK_TARGET_MAX', 4),
  BLINK_MIN_AMP: num('VITE_BLINK_MIN_AMP', 0.03),
  BLINK_MIN_INTERVAL_MS: num('VITE_BLINK_MIN_INTERVAL_MS', 120),

  // PAD depth ranges (good interval mapping)
  ZRANGE_GOOD_LO: num('VITE_ZRANGE_GOOD_LO', 0.02),
  ZRANGE_GOOD_HI: num('VITE_ZRANGE_GOOD_HI', 0.09),
  PLANE_GOOD_LO: num('VITE_PLANE_GOOD_LO', 0.01),
  PLANE_GOOD_HI: num('VITE_PLANE_GOOD_HI', 0.06),
  NOSE_GOOD_LO: num('VITE_NOSE_GOOD_LO', 0.07),
  NOSE_GOOD_HI: num('VITE_NOSE_GOOD_HI', 0.20),

  // Derived (do not env override)
  get ZRANGE_GOOD(){ return [this.ZRANGE_GOOD_LO, this.ZRANGE_GOOD_HI] },
  get PLANE_GOOD(){ return [this.PLANE_GOOD_LO, this.PLANE_GOOD_HI] },
  get NOSE_GOOD(){ return [this.NOSE_GOOD_LO, this.NOSE_GOOD_HI] },

  // Calibration
  CALIB_SECONDS: num('VITE_CALIB_SECONDS', 1.0),

  // Neutral capture / passport
  NEUTRAL_CAPTURE_DELAY_MS: num('VITE_NEUTRAL_CAPTURE_DELAY_MS', 1000),
  NEUTRAL_YAW: num('VITE_NEUTRAL_YAW', 0.06),
  NEUTRAL_MAR_DELTA: num('VITE_NEUTRAL_MAR_DELTA', 0.04),
  NEUTRAL_TIMEOUT_MS: num('VITE_NEUTRAL_TIMEOUT_MS', 2500),
  NEUTRAL_HOLD_FRAMES: num('VITE_NEUTRAL_HOLD_FRAMES', 6),

  // Score thresholds & weights
  PASS_THRESH_LIVE: num('VITE_PASS_THRESH_LIVE', 75),
  PASS_THRESH_PAD: num('VITE_PASS_THRESH_PAD', 70),

  LIVE_W_STEP: num('VITE_LIVE_W_STEP', 0.50),
  LIVE_W_TIMING: num('VITE_LIVE_W_TIMING', 0.22),
  LIVE_W_BLINK: num('VITE_LIVE_W_BLINK', 0.14),
  LIVE_W_SMOOTH: num('VITE_LIVE_W_SMOOTH', 0.14),

  PAD_W_DEPTH: num('VITE_PAD_W_DEPTH', 0.60),
  PAD_W_RESP: num('VITE_PAD_W_RESP', 0.25),
  PAD_W_ACTION: num('VITE_PAD_W_ACTION', 0.15),

  RESPONSE_MIN_S: num('VITE_RESPONSE_MIN_S', 0.18),
  RESPONSE_MAX_S: num('VITE_RESPONSE_MAX_S', 3.0),
  MIN_STEP_TIME_S: num('VITE_MIN_STEP_TIME_S', 0.33),

  // Face alignment gate (circle guide)
  ALIGN_CX: num('VITE_ALIGN_CX', 0.5),    // center x (ratio of width)
  ALIGN_CY: num('VITE_ALIGN_CY', 0.42),   // center y (ratio of height)
  ALIGN_R: num('VITE_ALIGN_R', 0.22),     // radius ratio relative to min(w,h)
  ALIGN_FACE_MIN: num('VITE_ALIGN_FACE_MIN', 0.36),  // min face-diameter / circle-diameter
  ALIGN_FACE_MAX: num('VITE_ALIGN_FACE_MAX', 0.90),  // max face-diameter / circle-diameter
  ALIGN_HOLD_FRAMES: num('VITE_ALIGN_HOLD_FRAMES', 18),

}
