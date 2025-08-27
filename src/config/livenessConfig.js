// src/config/livenessConfig.js
// Read config from env with safe parsing (Vite: import.meta.env, CRA: process.env)
const ENV = (typeof import.meta !== "undefined" && import.meta.env) ? import.meta.env : (typeof process !== "undefined" ? process.env : {})

// Safe numeric parser
const num = (key, def) => {
  const v = ENV[key]; if (v === undefined || v === null || v === "") return def
  const n = Number(v); return Number.isFinite(n) ? n : def
}
// Safe boolean parser
const bool = (key, def) => {
  const v = ENV[key]; if (v === undefined) return def
  const s = String(v).toLowerCase()
  if (["1","true","yes","on"].includes(s)) return true
  if (["0","false","no","off"].includes(s)) return false
  return def
}

export const CFG = {
  // Blink targets
  BLINK_TARGET_MIN: num("VITE_BLINK_TARGET_MIN", 3),
  BLINK_TARGET_MAX: num("VITE_BLINK_TARGET_MAX", 4),

  // Eye thresholds
  EAR_OPEN: num("VITE_EAR_OPEN", 0.21),
  EAR_CLOSED: num("VITE_EAR_CLOSED", 0.17),

  // Smile thresholds
  MAR_SMILE: num("VITE_MAR_SMILE", 0.28),
  SMILE_MAR_DELTA: num("VITE_SMILE_MAR_DELTA", 0.05),
  SMILE_W_DELTA: num("VITE_SMILE_W_DELTA", 0.035),
  SMILE_CURVE_DELTA: num("VITE_SMILE_CURVE_DELTA", 0.012),
  SMILE_MAR_DELTA2: num("VITE_SMILE_MAR_DELTA2", 0.06),

  // Yaw
  YAW_FLIP: bool("VITE_YAW_FLIP", true),
  YAW_RAD: num("VITE_YAW_RAD", 0.10),
  YAW_MARGIN: num("VITE_YAW_MARGIN", 0.03),

  // Timing gates
  RESPONSE_MIN_S: num("VITE_RESPONSE_MIN_S", 0.18),
  RESPONSE_MAX_S: num("VITE_RESPONSE_MAX_S", 3.00),
  MIN_STEP_TIME_S: num("VITE_MIN_STEP_TIME_S", 0.60),

  // Holds (by frames; ~60fps). We also expose seconds for docs; final frames below.
  HOLD_SEC_LEFT_RIGHT: num("VITE_HOLD_SEC_LEFT_RIGHT", 2.0),
  HOLD_SEC_SMILE: num("VITE_HOLD_SEC_SMILE", 1.0),

  // Liveness weights
  LIVE_W_STEP: num("VITE_LIVE_W_STEP", 0.50),
  LIVE_W_TIMING: num("VITE_LIVE_W_TIMING", 0.22),
  LIVE_W_BLINK: num("VITE_LIVE_W_BLINK", 0.14),
  LIVE_W_SMOOTH: num("VITE_LIVE_W_SMOOTH", 0.14),

  // PAD weights
  PAD_W_DEPTH: num("VITE_PAD_W_DEPTH", 0.60),
  PAD_W_RESP: num("VITE_PAD_W_RESP", 0.25),
  PAD_W_ACTION: num("VITE_PAD_W_ACTION", 0.15),

  // Depth good ranges
  ZRANGE_GOOD_LO: num("VITE_ZRANGE_GOOD_LO", 0.02),
  ZRANGE_GOOD_HI: num("VITE_ZRANGE_GOOD_HI", 0.09),
  PLANE_GOOD_LO: num("VITE_PLANE_GOOD_LO", 0.01),
  PLANE_GOOD_HI: num("VITE_PLANE_GOOD_HI", 0.06),
  NOSE_GOOD_LO: num("VITE_NOSE_GOOD_LO", 0.07),
  NOSE_GOOD_HI: num("VITE_NOSE_GOOD_HI", 0.20),

  // Calibration
  CALIB_SECONDS: num("VITE_CALIB_SECONDS", 1.0),

  // Passport capture
  NEUTRAL_CAPTURE_DELAY_MS: num("VITE_NEUTRAL_CAPTURE_DELAY_MS", 1000),
  NEUTRAL_YAW: num("VITE_NEUTRAL_YAW", 0.06),
  NEUTRAL_MAR_DELTA: num("VITE_NEUTRAL_MAR_DELTA", 0.04),
  NEUTRAL_TIMEOUT_MS: num("VITE_NEUTRAL_TIMEOUT_MS", 2500),
  NEUTRAL_HOLD_FRAMES: num("VITE_NEUTRAL_HOLD_FRAMES", 5),

  // Pass thresholds
  PASS_THRESH_LIVE: num("VITE_PASS_THRESH_LIVE", 75),
  PASS_THRESH_PAD: num("VITE_PASS_THRESH_PAD", 70),

  // Blink quality
  BLINK_AMP_GOOD_LOW: num("VITE_BLINK_AMP_GOOD_LOW", 0.05),
  BLINK_AMP_GOOD_HIGH: num("VITE_BLINK_AMP_GOOD_HIGH", 0.12),
  BLINK_MIN_AMP: num("VITE_BLINK_MIN_AMP", 0.015),
  BLINK_MIN_INTERVAL_MS: num("VITE_BLINK_MIN_INTERVAL_MS", 220),

  // Face alignment gate (NEW)
  FACE_ALIGN_ENABLED: bool("VITE_FACE_ALIGN_ENABLED", true),
  FACE_ALIGN_RADIUS_RATIO: num("VITE_FACE_ALIGN_RADIUS_RATIO", 0.32), // circle radius as fraction of min(canvasW, canvasH)
  FACE_ALIGN_CENTER_TOL_RATIO: num("VITE_FACE_ALIGN_CENTER_TOL_RATIO", 0.35), // allowed offset in radii
  FACE_ALIGN_EYE_DIST_MIN: num("VITE_FACE_ALIGN_EYE_DIST_MIN", 0.26), // eye distance / circle diameter
  FACE_ALIGN_EYE_DIST_MAX: num("VITE_FACE_ALIGN_EYE_DIST_MAX", 0.42),
  FACE_ALIGN_YAW_MAX: num("VITE_FACE_ALIGN_YAW_MAX", 0.22), // ~12.6°
  FACE_ALIGN_HOLD_FRAMES: num("VITE_FACE_ALIGN_HOLD_FRAMES", 18),
}

// Derived frame counts
export const FRAMES = {
  HOLD_FRAMES_TURN: Math.max(6, Math.round(CFG.HOLD_SEC_LEFT_RIGHT * 60)),
  HOLD_FRAMES: Math.max(6, Math.round(CFG.HOLD_SEC_SMILE * 60)),
}
