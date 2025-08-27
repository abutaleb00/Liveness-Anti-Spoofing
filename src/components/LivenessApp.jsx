import React, { useEffect, useRef, useState, useMemo } from "react";
import { motion, AnimatePresence } from "framer-motion";

// ---- TFJS + backends ----
import * as tf from "@tensorflow/tfjs";
import "@tensorflow/tfjs-backend-webgl";
import "@tensorflow/tfjs-backend-wasm";
import { setWasmPaths } from "@tensorflow/tfjs-backend-wasm";
setWasmPaths("https://cdn.jsdelivr.net/npm/@tensorflow/tfjs-backend-wasm@4.20.0/dist/");

// ---- Face landmarks model wrapper (supports tfjs & mediapipe runtimes) ----
import * as faceLandmarksDetection from "@tensorflow-models/face-landmarks-detection";

/* ----------------- utils ----------------- */
const dist = (a, b) => Math.hypot(a.x - b.x, a.y - b.y);
const clamp = (v, min, max) => Math.max(min, Math.min(max, v));
const lerp = (a, b, t) => a + (b - a) * t;
const scale01 = (x, lo, hi) => clamp((x - lo) / (hi - lo + 1e-12), 0, 1);
const percentile = (arr, p) => {
  if (!arr.length) return 0;
  const a = [...arr].sort((x, y) => x - y);
  const idx = clamp((a.length - 1) * p, 0, a.length - 1);
  const lo = Math.floor(idx), hi = Math.ceil(idx), t = idx - lo;
  return a[lo] * (1 - t) + a[hi] * t;
};
function shuffle(arr) { const a = [...arr]; for (let i = a.length - 1; i > 0; i--) { const j = Math.floor(Math.random() * (i + 1)); [a[i], a[j]] = [a[j], a[i]]; } return a; }

/* FaceMesh indices we use */
const IDX = {
  leftEyeOuter: 33, rightEyeOuter: 263,
  L: { left: 33, right: 133, top1: 159, top2: 160, bot1: 145, bot2: 144 },
  R: { left: 362, right: 263, top1: 386, top2: 385, bot1: 374, bot2: 380 },
  mouthLeft: 61, mouthRight: 291, mouthTop: 13, mouthBot: 14,
  cheekL: 234, cheekR: 454, noseTip: 4,
};
function eyeAspectRatio(pts, kp) {
  const p1 = pts[kp.left], p4 = pts[kp.right];
  const p2 = pts[kp.top1], p6 = pts[kp.bot1];
  const p3 = pts[kp.top2], p5 = pts[kp.bot2];
  const A = dist(p2, p6), B = dist(p3, p5), C = dist(p1, p4) + 1e-6;
  return (A + B) / (2 * C);
}
function mouthAspectRatio(pts) {
  const vertical = dist(pts[IDX.mouthTop], pts[IDX.mouthBot]);
  const horizontal = dist(pts[IDX.mouthLeft], pts[IDX.mouthRight]) + 1e-6;
  return vertical / horizontal;
}
// +rawYaw means head LEFT in camera space before mirroring
function rawYaw(pts) {
  const L = pts[IDX.cheekL], R = pts[IDX.cheekR];
  if (!L || !R || typeof L.z !== "number" || typeof R.z !== "number") return 0;
  return Math.atan2((L.z - R.z), Math.abs(R.x - L.x) + 1e-6);
}

/* ----------- depth features for PAD ----------- */
function planeResidualNorm(pts, keys, faceW) {
  const X = [], Z = [];
  for (const k of keys) {
    const p = pts[k]; if (!p || typeof p.z !== "number") continue;
    X.push([p.x / faceW, p.y / faceW, 1]); Z.push(p.z / faceW);
  }
  const n = X.length; if (n < 4) return 0;
  const xtx = [[0, 0, 0], [0, 0, 0], [0, 0, 0]], xtz = [0, 0, 0];
  for (let i = 0; i < n; i++) {
    const [x, y, o] = X[i], z = Z[i];
    xtx[0][0] += x * x; xtx[0][1] += x * y; xtx[0][2] += x * o;
    xtx[1][0] += y * x; xtx[1][1] += y * y; xtx[1][2] += y * o;
    xtx[2][0] += o * x; xtx[2][1] += o * y; xtx[2][2] += o * o;
    xtz[0] += x * z; xtz[1] += y * z; xtz[2] += o * z;
  }
  const m = xtx;
  const det = m[0][0] * (m[1][1] * m[2][2] - m[1][2] * m[2][1]) - m[0][1] * (m[1][0] * m[2][2] - m[1][2] * m[2][0]) + m[0][2] * (m[1][0] * m[2][1] - m[1][1] * m[2][0]);
  if (Math.abs(det) < 1e-9) return 0;
  const inv = [[0, 0, 0], [0, 0, 0], [0, 0, 0]];
  inv[0][0] = (m[1][1] * m[2][2] - m[1][2] * m[2][1]) / det;
  inv[0][1] = (m[0][2] * m[2][1] - m[0][1] * m[2][2]) / det;
  inv[0][2] = (m[0][1] * m[1][2] - m[0][2] * m[1][1]) / det;
  inv[1][0] = (m[1][2] * m[2][0] - m[1][0] * m[2][2]) / det;
  inv[1][1] = (m[0][0] * m[2][2] - m[0][2] * m[2][0]) / det;
  inv[1][2] = (m[0][2] * m[1][0] - m[0][0] * m[1][2]) / det;
  inv[2][0] = (m[1][0] * m[2][1] - m[1][1] * m[2][0]) / det;
  inv[2][1] = (m[0][1] * m[2][0] - m[0][0] * m[2][1]) / det;
  inv[2][2] = (m[0][0] * m[1][1] - m[0][1] * m[1][0]) / det;
  const beta = [
    inv[0][0] * xtz[0] + inv[0][1] * xtz[1] + inv[0][2] * xtz[2],
    inv[1][0] * xtz[0] + inv[1][1] * xtz[1] + inv[1][2] * xtz[2],
    inv[2][0] * xtz[0] + inv[2][1] * xtz[1] + inv[2][2] * xtz[2],
  ];
  let se = 0;
  for (let i = 0; i < n; i++) {
    const [x, y, o] = X[i], z = Z[i];
    const zhat = beta[0] * x + beta[1] * y + beta[2] * o;
    const r = z - zhat; se += r * r;
  }
  const rmse = Math.sqrt(se / n);
  return rmse;
}
function extractDepthFeatures(pts) {
  const keys = [IDX.noseTip, IDX.cheekL, IDX.cheekR, IDX.leftEyeOuter, IDX.rightEyeOuter, IDX.mouthTop, IDX.mouthBot];
  const faceW = dist(pts[IDX.leftEyeOuter], pts[IDX.rightEyeOuter]) + 1e-6;
  const Z = keys.map(k => pts[k]?.z).filter(z => typeof z === "number");
  if (Z.length < 4) return { zRangeN: 0, planeResN: 0, noseProtrusionN: 0 };
  const zRangeN = (Math.max(...Z) - Math.min(...Z)) / faceW;
  const planeResN = planeResidualNorm(pts, keys, faceW);
  const nose = pts[IDX.noseTip].z, cheeks = (pts[IDX.cheekL].z + pts[IDX.cheekR].z) / 2;
  const noseProtrusionN = Math.abs(nose - cheeks) / (Math.abs(Math.max(...Z) - Math.min(...Z)) + 1e-6);
  return { zRangeN, planeResN, noseProtrusionN };
}

/* --------------- Bilingual UI text --------------- */
export const INSTRUCTIONS = {
  blink: {
    en: "Please blink your eyes naturally 3 times (one after another)",
    bn: "অনুগ্রহ করে স্বাভাবিকভাবে ৩ বার চোখের পলক দিন (একের পর এক)",
    titleEn: "Blink",
    titleBn: "চোখের পলক",
    icon: "👀"
  },
  smile: {
    en: "Please smile naturally and hold for 1 second",
    bn: "অনুগ্রহ করে স্বাভাবিকভাবে হাসুন এবং ১ সেকেন্ড ধরে রাখুন",
    titleEn: "Smile!",
    titleBn: "হাসুন!",
    icon: "😊"
  },
  left: {
    en: "Please turn your face CLEARLY to the left and hold for 2 seconds",
    bn: "অনুগ্রহ করে আপনার মুখ স্পষ্টভাবে বামে ঘুরান এবং ২ সেকেন্ড ধরে রাখুন",
    titleEn: "Turn Left ←",
    titleBn: "বামে তাকান ←",
    icon: "↩️"
  },
  right: {
    en: "Please turn your face CLEARLY to the right and hold for 2 seconds",
    bn: "অনুগ্রহ করে আপনার মুখ স্পষ্টভাবে ডানে ঘুরান এবং ২ সেকেন্ড ধরে রাখুন",
    titleEn: "Turn Right →",
    titleBn: "ডানে তাকান →",
    icon: "↪️"
  },
};
const STEP_BASE = ["left", "right", "smile", "blink"];

/* ---------- Smile helpers ---------- */
function mouthFeatures(pts) {
  const faceW = dist(pts[IDX.leftEyeOuter], pts[IDX.rightEyeOuter]) + 1e-6;
  const wNorm = dist(pts[IDX.mouthLeft], pts[IDX.mouthRight]) / faceW;
  const hNorm = dist(pts[IDX.mouthTop], pts[IDX.mouthBot]) / faceW;
  const yCorners = (pts[IDX.mouthLeft].y + pts[IDX.mouthRight].y) / 2;
  const yCenter = (pts[IDX.mouthTop].y + pts[IDX.mouthBot].y) / 2;
  const curve = (yCenter - yCorners) / faceW;  // corners higher (smile) → positive
  return { wNorm, hNorm, curve };
}

/* ---------- Component ---------- */
export default function LivenessApp() {
  const videoRef = useRef(null);
  const canvasRef = useRef(null);
  const rafRef = useRef(0);
  const modelRef = useRef(null);

  const runtimeRef = useRef("tfjs");         // "tfjs" | "mediapipe"
  const noFaceFramesRef = useRef(0);         // for auto-fallback

  const [modelReady, setModelReady] = useState(false);
  const [cameraOn, setCameraOn] = useState(false);
  const [cameraReady, setCameraReady] = useState(false);

  const [phase, setPhase] = useState("calibrate");
  const phaseRef = useRef("calibrate");
  useEffect(() => { phaseRef.current = phase; }, [phase]);

  const [currentStep, setCurrentStep] = useState(0);
  const [steps, setSteps] = useState(() => shuffle(STEP_BASE));
  const stepsRef = useRef(steps); useEffect(() => { stepsRef.current = steps; }, [steps]);
  const currentStepRef = useRef(0);

  const [calibPct, setCalibPct] = useState(0);
  const [holdPct, setHoldPct] = useState(0);
  const [blinkCount, setBlinkCount] = useState(0);
  const [livenessScore, setLivenessScore] = useState(0);
  const [spoofScore, setSpoofScore] = useState(0);
  const [finalPass, setFinalPass] = useState(null);
  const finalPassRef = useRef(null); useEffect(() => { finalPassRef.current = finalPass; }, [finalPass]);
  const [error, setError] = useState("");
  const [faceCount, setFaceCount] = useState(0);
  const [photo, setPhoto] = useState(null);

  const blinkRef = useRef(0);
  const blinkTargetRef = useRef(3);
  const smooth = useRef({ EAR: 0, MAR: 0, YAW: 0, depthVar: 0, WNORM: 0, HNORM: 0, CURVE: 0 });

  const lastBBoxRef = useRef(null); // for passport crop

  const stateRef = useRef({
    prevEyesClosed: false,
    holdFrames: 0,
    depthSamples: [],

    calib: {
      samples: 0, secs: 0,
      earSum: 0, marSum: 0, mwSum: 0, curveSum: 0,
      ear: 0, mar: 0, mwBase: 0, curveBase: 0,
      EAR_OPEN_DYN: 0.21, EAR_CLOSED_DYN: 0.17
    },

    stepStartTime: performance.now(),
    stepStartBlink: 0,
    latencies: [],

    // liveness quality
    blinkAmps: [], lastOpenEAR: 0, closing: false, minEAR: 1, lastBlinkTs: 0,
    holdGoodFrames: 0, holdTotalFrames: 0,

    passed: { left: false, right: false, smile: false, blink: false },

    postCapture: { pending: false, due: 0, hold: 0, taken: false },
  });

  const finalizeScheduledRef = useRef(false);

  const cfg = useMemo(() => ({
    EAR_OPEN: 0.21, EAR_CLOSED: 0.17,
    MAR_SMILE: 0.28, SMILE_MAR_DELTA: 0.05, SMILE_W_DELTA: 0.035, SMILE_CURVE_DELTA: 0.012, SMILE_MAR_DELTA2: 0.06,
    YAW_FLIP: true, YAW_RAD: 0.10, YAW_MARGIN: 0.02,
    HOLD_FRAMES: 18, HOLD_FRAMES_TURN: 24, MIN_STEP_TIME_S: 0.35,
    BLINK_TARGET: 3, RESPONSE_MIN_S: 0.18, RESPONSE_MAX_S: 3.0,
    BLINK_AMP_GOOD: [0.05, 0.12], BLINK_MIN_INTERVAL_MS: 250, BLINK_MIN_AMP: 0.045,
    ZRANGE_GOOD: [0.02, 0.09], PLANE_GOOD: [0.01, 0.06], NOSE_GOOD: [0.07, 0.20],
    CALIB_SECONDS: 1.0,
    NEUTRAL_CAPTURE_DELAY_MS: 1000, NEUTRAL_YAW: 0.06, NEUTRAL_HOLD_FRAMES: 5, NEUTRAL_MAR_DELTA: 0.04, NEUTRAL_TIMEOUT_MS: 2500,
  }), []);

  /* ---------- detector builders & init ---------- */
  async function buildDetector(runtime = "tfjs") {
    if (modelRef.current?.dispose) {
      try { await modelRef.current.dispose(); } catch {}
    }

    if (runtime === "tfjs") {
      // Prefer WebGL → WASM → CPU (as last resort)
      const order = ["webgl", "wasm", "cpu"];
      for (const b of order) {
        try { await tf.setBackend(b); await tf.ready(); if (tf.getBackend() === b) break; } catch {}
      }
      console.info("[TFJS] backend:", tf.getBackend());

      modelRef.current = await faceLandmarksDetection.createDetector(
        faceLandmarksDetection.SupportedModels.MediaPipeFaceMesh,
        { runtime: "tfjs", refineLandmarks: true, maxFaces: 1 }
      );
      runtimeRef.current = "tfjs";
    } else {
      modelRef.current = await faceLandmarksDetection.createDetector(
        faceLandmarksDetection.SupportedModels.MediaPipeFaceMesh,
        {
          runtime: "mediapipe",
          refineLandmarks: true,
          maxFaces: 1,
          solutionPath: "https://cdn.jsdelivr.net/npm/@mediapipe/face_mesh"
        }
      );
      runtimeRef.current = "mediapipe";
      console.info("[MP] detector created via mediapipe runtime");
    }
  }

  /* model load */
  useEffect(() => {
    let cancelled = false;
    (async () => {
      try {
        await tf.ready();
        // Try TFJS first
        await buildDetector("tfjs");
        if (!cancelled) setModelReady(true);
      } catch (e) {
        console.error("Detector init failed on TFJS:", e);
        try {
          await buildDetector("mediapipe");
          if (!cancelled) setModelReady(true);
        } catch (e2) {
          console.error("Detector init failed on MediaPipe:", e2);
          setError(String(e2));
        }
      }
    })();

    return () => { cancelled = true; cancelAnimationFrame(rafRef.current); };
  }, []); // eslint-disable-line

  /* camera control */
  async function startCamera() {
    try {
      if (!modelReady) return;
      const v = videoRef.current;
      const stream = await navigator.mediaDevices.getUserMedia({
        video: { facingMode: "user", width: { ideal: 1280 }, height: { ideal: 720 } },
        audio: false
      });
      v.srcObject = stream;
      setCameraReady(false); await v.play();
      setCameraOn(true); setCameraReady(true);
      hardReset(); setPhase("calibrate");
    } catch (e) { console.error(e); setError(String(e)); }
  }
  async function stopCamera() {
    try {
      const v = videoRef.current;
      const stream = v?.srcObject;
      if (stream) stream.getTracks().forEach(t => t.stop());
      if (v) v.srcObject = null;
      cancelAnimationFrame(rafRef.current);
      setCameraOn(false); setCameraReady(false); setFaceCount(0);
      hardReset();
      const c = canvasRef.current; if (c) { const ctx = c.getContext("2d"); ctx.clearRect(0, 0, c.width, c.height); }
    } catch (e) { console.error(e); setError(String(e)); }
  }
  function restart() { if (cameraOn) softReset(); }

  /* resets */
  function softReset() {
    const newSteps = shuffle(STEP_BASE);
    setSteps(newSteps); stepsRef.current = newSteps;
    setCurrentStep(0); currentStepRef.current = 0;
    setHoldPct(0); setBlinkCount(0);
    setLivenessScore(0); setSpoofScore(0); setFinalPass(null); finalPassRef.current = null;
    setPhoto(null);
    blinkRef.current = 0; blinkTargetRef.current = 2 + Math.floor(Math.random() * 3);
    const st = stateRef.current;
    st.prevEyesClosed = false; st.holdFrames = 0;
    st.depthSamples = []; st.latencies = [];
    st.blinkAmps = []; st.lastOpenEAR = 0; st.closing = false; st.minEAR = 1; st.lastBlinkTs = 0;
    st.holdGoodFrames = 0; st.holdTotalFrames = 0;
    st.passed = { left: false, right: false, smile: false, blink: false };
    st.stepStartTime = performance.now(); st.stepStartBlink = 0;
    st.postCapture = { pending: false, due: 0, hold: 0, taken: false };
    finalizeScheduledRef.current = false;
    noFaceFramesRef.current = 0;
  }
  function hardReset() {
    const newSteps = shuffle(STEP_BASE);
    setSteps(newSteps); stepsRef.current = newSteps;
    setCurrentStep(0); currentStepRef.current = 0;
    setHoldPct(0); setBlinkCount(0);
    setLivenessScore(0); setSpoofScore(0); setFinalPass(null); finalPassRef.current = null;
    setPhoto(null); setCalibPct(0);
    blinkRef.current = 0; blinkTargetRef.current = 3;
    stateRef.current = {
      prevEyesClosed: false, holdFrames: 0,
      depthSamples: [],
      calib: { samples: 0, secs: 0, earSum: 0, marSum: 0, mwSum: 0, curveSum: 0, ear: 0, mar: 0, mwBase: 0, curveBase: 0, EAR_OPEN_DYN: 0.21, EAR_CLOSED_DYN: 0.17 },
      stepStartTime: performance.now(), stepStartBlink: 0,
      latencies: [],
      blinkAmps: [], lastOpenEAR: 0, closing: false, minEAR: 1, lastBlinkTs: 0,
      holdGoodFrames: 0, holdTotalFrames: 0,
      passed: { left: false, right: false, smile: false, blink: false },
      postCapture: { pending: false, due: 0, hold: 0, taken: false },
    };
    finalizeScheduledRef.current = false;
    noFaceFramesRef.current = 0;
  }

  /* ----- PAD + reasons ----- */
  function computePadFromSamples(st) {
    const arr = st.depthSamples;
    let depthScore = 0;
    if (arr.length >= 12) {
      const zRangeN_p75 = percentile(arr.map(o => o.zRangeN), 0.75);
      const planeN_p25 = percentile(arr.map(o => o.planeResN), 0.25);
      const noseN_p50 = percentile(arr.map(o => o.noseProtrusionN), 0.50);
      const sZ = scale01(zRangeN_p75, cfg.ZRANGE_GOOD[0], cfg.ZRANGE_GOOD[1]);
      const sPlan = scale01(planeN_p25, cfg.PLANE_GOOD[0], cfg.PLANE_GOOD[1]);
      const sNose = scale01(noseN_p50, cfg.NOSE_GOOD[0], cfg.NOSE_GOOD[1]);
      depthScore = (0.5 * sZ + 0.3 * sPlan + 0.2 * sNose) * 100;
    }
    const totalSteps = 4;
    const passedSteps = ["left", "right", "smile", "blink"].reduce((a, k) => a + (st.passed[k] ? 1 : 0), 0);
    const actionQuality = (passedSteps / totalSteps) * 100;
    const timely = st.latencies.filter(t => t >= cfg.RESPONSE_MIN_S && t <= cfg.RESPONSE_MAX_S).length;
    const respScore = (st.latencies.length > 0) ? (timely / st.latencies.length) * 100 : 0;
    const pad = 0.60 * depthScore + 0.25 * respScore + 0.15 * actionQuality;
    return Math.round(pad);
  }
  function padReasons(st) {
    const arr = st.depthSamples; const reasons = [];
    if (arr.length >= 12) {
      const zRangeN_p75 = percentile(arr.map(o => o.zRangeN), 0.75);
      const planeN_p25 = percentile(arr.map(o => o.planeResN), 0.25);
      const noseN_p50 = percentile(arr.map(o => o.noseProtrusionN), 0.50);
      if (zRangeN_p75 < cfg.ZRANGE_GOOD[0]) reasons.push("Flat depth (possible screen)");
      if (planeN_p25 < cfg.PLANE_GOOD[0]) reasons.push("Near-planar surface");
      if (noseN_p50 < cfg.NOSE_GOOD[0]) reasons.push("Weak nose protrusion");
    }
    const timely = stateRef.current.latencies.filter(t => t >= cfg.RESPONSE_MIN_S && t <= cfg.RESPONSE_MAX_S).length;
    if (stateRef.current.latencies.length > 0 && (timely / stateRef.current.latencies.length) < 0.7) reasons.push("Prompt timing mismatch");
    return reasons;
  }

  /* ---------- Continuous liveness ---------- */
  function computeLivenessComposite(st) {
    const totalSteps = 4;
    const passedSteps = ["left", "right", "smile", "blink"].reduce((a, k) => a + (st.passed[k] ? 1 : 0), 0);
    let progressCurrent = 0;
    const step = stepsRef.current[currentStepRef.current];
    if (step) {
      if (step === "blink") {
        const blinksInStep = blinkRef.current - st.stepStartBlink;
        const target = blinkTargetRef.current || cfg.BLINK_TARGET;
        progressCurrent = clamp(blinksInStep / target, 0, 1);
      } else {
        progressCurrent = clamp(holdPct / 100, 0, 1);
      }
    }
    const stepScore = ((passedSteps + progressCurrent) / totalSteps) * 100;

    const timely = st.latencies.filter(t => t >= cfg.RESPONSE_MIN_S && t <= cfg.RESPONSE_MAX_S).length;
    const timingScore = (st.latencies.length > 0) ? (timely / st.latencies.length) * 100 : 65;

    let blinkScore = 60;
    if (st.blinkAmps.length) {
      const med = percentile(st.blinkAmps, 0.5);
      blinkScore = scale01(med, cfg.BLINK_AMP_GOOD[0], cfg.BLINK_AMP_GOOD[1]) * 100;
    }

    const smoothness = st.holdTotalFrames ? (st.holdGoodFrames / st.holdTotalFrames) : 0.65;
    const smoothnessScore = smoothness * 100;

    const live = 0.50 * stepScore + 0.22 * timingScore + 0.14 * blinkScore + 0.14 * smoothnessScore;
    return Math.round(clamp(live, 0, 100));
  }

  /* -------- RAF loop -------- */
  useEffect(() => {
    if (!cameraOn) return;
    const loop = async () => {
      await tick();
      setLivenessScore(computeLivenessComposite(stateRef.current));
      rafRef.current = requestAnimationFrame(loop);
    };
    rafRef.current = requestAnimationFrame(loop);
    return () => cancelAnimationFrame(rafRef.current);
  }, [cameraOn]); // eslint-disable-line

  /* -------- main tick -------- */
  async function tick() {
    const v = videoRef.current, c = canvasRef.current;
    if (!v || !c || !modelRef.current || v.readyState < 2) return;
    if (!v.videoWidth || !v.videoHeight) return;

    const ctx = c.getContext("2d");
    c.width = v.videoWidth; c.height = v.videoHeight;

    let faces = [];
    try {
      faces = await modelRef.current.estimateFaces(v, { flipHorizontal: true, staticImageMode: false });
    } catch (e) {
      console.error("estimateFaces error:", e);
      // If TFJS path threw, try switching to MediaPipe
      if (runtimeRef.current === "tfjs") {
        try { await buildDetector("mediapipe"); return; } catch {}
      }
      return;
    }

    ctx.drawImage(v, 0, 0, c.width, c.height);

    const nFaces = faces?.length || 0;
    if (nFaces !== faceCount) setFaceCount(nFaces);

    // Auto-fallback: TFJS returns 0 faces for a while → try MediaPipe runtime
    if (!nFaces) {
      if (runtimeRef.current === "tfjs") {
        noFaceFramesRef.current++;
        if (noFaceFramesRef.current > 40) { // ~0.7s at 60fps
          console.info("No faces on TFJS → switching to MediaPipe runtime");
          await buildDetector("mediapipe");
          noFaceFramesRef.current = 0;
        }
      }
      return;
    } else {
      noFaceFramesRef.current = 0;
    }

    const face = faces[0];
    const pts2D = face.keypoints;
    const pts = face.keypoints3D ?? face.keypoints;

    // bbox for passport capture
    if (pts2D && pts2D.length) {
      let minX = 1e9, minY = 1e9, maxX = -1e9, maxY = -1e9;
      for (const p of pts2D) { if (p.x < minX) minX = p.x; if (p.y < minY) minY = p.y; if (p.x > maxX) maxX = p.x; if (p.y > maxY) maxY = p.y; }
      lastBBoxRef.current = { x: minX, y: minY, w: maxX - minX, h: maxY - minY };
    }

    const EAR = (eyeAspectRatio(pts, IDX.L) + eyeAspectRatio(pts, IDX.R)) / 2;
    const MAR = mouthAspectRatio(pts);
    const YAW = (cfg.YAW_FLIP ? -1 : 1) * rawYaw(pts);
    const { wNorm, hNorm, curve } = mouthFeatures(pts);

    const a = 0.2;
    smooth.current.EAR = lerp(smooth.current.EAR || EAR, EAR, a);
    smooth.current.MAR = lerp(smooth.current.MAR || MAR, MAR, a);
    smooth.current.YAW = lerp(smooth.current.YAW || YAW, YAW, a);
    smooth.current.WNORM = lerp(smooth.current.WNORM || wNorm, wNorm, a);
    smooth.current.HNORM = lerp(smooth.current.HNORM || hNorm, hNorm, a);
    smooth.current.CURVE = lerp(smooth.current.CURVE || curve, curve, a);

    drawMesh(ctx, face);

    if (phaseRef.current === "calibrate") {
      const st = stateRef.current;
      st.calib.samples += 1; st.calib.secs += (1 / 60);
      st.calib.earSum += smooth.current.EAR;
      st.calib.marSum += smooth.current.MAR;
      st.calib.mwSum += smooth.current.WNORM;
      st.calib.curveSum += smooth.current.CURVE;

      const pct = Math.min(100, Math.round((st.calib.secs / cfg.CALIB_SECONDS) * 100)); setCalibPct(pct);
      if (st.calib.secs >= cfg.CALIB_SECONDS) {
        st.calib.ear = st.calib.earSum / st.calib.samples;
        st.calib.mar = st.calib.marSum / st.calib.samples;
        st.calib.mwBase = st.calib.mwSum / st.calib.samples;
        st.calib.curveBase = st.calib.curveSum / st.calib.samples;
        st.calib.EAR_CLOSED_DYN = Math.max(cfg.EAR_CLOSED, st.calib.ear - 0.06);
        st.calib.EAR_OPEN_DYN = Math.max(cfg.EAR_OPEN, st.calib.ear - 0.02);
        setPhase("run");
        st.stepStartTime = performance.now();
        st.stepStartBlink = blinkRef.current;
        blinkTargetRef.current = 2 + Math.floor(Math.random() * 3);
      }
      return;
    }

    updateBlink(EAR);
    updateStepProgress();
    updateDepthSamples(pts);
    setSpoofScore(computePadFromSamples(stateRef.current));

    // neutral capture
    const st = stateRef.current;
    const done = currentStepRef.current >= stepsRef.current.length;
    if (done && st.postCapture.pending && !st.postCapture.taken) {
      const now = performance.now();
      const due = st.postCapture.due;
      if (now >= due) {
        const neutralYaw = Math.abs(smooth.current.YAW) < cfg.NEUTRAL_YAW;
        const neutralSmile = (smooth.current.MAR < st.calib.mar + cfg.NEUTRAL_MAR_DELTA);
        const eyesOk = smooth.current.EAR > (st.calib.ear - 0.03);
        if (neutralYaw && neutralSmile && eyesOk) {
          st.postCapture.hold++;
          if (st.postCapture.hold >= cfg.NEUTRAL_HOLD_FRAMES) {
            const shot = await capturePassport();
            setPhoto(shot?.data_url || null);
            st.postCapture.taken = true;
          }
        } else {
          st.postCapture.hold = Math.max(0, st.postCapture.hold - 1);
          if (now - due > cfg.NEUTRAL_TIMEOUT_MS) {
            const shot = await capturePassport();
            setPhoto(shot?.data_url || null);
            st.postCapture.taken = true;
          }
        }
      }
    }
  }

  function updateBlink(EAR) {
    const st = stateRef.current;
    const eyesClosed = EAR < (st.calib?.EAR_CLOSED_DYN ?? cfg.EAR_CLOSED);
    const reOpen = EAR > (st.calib?.EAR_OPEN_DYN ?? cfg.EAR_OPEN);

    if (!eyesClosed) st.lastOpenEAR = EAR;

    if (eyesClosed && !st.prevEyesClosed) { st.prevEyesClosed = true; st.closing = true; st.minEAR = EAR; }
    else if (eyesClosed && st.closing) { if (EAR < st.minEAR) st.minEAR = EAR; }

    if (st.prevEyesClosed && reOpen) {
      st.prevEyesClosed = false;
      if (st.closing) {
        const open = st.lastOpenEAR || st.calib.ear || EAR;
        const amp = Math.max(0, open - st.minEAR);
        const now = performance.now(); const dt = now - (st.lastBlinkTs || 0);
        if (amp >= cfg.BLINK_MIN_AMP && dt >= cfg.BLINK_MIN_INTERVAL_MS) {
          st.blinkAmps.push(amp); if (st.blinkAmps.length > 20) st.blinkAmps.shift();
          st.lastBlinkTs = now;
          blinkRef.current += 1; setBlinkCount(b => b + 1);
        }
      }
      st.closing = false;
    }
  }

  function getHoldFramesForStep(step) { return (step === "left" || step === "right") ? cfg.HOLD_FRAMES_TURN : cfg.HOLD_FRAMES; }

  function updateStepProgress() {
    const st = stateRef.current;
    const step = stepsRef.current[currentStepRef.current];
    if (!step) return;

    const timeSinceStart = (performance.now() - st.stepStartTime) / 1000;
    const minGate = Math.max(cfg.RESPONSE_MIN_S, cfg.MIN_STEP_TIME_S);

    let ok = false;
    if (step === "left") { ok = smooth.current.YAW > (cfg.YAW_RAD + cfg.YAW_MARGIN); }
    else if (step === "right") { ok = smooth.current.YAW < -(cfg.YAW_RAD + cfg.YAW_MARGIN); }
    else if (step === "smile") {
      const wDelta = (smooth.current.WNORM || 0) - (st.calib.mwBase || 0);
      const curveDelta = (smooth.current.CURVE || 0) - (st.calib.curveBase || 0);
      const marDelta = (smooth.current.MAR || 0) - (st.calib.mar || 0);
      ok = wDelta >= cfg.SMILE_W_DELTA || curveDelta >= cfg.SMILE_CURVE_DELTA || marDelta >= cfg.SMILE_MAR_DELTA2 || (smooth.current.MAR || 0) > Math.max(cfg.MAR_SMILE, st.calib.mar + cfg.SMILE_MAR_DELTA);
    } else if (step === "blink") {
      const blinksInStep = blinkRef.current - st.stepStartBlink;
      const target = blinkTargetRef.current || cfg.BLINK_TARGET;
      setHoldPct(clamp((blinksInStep / target) * 100, 0, 100));
      if (blinksInStep >= target && timeSinceStart >= minGate) {
        st.passed.blink = true;
        const latency = (performance.now() - st.stepStartTime) / 1000;
        st.latencies.push(latency);
        nextStep();
      }
      return;
    }

    // steadiness and progress
    if (ok) st.holdGoodFrames++;
    st.holdTotalFrames++;

    const neededHold = getHoldFramesForStep(step);
    if (timeSinceStart < minGate) { setHoldPct(clamp((st.holdFrames / neededHold) * 100, 0, 100)); return; }

    if (ok) st.holdFrames++;
    else st.holdFrames = Math.max(0, st.holdFrames - 1);

    setHoldPct(clamp((st.holdFrames / neededHold) * 100, 0, 100));

    if (st.holdFrames >= neededHold) {
      st.holdFrames = 0;
      st.passed[step] = true;
      const latency = (performance.now() - st.stepStartTime) / 1000;
      st.latencies.push(latency);
      nextStep();
    }
  }

  function nextStep() {
    const st = stateRef.current;
    currentStepRef.current += 1;
    setCurrentStep(i => {
      const n = i + 1;
      if (n >= stepsRef.current.length) {
        const pass = (computePadFromSamples(st) >= 70) && (computeLivenessComposite(st) >= 75);
        setFinalPass(pass); finalPassRef.current = pass;
        st.postCapture = { pending: true, due: performance.now() + cfg.NEUTRAL_CAPTURE_DELAY_MS, hold: 0, taken: false };
        scheduleFinalizeResults();
      } else {
        st.stepStartTime = performance.now();
        st.stepStartBlink = blinkRef.current;
        st.holdFrames = 0;
      }
      return n;
    });
  }

  function scheduleFinalizeResults() {
    if (finalizeScheduledRef.current) return;
    finalizeScheduledRef.current = true;
    setTimeout(() => {
      const st = stateRef.current;
      const padNow = computePadFromSamples(st);
      setSpoofScore(padNow);
      const passNow = padNow >= 70 && (computeLivenessComposite(st) >= 75);
      setFinalPass(passNow); finalPassRef.current = passNow;
    }, 700);
  }

  function updateDepthSamples(pts) {
    const f = extractDepthFeatures(pts);
    stateRef.current.depthSamples.push(f);
    if (stateRef.current.depthSamples.length > 200) stateRef.current.depthSamples.shift();
    smooth.current.depthVar = f.zRangeN || 0;
  }

  /* ------------- drawing / overlays ------------- */
  function drawMesh(ctx, face) {
    const kp = face.keypoints;
    ctx.save();
    ctx.fillStyle = "rgba(45,212,191,0.5)";
    for (let i = 0; i < kp.length; i += 8) { const p = kp[i]; ctx.beginPath(); ctx.arc(p.x, p.y, 1.6, 0, Math.PI * 2); ctx.fill(); }
    ctx.restore();
  }

  /* ---------- Passport capture (no UI overlays) ---------- */
  async function capturePassport() {
    const v = videoRef.current;
    if (!v || v.readyState < 2) return { data_url: null, base64: null };

    const frame = await grabBestFrameCanvas(v); // mirrored already
    const ow = frame.width, oh = frame.height;

    const box = lastBBoxRef.current || { x: ow * 0.25, y: oh * 0.20, w: ow * 0.50, h: oh * 0.60 };
    const padLeft = box.w * 0.50;
    const padRight = box.w * 0.50;
    const padTop = box.h * 1.00;
    const padBottom = box.h * 0.60;

    let sx = Math.max(0, Math.floor(box.x - padLeft));
    let sy = Math.max(0, Math.floor(box.y - padTop));
    let sw = Math.min(ow - sx, Math.ceil(box.w + padLeft + padRight));
    let sh = Math.min(oh - sy, Math.ceil(box.h + padTop + padBottom));

    const targetAR = 3 / 4;
    const curAR = sw / sh;
    if (curAR > targetAR) {
      const newH = Math.round(sw / targetAR);
      const delta = newH - sh;
      sy = Math.max(0, sy - Math.round(delta * 0.60));
      sh = Math.min(oh - sy, newH);
    } else {
      const newW = Math.round(sh * targetAR);
      const delta = newW - sw;
      sx = Math.max(0, sx - Math.round(delta / 2));
      sw = Math.min(ow - sx, newW);
    }

    const outW = 900, outH = 1200;
    const out = document.createElement("canvas");
    out.width = outW; out.height = outH;
    const outctx = out.getContext("2d");
    outctx.imageSmoothingEnabled = true;
    outctx.imageSmoothingQuality = "high";
    outctx.drawImage(frame, sx, sy, sw, sh, 0, 0, outW, outH);

    enhanceCanvasInPlace(outctx, outW, outH);

    const dataUrl = out.toDataURL("image/jpeg", 0.96);
    return { data_url: dataUrl, base64: dataUrl.split(",")[1] };
  }

  async function grabBestFrameCanvas(video) {
    const makeCanvas = (w, h) => { const c = document.createElement("canvas"); c.width = w; c.height = h; return c; };
    try {
      const track = video.srcObject?.getVideoTracks?.()[0];
      if (track && "ImageCapture" in window) {
        const ic = new window.ImageCapture(track);
        const bmp = await ic.grabFrame();
        const c = makeCanvas(bmp.width, bmp.height);
        const cx = c.getContext("2d");
        cx.save(); cx.translate(c.width, 0); cx.scale(-1, 1);
        cx.drawImage(bmp, 0, 0);
        cx.restore();
        return c;
      }
    } catch { /* fallback */ }

    const c = makeCanvas(video.videoWidth, video.videoHeight);
    const cx = c.getContext("2d");
    cx.save(); cx.translate(c.width, 0); cx.scale(-1, 1);
    cx.drawImage(video, 0, 0);
    cx.restore();
    return c;
  }

  function enhanceCanvasInPlace(ctx, w, h) {
    const img = ctx.getImageData(0, 0, w, h);
    const d = img.data; const n = d.length;
    let rSum = 0, gSum = 0, bSum = 0, count = n / 4;
    for (let i = 0; i < n; i += 4) { rSum += d[i]; gSum += d[i + 1]; bSum += d[i + 2]; }
    const rMean = rSum / count, gMean = gSum / count, bMean = bSum / count;
    const avg = (rMean + gMean + bMean) / 3 || 1;
    const gR = avg / (rMean || 1), gG = avg / (gMean || 1), gB = avg / (bMean || 1);

    const hist = new Uint32Array(256);
    for (let i = 0; i < n; i += 4) {
      let r = clamp(Math.round(d[i] * gR), 0, 255);
      let g = clamp(Math.round(d[i + 1] * gG), 0, 255);
      let b = clamp(Math.round(d[i + 2] * gB), 0, 255);
      d[i] = r; d[i + 1] = g; d[i + 2] = b;
      const y = Math.round(0.2126 * r + 0.7152 * g + 0.0722 * b);
      hist[y]++;
    }
    const total = count;
    const lowCut = Math.round(total * 0.02), highCut = Math.round(total * 0.98);
    let acc = 0, low = 0, high = 255;
    for (let v = 0; v < 256; v++) { acc += hist[v]; if (acc >= lowCut) { low = v; break; } }
    acc = 0;
    for (let v = 255; v >= 0; v--) { acc += hist[v]; if (acc >= total - highCut) { high = v; break; } }
    const eps = Math.max(1, high - low);
    const gain = 255 / eps, bias = -low * gain;
    const gamma = 0.95;

    for (let i = 0; i < n; i += 4) {
      let r = clamp((d[i] * gain + bias), 0, 255);
      let g = clamp((d[i + 1] * gain + bias), 0, 255);
      let b = clamp((d[i + 2] * gain + bias), 0, 255);
      d[i] = Math.round(255 * Math.pow(r / 255, gamma));
      d[i + 1] = Math.round(255 * Math.pow(g / 255, gamma));
      d[i + 2] = Math.round(255 * Math.pow(b / 255, gamma));
    }
    ctx.putImageData(img, 0, 0);

    const src = ctx.getImageData(0, 0, w, h);
    const sb = src.data;
    const blur = new Uint8ClampedArray(sb.length);
    const k = [1, 2, 1, 2, 4, 2, 1, 2, 1];
    for (let yy = 1; yy < h - 1; yy++) {
      for (let xx = 1; xx < w - 1; xx++) {
        let r = 0, g = 0, b = 0, ki = 0;
        for (let j = -1; j <= 1; j++) {
          for (let i = -1; i <= 1; i++) {
            const idx = ((yy + j) * w + (xx + i)) << 2;
            const kv = k[ki++];
            r += sb[idx] * kv;
            g += sb[idx + 1] * kv;
            b += sb[idx + 2] * kv;
          }
        }
        const o = (yy * w + xx) << 2;
        blur[o] = r >> 4;
        blur[o + 1] = g >> 4;
        blur[o + 2] = b >> 4;
        blur[o + 3] = 255;
      }
    }
    const amount = 0.55;
    for (let y = 1; y < h - 1; y++) {
      for (let x = 1; x < w - 1; x++) {
        const i4 = (y * w + x) << 2;
        const r = sb[i4], g = sb[i4 + 1], b = sb[i4 + 2];
        const br = blur[i4], bg = blur[i4 + 1], bb = blur[i4 + 2];
        src.data[i4] = clamp(Math.round(r + amount * (r - br)), 0, 255);
        src.data[i4 + 1] = clamp(Math.round(g + amount * (g - bg)), 0, 255);
        src.data[i4 + 2] = clamp(Math.round(b + amount * (b - bb)), 0, 255);
      }
    }
    ctx.putImageData(src, 0, 0);
  }

  async function copySnapshotBase64() {
    const shot = await capturePassport();
    if (!shot?.base64) return;
    try { await navigator.clipboard.writeText(shot.base64); alert("Passport photo base64 copied."); }
    catch { console.log("Photo base64:", shot.base64.slice(0, 64) + "..."); }
  }

  const allDone = currentStep >= steps.length;
  const decided = finalPass !== null;
  const liveNow = livenessScore;
  const showPass = decided ? finalPass : (spoofScore >= 70 && liveNow >= 75);

  const activeStep = steps[currentStep];

  return (
    <div className="grid lg:grid-cols-5 gap-6">
      <div className="lg:col-span-3 space-y-3">
        <div className="relative rounded-2xl overflow-hidden card min-h-[360px]">
          {/* Hidden <video> keeps layout clean while we draw on the canvas */}
          <video
            ref={videoRef}
            className="w-[1px] h-[1px] absolute opacity-0 -z-10"
            muted
            playsInline
            autoPlay
          />
          <canvas ref={canvasRef} className="w-full h-auto block" />

          {/* Animated HUD */}
          <PromptHUD
            phase={phase}
            calibPct={calibPct}
            cameraOn={cameraOn}
            cameraReady={cameraReady}
            faces={faceCount}
            done={allDone}
            finalPass={finalPass}
            stepKey={activeStep}
          />

          {!cameraOn && (
            <div className="absolute inset-0 grid place-items-center bg-slate-900/40 backdrop-blur">
              <div className="p-8 text-center">
                <div className="text-xl font-semibold mb-2">Webcam is off</div>
                <div className="text-slate-300 mb-4">Click start to begin liveness verification.</div>
                <button className="btn" disabled={!modelReady} onClick={startCamera}>
                  {modelReady ? "Start verification" : "Loading model…"}
                </button>
                {error && <div className="text-rose-400 mt-3 text-sm">{error}</div>}
              </div>
            </div>
          )}

          {cameraOn && (
            <div className="absolute bottom-0 left-0 right-0 p-4 backdrop-blur bg-slate-900/40">
              <div className="progress"><div style={{ width: `${holdPct}%` }} /></div>
              <div className="mt-2 flex items-center justify-between text-xs text-slate-300">
                <span>Step {Math.min(currentStep + 1, steps.length)}/{steps.length}</span>
                <span>Blinks: <b className="text-slate-100">{blinkCount}</b></span>
              </div>
            </div>
          )}
        </div>

        {/* Controls */}
        <div className="flex items-center gap-2">
          {!cameraOn ? (
            ""
          ) : (
            <>
              <button className="btn" onClick={restart}>Restart</button>
              <button className="btn-outline" onClick={stopCamera}>Stop</button>
              <button className="btn-outline" onClick={copySnapshotBase64}>Copy Passport Base64</button>
            </>
          )}
        </div>

        {/* Results */}
        {cameraOn && allDone && (
          <div className={`card p-4 border ${showPass ? "border-emerald-600" : "border-rose-600"}`}>
            <div className="flex items-center justify-between">
              <div>
                <div className="text-lg font-semibold">Result</div>
                <div className="text-slate-300 text-sm">Liveness score & anti-spoof checks</div>
              </div>
              <div className="text-right">
                <div className="text-2xl font-bold">{showPass ? "PASSED" : "FAILED"}</div>
                <div className="text-xs text-slate-400">{decided ? "Reviewed" : "Processing…"}</div>
              </div>
            </div>
            <div className="grid grid-cols-2 md:grid-cols-4 gap-3 mt-4">
              <Metric label="Liveness Score" value={`${liveNow}/100`} ok={liveNow >= 75} />
              <Metric label="Anti-Spoof Score" value={`${spoofScore}/100`} ok={spoofScore >= 70} />
              <Metric label="Depth Var (zRangeN)" value={(smooth.current.depthVar || 0).toFixed(4)} ok={(smooth.current.depthVar || 0) >= 0.02} />
              <Metric label="Blink Count" value={blinkCount} ok={blinkCount >= 2} />
            </div>
          </div>
        )}

        {/* Captured passport photo */}
        {cameraOn && allDone && (
          <div className="card p-4">
            <div className="flex items-center justify-between">
              <div className="text-lg font-semibold">Captured Photo (Passport)</div>
              <button className="btn-outline" onClick={async () => setPhoto((await capturePassport())?.data_url || null)}>
                Retake
              </button>
            </div>
            <div className="mt-3 rounded-xl overflow-hidden border border-slate-800 bg-white">
              {photo ? (
                <img src={photo} alt="passport" className="w-full h-auto block" />
              ) : (
                <div className="p-6 text-slate-600 text-sm">
                  Preparing a neutral snapshot… Please face the camera naturally.
                </div>
              )}
            </div>
          </div>
        )}
      </div>

      {/* RIGHT COLUMN */}
      <div className="lg:col-span-2 space-y-4">
        {cameraOn && (
          <>
            <LiveCard getIndicator={() => {
              const st = stateRef.current;
              const score = computeLivenessComposite(st);
              const label = score >= 75 ? "Strong" : score >= 50 ? "In Progress" : "Low";
              const reasons = liveReasons(st);
              return { score, label, reasons };
            }} />

            <PadCard getIndicator={() => {
              const st = stateRef.current;
              const score = computePadFromSamples(st);
              const label = score >= 75 ? "Live" : (score < 50 ? "Block" : "Suspicious");
              const reasons = padReasons(st);
              return { score, label, reasons };
            }} />
          </>
        )}

        {/* Checklist (bilingual) */}
        <div className="card p-4">
          <div className="flex items-center justify-between">
            <div>
              <div className="text-lg font-semibold">Checklist</div>
              <div className="text-slate-400 text-xs">Follow the prompts in random order</div>
            </div>
            <span className={`badge ${Object.values(stateRef.current.passed).filter(Boolean).length === 4 ? "badge-green" : "badge-yellow"}`}>
              {Object.values(stateRef.current.passed).filter(Boolean).length}/4 done
            </span>
          </div>
          <ul className="mt-3 space-y-2">
            {steps.map((s) => {
              const isCurrent = steps[currentStep] === s;
              const passed = stateRef.current.passed[s];
              const data = INSTRUCTIONS[s];
              return (
                <li key={s} className={`flex items-center gap-3 p-3 rounded-xl border ${isCurrent ? "bg-slate-800/60 border-slate-700" : "border-slate-800"}`}>
                  <div className={`w-8 h-8 grid place-items-center rounded-full ${passed ? "bg-emerald-600/20" : isCurrent ? "bg-yellow-600/20" : "bg-slate-600/20"}`}>
                    <span className="text-lg">{data.icon}</span>
                  </div>
                  <div className="flex-1">
                    <div className="text-sm font-medium text-slate-200">{data.titleEn} <span className="text-slate-400">· {data.titleBn}</span></div>
                    <div className="text-[12px] text-slate-400 leading-tight">{data.en}</div>
                    <div className="text-[12px] text-slate-400 leading-tight">{data.bn}</div>
                  </div>
                  <span className={`badge ${passed ? "badge-green" : (isCurrent ? "badge-yellow" : "badge-red")}`}>
                    {passed ? "Done" : (isCurrent ? "Now" : "Wait")}
                  </span>
                </li>
              );
            })}
          </ul>
        </div>

        {/* Debug */}
        <div className="card p-4">
          <div className="text-lg font-semibold mb-2">Debug Panel</div>
          <div className="grid grid-cols-2 gap-x-4 gap-y-2 text-sm">
            <div className="label">EAR:</div><div className="value">{(smooth.current.EAR || 0).toFixed(3)}</div>
            <div className="label">MAR:</div><div className="value">{(smooth.current.MAR || 0).toFixed(3)}</div>
            <div className="label">Mouth width (norm):</div><div className="value">{(smooth.current.WNORM || 0).toFixed(3)}</div>
            <div className="label">Corner lift (curve):</div><div className="value">{(smooth.current.CURVE || 0).toFixed(3)}</div>
            <div className="label">YAW(rad):</div><div className="value">{(smooth.current.YAW || 0).toFixed(3)}</div>
            <div className="label">YAW(°):</div><div className="value">{(((smooth.current.YAW || 0) * 180) / Math.PI).toFixed(1)}</div>
            <div className="label">Depth Var:</div><div className="value">{(smooth.current.depthVar || 0).toExponential(3)}</div>
          </div>
        </div>
      </div>
    </div>
  );
}

/* -------- small components -------- */
function Metric({ label, value, ok }) {
  return (
    <div className="p-3 rounded-xl border border-slate-800 bg-slate-900/60">
      <div className="text-xs text-slate-400">{label}</div>
      <div className={`text-lg font-semibold ${ok ? "text-emerald-300" : "text-rose-300"}`}>{value}</div>
    </div>
  );
}
function PadCard({ getIndicator }) {
  const ind = getIndicator();
  const color = ind.label === "Live" ? "text-emerald-300" : ind.label === "Suspicious" ? "text-amber-300" : "text-rose-300";
  const badge = ind.label === "Live" ? "badge-green" : ind.label === "Suspicious" ? "badge-yellow" : "badge-red";
  return (
    <div className="card p-4">
      <div className="flex items-center justify-between">
        <div className="text-lg font-semibold">Anti-Spoof Status</div>
        <span className={`badge ${badge}`}>{ind.label}</span>
      </div>
      <div className="mt-2 text-sm">
        <div>PAD Score: <span className={`font-semibold ${color}`}>{ind.score}</span></div>
        {ind.reasons?.length > 0
          ? <ul className="list-disc pl-5 text-slate-300 mt-1">{ind.reasons.map((r, i) => <li key={i}>{r}</li>)}</ul>
          : <div className="text-slate-400">No issues detected so far.</div>}
      </div>
    </div>
  );
}
function LiveCard({ getIndicator }) {
  const ind = getIndicator();
  const labelToUI = (label) => label === "Strong" ? ["badge-green", "text-emerald-300"]
    : label === "In Progress" ? ["badge-yellow", "text-amber-300"]
      : ["badge-red", "text-rose-300"];
  const [badge, color] = labelToUI(ind.label);
  return (
    <div className="card p-4">
      <div className="flex items-center justify-between">
        <div className="text-lg font-semibold">Liveness Status</div>
        <span className={`badge ${badge}`}>{ind.label}</span>
      </div>
      <div className="mt-2 text-sm">
        <div>Live Score: <span className={`font-semibold ${color}`}>{ind.score}</span></div>
        {ind.reasons?.length > 0
          ? <ul className="list-disc pl-5 text-slate-300 mt-1">{ind.reasons.map((r, i) => <li key={i}>{r}</li>)}</ul>
          : <div className="text-slate-400">All good.</div>}
      </div>
    </div>
  );
}

/* ---------- Animated HUD ---------- */
function PromptHUD({ phase, calibPct, cameraOn, cameraReady, faces, done, finalPass, stepKey }) {
  const commonPanel = {
    initial: { opacity: 0, y: -12, scale: 0.98 },
    animate: { opacity: 1, y: 0, scale: 1, transition: { type: "spring", stiffness: 220, damping: 20 } },
    exit: { opacity: 0, y: -8, scale: 0.98, transition: { duration: 0.18 } },
  };
  const chipFx = {
    initial: { opacity: 0, y: 12 },
    animate: { opacity: 1, y: 0, transition: { type: "spring", stiffness: 200, damping: 20 } },
    exit: { opacity: 0, y: 12, transition: { duration: 0.18 } },
  };
  const step = stepKey ? INSTRUCTIONS[stepKey] : null;

  return (
    <div className="pointer-events-none absolute inset-0">
      {/* Top-right status / prompt */}
      <div className="absolute top-4 right-4 flex flex-col items-end gap-2">
        <AnimatePresence mode="wait">
          {phase === "calibrate" && (
            <motion.div key="calib" {...commonPanel} className="px-5 py-3 rounded-2xl bg-slate-900/80 backdrop-blur border border-white/10 shadow-xl">
              <div className="text-xl font-semibold text-slate-50">Calibrating…</div>
              <div className="text-slate-300 text-sm">{calibPct}%</div>
            </motion.div>
          )}

          {phase === "run" && !done && step && (
            <motion.div key={stepKey} {...commonPanel} className="px-5 py-4 rounded-2xl bg-slate-900/80 backdrop-blur border border-white/10 shadow-xl w-[min(92vw,360px)]">
              <div className="text-xl font-semibold text-slate-50 flex items-center gap-2">
                <span className="text-2xl" aria-hidden>{step.icon}</span>
                <span>{step.titleEn}</span>
              </div>
              <div className="mt-1 text-slate-200 text-sm">{step.en}</div>
              <div className="text-slate-300 text-xs">{step.bn}</div>
            </motion.div>
          )}

          {done && (
            <motion.div key="done" {...commonPanel} className={`px-5 py-4 rounded-2xl backdrop-blur border shadow-xl ${finalPass ? "bg-emerald-900/70 border-emerald-500/30" : "bg-rose-900/70 border-rose-500/30"}`}>
              <div className="text-xl font-semibold text-slate-50">{finalPass ? "Verification Passed" : "Check Failed — Try Again"}</div>
            </motion.div>
          )}
        </AnimatePresence>

        {/* Small badges */}
        <div className="flex items-center gap-2">
          <div className="px-3 py-1 rounded-full bg-slate-900/70 border border-white/10 text-xs text-slate-200 shadow">{cameraOn ? (cameraReady ? "Cam: Ready" : "Cam: Starting…") : "Cam: Off"}</div>
          <div className="px-3 py-1 rounded-full bg-slate-900/70 border border-white/10 text-xs text-slate-200 shadow">Faces: {faces}</div>
        </div>
      </div>

      {/* Bottom-center chip */}
      <div className="absolute bottom-4 left-0 right-0 grid place-items-center">
        <AnimatePresence mode="wait">
          {phase === "run" && step && !done && (
            <motion.div key={`chip-${stepKey}`} {...chipFx} className="px-4 py-2 rounded-full bg-slate-900/80 border border-white/10 text-slate-100 text-sm backdrop-blur flex items-center gap-2 shadow">
              <span className="text-xl" aria-hidden>{step.icon}</span>
              <span className="font-medium">{step.titleEn}</span>
            </motion.div>
          )}
        </AnimatePresence>
      </div>
    </div>
  );
}

/* -------- helpers used in LiveCard -------- */
function liveReasons(st) {
  const reasons = [];
  if (st.latencies.length) {
    const timely = st.latencies.filter(t => t >= 0.18 && t <= 3.0).length;
    if (timely / st.latencies.length < 0.7) reasons.push("Responses too fast/slow");
  }
  if (st.blinkAmps.length) {
    const med = percentile(st.blinkAmps, 0.5);
    if (med < 0.05) reasons.push("Blinks too shallow");
  }
  if (st.holdTotalFrames && (st.holdGoodFrames / st.holdTotalFrames) < 0.7) reasons.push("Hold not steady enough");
  return reasons.length ? reasons : ["All good"];
}