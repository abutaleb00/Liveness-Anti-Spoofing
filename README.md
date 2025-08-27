# Liveness + Anti‑Spoofing (React + Vite + Tailwind)

A practical, **workable** browser demo for liveness and simple anti‑spoofing, using **TensorFlow.js** and **MediaPipe FaceMesh**.

## Features

- Randomized challenge: **Blink (x3)**, **Head turn left/right**, **Smile**
- Hold‑to‑pass for pose steps (prevents single‑frame spoofs)
- **Blink detection** via Eye Aspect Ratio (EAR)
- **Smile detection** via Mouth Aspect Ratio (MAR)
- **Yaw (left/right)** via face symmetry score around eye corners
- **Passive anti‑spoof checks**: depth variance across key points during motion + multi‑action challenge
- Clean UI with **Tailwind**, **debug panel**, and progress bar

> Note: This is a client‑side demo. For production, combine with server‑side checks and/or 3D depth sensors.

## Run locally

```bash
npm install
npm run dev
```

Open the URL printed in the terminal (usually http://localhost:5173). Allow camera access.

## Production build

```bash
npm run build
npm run preview
```

## Tuning

Open **`src/components/LivenessApp.jsx`**, adjust the `cfg` block:

- `EAR_OPEN`, `EAR_CLOSED`: blink thresholds (typical EAR open ~0.25, closed ~0.15–0.20)
- `MAR_SMILE`: smile detection threshold (~0.32–0.4)
- `YAW_THRESH`: how far to turn head (0.08–0.15)
- `HOLD_FRAMES`: how long to hold a pose to pass (frames)
- `BLINK_TARGET`: number of blinks to count
- `SPOOF_DEPTH_MIN`: minimal depth variance expected during motion

## FAQ

**Camera mirrored?** We flip horizontally for a natural selfie view. The yaw sign is computed accordingly.

**MediaPipe assets not loading?** We point to the official CDN in `solutionPath`. Ensure you’re online and not blocking CDN requests.

**Accuracy tips**
- Use a well‑lit environment (soft, frontal light).
- Keep the face within the frame; avoid strong backlight.
- Webcam quality matters (720p minimum recommended).

## License

MIT (for demo purposes) – You are responsible for any regulatory compliance for your use case.