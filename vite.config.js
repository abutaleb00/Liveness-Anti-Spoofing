// vite.config.js
import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

export default defineConfig({
  plugins: [react()],
  server: { port: 5173, host: true },

  // Keep Vite from touching Mediapipe bundles
  optimizeDeps: {
    exclude: ['@mediapipe/face_mesh', '@mediapipe/camera_utils'],
  },

  // (Optional) extra safety: avoid CJS transforms that have mangled FaceMesh in prod
  build: {
    commonjsOptions: { include: [] },
  },
})