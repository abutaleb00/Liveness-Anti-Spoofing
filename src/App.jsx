import React from 'react'
import LivenessApp from './components/LivenessApp.jsx'

export default function App() {
  return (
    <div className="min-h-screen p-4 md:p-8">
      <div className="max-w-6xl mx-auto space-y-6">
        <header className="flex items-center justify-between">
          <h1 className="text-2xl md:text-3xl font-bold tracking-tight">
            Liveness <span className="text-cyan-400">&</span> Anti‑Spoofing Demo
          </h1>
          <a className="btn-outline" href="https://github.com/tensorflow/tfjs-models/tree/master/face-landmarks-detection" target="_blank" rel="noreferrer">
            Docs
          </a>
        </header>
        <LivenessApp />
        <footer className="text-center text-xs text-slate-400 pt-6">
          2025 © Commlink Infotech LLC
        </footer>
      </div>
    </div>
  )
}