import { useEffect, useState } from 'react'
import { api, ScenarioMeta, SimConfig } from '../api/client'

type Props = {
  cfg: SimConfig
  setCfg: (c: SimConfig) => void
  run: (c?: Partial<SimConfig>) => void
  startLive: (c?: Partial<SimConfig>) => Promise<void>
  loadMock: () => void
  loading: boolean
  error: string | null
  source: 'live' | 'replay' | null
}

const NUM_FIELDS: { key: keyof SimConfig; label: string; step?: string }[] = [
  { key: 'n_bands', label: 'Bands', step: '1' },
  { key: 'n_steps', label: 'Steps', step: '10' },
  { key: 'seed', label: 'Seed', step: '1' },
  { key: 'alpha', label: 'Decay α', step: '0.01' },
  { key: 'floor', label: 'Sweep floor', step: '0.01' },
  { key: 'step_ms', label: 'Frame ms', step: '10' },
]

export default function ConfigPanel({ cfg, setCfg, run, startLive, loadMock, loading, error, source }: Props) {
  const [scenarios, setScenarios] = useState<ScenarioMeta[]>([])

  useEffect(() => {
    api.scenarios().then((r) => setScenarios(r.scenarios)).catch(() => {})
  }, [])

  useEffect(() => {
    if (cfg.scheduler !== 'rfi_ucb') setCfg({ ...cfg, scheduler: 'rfi_ucb' })
  }, []) // eslint-disable-line react-hooks/exhaustive-deps

  const patch = (p: Partial<SimConfig>) => setCfg({ ...cfg, ...p })

  return (
    <div className="panel flex flex-col gap-3 p-4">
      <div className="panel-title">SIMULATION CONFIG</div>

      <div className="rounded-lg border border-amber/30 bg-amber/5 px-2.5 py-2 text-[11px] leading-snug text-amber">
        Algorithm fixed to <span className="font-bold">UCB1 + Random Forest</span> — a UCB1 bandit whose per-band
        priority is blended with a Random Forest prior trained on the TSRD radar dataset.
      </div>

      <div className="grid grid-cols-3 gap-2">
        {NUM_FIELDS.map((f) => (
          <label key={f.key} className="flex flex-col gap-1">
            <span className="text-[10px] text-dim uppercase">{f.label}</span>
            <input
              type="number"
              step={f.step ?? '1'}
              value={Number(cfg[f.key])}
              onChange={(e) => patch({ [f.key]: Number(e.target.value) } as Partial<SimConfig>)}
              className="num rounded-md border border-line bg-inner px-2 py-1.5 text-sm text-slate-200 outline-none focus:border-neon/60"
            />
          </label>
        ))}
      </div>

      <div>
        <div className="mb-1.5 text-[10px] text-dim uppercase">Scenario presets</div>
        <div className="grid max-h-44 grid-cols-1 gap-1.5 overflow-y-auto pr-1 scroll-slim">
          {scenarios.map((s) => {
            const active = cfg.scenario === s.id
            return (
              <button
                key={s.id}
                onClick={() => patch({ scenario: s.id })}
                className={`flex flex-col rounded-lg border px-2.5 py-1.5 text-left transition-all ${
                  active ? 'border-neon/70 bg-neon/10' : 'border-line bg-inner/40 hover:border-neon/40'
                }`}
              >
                <span className={`text-xs font-medium ${active ? 'text-neon' : 'text-slate-200'}`}>{s.label}</span>
                <span className="text-[10px] text-dim">{s.desc}</span>
              </button>
            )
          })}
        </div>
      </div>

      <button onClick={() => startLive()} disabled={loading} className="btn btn-primary w-full py-2.5 disabled:opacity-60">
        {loading ? (
          <span className="inline-block h-3.5 w-3.5 animate-spin rounded-full border-2 border-neon border-t-transparent" />
        ) : (
          '▶ START LIVE (UCB1 + RF)'
        )}
      </button>
      <button onClick={() => run()} disabled={loading} className="btn btn-ghost w-full py-2 text-xs">
        ⏪ RUN AS REPLAY
      </button>
      <button onClick={loadMock} disabled={loading} className="btn btn-ghost w-full py-2 text-xs">
        ⚡ LOAD MOCK DATA (no backend)
      </button>

      {source && (
        <div className={`text-center text-[10px] tracking-widest uppercase ${source === 'live' ? 'text-hit' : 'text-neon'}`}>
          {source === 'live' ? '◉ live feed from backend' : '▸ replaying precomputed run'}
        </div>
      )}

      {error && <div className="rounded-md border border-miss/40 bg-miss/10 px-2 py-1.5 text-xs text-miss">{error}</div>}
    </div>
  )
}