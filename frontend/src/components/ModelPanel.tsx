import { useEffect, useState } from 'react'
import { api, ModelStatus, SimConfig } from '../api/client'
import EChart from './EChart'

type Props = { cfg: SimConfig; setCfg: (c: SimConfig) => void }

type CurvePoint = { timestep?: number; mean_interception?: number; epoch?: number; val_acc?: number }

export default function ModelPanel({ cfg, setCfg }: Props) {
  const [models, setModels] = useState<ModelStatus[]>([])
  const [curves, setCurves] = useState<Record<string, CurvePoint[]>>({})

  useEffect(() => {
    api.models().then((r) => {
      setModels(r.models)
      r.models.forEach((m) => {
        if (m.present && m.curves_file) {
          api.curves(m.curves_file).then((c) => setCurves((prev) => ({ ...prev, [m.scheduler]: c as CurvePoint[] }))).catch(() => {})
        }
      })
    }).catch(() => {})
  }, [])

  return (
    <div className="panel flex flex-col gap-2 p-4">
      <div className="panel-title">TRAINED MODELS</div>
      {models.filter((m) => m.scheduler !== 'bandit_baseline').map((m) => {
        const active = cfg.scheduler === m.scheduler
        return (
          <div key={m.scheduler} className={`rounded-xl border p-3 transition-colors ${active ? 'border-hit/60 bg-hit/5' : 'border-line bg-inner/40'}`}>
            <div className="flex items-center justify-between">
              <div className="font-mono text-sm font-semibold text-slate-100">{m.scheduler}</div>
              <div className={`text-[10px] px-2 py-0.5 rounded-full border ${m.present ? 'border-hit/50 text-hit' : 'border-line text-dim'}`}>
                {m.present ? 'READY' : 'NOT TRAINED'}
              </div>
            </div>
            {m.present && m.meta && (
              <div className="mt-1.5 flex flex-wrap gap-x-3 gap-y-0.5 text-[10px] text-dim">
                <span>algo {m.meta.algo ?? 'lstm'}</span>
                {m.meta.final_interception_pct !== undefined && <span className="num">eval {m.meta.final_interception_pct}%</span>}
                {m.meta.final_val_acc !== undefined && <span className="num">val acc {m.meta.final_val_acc}</span>}
                {m.meta.wall_seconds !== undefined && <span className="num">{m.meta.wall_seconds}s train</span>}
              </div>
            )}
            {curves[m.scheduler]?.length ? (
              <EChart
                className="mt-2 h-14 w-full"
                option={{
                  grid: { left: 4, right: 4, top: 6, bottom: 4 },
                  xAxis: { type: 'category', show: false, data: curves[m.scheduler].map((p) => p.timestep ?? p.epoch ?? 0) },
                  yAxis: { type: 'value', show: false, min: 0 },
                  series: [
                    {
                      type: 'line',
                      smooth: true,
                      symbol: 'none',
                      data: curves[m.scheduler].map((p) => p.mean_interception ?? (p.val_acc ?? 0) * 100),
                      lineStyle: { color: '#22e584', width: 1.5 },
                      areaStyle: { color: 'rgba(34,229,132,0.15)' },
                    },
                  ],
                }}
              />
            ) : null}
            <button
              onClick={() => setCfg({ ...cfg, scheduler: m.scheduler })}
              disabled={!m.present}
              className={`btn mt-2 w-full py-1.5 text-xs ${active ? 'btn-primary' : 'btn-ghost'} disabled:opacity-40`}
            >
              {active ? '✓ ACTIVE' : m.present ? 'USE AS SMART' : 'TRAIN REQUIRED'}
            </button>
          </div>
        )
      })}
    </div>
  )
}