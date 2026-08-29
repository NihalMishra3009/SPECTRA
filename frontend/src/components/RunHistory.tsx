import { useCallback, useEffect, useState } from 'react'
import { api, DbRunSummary, DbStats, SimResult } from '../api/client'

type Props = { loadRun: (res: SimResult) => void }

export default function RunHistory({ loadRun }: Props) {
  const [runs, setRuns] = useState<DbRunSummary[]>([])
  const [stats, setStats] = useState<DbStats | null>(null)
  const [loadingId, setLoadingId] = useState<string | null>(null)
  const [busy, setBusy] = useState(false)

  const refresh = useCallback(() => {
    if (busy) return
    setBusy(true)
    api.dbRuns(12).then((r) => setRuns(r.runs)).catch(() => {})
    api.dbStats().then(setStats).catch(() => {}).finally(() => setBusy(false))
  }, [busy])

  useEffect(() => {
    refresh()
    const id = setInterval(refresh, 8000)
    return () => clearInterval(id)
  }, [refresh])

  const open = async (summary: DbRunSummary) => {
    setLoadingId(summary.run_id)
    try {
      const res = await api.dbRun(summary.run_id)
      loadRun(res)
    } catch {
      /* keep list as-is */
    } finally {
      setLoadingId(null)
    }
  }

  return (
    <div className="panel flex flex-col gap-2 p-4">
      <div className="flex items-center justify-between">
        <span className="panel-title">RECORDED RUNS · SQLite</span>
        <button onClick={refresh} className="btn h-6 px-2 py-0 text-[10px] btn-ghost" disabled={busy}>
          ⟳ refresh
        </button>
      </div>

      {stats && (
        <div className="rounded-lg border border-line bg-inner/60 px-2.5 py-1.5 text-[10px] text-dim">
          <span className="num text-neon2">{stats.total_runs}</span> runs · <span className="num">{stats.telemetry_samples.toLocaleString()}</span> RF samples
          {stats.best_run?.scheduler ? (
            <>
              {' '}· best <span className="num text-amber">{stats.best_run.scheduler}</span> @{' '}
              <span className="num text-hit">{Math.round(stats.best_run.ir ?? 0)}%</span> ({stats.best_run.scenario_id})
            </>
          ) : null}
        </div>
      )}

      <div className="flex max-h-52 flex-col gap-1.5 overflow-y-auto pr-1 scroll-slim">
        {runs.length === 0 && <div className="text-[11px] text-dim">No runs recorded yet — RUN one, it's saved automatically.</div>}
        {runs.map((r) => {
          const gain = (r.outcome.smart_ir ?? 0) - (r.outcome.baseline_ir ?? 0)
          return (
            <button
              key={r.run_id}
              onClick={() => open(r)}
              className={`flex items-center justify-between gap-2 rounded-lg border border-line bg-inner/40 px-2.5 py-1.5 text-left transition-all hover:border-neon/40 ${
                loadingId === r.run_id ? 'opacity-60' : ''
              }`}
            >
              <div className="min-w-0">
                <div className="truncate text-[11px] font-medium text-slate-200">
                  {r.scenario_label ?? r.scenario_id} <span className="text-dim">· {r.scheduler}</span>
                </div>
                <div className="num text-[10px] text-dim">
                  {new Date(r.created_at).toLocaleTimeString()} · {r.n_steps}t · seed {r.seed ?? '—'}
                </div>
              </div>
              <div className="text-right">
                <div className="num text-xs text-slate-100">{Math.round(r.outcome.smart_ir ?? 0)}%</div>
                <div className={`num text-[10px] ${gain >= 0 ? 'text-hit' : 'text-miss'}`}>
                  Δ{gain >= 0 ? '+' : ''}{gain.toFixed(0)}
                </div>
              </div>
            </button>
          )
        })}
      </div>
      <div className="text-[10px] text-dim">Click a run → loads full result from <span className="font-mono">database/spectra.db</span></div>
    </div>
  )
}