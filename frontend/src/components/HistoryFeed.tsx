import { useEffect, useMemo, useRef } from 'react'
import type { StepLog, ModelInfo } from '../api/client'

type Props = {
  log: StepLog[]
  cur: number
  edges?: [number, number][]
  model?: ModelInfo | null
}

export default function HistoryFeed({ log, cur, edges, model }: Props) {
  const scan = useRef<HTMLDivElement>(null)
  const upTo = useMemo(() => log.slice(0, Math.min(cur, log.length - 1) + 1), [log, cur])
  const recent = useMemo(() => [...upTo].reverse().slice(0, 48), [upTo])

  useEffect(() => {
    const el = scan.current
    if (el) el.scrollTop = 0
  }, [cur])

  return (
    <div className="panel flex min-h-0 flex-col p-4">
      <div className="panel-title mb-2">
        SCAN HISTORY <span className="num text-dim/70">· {upTo.length} visits</span>
      </div>
      <div ref={scan} className="scroll-slim min-h-0 flex-1 space-y-1 overflow-y-auto pr-1">
        {recent.length === 0 && <div className="text-sm text-dim">Scanned bands appear here as they are visited.</div>}
        {recent.map((e) => {
          const ghz = edges?.[e.band] ? `${edges[e.band][0]}–${edges[e.band][1]} GHz` : ''
          return (
            <div
              key={e.t}
              className={`flex items-center justify-between rounded-lg border px-2.5 py-1.5 text-[11px] ${
                e.hit ? 'border-hit/30 bg-hit/5' : 'border-line bg-inner/40'
              }`}
            >
              <span className="num text-dim">t={e.t}</span>
              <span className="font-semibold text-slate-200">Band B{e.band}</span>
              <span className="hidden text-dim sm:inline">{ghz}</span>
              <span className={`num font-bold ${e.hit ? 'text-hit' : 'text-miss'}`}>{e.hit ? 'HIT' : 'MISS'}</span>
            </div>
          )
        })}
      </div>

      <div className="mt-2 flex flex-wrap items-center gap-x-3 gap-y-1 border-t border-line/60 pt-2 text-[10px] text-dim">
        <span className="flex items-center gap-1.5 text-hit">
          <span className="relative flex h-1.5 w-1.5">
            <span className="absolute inline-flex h-full w-full animate-ping rounded-full bg-hit opacity-60" />
            <span className="relative inline-flex h-1.5 w-1.5 rounded-full bg-hit" />
          </span>
          MODEL LIVE
        </span>
        <span className="num">{model?.model_type ?? 'UCB1'}+RF</span>
        {model?.trees ? <span className="num">{model.trees} trees</span> : null}
        {model?.predict_calls ? <span className="num">{model.predict_calls} calls</span> : null}
        {model && model.avg_latency_ms != null ? <span className="num">{model.avg_latency_ms}ms/pred</span> : null}
        <span className="num">{model?.model_file ?? 'band_activity_model.pkl'}</span>
      </div>
    </div>
  )
}