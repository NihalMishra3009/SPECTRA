import { useMemo } from 'react'
import { motion } from 'framer-motion'
import type { SimResult, SimConfig, TsrdBand } from '../api/client'

type Props = {
  result: SimResult | null
  cur: number
  cfg: SimConfig
  tsrd: TsrdBand[]
  source: 'live' | 'replay' | null
}

type BandStat = {
  band: number
  hits: number
  scans: number
  sinceHit: number
  last: 'hit' | 'miss' | null
  lastT: number
}

function computeBandStats(log: { t: number; band: number; hit: boolean }[], cur: number, n: number): BandStat[] {
  const stats = Array.from({ length: n }, (_, b) => ({ band: b, hits: 0, scans: 0, sinceHit: cur + 1, last: null as 'hit' | 'miss' | null, lastT: -1 }))
  for (let t = 0; t <= Math.min(cur, log.length - 1); t++) {
    const e = log[t]
    const s = stats[e.band]
    s.scans += 1
    s.last = e.hit ? 'hit' : 'miss'
    s.lastT = t
    if (e.hit) {
      s.hits += 1
      s.sinceHit = 0
    } else if (t > 0) {
      s.sinceHit += 1
    }
  }
  return stats
}

export default function BandPanel({ result, cur, cfg, tsrd, source }: Props) {
  const n = cfg.n_bands
  const smart = result?.smart ?? null
  const log = smart?.log ?? []
  const truthGrid = result?.ground_truth ?? []
  const prioGrid = smart?.priorities ?? []
  const edges = result?.band_edges_ghz ?? []

  const frame = useMemo(() => {
    if (!result || log.length === 0) return null
    const t = Math.min(cur, log.length - 1)
    const entry = log[t]
    const truth = truthGrid[t] ?? []
    const prio = prioGrid[t] ?? []
    return { t, band: entry.band, hit: entry.hit, snr: entry.snr, truth, prio }
  }, [result, cur, log, truthGrid, prioGrid])

  const stats = useMemo(() => computeBandStats(log, cur, n), [log, cur, n])

  const totals = useMemo(() => {
    const hits = stats.reduce((a, s) => a + s.hits, 0)
    const scans = stats.reduce((a, s) => a + s.scans, 0)
    return { hits, scans, ir: scans ? hits / scans : 0 }
  }, [stats])

  const focusMeta = frame ? tsrd[frame.band] ?? null : null
  const focusEdge = frame ? edges[frame.band] ?? null : null

  return (
    <div className="flex min-h-0 flex-col gap-3">
      {/* running counters ------------------------------------------------ */}
      <div className="grid grid-cols-3 gap-3">
        <CounterBox label="TOTAL SCANS" value={totals.scans} hint="receiver dwells" accent="text-neon" />
        <CounterBox label="TOTAL HITS" value={totals.hits} hint="intercepted transmissions" accent="text-hit" />
        <CounterBox label="INTERCEPTION" value={`${(totals.ir * 100).toFixed(1)}%`} hint="hits / scans" accent="text-amber" />
      </div>

      {/* current focus banner -------------------------------------------- */}
      <div
        className={`panel relative flex items-center justify-between overflow-hidden px-5 py-3 transition-colors ${
          frame?.hit ? 'border-hit/50' : frame ? 'border-miss/50' : 'border-line'
        }`}
      >
        <div className="flex items-center gap-4">
          <span className={`relative flex h-3 w-3 ${frame ? '' : 'opacity-40'}`}>
            {frame && (
              <span className={`absolute inline-flex h-full w-full animate-ping rounded-full ${frame.hit ? 'bg-hit' : 'bg-neon'} opacity-60`} />
            )}
            <span className={`relative inline-flex h-3 w-3 rounded-full ${frame?.hit ? 'bg-hit' : 'bg-neon'}`} />
          </span>
          <div>
            <div className="text-[10px] tracking-[0.22em] text-dim uppercase">Currently focusing</div>
            <div className="font-display text-xl font-bold leading-tight text-slate-50">
              Band B{frame ? frame.band : '—'}
              <span className="ml-3 text-sm font-medium text-neon2">
                {focusEdge ? `${focusEdge[0]}–${focusEdge[1]} GHz` : '— GHz'}
              </span>
            </div>
          </div>
          <div className="hidden text-[11px] text-dim md:block">
            {focusMeta ? (
              <span className="num">
                {focusMeta.config_id} · {focusMeta.n_emitters} emitters · {focusMeta.pulse_width_us} µs ·{' '}
                {focusMeta.amplitude_dbm} dBm
              </span>
            ) : (
              'TSRD reference band'
            )}
          </div>
        </div>
        <div className="flex items-center gap-3">
          {frame ? (
            <>
              <div className={`rounded-lg border px-3 py-1.5 text-sm font-bold ${frame.hit ? 'border-hit/60 bg-hit/10 text-hit' : 'border-miss/60 bg-miss/10 text-miss'}`}>
                {frame.hit ? 'HIT' : 'MISS'}
              </div>
              <div className="num text-[11px] text-dim">SNR {frame.snr.toFixed(2)}</div>
              <div className="num text-[11px] text-dim">t={frame.t}</div>
            </>
          ) : (
            <div className="text-sm text-dim">{source ? 'waiting for stream…' : 'press ▶ RUN SIMULATION'}</div>
          )}
        </div>
      </div>

      {/* the 10 band boxes ------------------------------------------------ */}
      <div className={`grid grid-cols-2 gap-3 sm:grid-cols-5`}>
        {Array.from({ length: n }, (_, b) => {
          const s = stats[b]
          const emitting = frame?.truth[b] ?? false
          const focused = frame?.band === b
          const currentHit = focused ? frame?.hit : null
          const prio = frame?.prio[b] ?? null
          const meta = tsrd[b] ?? null
          const edge = edges[b] ?? null
          const delay = (i: number) => i * 0.03
          return (
            <motion.div
              key={b}
              initial={{ opacity: 0, scale: 0.94 }}
              animate={{ opacity: 1, scale: 1 }}
              transition={{ delay: delay(b) }}
              className={`relative flex flex-col justify-between overflow-hidden rounded-2xl border bg-panel/85 p-3 backdrop-blur-md transition-all duration-200 ${
                focused
                  ? currentHit
                    ? 'border-hit/80 shadow-[0_0_30px_-8px_rgba(34,229,132,0.55)]'
                    : 'border-miss/80 shadow-[0_0_30px_-8px_rgba(255,77,109,0.5)]'
                  : 'border-line'
              }`}
            >
              {/* emission pulse */}
              <div
                className={`pointer-events-none absolute inset-0 transition-opacity ${
                  emitting && !focused ? 'bg-neon/[0.05]' : ''
                }`}
              />
              {emitting && (
                <span className="absolute right-3 top-3 flex h-2 w-2">
                  <span className="absolute inline-flex h-full w-full animate-ping rounded-full bg-neon opacity-60" />
                  <span className="relative inline-flex h-2 w-2 rounded-full bg-neon" />
                </span>
              )}

              <div className="flex items-start justify-between">
                <div>
                  <div className="font-display text-base font-bold text-slate-50">B{b}</div>
                  <div className="num text-[10px] text-dim">{edge ? `${edge[0]}–${edge[1]}` : ''} GHz</div>
                </div>
                {focused && (
                  <span className={`rounded-md px-1.5 py-0.5 text-[9px] font-bold tracking-widest ${currentHit ? 'bg-hit/15 text-hit' : 'bg-miss/15 text-miss'}`}>
                    ◉ FOCUS
                  </span>
                )}
              </div>

              {/* live activity strip (last 16 steps) */}
              <div className="my-2 flex h-4 items-end gap-[2px]">
                {Array.from({ length: 16 }, (_, k) => {
                  const t = Math.max(0, Math.min(cur, log.length - 1)) - 15 + k
                  const on = t >= 0 && (truthGrid[t]?.[b] ?? false)
                  const scanMatch = log[t]?.band === b
                  const col = scanMatch ? (log[t].hit ? 'bg-hit' : 'bg-miss') : on ? 'bg-neon/80' : 'bg-line/60'
                  return <span key={k} className={`flex-1 rounded-sm ${col} opacity-80`} style={{ height: on || scanMatch ? '100%' : '30%' }} />
                })}
              </div>

              {/* priority score */}
              <div>
                <div className="mb-1 flex items-baseline justify-between">
                  <span className="text-[9px] tracking-widest text-dim uppercase">priority</span>
                  <span className="num text-sm font-semibold text-neon2">{prio != null ? `${Math.round(prio)}%` : '–'}</span>
                </div>
                <div className="h-1.5 w-full overflow-hidden rounded-full bg-inner">
                  <motion.div
                    className="h-full rounded-full bg-gradient-to-r from-neon/70 to-neon"
                    animate={{ width: `${prio ?? 0}%` }}
                    transition={{ duration: 0.25 }}
                  />
                </div>
              </div>

              {/* hit/miss + tsrd reference */}
              <div className="mt-2 flex items-center justify-between text-[10px]">
                <span className="flex items-center gap-1">
                  {s.last ? (
                    <>
                      <span className={`inline-block h-1.5 w-1.5 rounded-full ${s.last === 'hit' ? 'bg-hit' : 'bg-miss'}`} />
                      <span className={`num ${s.last === 'hit' ? 'text-hit' : 'text-miss'}`}>
                        {s.last === 'hit' ? 'HIT' : 'MISS'}
                      </span>
                      <span className="text-dim">· {s.hits}/{s.scans}</span>
                    </>
                  ) : (
                    <span className="text-dim">no scan yet</span>
                  )}
                </span>
                <span className="num text-dim">
                  {s.sinceHit <= cur + 1 ? `${s.sinceHit}t idle` : ''}
                </span>
              </div>
              {meta && (
                <div className="num mt-1.5 border-t border-line/60 pt-1.5 text-[9px] leading-tight text-dim">
                  {meta.config_id} · {meta.n_emitters} em · {meta.pulse_width_us}µs · {meta.amplitude_dbm}dBm
                </div>
              )}
            </motion.div>
          )
        })}
      </div>
    </div>
  )
}

function CounterBox({ label, value, hint, accent }: { label: string; value: number | string; hint: string; accent: string }) {
  return (
    <div className="panel flex flex-col items-center justify-center gap-0.5 px-4 py-3">
      <div className="text-[9px] tracking-[0.22em] text-dim uppercase">{label}</div>
      <div className={`num text-3xl font-bold ${accent}`}>{typeof value === 'number' ? value.toLocaleString() : value}</div>
      <div className="text-[9px] text-dim/70">{hint}</div>
    </div>
  )
}