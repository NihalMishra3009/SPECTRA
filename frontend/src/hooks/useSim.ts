import { useCallback, useEffect, useRef, useState } from 'react'
import { api, LiveEvent, Metrics, SimConfig, SimEvent, SimResult, StepLog, wsStream } from '../api/client'

export const DEFAULT_CFG: SimConfig = {
  n_bands: 10,
  n_steps: 300,
  seed: 2024,
  scenario: 'tsrd',
  scheduler: 'rfi_ucb',
  alpha: 0.25,
  epsilon: 0.05,
  window: 40,
  floor: 0.15,
  step_ms: 350,
}

export type Source = 'live' | 'replay'

export type AlertItem = {
  t: number
  type: string
  bands: number[]
  smartDelay: number | null
  baselineDelay: number | null
}

export function computeAlerts(result: SimResult | null): AlertItem[] {
  if (!result) return []
  const alerts: AlertItem[] = []
  for (const ev of result.events) {
    if (!ev.bands_on.length) continue
    const smartFirst = firstHitAfter(result.smart.log, ev.bands_on, ev.t)
    const baseFirst = firstHitAfter(result.baseline.log, ev.bands_on, ev.t)
    alerts.push({
      t: ev.t,
      type: ev.type,
      bands: ev.bands_on,
      smartDelay: smartFirst !== null ? smartFirst - ev.t : null,
      baselineDelay: baseFirst !== null ? baseFirst - ev.t : null,
    })
  }
  return alerts.reverse()
}

function firstHitAfter(log: { t: number; band: number; hit: boolean }[], bands: number[], t0: number): number | null {
  for (const e of log) {
    if (e.t >= t0 && e.hit && bands.includes(e.band)) return e.t
  }
  return null
}

type Acc = {
  truth: boolean[][]
  smartLog: StepLog[]
  baseLog: StepLog[]
  prio: number[][]
  scores: number[][]
  prior: number[][]
  ucb: number[][]
  events: SimEvent[]
  hits: number
  model_info: SimResult['smart']['model_info']
}

function scenarioLabel(id: string): string {
  if (id === 'tsrd') return 'TSRD · 10 real reference bands'
  return id
}

function bandEdges(n: number, start = 2, end = 18): [number, number][] {
  const w = (end - start) / n
  return Array.from({ length: n }, (_, i) => [+(start + i * w).toFixed(2), +(start + (i + 1) * w).toFixed(2)])
}

export function buildLiveResult(cfg: SimConfig, acc: Acc): SimResult {
  const n = cfg.n_bands
  const scans = acc.smartLog.length
  const hits = acc.hits
  const misses = scans - hits
  const metrics: Metrics = {
    interception_ratio: scans ? (hits / scans) * 100 : 0,
    avg_intercept_time: 0,
    miss_count: misses,
    probability_of_detection: scans ? hits / Math.max(1, misses + hits) : 0,
    probability_of_false_alarm: 0,
    total_reward: hits - 0.5 * misses,
    avg_reward: scans ? (hits - 0.5 * misses) / scans : 0,
    hits,
    scans,
    adaptation_speed: null,
    correct_predictions_pct: 0,
  }
  const activity = Array(n).fill(0)
  for (const row of acc.truth) row.forEach((on, b) => void (on && activity[b]++))
  return {
    config: cfg as unknown as Record<string, unknown>,
    scenario_id: cfg.scenario,
    scenario_label: scenarioLabel(cfg.scenario),
    ground_truth: acc.truth,
    band_edges_ghz: bandEdges(n),
    activity_profile: activity,
    events: acc.events,
    n_segments: 0,
    total_transmissions: 0,
    baseline: { label: 'Round-robin (Open-loop)', scheduler: 'round_robin', log: acc.baseLog, metrics },
    smart: {
      label: 'UCB1 + Random Forest',
      scheduler: cfg.scheduler,
      log: acc.smartLog,
      metrics,
      priorities: acc.prio,
      model_info: acc.model_info,
    },
    meta: { n_bands: n, n_steps: cfg.n_steps, seed: cfg.seed, alpha: cfg.alpha, epsilon: cfg.epsilon },
  }
}

export function useSim() {
  const [cfg, setCfg] = useState<SimConfig>(DEFAULT_CFG)
  const [result, setResult] = useState<SimResult | null>(null)
  const [cur, setCur] = useState(0)
  const [playing, setPlaying] = useState(false)
  const [speed, setSpeed] = useState(1)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [source, setSource] = useState<Source | null>(null)
  const stopRef = useRef(false)
  const liveRef = useRef<{ close: () => void } | null>(null)

  const T = result?.meta.n_steps ?? 0

  const closeLive = useCallback(() => {
    liveRef.current?.close()
    liveRef.current = null
  }, [])

  // ------------------------------------------------------------------ replay
  const run = useCallback(
    async (next?: Partial<SimConfig>) => {
      const merged = { ...DEFAULT_CFG, ...cfg, ...next } as SimConfig
      if (next) setCfg(merged)
      setLoading(true)
      setError(null)
      stopRef.current = false
      closeLive()
      try {
        const res = await api.simulate(merged)
        if (stopRef.current) return
        setSource('replay')
        setResult(res)
        setCur(0)
        setPlaying(true)
      } catch (e) {
        // network/backend failure -> fall back to bundled offline mock
        const msg = e instanceof Error ? e.message : String(e)
        try {
          const mock = await api.mock()
          if (stopRef.current) return
          setSource('replay')
          setResult(mock)
          setCur(0)
          setPlaying(true)
          setError(`Backend offline — loaded bundled MOCK data (${msg})`)
        } catch {
          setError(msg)
        }
      } finally {
        setLoading(false)
      }
    },
    [cfg, closeLive],
  )

  // -------------------------------------------------------------------- live
  const startLive = useCallback(
    async (next?: Partial<SimConfig>) => {
      const merged = { ...DEFAULT_CFG, ...cfg, ...next } as SimConfig
      if (next) setCfg(merged)
      setLoading(true)
      setError(null)
      stopRef.current = false
      closeLive()

      // quick probe — if the backend is unreachable, fall back to replay/mock
      try {
        await api.health()
      } catch {
        await run(merged)
        return
      }

      const acc: Acc = { truth: [], smartLog: [], baseLog: [], prio: [], scores: [], prior: [], ucb: [], events: [], hits: 0, model_info: null }
      setSource('live')
      setPlaying(true)

      const close = wsStream(
        merged,
        (ev) => {
          if (stopRef.current) return
          const e = ev as LiveEvent
          acc.truth.push(e.truth)
          acc.baseLog.push({ t: e.t, band: e.baseline.band, hit: e.baseline.hit, snr: e.baseline.snr, reward: 0, ratio: 0 })
          acc.smartLog.push({ t: e.t, band: e.smart.band, hit: e.smart.hit, snr: e.smart.snr, reward: e.smart.hit ? 1 : -0.5, ratio: 0 })
          acc.prio.push(e.priorities ?? [])
          acc.scores.push(e.scores ?? [])
          acc.prior.push(e.prior ?? [])
          acc.ucb.push(e.ucb ?? [])
          if (e.smart.hit) acc.hits += 1
          if (e.model) acc.model_info = e.model
          if (e.event) acc.events.push(e.event)
          setResult(buildLiveResult(merged, acc))
          setCur(e.t)
        },
        () => {
          if (!stopRef.current) setPlaying(false)
        },
      )
      liveRef.current = { close }
      setLoading(false)
    },
    [cfg, closeLive, run],
  )

  const loadMock = useCallback(async () => {
    setLoading(true)
    setError(null)
    stopRef.current = false
    closeLive()
    try {
      const mock = await api.mock()
      if (stopRef.current) return
      setSource('replay')
      setResult(mock)
      setCur(0)
      setPlaying(true)
      setError('Loaded bundled MOCK data (offline demo — no backend needed)')
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e))
    } finally {
      setLoading(false)
    }
  }, [closeLive])

  const loadRun = useCallback(
    (res: SimResult) => {
      stopRef.current = false
      closeLive()
      setSource('replay')
      setResult(res)
      setCur(0)
      setPlaying(false)
      setError(null)
    },
    [closeLive],
  )

  // replay animation timer (live mode is driven by the WebSocket instead)
  useEffect(() => {
    if (source !== 'replay' || !playing || !result) return
    const id = setInterval(() => {
      setCur((c) => {
        if (c >= T - 1) {
          setPlaying(false)
          return c
        }
        return c + 1
      })
    }, 1000 / speed)
    return () => clearInterval(id)
  }, [source, playing, result, speed, T])

  const stop = useCallback(() => {
    stopRef.current = true
    closeLive()
    setPlaying(false)
  }, [closeLive])

  const seek = useCallback(
    (t: number) => {
      if (source !== 'replay') return
      setCur(Math.max(0, Math.min(t, T > 0 ? T - 1 : 0)))
    },
    [source, T],
  )

  const toggle = useCallback(() => setPlaying((p) => !p), [])

  return { cfg, setCfg, result, cur, playing, speed, setSpeed, loading, error, source, run, startLive, toggle, seek, stop, loadMock, loadRun, T }
}