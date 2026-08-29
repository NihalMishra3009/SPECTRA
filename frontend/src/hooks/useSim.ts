import { useCallback, useEffect, useRef, useState } from 'react'
import { api, SimConfig, SimResult } from '../api/client'

export const DEFAULT_CFG: SimConfig = {
  n_bands: 10,
  n_steps: 300,
  seed: 2024,
  scenario: 'stable_switch_surprise',
  scheduler: 'thompson',
  alpha: 0.25,
  epsilon: 0.05,
  window: 40,
  floor: 0.15,
}

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

export function useSim() {
  const [cfg, setCfg] = useState<SimConfig>(DEFAULT_CFG)
  const [result, setResult] = useState<SimResult | null>(null)
  const [cur, setCur] = useState(0)
  const [playing, setPlaying] = useState(false)
  const [speed, setSpeed] = useState(1)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const stopRef = useRef(false)

  const T = result?.meta.n_steps ?? 0

  const run = useCallback(async (next?: Partial<SimConfig>) => {
    const merged = { ...DEFAULT_CFG, ...cfg, ...next } as SimConfig
    if (next) setCfg(merged)
    setLoading(true)
    setError(null)
    stopRef.current = false
    try {
      const res = await api.simulate(merged)
      if (stopRef.current) return
      setResult(res)
      setCur(0)
      setPlaying(true)
    } catch (e) {
      // network/backend failure -> fall back to bundled offline mock
      const msg = e instanceof Error ? e.message : String(e)
      try {
        const mock = await api.mock()
        if (stopRef.current) return
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
  }, [cfg])

  const loadMock = useCallback(async () => {
    setLoading(true)
    setError(null)
    stopRef.current = false
    try {
      const mock = await api.mock()
      if (stopRef.current) return
      setResult(mock)
      setCur(0)
      setPlaying(true)
      setError('Loaded bundled MOCK data (offline demo — no backend needed)')
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e))
    } finally {
      setLoading(false)
    }
  }, [])

  const loadRun = useCallback((res: SimResult) => {
    stopRef.current = false
    setResult(res)
    setCur(0)
    setPlaying(false)
    setError(null)
  }, [])

  useEffect(() => {
    if (!playing || !result) return
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
  }, [playing, result, speed, T])

  const stop = useCallback(() => {
    stopRef.current = true
    setPlaying(false)
  }, [])

  const seek = useCallback((t: number) => {
    setCur(Math.max(0, Math.min(t, T > 0 ? T - 1 : 0)))
  }, [T])

  const toggle = useCallback(() => setPlaying((p) => !p), [])

  return { cfg, setCfg, result, cur, playing, speed, setSpeed, loading, error, run, toggle, seek, stop, loadMock, loadRun, T }
}