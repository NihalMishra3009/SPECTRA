export type StepLog = {
  t: number
  band: number
  hit: boolean
  snr: number
  reward: number
  ratio: number
}

export type Metrics = {
  interception_ratio: number
  avg_intercept_time: number
  miss_count: number
  probability_of_detection: number
  probability_of_false_alarm: number
  total_reward: number
  avg_reward: number
  hits: number
  scans: number
  adaptation_speed: number | null
  correct_predictions_pct: number
}

export type ScannerResult = {
  label: string
  scheduler: string
  log: StepLog[]
  metrics: Metrics
  priorities?: number[][]
}

export type SimEvent = {
  t: number
  type: 'change' | 'surprise'
  bands_on: number[]
  bands_off: number[]
  surprise: number[]
}

export type SimConfig = {
  n_bands: number
  n_steps: number
  seed: number
  scenario: string
  scheduler: string
  alpha: number
  epsilon: number
  window: number
  floor: number
}

export type SimResult = {
  run_id?: string | null
  config: Record<string, unknown>
  scenario_id: string
  scenario_label: string
  ground_truth: boolean[][]
  band_edges_ghz?: [number, number][]
  activity_profile: number[]
  events: SimEvent[]
  n_segments: number
  total_transmissions: number
  baseline: ScannerResult
  smart: ScannerResult
  meta: { n_bands: number; n_steps: number; seed: number; alpha: number; epsilon: number }
}

export type ScenarioMeta = { id: string; label: string; desc: string }
export type SchedulerMeta = { id: string; desc: string }
export type ModelStatus = {
  scheduler: string
  present: boolean
  meta: {
    algo?: string
    final_interception_pct?: number
    wall_seconds?: number
    total_timesteps?: number
    final_val_acc?: number
    seed?: number
    threshold?: number
    n_bands?: number
  } | null
  curves_file?: string
}

export type DbRunSummary = {
  run_id: string
  created_at: string
  scenario_id: string
  scenario_label: string
  scheduler: string
  n_bands: number
  n_steps: number
  seed: number | null
  outcome: {
    smart_ir?: number
    baseline_ir?: number
    smart_reward?: number
    baseline_reward?: number
    miss_saved?: number
    events?: number
  }
}

export type DbStats = {
  total_runs: number
  telemetry_samples: number
  scenario_breakdown: Record<string, number>
  best_run: { scenario_id?: string; scheduler?: string; ir?: number | null } | null
  db_path?: string
}

const BASE = (import.meta.env.VITE_API_URL as string | undefined) ?? ''

async function j<T>(path: string, init?: RequestInit): Promise<T> {
  const res = await fetch(`${BASE}${path}`, init)
  if (!res.ok) throw new Error(`${path} -> ${res.status}`)
  return res.json() as Promise<T>
}

export const api = {
  scenarios: () => j<{ scenarios: ScenarioMeta[] }>('/api/scenarios'),
  schedulers: () => j<{ schedulers: SchedulerMeta[] }>('/api/schedulers'),
  models: () => j<{ models: ModelStatus[] }>('/api/models'),
  curves: (name: string) => j<unknown[]>(`/api/curves/${name}`),
  // Bundled offline fixture (served from public/mock — works with no backend).
  mock: () =>
    fetch(`${BASE}/mock/demo.json`).then((r) => {
      if (!r.ok) throw new Error(`mock -> ${r.status}`)
      return r.json() as Promise<SimResult>
    }),
  simulate: (cfg: Partial<SimConfig>) =>
    j<SimResult>('/api/simulate', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(cfg),
    }),
  dbRuns: (limit = 12) => j<{ runs: DbRunSummary[] }>(`/api/db/runs?limit=${limit}`),
  dbRun: (runId: string) => j<SimResult>(`/api/db/runs/${runId}`),
  dbStats: () => j<DbStats>('/api/db/stats'),
}

export function wsStream(cfg: Partial<SimConfig>, onEvent: (ev: Record<string, unknown>) => void, onDone: () => void) {
  const proto = location.protocol === 'https:' ? 'wss' : 'ws'
  const url = `${proto}://${location.host}/ws/simulate`
  const ws = new WebSocket(url)
  ws.onopen = () => ws.send(JSON.stringify(cfg))
  ws.onmessage = (msg) => onEvent(JSON.parse(msg.data as string))
  ws.onclose = onDone
  return () => ws.close()
}