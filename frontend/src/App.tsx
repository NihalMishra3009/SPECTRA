import { useCallback, useEffect, useMemo, useRef, useState } from 'react'
import { motion, AnimatePresence } from 'framer-motion'
import { useSim, computeAlerts } from './hooks/useSim'
import TopBar from './components/TopBar'
import ConfigPanel from './components/ConfigPanel'
import WaterfallChart from './components/WaterfallChart'
import ComparisonChart from './components/ComparisonChart'
import SpectrumMiniMap from './components/SpectrumMiniMap'
import KPIs from './components/KPIs'
import AlertFeed from './components/AlertFeed'
import PlaybackControls from './components/PlaybackControls'
import ModelPanel from './components/ModelPanel'
import RunHistory from './components/RunHistory'

const DEMO_STEPS = [
  { cfg: { scenario: 'stable_switch_surprise', scheduler: 'thompson' }, label: 'Stage 1-3 · Stable + Switch + Surprise' },
  { cfg: { scenario: 'surprise', scheduler: 'epsilon_greedy' }, label: 'Surprise-emitter detection' },
  { cfg: { scenario: 'periodic_only', scheduler: 'sequence' }, label: 'LSTM timing prediction' },
  { cfg: { scenario: 'hopper', scheduler: 'sequence' }, label: 'Frequency-hopping interception' },
]

export default function App() {
  const sim = useSim()
  const [demoIdx, setDemoIdx] = useState(0)
  const [demoActive, setDemoActive] = useState(false)
  const [waterfallSide, setWaterfallSide] = useState<'base' | 'smart'>('smart')
  const finishedRef = useRef(true)

  const { result, cur, playing, speed, cfg, setCfg } = sim
  const alerts = useMemo(() => computeAlerts(result), [result])
  const base = result?.baseline ?? null
  const smart = result?.smart ?? null
  const truth = result?.ground_truth ?? null
  const T = result?.meta.n_steps ?? 0

  const startDemo = useCallback(() => {
    setDemoActive(true)
    setDemoIdx(0)
    sim.setSpeed(2)
    finishedRef.current = false
    sim.run({ ...DEMO_STEPS[0].cfg, n_steps: 300 })
  }, [sim])

  useEffect(() => {
    if (!demoActive || !result || !playing) return
    if (cur < T - 1) return
    if (finishedRef.current) return
    finishedRef.current = true
    const id = setTimeout(() => {
      const nextIdx = demoIdx + 1
      if (nextIdx >= DEMO_STEPS.length) {
        setDemoActive(false)
        return
      }
      setDemoIdx(nextIdx)
      finishedRef.current = false
      sim.setSpeed(2)
      sim.run({ ...DEMO_STEPS[nextIdx].cfg, n_steps: 300 })
    }, 1200)
    return () => clearTimeout(id)
  }, [demoActive, demoIdx, result, cur, playing, T, sim])

  const waterfall = waterfallSide === 'base' ? base : smart
  const hitCounts = useMemo(() => {
    if (!smart) return undefined
    const counts: number[] = []
    for (let b = 0; b < cfg.n_bands; b++) counts.push(smart.log.filter((e) => e.band === b && e.hit).length)
    return counts
  }, [smart, cfg.n_bands])

  return (
    <div className="flex h-full flex-col gap-3 p-3">
      <TopBar
        scenarioLabel={result?.scenario_label ?? cfg.scenario}
        scheduler={cfg.scheduler}
        playing={playing}
        onDemo={startDemo}
        demoActive={demoActive}
        smartAlgo={smart?.scheduler === 'sequence' ? 'LSTM' : smart?.scheduler === 'rl_dqn' || smart?.scheduler === 'rl_ppo' ? smart.scheduler : null}
        isMock={!!(result as { mock?: boolean } | null)?.mock}
      />

      {demoActive && (
        <div className="flex items-center gap-2 text-[11px] text-amber">
          <span className="animate-pulse">●</span> DEMO MODE — {demoIdx + 1}/{DEMO_STEPS.length}: {DEMO_STEPS[demoIdx].label}
        </div>
      )}

      <div className="grid min-h-0 flex-1 grid-cols-1 gap-3 lg:grid-cols-[300px_1fr_330px]">
        {/* left column */}
        <div className="flex min-h-0 flex-col gap-3 overflow-y-auto pr-1 scroll-slim">
          <ConfigPanel cfg={cfg} setCfg={setCfg} run={sim.run} loadMock={sim.loadMock} loading={sim.loading} error={sim.error} />
          <ModelPanel cfg={cfg} setCfg={setCfg} />
        </div>

        {/* center column */}
        <div className="flex min-h-0 flex-col gap-3">
          <div className="panel flex min-h-0 flex-1 flex-col p-3">
            <div className="mb-1 flex items-center justify-between px-1">
              <div className="flex items-center gap-2">
                <button
                  onClick={() => setWaterfallSide('base')}
                  className={`btn h-7 px-2 py-0 text-[11px] ${waterfallSide === 'base' ? 'btn-ghost border-neon/50 text-neon2' : 'btn-ghost'}`}
                >
                  baseline
                </button>
                <button
                  onClick={() => setWaterfallSide('smart')}
                  className={`btn h-7 px-2 py-0 text-[11px] ${waterfallSide === 'smart' ? 'btn-primary' : 'btn-ghost'}`}
                >
                  smart
                </button>
              </div>
              <span className="text-[10px] text-dim uppercase">ground truth + scan overlay</span>
            </div>
            {truth && waterfall ? (
              <WaterfallChart
                truth={truth}
                log={waterfall.log}
                events={result!.events}
                cur={cur}
                label={waterfall.label}
                algoName={waterfall.scheduler}
                bandEdges={result!.band_edges_ghz}
              />
            ) : (
              <EmptyHint />
            )}
            {result && <PlaybackControls cur={cur} T={T} playing={playing} speed={speed} onToggle={sim.toggle} onSeek={sim.seek} onSpeed={() => sim.setSpeed([0.5, 1, 2, 4][([0.5, 1, 2, 4].indexOf(speed) + 1) % 4])} />}
          </div>

          <div className="panel p-3">
            <div className="panel-title mb-1 px-1">SMART vs BASELINE · time evolution</div>
            {result ? (
              <ComparisonChart baseline={base!.log} smart={smart!.log} />
            ) : (
              <div className="flex h-40 items-center justify-center text-sm text-dim">Waiting for a run…</div>
            )}
          </div>
        </div>

        {/* right column */}
        <div className="flex min-h-0 flex-col gap-3 overflow-y-auto pl-1 scroll-slim">
          <div className="panel p-4">
            <div className="panel-title mb-2">LIVE METRICS</div>
            <KPIs base={base?.metrics ?? null} smart={smart?.metrics ?? null} />
          </div>
          <div className="panel p-4">
            <div className="panel-title mb-2">SPECTRUM THREAT MINI-MAP</div>
            {result ? (
              <SpectrumMiniMap profile={result.activity_profile} hits={hitCounts} nColumns={cfg.n_bands} bandEdges={result.band_edges_ghz} />
            ) : (
              <div className="flex h-28 items-center justify-center text-sm text-dim">—</div>
            )}
          </div>
          <div className="panel p-4">
            <div className="panel-title mb-2">EVENT DETECTION LOG</div>
            {alerts.length ? <AlertFeed alerts={alerts} cur={cur} /> : <div className="text-sm text-dim">No events yet — run a scenario with a switch/surprise.</div>}
          </div>
          <RunHistory loadRun={sim.loadRun} />
        </div>
      </div>
    </div>
  )
}

function EmptyHint() {
  return (
    <div className="flex flex-1 items-center justify-center">
      <div className="text-center">
        <div className="mb-2 text-3xl">📡</div>
        <div className="text-sm text-dim">Configure and run a simulation — the animated waterfall appears here.</div>
      </div>
    </div>
  )
}