import { useEffect, useState } from 'react'
import { useSim } from './hooks/useSim'
import TopBar from './components/TopBar'
import ConfigPanel from './components/ConfigPanel'
import BandPanel from './components/BandPanel'
import HistoryFeed from './components/HistoryFeed'
import { api, TsrdBand } from './api/client'

export default function App() {
  const sim = useSim()
  const [tsrd, setTsrd] = useState<TsrdBand[]>([])

  useEffect(() => {
    api.tsrdBands().then((r) => setTsrd(r.bands)).catch(() => {})
  }, [])

  const { result, cur, playing, cfg, setCfg, source } = sim
  const smart = result?.smart ?? null
  const isMock = !!(result as { mock?: boolean } | null)?.mock

  return (
    <div className="flex h-full flex-col gap-3 p-3">
      <TopBar
        scenarioLabel={result?.scenario_label ?? cfg.scenario}
        playing={playing}
        isMock={isMock}
        source={source}
        modelType={smart?.model_info?.model_type ?? null}
      />

      <div className="grid min-h-0 flex-1 grid-cols-1 gap-3 xl:grid-cols-[300px_1fr_340px]">
        {/* left column — config */}
        <div className="flex min-h-0 flex-col gap-3 overflow-y-auto pr-1 scroll-slim">
          <ConfigPanel
            cfg={cfg}
            setCfg={setCfg}
            run={sim.run}
            startLive={sim.startLive}
            loadMock={sim.loadMock}
            loading={sim.loading}
            error={sim.error}
            source={source}
          />
        </div>

        {/* center column — the 10-band box */}
        <div className="flex min-h-0 flex-col overflow-y-auto scroll-slim">
          <BandPanel result={result} cur={cur} cfg={cfg} tsrd={tsrd} source={source} />
        </div>

        {/* right column — history + model status */}
        <div className="flex min-h-0 flex-col gap-3 overflow-y-auto pl-1 scroll-slim">
          <HistoryFeed
            log={smart?.log ?? []}
            cur={cur}
            edges={result?.band_edges_ghz}
            model={smart?.model_info ?? null}
          />
        </div>
      </div>
    </div>
  )
}