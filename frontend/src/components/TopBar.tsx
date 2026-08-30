import { motion } from 'framer-motion'

type Props = {
  scenarioLabel: string
  playing: boolean
  isMock?: boolean
  source: 'live' | 'replay' | null
  modelType?: string | null
}

export default function TopBar({ scenarioLabel, playing, isMock, source, modelType }: Props) {
  return (
    <motion.header
      initial={{ opacity: 0, y: -10 }}
      animate={{ opacity: 1, y: 0 }}
      className="panel flex items-center justify-between gap-4 px-5 py-3"
    >
      <div className="flex items-center gap-3">
        <div className="flex h-9 w-9 items-center justify-center rounded-xl bg-neon/10 ring-1 ring-neon/40">
          <svg width="20" height="20" viewBox="0 0 32 32" fill="none">
            <path d="M8 22V10h2.4l5.6 7.4V10h2.4v12h-2.4l-5.6-7.4V22zM22 10h2.6l4.4 12h-2.7l-.8-2.4H20.7l-.8 2.4h-2.7l4.4-12H22zm.3 7.3l-1-3-1 3h2z" fill="#00e5ff" />
          </svg>
        </div>
        <div>
          <h1 className="font-display text-lg font-bold leading-none tracking-tight text-slate-50">
            SPECTRA
          </h1>
          <p className="text-[11px] tracking-widest text-dim uppercase">UCB1 + Random Forest · EW Smart Scan</p>
        </div>
      </div>

      <div className="hidden items-center gap-2 lg:flex">
        <Chip label="scenario" value={scenarioLabel} accent="text-neon2" />
        <Chip label="algorithm" value="UCB1 + Random Forest" accent="text-amber" />
        {modelType && <Chip label="model" value={modelType} accent="text-hit" />}
        {source === 'live' && <Chip label="feed" value="live ws" accent="text-hit" />}
        {source === 'replay' && <Chip label="feed" value="replay" accent="text-neon" />}
        {isMock && <Chip label="source" value="offline mock" accent="text-violet-300" />}
      </div>

      <div className={`flex items-center gap-1.5 rounded-full border px-3 py-1 text-[11px] ${playing ? 'border-hit/60 bg-hit/10 text-hit' : 'border-line bg-inner/50 text-dim'}`}>
        <span className={`relative flex h-2 w-2 ${playing ? '' : 'opacity-40'}`}>
          {playing && <span className="absolute inline-flex h-full w-full animate-ping rounded-full bg-hit opacity-60" />}
          <span className="relative inline-flex h-2 w-2 rounded-full bg-current" />
        </span>
        {playing ? 'LIVE' : 'READY'}
      </div>
    </motion.header>
  )
}

function Chip({ label, value, accent }: { label: string; value: string; accent: string }) {
  return (
    <div className="rounded-lg border border-line bg-inner/60 px-3 py-1.5">
      <div className="text-[9px] tracking-widest text-dim uppercase">{label}</div>
      <div className={`text-xs font-semibold ${accent}`}>{value}</div>
    </div>
  )
}