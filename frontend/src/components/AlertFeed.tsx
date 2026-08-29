import { AnimatePresence, motion } from 'framer-motion'
import type { AlertItem } from '../hooks/useSim'

type Props = { alerts: AlertItem[]; cur: number }

export default function AlertFeed({ alerts, cur }: Props) {
  const relevant = alerts.filter((a) => a.t <= cur).slice(0, 8)
  return (
    <div className="flex flex-col gap-2">
      <AnimatePresence mode="popLayout">
        {relevant.map((a) => {
          const fresh = cur - a.t < 30
          const detected = a.smartDelay !== null
          return (
            <motion.div
              key={`${a.t}-${a.bands.join('.')}`}
              layout
              initial={{ opacity: 0, x: 24 }}
              animate={{ opacity: 1, x: 0 }}
              exit={{ opacity: 0, x: -24 }}
              className={`rounded-xl border px-3 py-2 text-xs ${
                fresh ? 'border-amber/60 bg-amber/10' : a.type === 'surprise' ? 'border-amber/40 bg-inner' : 'border-line bg-inner'
              }`}
            >
              <div className="flex items-center justify-between">
                <span className={`font-display font-semibold ${a.type === 'surprise' ? 'text-amber' : 'text-violet-300'}`}>
                  {a.type === 'surprise' ? '⚠ SURPRISE EMITTER' : '◈ PATTERN CHANGE'}
                </span>
                <span className="num text-dim">t={a.t}</span>
              </div>
              <div className="mt-1 text-dim">new active bands: {a.bands.map((b) => `B${b}`).join(' · ')}</div>
              <div className="num mt-1 flex items-center justify-between">
                {detected ? (
                  <span className="text-hit">smart detected +{a.smartDelay}t</span>
                ) : (
                  <span className="text-miss">smart MISSED</span>
                )}
                <span className={a.baselineDelay !== null ? 'text-dim' : 'text-miss'}>
                  baseline {a.baselineDelay !== null ? `+${a.baselineDelay}t` : 'missed'}
                </span>
              </div>
            </motion.div>
          )
        })}
      </AnimatePresence>
    </div>
  )
}