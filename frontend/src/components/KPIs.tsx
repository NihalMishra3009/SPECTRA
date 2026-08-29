import { MotionConfig, motion } from 'framer-motion'
import type { Metrics } from '../api/client'

type Props = { base: Metrics | null; smart: Metrics | null }

function Delta({ higherBetter, base, smart }: { higherBetter: boolean; base: number; smart: number }) {
  const d = smart - base
  if (Math.abs(d) < 0.005) return <span className="num text-[11px] text-dim">■ ±0</span>
  const good = higherBetter ? d > 0 : d < 0
  return (
    <span className={`num text-[11px] ${good ? 'text-hit' : 'text-miss'}`}>
      {d > 0 ? '▲' : '▼'} {Math.abs(d).toFixed(2)}
    </span>
  )
}

const fmt = (v: number, digits = 2) => v.toLocaleString(undefined, { maximumFractionDigits: digits })

export default function KPIs({ base, smart }: Props) {
  if (!base || !smart) {
    return (
      <div className="panel flex h-full items-center justify-center p-4">
        <span className="text-sm text-dim">Run a simulation to see live KPIs</span>
      </div>
    )
  }

  const cells: { k: string; hb: boolean; sv: number; bv: number; f: string; skip?: boolean }[] = [
    { k: 'Interception ratio', hb: true, sv: smart.interception_ratio, bv: base.interception_ratio, f: `${fmt(smart.interception_ratio, 1)}%` },
    { k: 'Avg intercept time', hb: false, sv: smart.avg_intercept_time, bv: base.avg_intercept_time, f: `${fmt(smart.avg_intercept_time, 1)}t` },
    { k: 'Miss count', hb: false, sv: smart.miss_count, bv: base.miss_count, f: `${smart.miss_count}` },
    { k: 'Avg reward · waste cut', hb: true, sv: smart.avg_reward, bv: base.avg_reward, f: fmt(smart.avg_reward) },
    { k: 'Probability Pd', hb: true, sv: smart.probability_of_detection * 100, bv: base.probability_of_detection * 100, f: `${fmt(smart.probability_of_detection * 100, 1)}%` },
    { k: 'Correct prediction', hb: true, sv: smart.correct_predictions_pct, bv: base.correct_predictions_pct, f: `${fmt(smart.correct_predictions_pct, 1)}%` },
    { k: 'Hits', hb: true, sv: smart.hits, bv: base.hits, f: `${smart.hits}` },
    {
      k: 'Adaptation speed',
      hb: false,
      sv: smart.adaptation_speed ?? 0,
      bv: base.adaptation_speed ?? 0,
      f: smart.adaptation_speed == null ? '—' : `${fmt(smart.adaptation_speed, 0)}t`,
      skip: smart.adaptation_speed == null,
    },
  ]

  return (
    <MotionConfig reducedMotion="user">
      <div className="grid grid-cols-2 gap-2">
        {cells.map((c, i) => (
          <motion.div
            key={c.k}
            initial={{ opacity: 0, y: 6 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: i * 0.04 }}
            className="kpi-card"
          >
            <div className="text-[10px] font-medium tracking-wide text-dim uppercase">{c.k}</div>
            <div className="num mt-1 text-xl font-semibold text-slate-100">{c.f}</div>
            <div className="mt-1 flex items-center justify-between text-[11px]">
              <span className="text-dim">base {fmt(c.bv, 1)}</span>
              {!c.skip && <Delta higherBetter={c.hb} base={c.bv} smart={c.sv} />}
            </div>
          </motion.div>
        ))}
      </div>
    </MotionConfig>
  )
}