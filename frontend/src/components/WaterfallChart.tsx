import { useMemo } from 'react'
import * as echarts from 'echarts'
import EChart from './EChart'
import type { SimEvent, StepLog } from '../api/client'

type Props = {
  truth: boolean[][]
  log: StepLog[]
  events: SimEvent[]
  cur: number
  label: string
  algoName?: string
  bandEdges?: [number, number][]
}

export default function WaterfallChart({ truth, log, events, cur, label, algoName, bandEdges }: Props) {
  const edges = bandEdges ?? Array.from({ length: truth[0]?.length ?? 1 }, (_, i) => [i * 1.6, (i + 1) * 1.6])
  const option = useMemo(() => {
    const nBands = truth[0]?.length ?? 1
    const data: [number, number, number][] = []
    for (let b = 0; b < nBands; b++) {
      for (let t = 0; t <= cur; t++) {
        let v = truth[t]?.[b] ? 1 : 0
        const sc = log[t]
        if (sc) v = sc.hit ? 2 : 3
        data.push([t, b, v])
      }
    }
    const eventLines = events
      .filter((e) => e.t <= cur)
      .map((e) => ({
        xAxis: e.t,
        lineStyle: {
          color: e.type === 'surprise' ? '#ffb020' : '#8b5cf6',
          width: 2,
          type: 'dashed',
        },
        label: {
          formatter: e.type === 'surprise' ? '! SURPRISE' : 'switch',
          position: 'insideTop',
          color: '#ffb020',
          fontSize: 10,
        },
      }))

    const edgeOf = (b: number) => ({
      lo: edges[b]?.[0] ?? 0,
      hi: edges[b]?.[1] ?? 0,
      mid: (((edges[b]?.[0] ?? 0) + (edges[b]?.[1] ?? 0)) / 2).toFixed(1),
    })

    const option = {
      animation: false,
      grid: { left: 52, right: 16, top: 34, bottom: 30 },
      tooltip: {
        formatter: (p: any) => {
          const b = p.data[1]
          const t = p.data[0]
          const v = p.data[2]
          const e = edgeOf(b)
          const label = v === 1 ? '● emitter active (signal)' : v === 2 ? '◉ scan + HIT' : v === 3 ? '◌ scan MISS' : '—'
          return `t=${t}  B${b} · ${e.lo.toFixed(1)}–${e.hi.toFixed(1)} GHz<br/>${label}`
        },
        backgroundColor: '#0a111f',
        borderColor: '#1b2a44',
        textStyle: { color: '#dbe4f5', fontSize: 12 },
      },
      xAxis: {
        type: 'category',
        data: Array.from({ length: cur + 1 }, (_, i) => i),
        axisLabel: { color: '#7b8aa6', fontSize: 10, interval: Math.max(1, Math.floor(cur / 12)) },
        axisLine: { lineStyle: { color: '#1b2a44' } },
        splitLine: { show: false },
        name: 'time step',
        nameTextStyle: { color: '#7b8aa6', fontSize: 10 },
      },
      yAxis: {
        type: 'category',
        data: Array.from({ length: nBands }, (_, i) => i).reverse(),
        axisLabel: {
          color: '#7b8aa6',
          fontSize: 11,
          formatter: (val: string) => {
            const b = Number(val)
            const e = edgeOf(b)
            return `B${b} ${e.mid}G`
          },
        },
        axisLine: { lineStyle: { color: '#1b2a44' } },
        name: 'band · freq (GHz)',
        nameTextStyle: { color: '#7b8aa6', fontSize: 10 },
      },
      visualMap: {
        show: false,
        min: 0,
        max: 3,
        pieces: [
          { value: 1, color: 'rgba(0,229,255,0.17)' },
          { value: 2, color: '#22e584' },
          { value: 3, color: '#ff4d6d' },
        ],
      },
      series: [
        {
          type: 'heatmap',
          data: data as any,
          markLine: {
            symbol: 'none',
            silent: true,
            data: [
              {
                xAxis: cur,
                lineStyle: { color: '#ffffff', width: 2, type: 'solid', shadowColor: '#00e5ff', shadowBlur: 8 },
                label: { formatter: '◄ scan', position: 'insideTop', color: '#00e5ff', fontSize: 10 },
              },
              ...eventLines,
            ],
          },
        },
      ],
    }
    return option as unknown as echarts.EChartsOption
  }, [truth, log, events, cur, edges])

  return (
    <div className="flex h-full w-full flex-col">
      <div className="mb-1 flex items-baseline justify-between px-1">
        <span className="panel-title">
          {label} {algoName ? `/ ${algoName}` : ''}
        </span>
        <span className="num text-[11px] text-neon2">t={cur}</span>
      </div>
      <EChart option={option} className="min-h-0 flex-1" />
    </div>
  )
}