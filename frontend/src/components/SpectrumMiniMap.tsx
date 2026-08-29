import { useMemo } from 'react'
import * as echarts from 'echarts'
import EChart from './EChart'

type Props = {
  profile: number[]
  hits?: number[]
  nColumns?: number
  height?: string
  bandEdges?: [number, number][]
}

export default function SpectrumMiniMap({ profile, hits, nColumns, height = 'h-32', bandEdges }: Props) {
  const option = useMemo(() => {
    const bands = profile.map((_, i) => {
      const e = bandEdges?.[i]
      return e ? `B${i} ${((e[0] + e[1]) / 2).toFixed(1)}G` : `B${i}`
    })
    const maxV = Math.max(1, ...profile)
    const option: echarts.EChartsOption = {
      backgroundColor: 'transparent',
      grid: { left: 30, right: 8, top: 10, bottom: 26 },
      tooltip: {
        trigger: 'axis',
        backgroundColor: '#0a111f',
        borderColor: '#1b2a44',
        textStyle: { color: '#dbe4f5', fontSize: 11 },
        formatter: (p: any) => {
          const i = p[0]?.dataIndex ?? 0
          const e = bandEdges?.[i]
          const rng = e ? `${e[0].toFixed(1)}–${e[1].toFixed(1)} GHz` : ''
          const hitsTxt = hits ? ` · smart hits ${hits[i]}` : ''
          return `B${i}${rng ? ` (${rng})` : ''}: ${profile[i]} signals${hitsTxt}`
        },
      },
      xAxis: {
        type: 'category',
        data: bands,
        axisLabel: {
          color: '#7b8aa6',
          fontSize: 9,
          interval: nColumns ? Math.max(0, Math.floor(profile.length / nColumns) - 1) : 'auto',
        },
        axisLine: { lineStyle: { color: '#1b2a44' } },
      },
      yAxis: {
        type: 'value',
        max: maxV * 1.1 || 1,
        axisLabel: { color: '#6b7a96', fontSize: 9 },
        splitLine: { lineStyle: { color: '#101a2c' } },
      },
      series: [
        {
          type: 'bar',
          data: profile.map((v, i) => ({
            value: v,
            itemStyle: {
              borderRadius: [3, 3, 0, 0],
              color: v > 0.6 * maxV ? '#00e5ff' : v > 0.25 * maxV ? 'rgba(0,229,255,0.6)' : 'rgba(0,229,255,0.25)',
            },
          })),
          barMaxWidth: 14,
          emphasis: { itemStyle: { color: '#ffb020' } },
          z: 2,
        },
        ...(hits
          ? [
              {
                type: 'line',
                data: hits,
                symbol: 'circle',
                symbolSize: 4,
                lineStyle: { color: '#22e584', width: 1.4 },
                itemStyle: { color: '#22e584' },
                z: 3,
              } as echarts.SeriesOption,
            ]
          : []),
      ],
    }
    return option
  }, [profile, hits, nColumns])

  return <EChart option={option} className={height} />
}