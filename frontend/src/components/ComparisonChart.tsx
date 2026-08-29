import { useMemo } from 'react'
import * as echarts from 'echarts'
import EChart from './EChart'
import type { StepLog } from '../api/client'

type Props = {
  baseline: StepLog[]
  smart: StepLog[]
  height?: string
}

export default function ComparisonChart({ baseline, smart, height = 'h-44' }: Props) {
  const option = useMemo(() => {
    const x = baseline.map((e) => e.t)
    const baseRatio = baseline.map((e) => e.ratio)
    const smartRatio = smart.map((e) => e.ratio)
    const baseReward = cumulativeReward(baseline)
    const smartReward = cumulativeReward(smart)

    const option: echarts.EChartsOption = {
      backgroundColor: 'transparent',
      legend: {
        data: ['baseline %', 'smart %', 'smart reward', 'baseline reward'],
        textStyle: { color: '#7b8aa6', fontSize: 10 },
        top: 0,
        itemWidth: 12,
        itemHeight: 8,
      },
      grid: { left: 42, right: 42, top: 26, bottom: 22 },
      tooltip: {
        trigger: 'axis',
        backgroundColor: '#0a111f',
        borderColor: '#1b2a44',
        textStyle: { color: '#dbe4f5', fontSize: 12 },
      },
      xAxis: {
        type: 'category',
        data: x,
        axisLabel: { color: '#7b8aa6', fontSize: 10 },
        axisLine: { lineStyle: { color: '#1b2a44' } },
        name: 't',
        nameTextStyle: { color: '#7b8aa6', fontSize: 10 },
      },
      yAxis: [
        {
          type: 'value',
          min: 0,
          max: 100,
          name: 'interception %',
          nameTextStyle: { color: '#7b8aa6', fontSize: 10 },
          axisLabel: { color: '#6b7a96', fontSize: 10 },
          splitLine: { lineStyle: { color: '#101a2c' } },
        },
        {
          type: 'value',
          name: 'avg reward',
          nameTextStyle: { color: '#7b8aa6', fontSize: 10 },
          axisLabel: { color: '#6b7a96', fontSize: 10 },
          splitLine: { show: false },
        },
      ],
      series: [
        {
          name: 'baseline %',
          type: 'line',
          data: baseRatio,
          symbol: 'none',
          lineStyle: { color: '#4d6b9f', width: 1.5 },
          itemStyle: { color: '#4d6b9f' },
          areaStyle: { color: 'rgba(77,107,159,0.12)' },
        },
        {
          name: 'smart %',
          type: 'line',
          data: smartRatio,
          symbol: 'none',
          lineStyle: { color: '#00e5ff', width: 2, shadowColor: '#00e5ff', shadowBlur: 6 },
          itemStyle: { color: '#00e5ff' },
          areaStyle: { color: 'rgba(0,229,255,0.14)' },
        },
        {
          name: 'smart reward',
          type: 'line',
          yAxisIndex: 1,
          data: smartReward,
          symbol: 'none',
          lineStyle: { color: '#22e584', width: 1.2, type: 'dashed' },
        },
        {
          name: 'baseline reward',
          type: 'line',
          yAxisIndex: 1,
          data: baseReward,
          symbol: 'none',
          lineStyle: { color: '#ff4d6d', width: 1, type: 'dashed' },
        },
      ],
    }
    return option
  }, [baseline, smart])

  return <EChart option={option} className={height} />
}

function cumulativeReward(log: StepLog[]): number[] {
  const out: number[] = []
  let acc = 0
  for (const e of log) {
    acc += e.reward
    out.push(Number((acc / (e.t + 1)).toFixed(3)))
  }
  return out
}