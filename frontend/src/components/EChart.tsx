import { useEffect, useRef } from 'react'
import * as echarts from 'echarts'

type Props = {
  option: echarts.EChartsOption
  className?: string
}

export default function EChart({ option, className }: Props) {
  const ref = useRef<HTMLDivElement>(null)
  const inst = useRef<echarts.ECharts | null>(null)

  useEffect(() => {
    const el = ref.current
    if (!el) return
    const chart = echarts.init(el)
    inst.current = chart
    const ro = new ResizeObserver(() => chart.resize())
    ro.observe(el)
    return () => {
      ro.disconnect()
      chart.dispose()
      inst.current = null
    }
  }, [])

  useEffect(() => {
    inst.current?.setOption(option as echarts.EChartsOption, { notMerge: true })
  }, [option])

  return <div ref={ref} className={className} />
}