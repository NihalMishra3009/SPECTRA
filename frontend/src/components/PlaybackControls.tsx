type Props = {
  cur: number
  T: number
  playing: boolean
  speed: number
  onToggle: () => void
  onSeek: (t: number) => void
  onSpeed: () => void
}

const SPEEDS = [0.5, 1, 2, 4]

export default function PlaybackControls({ cur, T, playing, speed, onToggle, onSeek, onSpeed }: Props) {
  const pct = T > 1 ? (cur / (T - 1)) * 100 : 0
  const nextSpeed = SPEEDS[(SPEEDS.indexOf(speed) + 1) % SPEEDS.length]
  return (
    <div className="flex items-center gap-3 px-1 pb-1">
      <button onClick={onToggle} className="btn btn-primary h-10 w-10 rounded-full p-0 text-lg">
        {playing ? '⏸' : '▶'}
      </button>
      <button onClick={() => onSeek(0)} className="btn btn-ghost h-10 px-3 py-0 text-xs">⏮ RESTART</button>
      <div className="relative h-1.5 flex-1 rounded-full bg-line">
        <div className="absolute h-1.5 rounded-full bg-gradient-to-r from-neon to-hit" style={{ width: `${pct}%` }} />
        <input
          type="range"
          min={0}
          max={Math.max(1, T - 1)}
          value={cur}
          onChange={(e) => onSeek(Number(e.target.value))}
          className="absolute -top-1.5 left-0 h-4 w-full cursor-pointer opacity-0"
        />
      </div>
      <span className="num min-w-24 text-right text-xs text-neon2">
        t {cur} / {T - 1}
      </span>
      <button onClick={onSpeed} className="btn btn-ghost h-10 px-3 py-0 text-xs">
        {speed}×
      </button>
    </div>
  )
}