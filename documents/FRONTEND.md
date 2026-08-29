# FRONTEND — SPECTRA Dashboard (React + Vite + Tailwind)

Dark **EW Command Center** UI: React 18 · Vite · TypeScript · Tailwind v4 ·
ECharts · Framer Motion. Fonts: **Space Grotesk** (display), **Inter** (body),
**JetBrains Mono** (telemetry numbers).

## Run / build
```powershell
cd frontend
npm install
npm run dev        # http://localhost:5173  (proxies /api + /ws → localhost:8000)
npm run build      # type-check + production bundle to dist/
npm run preview    # serve the production build
```

Set `VITE_API_URL` at build time to point at a deployed backend
(e.g. `VITE_API_URL=https://your-app.hf.space npm run build`).

### Offline mock mode (no backend needed)

A deterministic demo fixture is bundled at `public/mock/demo.json` (92 KB —
full `stable_switch_surprise` run, thompson scheduler, seed 2024). Two ways
the dashboard uses it:

- **Explicit**: Config panel **⚡ LOAD MOCK DATA (no backend)** button loads it.
- **Fallback**: if `POST /api/simulate` fails (backend down), the UI auto-falls
  back to the fixture and shows a "loaded bundled MOCK data" notice.

Regenerate the fixture from the backend:
```powershell
cd backend
.\.venv\Scripts\python -c "from app.config import SimConfig; from app.sim import engine; import json; open(r'..\frontend\public\mock\demo.json','w').write(json.dumps(engine.run_simulation(SimConfig(n_bands=10,n_steps=300,seed=2024,scenario='stable_switch_surprise',scheduler='thompson'))))"
```

## Structure
```
src/
├─ main.tsx / App.tsx      layout grid + Demo Mode orchestrator
├─ index.css               Tailwind v4 @theme tokens + component classes
├─ api/client.ts           typed REST client + WebSocket stream helper
├─ hooks/useSim.ts         playback state machine + alert derivation
├─ components/
│  ├─ EChart.tsx           thin ECharts wrapper (init/resize/dispose)
│  ├─ TopBar.tsx           brand, LIVE badge, scenario/scheduler chips, DEMO button
│  ├─ ConfigPanel.tsx      numeric config, scheduler select, scenario grid, RUN
│  ├─ WaterfallChart.tsx   animated time×band heatmap + sweep line + event marks
│  ├─ ComparisonChart.tsx  cumulative interception % + avg-reward lines
│  ├─ SpectrumMiniMap.tsx  activity profile bars + smart hit line
│  ├─ KPIs.tsx             metric cards with Δ-vs-baseline badges
│  ├─ AlertFeed.tsx        surprise/switch detection log with live flash
│  ├─ PlaybackControls.tsx play/pause · restart · scrub · speed (0.5/1/2/4×)
│  └─ ModelPanel.tsx       trained-model cards + training-curve sparklines
```

## Playback model

The backend returns a **deterministic full run** (`POST /api/simulate`). The UI
replays it client-side: `useSim` ticks `cur` on an interval scaled by `speed`
(1000ms/speed). Charts slice up to `cur` → smooth animation, scrubber, restart,
and frame-accurate event replay. A WebSocket mode (`wsStream`) is also available
for a live-telemetry feel when needed.

When playback passes a ground-truth **change/surprise** event, `AlertFeed`
lights it up and shows **smart vs baseline detection delay** (e.g. `smart +3t`,
`baseline missed`).

## Real frequency labels

`SimResult.band_edges_ghz` holds each band's GHz range (spectrum 2–18 GHz
split across `n_bands`). The waterfall Y-axis shows **band + center GHz**
(`B7 14.0G`), and tooltips show the full sender-side range
(`B7 (13.2–14.8 GHz)`). The spectrum mini-map glues the same ranges to its
bars (`2.8G`, `4.4G`, …). If `band_edges_ghz` is missing (older fixtures) the
UI falls back to a default 1.6 GHz grid.

## Demo Mode

`App.tsx` auto-plays 4 guided scenarios at 2×:
1. Stable + Switch + Surprise → `thompson`
2. Surprise emitter → `epsilon_greedy`
3. Periodic timing → `sequence` (LSTM)
4. Frequency hopper → `sequence`

Each run advances automatically when playback reaches the end — zero fumbling
during the SIH presentation.

## Theming (Tailwind v4 tokens in index.css)

```css
@theme {
  --color-void #04060c   --color-neon #00e5ff   --color-amber #ffb020
  --color-hit #22e584    --color-miss #ff4d6d   --color-dim #7b8aa6
  --color-panel/-inner/-line ...
  --font-display/-sans/-mono ...
}
```
Radar-grid background uses two repeating linear-gradients in `body`.

## Deploy (free)
- **Vercel**: framework preset "Vite", build `npm run build`, output `dist`.
- Backend on HuggingFace Spaces / Render → set `VITE_API_URL`.

## Adding a chart
1. Write an ECharts `option` object in a component.
2. Wrap with `<EChart option={option} className="h-44" />` (auto-resizes).
3. Heavy literals/typed options → cast `as unknown as echarts.EChartsOption`.