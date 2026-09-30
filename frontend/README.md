# Sentinel AI - Dashboard

React + Vite + Tailwind CSS v4 dashboard for the Sentinel AI backend. It shows
the annotated live feed, per-frame stats, active tracks (with fall state),
zones, and alerts with incident snapshots and clips. Everything shown comes
from the backend; there is no simulated data. When the backend is not running
the dashboard says so.

## Run

Start the backend first (from the repository root). With `--demo` it runs the
full pipeline on the bundled sample videos, so no camera is needed:

```bash
python backend/main.py --demo
```

Then either:

- **Served by the backend:** `npm install && npm run build`, then open
  http://localhost:8000. The backend serves `frontend/dist` at `/`.
- **Dev server:** `npm install && npm run dev`, then open the printed URL.
  Vite proxies `/api` and `/ws` to `http://localhost:8000` (override with the
  `SENTINEL_BACKEND` environment variable).

## Pages

A left sidebar links the pages (client-side routes; the backend serves
`index.html` for unknown non-API paths, so reloading any route works):

- `/` **Live**: annotated feed, stats (with per-layer timings), tracks, live
  alert strip, alert timeline and zones. Panels can be shown or hidden; the
  choice is remembered in localStorage.
- `/events` **Events**: filterable, paged event table with thumbnails, an
  events-per-hour chart and a clip player. `/events?event=<id>` opens one event.
- `/analytics` **Analytics**: events per hour, zone, type and severity.
- `/settings` **Settings**: read-only camera, zones, notification channels and
  thresholds from `/api/meta`.
- `/search`, `/rules`: placeholders for later phases.

One WebSocket connection (`src/context/WebSocketProvider.jsx`) is shared by all
pages. Frames go to a separate context so only the video and stats re-render
per frame.

## Configuration

Optional build-time environment variables:

| Variable       | Default                                  |
| -------------- | ---------------------------------------- |
| `VITE_WS_URL`  | `ws(s)://<page host>/ws/feed`            |
| `VITE_API_URL` | same origin as the page                  |

## Data sources

- `WS /ws/feed`: `history`, `alert`, and `frame` messages (annotated JPEG + stats).
- `GET /api/stats`, `/api/health`: source info, uptime, source errors.
- `GET /api/alerts`: persisted alert history.
- `GET /api/tracks`, `/api/zones`: track registry and zone list.
- `GET /api/snapshots/{id}`, `/api/clips/{id}`: incident media.
- `GET /api/events`, `/api/events/{id}`, `/api/events/stats`: event log and counts.
- `GET /api/meta`: filter options and the Settings page.

## Scripts

- `npm run dev`: development server
- `npm run build`: production build to `dist/`
- `npm run lint`: ESLint
