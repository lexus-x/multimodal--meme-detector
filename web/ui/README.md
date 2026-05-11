# MemeGuardian AI — React Frontend

A modern React + Vite frontend for the Multimodal Offensive Meme Detector.

## Quick Start

```bash
# From the project root:
cd app/ui

# Install dependencies
npm install

# Start dev server (port 3000)
npm run dev
```

The dev server proxies `/predict` requests to the FastAPI backend at `localhost:8000`.

## Prerequisites

- **Node.js** >= 18
- **Backend running** at `http://localhost:8000`

Start the backend first:
```bash
# From project root
python -m app.api.main
# or
make api
```

## Build for Production

```bash
npm run build
```

Output goes to `dist/`. Serve with any static file server or use the Docker setup.

## Docker

```bash
# From project root — builds both backend and frontend
docker-compose up --build
```

- Frontend: http://localhost:3000
- Backend API: http://localhost:8000

## Project Structure

```
app/ui/
├── public/           # Static assets (favicon, example images)
├── src/
│   ├── App.jsx       # Main application component
│   ├── main.jsx      # React entry point
│   └── index.css     # Global styles
├── index.html        # HTML template
├── vite.config.js    # Vite config with API proxy
├── nginx.conf        # Nginx config for Docker production build
├── Dockerfile.ui     # Multi-stage Docker build
└── package.json
```

## API Proxy

In development, Vite proxies requests to the backend:

| Frontend Path | Proxied To | Purpose |
|---------------|------------|---------|
| `/predict` | `http://localhost:8000/predict` | Meme classification |
| `/api/*` | `http://localhost:8000/*` | General API access |

In production (Docker), Nginx handles the proxying.

## Tech Stack

- **React 19** — UI framework
- **Vite 8** — Build tool & dev server
- **Framer Motion** — Animations
- **Lucide React** — Icons
- **Axios** — HTTP client
