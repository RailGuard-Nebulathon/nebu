# RailGuard AI application submission

This folder contains the compulsory NebulaX PS3 app source and deployment
materials: React interface, FastAPI backend, RailGuard inference package,
configurations, and trained bundles for all four subsystems. Raw datasets,
predictions, dependency directories, and caches are excluded.

## Run locally on Windows

Requirements: Python 3.11+, Node.js 22+, and npm.

```powershell
python -m pip install -e ".[api]"
npm --prefix web/frontend install
Copy-Item .env.example .env
```

Start the API:

```powershell
python -m uvicorn railguard_api.main:app --app-dir web/backend --reload
```

In another terminal, start the web interface:

```powershell
npm --prefix web/frontend run dev
```

Open `http://localhost:5173`; API documentation is at
`http://localhost:8000/docs`. Included bundles are discovered under
`outputs/checkpoints/competition/<task>`. Demonstration fallback is disabled in
the supplied environment example.

## Docker

```powershell
Copy-Item .env.example .env
docker compose up --build
```

Only load the included Joblib files as trusted local artifacts. Results provide
decision support and are not automatic maintenance orders. No hidden-test score
is claimed.

