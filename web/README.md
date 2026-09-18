# RailGuard web application

The React application is the non-technical upload workflow required by NebulaX PS3. A FastAPI
boundary calls the existing `RailGuardPredictor`, converts results to the official task-specific
schemas, validates them, and returns both display data and downloadable CSV text.

## Local development

Install the Python API and JavaScript dependencies:

```bash
python -m pip install -e ".[api]"
npm --prefix web/frontend install
```

Run the services in separate terminals:

```bash
make api
make web
```

Open `http://localhost:5173`. FastAPI documentation is available at
`http://localhost:8000/docs`.

## Model bundles

Set the applicable variables in `.env` or the shell:

```text
RAILGUARD_DOOR_BUNDLE=outputs/checkpoints/door
RAILGUARD_ACV_BUNDLE=outputs/checkpoints/acv
RAILGUARD_CORRUGATION_BUNDLE=outputs/checkpoints/corrugation
RAILGUARD_SHM_BUNDLE=outputs/checkpoints/shm
```

Only trusted locally produced classical bundles are loaded. When a bundle is absent, Auto mode
uses an unmistakably labelled demonstration result if `RAILGUARD_ALLOW_DEMO=true`. Demo output is
never described as competition-ready and must not be submitted.

## Containers

```bash
docker compose up --build
```

The web application is served on port 5173 and the API on port 8000.
