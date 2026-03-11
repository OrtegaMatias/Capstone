# Capstone Weekly Decision Analytics Framework

Framework repo-local para trabajar el proyecto del ramo por semanas. El repo arranca con seeds versionados, genera una `workspace/` local ignorada por Git y expone una experiencia semanal para:

- Week 1: EDA
- Week 2: ML con historia temporal
- Weeks 3-6: scaffold guiado para modelamiento matematico y optimizacion

## Stack
- Backend: FastAPI, pandas, numpy, scipy, statsmodels, scikit-learn
- Frontend: React + TypeScript + Vite
- Runtime recomendado: Docker Compose

## Estructura
- `framework/manifest.json`: manifiesto versionado del roadmap semanal
- `seed/`: datasets base de Week 1 y Week 2
- `workspace/`: artefactos generados por semana (`canonical.csv`, `metadata.json`, `notes.md`, `report.md`, `report.html`)
- `backend/`: API, ETL, analitica y tests
- `frontend/`: roadmap UI y vistas semanales

## Levantar con Docker Compose
```bash
docker compose up --build
```

Servicios:
- Frontend: `http://localhost:5173`
- Backend: `http://localhost:8000`
- Health: `http://localhost:8000/health`

## Levantar localmente
### Backend
```bash
cd backend
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
uvicorn app.main:app --reload --port 8000
```

### Frontend
```bash
cd frontend
npm install
npm run dev
```

## Flujo del framework
1. Al iniciar el backend se valida el manifiesto y se bootstrappea la `workspace/`.
2. Weeks 1 y 2 cargan automáticamente desde `seed/`.
3. Cada semana mantiene sus notas y reportes dentro de `workspace/<week-id>/`.
4. El frontend consume endpoints por semana, no por `dataset_id`.

## API semanal
- `GET /api/v1/framework`
- `GET /api/v1/weeks/{week_id}`
- `GET /api/v1/weeks/{week_id}/preview`
- `GET /api/v1/weeks/{week_id}/eda`
- `GET /api/v1/weeks/{week_id}/ml/overview`
- `GET /api/v1/weeks/{week_id}/notes`
- `PUT /api/v1/weeks/{week_id}/notes`
- `POST /api/v1/weeks/{week_id}/report/refresh`
- `GET /api/v1/weeks/{week_id}/report`

## Estado actual del roadmap
- `week-1`: activa, usa `seed/Week1`
- `week-2`: activa, usa `seed/Week2` con holdout temporal sobre la ultima semana disponible
- `week-3` a `week-6`: scaffold guiado con checklist, entregables y reporte persistente

## Tests
```bash
cd backend
pytest
```

```bash
cd frontend
npm run build
```

## Compatibilidad heredada
Los endpoints legacy centrados en `dataset_id` siguen presentes como compatibilidad temporal, pero la UI principal ya no depende de ellos.
