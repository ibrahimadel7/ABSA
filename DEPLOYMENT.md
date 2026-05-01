# Deployment notes

Quick instructions to run the backend and Streamlit frontend locally or on common hosts.

Run locally

- Start the backend API:

```bash
uvicorn backend.main:app --host 0.0.0.0 --port 8000
```

- Start the Streamlit frontend (in a separate shell):

```bash
streamlit run frontend/streamlit_app.py --server.port 8501
```

Environment

- Install dependencies (recommended in a virtualenv):

```bash
python -m pip install -r requirements.txt
```

Notes for hosting platforms

- Heroku / Render: the repository includes a `Procfile` with two process types: `web` (the FastAPI app) and `streamlit` (the Streamlit UI). These are two independent process types and should be deployed as separate services/dynos where required.

- Combined deployment: for convenience deploy the API as the primary service and host the Streamlit app as a separate app (so each has its own port and process).

- Docker: if you prefer containerized deployment, create a `Dockerfile` and optionally a `docker-compose.yml` with two services (`api` and `streamlit`). If you want, I can add example Dockerfiles.

API base URL

- The Streamlit app reads `ABSA_API_BASE_URL` or `ABSA_API_URL` environment variables to locate the API. Set that to the deployed API host (for example `https://your-api.example.com`).

Dependency placement for Streamlit Community Cloud

- Place a `requirements.txt` either at the repository root or in the same directory as the entrypoint file. Community Cloud will search the entrypoint directory first, then the repo root. Because this app's entrypoint is `frontend/streamlit_app.py`, a `frontend/requirements.txt` is provided and will be used by Community Cloud.

- If you require system (apt) packages, add a `packages.txt` file at the repository root listing Debian package names (one per line).
