# Streamlit frontend

Entrypoint: `frontend/streamlit_app.py`

- Dependencies: `requirements.txt` at repository root.
- To run locally (from repo root):

```bash
python -m pip install -r requirements.txt
streamlit run frontend/streamlit_app.py
```

- For Streamlit Community Cloud: create an app and set the entrypoint to `frontend/streamlit_app.py`. The repository already contains `.streamlit/config.toml` and `requirements.txt` at the root.
