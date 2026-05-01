"""Backend entrypoint exposing the FastAPI app for hosting.

Run with an ASGI server such as:

    uvicorn backend.main:app --host 0.0.0.0 --port 8000

This module imports the existing FastAPI app defined in
`deepx_submission.api` so deployment targets can reference a stable path.
"""

from deepx_submission.api import app  # re-export FastAPI app

__all__ = ["app"]
