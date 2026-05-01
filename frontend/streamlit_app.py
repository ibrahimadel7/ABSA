"""Wrapper entrypoint for Streamlit hosting.

This file is a small shim so hosting platforms can run `streamlit run
frontend/streamlit_app.py` while keeping the original app implementation in
`deepx_submission.streamlit_app`.
"""

from deepx_submission import streamlit_app  # noqa: F401
