"""Wrapper entrypoint for Streamlit hosting.

This file is a small shim so hosting platforms can run `streamlit run
frontend/streamlit_app.py` while keeping the original app implementation in
`deepx_submission.streamlit_app`.
"""

import sys
from pathlib import Path

# Ensure repository root is on sys.path when Streamlit runs this shim
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
	sys.path.insert(0, str(ROOT))

from deepx_submission import streamlit_app  # noqa: F401
