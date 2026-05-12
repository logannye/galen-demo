"""Streamlit Cloud entry point.

This file exists at the repo root so Streamlit Cloud knows where to
start. It simply re-exports the existing demo's main() so the
deployment repo's structure is independent of the development repo's
layout.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

# Push the project root onto sys.path so the existing demo imports work.
WORKSPACE = Path(__file__).resolve().parent
sys.path.insert(0, str(WORKSPACE))

# Promote Streamlit secrets to environment variables so the existing
# code paths that read os.environ (Bedrock SDK, the password gate,
# the cloud-mode flag) all work uniformly.
import streamlit as st
for k in ("DEMO_PASSWORD", "DEMO_CLOUD_MODE", "AWS_ACCESS_KEY_ID",
          "AWS_SECRET_ACCESS_KEY", "AWS_REGION", "GALEN_BEDROCK_MODEL_ID"):
    try:
        v = st.secrets.get(k)
    except Exception:
        v = None
    if v and k not in os.environ:
        os.environ[k] = str(v)

# Default to cloud mode in the hosted deployment (free-text input
# disabled, pre-computed profiles only).
os.environ.setdefault("DEMO_CLOUD_MODE", "true")

from scripts.demo.streamlit_app_v4 import main

if __name__ == "__main__" or True:
    main()
