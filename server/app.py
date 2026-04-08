# server/app.py
# Required by OpenEnv multi-mode deployment validator.
# The actual server runs from the root app.py via uvicorn.

from app import app, main
