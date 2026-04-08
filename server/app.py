"""
server/app.py
=============
Entry point required by OpenEnv multi-mode deployment validator.
Imports and re-exports the main FastAPI app and main() launcher
from the root app.py.
"""

import sys
import os

# Make sure root directory is in path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app import app  # noqa: F401


def main():
    """Entry point called by openenv validate and [project.scripts] server."""
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=7860)


if __name__ == "__main__":
    main()
