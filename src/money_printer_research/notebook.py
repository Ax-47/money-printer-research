import sys
from pathlib import Path

# src/money_printer_research/notebook.py -> project root
ROOT = Path(__file__).resolve().parents[2]


def exec_notebook() -> None:
    """Open JupyterLab at the project root, using this project's environment."""
    from jupyterlab.labapp import main

    sys.exit(main(["--notebook-dir", str(ROOT)]))
