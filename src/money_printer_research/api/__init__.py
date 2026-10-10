"""HTTP API over the pipeline's outputs: the clean tables and the Elo model's predictions."""

from money_printer_research.api.app import create_app, exec_api

__all__ = ["create_app", "exec_api"]
