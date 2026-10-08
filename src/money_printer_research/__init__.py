"""Entry points for the console scripts, imported only when used.

Importing the package (for example money_printer_research.clean_data in a notebook)
must not pull in the scraping libraries (soccerdata, kagglehub) that only
collect_data needs, so each entry point loads its module on first access.
"""

from importlib import import_module
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from money_printer_research.analysis import exec_feature_importance
    from money_printer_research.clean_data import clean_data
    from money_printer_research.data_collection import collect_data
    from money_printer_research.notebook import exec_notebook

# Entry point name -> module that defines it.
_ENTRY_POINTS = {
    "collect_data": "money_printer_research.data_collection",
    "clean_data": "money_printer_research.clean_data",
    "exec_feature_importance": "money_printer_research.analysis",
    "exec_notebook": "money_printer_research.notebook",
}


def __getattr__(name: str) -> Any:
    if name not in _ENTRY_POINTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = getattr(import_module(_ENTRY_POINTS[name]), name)
    # Importing money_printer_research.clean_data binds the *subpackage* to this
    # name; overwrite it so `from money_printer_research import clean_data` still
    # gets the function the console script expects.
    globals()[name] = value
    return value


__all__ = ["clean_data", "collect_data", "exec_feature_importance", "exec_notebook"]
