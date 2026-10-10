"""MkDocs hook: render a pandera schema's columns as a Markdown table.

A line holding only `<!-- columns: money_printer_research.schema.ClubMapCols -->`
becomes a table of that schema's columns, with dtype, nullability and the
description from pa.Field. The target is a DataFrameModel class, a
DataFrameSchema, or a function returning one (match_features_schema).
"""

import re
from importlib import import_module
from typing import Any

import pandera.polars as pa

MARKER = re.compile(r"^<!-- columns: ([\w.]+) -->$", re.MULTILINE)


def _load(path: str) -> pa.DataFrameSchema:
    module, _, name = path.rpartition(".")
    target: Any = getattr(import_module(module), name)
    if isinstance(target, pa.DataFrameSchema):
        return target
    if isinstance(target, type) and issubclass(target, pa.DataFrameModel):
        return target.to_schema()
    return target()


def _cell(text: str | None) -> str:
    text = (text or "").replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
    return text.replace("|", "\\|").replace("\n", " ")


def _table(path: str) -> str:
    rows = ["| Column | Type | Nullable | Description |", "| --- | --- | --- | --- |"]
    for name, col in _load(path).columns.items():
        nullable = "yes" if col.nullable else ""
        rows.append(f"| `{name}` | `{col.dtype}` | {nullable} | {_cell(col.description)} |")
    return "\n".join(rows)


def on_page_markdown(markdown: str, **_: Any) -> str:
    return MARKER.sub(lambda m: _table(m.group(1)), markdown)
