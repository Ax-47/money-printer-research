"""Save collected files only after they match their full raw schema."""

from pathlib import Path

import pandas as pd

from money_printer_research.schema.raw import RawSchema, check_collected


def save_checked(df: pd.DataFrame, out: Path, schema: RawSchema) -> None:
    """Write `df` to `out` (.parquet or .csv) only if it matches every column of `schema`.

    The file is written next to `out` first and checked from disk, so a frame
    that fails the check never leaves a file that later runs would skip.
    """
    tmp = out.with_name(f"{out.stem}.part{out.suffix}")
    if out.suffix == ".parquet":
        df.to_parquet(tmp)
    else:
        df.to_csv(tmp, index=False)
    try:
        check_collected(tmp, schema)
    except ValueError:
        tmp.unlink()
        raise
    tmp.replace(out)
