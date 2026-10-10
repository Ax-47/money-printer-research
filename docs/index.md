# money-printer-research

Collect, clean and model football match data from Understat and Transfermarkt.

The project builds one row per league match in the five big European leagues
(England, Spain, Germany, Italy, France) from 2014/15 on, with the full-time
result (home / draw / away) as the target and only pre-match features as inputs:
Elo ratings, team and player fatigue, schedule congestion and each starter's form.

## Quick start

```bash
make install               # uv sync --all-groups
uv run collect_data        # download the sources into data/
uv run clean_data          # write the cleaned tables and features to data/clean
uv run check_clean         # validate every clean CSV against its schema
```

## Where to go next

- [Getting started](getting-started.md): install and run the project.
- [Data pipeline](pipeline/index.md): what each step reads, does and writes.
- [Schemas](schema/index.md): every table and column, grouped by entity.
- [Commands](cli.md): the console scripts.
- [API reference](api/index.md): generated from the docstrings.
