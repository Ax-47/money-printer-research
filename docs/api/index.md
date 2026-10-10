# Source Code API Reference Overview

Generated from the docstrings in `src/money_printer_research`. Each page covers
one package:

| Package | What it holds |
| --- | --- |
| [`data_collection`](data_collection.md) | The collectors behind `collect_data` |
| [`clean_data`](clean_data.md) | The cleaning steps and the `clean_data` entry point |
| [`features`](features.md) | Elo, fatigue, schedule and player features |
| [`schema`](schema.md) | Pandera models for every frame, and `check_clean` |
| [`analysis`](analysis.md) | Feature importance |
| [`cool_stuff`](cool_stuff.md) | `Result`, Kleisli arrows and schema-checked steps |

For what each step does and in what order, start with the
[Data pipeline](../pipeline/index.md). For column-level detail, see
[Schemas](../schema/index.md).
