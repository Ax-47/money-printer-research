from pathlib import Path

import kagglehub

from money_printer_research.config import Settings


def _collect_matches(settings: Settings) -> None:
    cfg = (settings or Settings()).kaggle
    Path(kagglehub.dataset_download(cfg.handle, output_dir=str(cfg.output_dir)))
