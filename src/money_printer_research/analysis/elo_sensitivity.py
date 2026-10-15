"""How much the Elo settings matter: walk-forward log loss over a grid of K and home_adv.

features.elo.EloParams were picked on seasons that come before every game the models are
validated or tested on, so the choice cannot leak the test seasons. This module redoes that
search so it can be checked:

    1. football-data.co.uk results for the five leagues from WARMUP_FROM, read from
       settings.modeling.odds_dir (fetched there first when missing, as collect_odds does)
    2. Elo replayed over all of them for each (K, home_adv) in the grid, as clean_data does
    3. the Elo-driven double Poisson model of features.elo, walk-forward over TUNE_SEASONS:
       each season predicted from a model fitted on the seasons before it only
    4. log loss, RPS and accuracy per setting

The seasons before TUNE_SEASONS only warm the ratings up. `uv run elo_sensitivity` prints
the table, the best setting and a LaTeX tabular of the K x home_adv log losses.
"""

import itertools
from pathlib import Path

import polars as pl
from pandera.typing.polars import DataFrame

from money_printer_research.config import Settings
from money_printer_research.config import settings as default_settings
from money_printer_research.data_collection.scrape_odds import (
    DIVISIONS,
    _download,  # pyright: ignore[reportPrivateUsage]
    odds_file,
    odds_url,
)
from money_printer_research.features.elo import (
    DEFAULT_ELO,
    EloParams,
    evaluate,
    get_elo,
    season_label,
    walk_forward,
)
from money_printer_research.schema import EloInputCols

WARMUP_FROM = 2000
"""First season read; ratings need a few seasons to settle before they are scored."""

TUNE_SEASONS = [f"{y}/{str(y + 1)[-2:]}" for y in range(2005, 2015)]
"""2005/06-2014/15: where EloParams were chosen, before the validation and test seasons."""

K_GRID = [5.0, 10.0, 15.0, 20.0, 25.0, 30.0]
HOME_ADV_GRID = [0.0, 20.0, 40.0, 60.0, 80.0, 100.0]
REGRESS_GRID = [0.0, 0.1, 0.2, 0.3]
"""season_regress values tried at the default K and home_adv."""

DATE_FORMATS = ["%d/%m/%Y", "%d/%m/%y"]
"""football-data writes two-digit years in its older files."""


def _season_file(settings: Settings, league: str, year: int) -> Path:
    """A league season's football-data CSV, downloaded first if it is not there yet."""
    path = odds_file(settings.modeling.odds_dir, league, year)
    if not path.exists() and not _download(odds_url(DIVISIONS[league], year), path):
        raise FileNotFoundError(f"football-data has no {league} {year}")
    return path


def _read_season(path: Path, year: int) -> pl.DataFrame:
    """One football-data CSV -> EloInputCols rows; blank trailing rows are dropped."""
    raw = pl.read_csv(path, encoding="utf8-lossy", infer_schema=False, truncate_ragged_lines=True)
    hg, ag = pl.col("FTHG").cast(pl.Int64), pl.col("FTAG").cast(pl.Int64)
    # "%d/%m/%Y" would read "13/08/05" as year 5, so pick the format by the string's length.
    day = pl.col("Date")
    date = (
        pl.when(day.str.len_chars() <= 8)
        .then(day.str.strptime(pl.Date, DATE_FORMATS[1], strict=False))
        .otherwise(day.str.strptime(pl.Date, DATE_FORMATS[0], strict=False))
    )
    return raw.filter(pl.col("HomeTeam").is_not_null() & pl.col("FTHG").is_not_null()).select(
        season_label(pl.lit(year)).alias(EloInputCols.season),
        date.alias(EloInputCols.match_date),
        pl.col("HomeTeam").alias(EloInputCols.home_team),
        pl.col("AwayTeam").alias(EloInputCols.away_team),
        hg.alias(EloInputCols.full_time_home_goals),
        ag.alias(EloInputCols.full_time_away_goals),
        pl.when(hg > ag)
        .then(pl.lit("H"))
        .when(hg == ag)
        .then(pl.lit("D"))
        .otherwise(pl.lit("A"))
        .alias(EloInputCols.full_time_result),
    )


def load_history(settings: Settings = default_settings) -> DataFrame[EloInputCols]:
    """Every league match from WARMUP_FROM through the last tuning season."""
    last = int(TUNE_SEASONS[-1][:4])
    frames = [
        _read_season(_season_file(settings, league, year), year)
        for league in DIVISIONS
        for year in range(WARMUP_FROM, last + 1)
    ]
    matches = pl.concat(frames)
    year = pl.col(EloInputCols.match_date).dt.year()
    start = pl.col(EloInputCols.season).str.slice(0, 4).cast(pl.Int32)
    bad = matches.filter(year.is_null() | (year < start) | (year > start + 1))
    if bad.height:
        raise ValueError(f"{bad.height} matches dated outside their season:\n{bad.head()}")
    return EloInputCols.validate(matches, lazy=True)


def score(matches: DataFrame[EloInputCols], params: EloParams) -> dict[str, float]:
    """Walk-forward log loss, RPS and accuracy over TUNE_SEASONS for one Elo setting."""
    return evaluate(walk_forward(get_elo(matches, params), TUNE_SEASONS))


def sensitivity(matches: DataFrame[EloInputCols]) -> pl.DataFrame:
    """One row per setting: the K x home_adv grid, then season_regress at the default."""
    settings = [
        EloParams(k=k, home_adv=h) for k, h in itertools.product(K_GRID, HOME_ADV_GRID)
    ] + [
        EloParams(k=DEFAULT_ELO.k, home_adv=DEFAULT_ELO.home_adv, season_regress=r)
        for r in REGRESS_GRID
        if r != DEFAULT_ELO.season_regress
    ]
    rows = [
        {"k": p.k, "home_adv": p.home_adv, "season_regress": p.season_regress, **score(matches, p)}
        for p in settings
    ]
    return pl.DataFrame(rows)


def latex_grid(table: pl.DataFrame) -> str:
    """K down, home_adv across, log loss in each cell; the default setting in bold."""
    grid = table.filter(pl.col("season_regress") == 0)
    lines = [
        r"\begin{tabular}{|c|" + "c|" * len(HOME_ADV_GRID) + "}",
        r"\hline",
        r"$K$ \textbackslash{} $h$ & " + " & ".join(f"{h:g}" for h in HOME_ADV_GRID) + r" \\",
        r"\hline",
    ]
    for k in K_GRID:
        cells = []
        for h in HOME_ADV_GRID:
            loss = grid.filter((pl.col("k") == k) & (pl.col("home_adv") == h))["log_loss"][0]
            cell = f"{loss:.4f}"
            if (k, h) == (DEFAULT_ELO.k, DEFAULT_ELO.home_adv):
                cell = rf"\textbf{{{cell}}}"
            cells.append(f"${cell}$")
        lines.append(f"{k:g} & " + " & ".join(cells) + r" \\")
    lines += [r"\hline", r"\end{tabular}"]
    return "\n".join(lines)


def exec_elo_sensitivity(settings: Settings = default_settings) -> None:
    matches = load_history(settings)
    table = sensitivity(matches).sort("log_loss")
    default = table.filter(
        (pl.col("k") == DEFAULT_ELO.k)
        & (pl.col("home_adv") == DEFAULT_ELO.home_adv)
        & (pl.col("season_regress") == DEFAULT_ELO.season_regress)
    )
    print(f"{matches.height} matches, scored on {TUNE_SEASONS[0]}-{TUNE_SEASONS[-1]}")
    with pl.Config(tbl_rows=-1, float_precision=4):
        print(table)
    best = table.row(0, named=True)
    print(
        f"best k={best['k']:g} home_adv={best['home_adv']:g} "
        f"regress={best['season_regress']:g}: {best['log_loss']:.4f}; "
        f"default: {default['log_loss'][0]:.4f}"
    )
    print(latex_grid(table))
