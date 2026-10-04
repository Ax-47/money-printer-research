import polars as pl

ALIASES = {
    "man united": "manchester united",
    "man utd": "manchester united",
    "manchester utd": "manchester united",
    "man city": "manchester city",
    "newcastle": "newcastle united",
    "newcastle utd": "newcastle united",
    "nott'm forest": "nottingham forest",
    "nott'ham forest": "nottingham forest",
    "sheffield utd": "sheffield united",
    "sheffield weds": "sheffield wednesday",
    "leeds": "leeds united",
    "spurs": "tottenham",
    "tottenham hotspur": "tottenham",
    "west brom": "west bromwich albion",
    "wolves": "wolverhampton wanderers",
    "west ham united": "west ham",
    "brighton & hove albion": "brighton",
    "brighton and hove albion": "brighton",
    "qpr": "queens park rangers",
    "leicester": "leicester city",
    "norwich": "norwich city",
    "cardiff": "cardiff city",
    "swansea": "swansea city",
    "stoke": "stoke city",
    "hull": "hull city",
    "huddersfield": "huddersfield town",
    "luton": "luton town",
    "ipswich": "ipswich town",
    "wigan": "wigan athletic",
    "bolton": "bolton wanderers",
    "birmingham": "birmingham city",
    "blackburn": "blackburn rovers",
    "charlton": "charlton athletic",
    "derby": "derby county",
    "coventry": "coventry city",
    "bradford": "bradford city",
    "arsenal": "arsenal",
    "aston villa": "aston villa",
    "blackpool": "blackpool",
    "bournemouth": "bournemouth",
    "brentford": "brentford",
    "burnley": "burnley",
    "chelsea": "chelsea",
    "crystal palace": "crystal palace",
    "everton": "everton",
    "fulham": "fulham",
    "liverpool": "liverpool",
    "middlesbrough": "middlesbrough",
    "portsmouth": "portsmouth",
    "reading": "reading",
    "southampton": "southampton",
    "sunderland": "sunderland",
    "watford": "watford",
}
ALIASES = {**{v: v for v in ALIASES.values()}, **ALIASES}
TEAM_COLUMNS = ("team", "home_team", "away_team")


def _get_team_cols(df: pl.DataFrame) -> list[str]:
    return [c for c in TEAM_COLUMNS if c in df.columns]


def _get_team_names(df: pl.DataFrame) -> set[str]:
    cols = [c for c in TEAM_COLUMNS if c in df.columns]
    if not cols:
        raise ValueError(f"No team column found in {df.columns}")
    return {n for c in cols for n in df[c].drop_nulls().to_list()}


def _alias_key(name: str) -> str:
    """Python twin of _clean below. Keep the two in sync."""
    return name.lower().replace(".", "").strip()


def _clean(col: pl.Expr) -> pl.Expr:
    return col.str.to_lowercase().str.replace_all(".", "", literal=True).str.strip_chars()


def _normalize(col: pl.Expr) -> pl.Expr:
    # replace_strict raises on any name that is not a key in ALIASES.
    return _clean(col).replace_strict(ALIASES)


def format_alias_lines(missing: dict[str, pl.DataFrame]) -> str:
    """Paste-ready ALIASES lines. The target defaults to the key itself."""
    sources_by_key: dict[str, set[str]] = {}
    for src, df in missing.items():
        for key in df["key"]:
            sources_by_key.setdefault(key, set()).add(src)
    return "\n".join(
        f'    "{key}": "{key}",  # {", ".join(sorted(srcs))}'
        for key, srcs in sorted(sources_by_key.items())
    )


def find_missing_aliases(df: pl.DataFrame) -> pl.DataFrame:
    """Team names (per source) whose cleaned key is not in ALIASES."""
    rows = [
        (name, _alias_key(name))
        for name in sorted(_get_team_names(df))
        if _alias_key(name) not in ALIASES
    ]
    return pl.DataFrame(rows, schema={"name": pl.String, "key": pl.String}, orient="row")


def rename_team_names(df: pl.DataFrame) -> pl.DataFrame:
    cols = _get_team_cols(df)
    return df.with_columns(_normalize(pl.col(cols)))
