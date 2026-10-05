import polars as pl

ALIASES = {
    # ENG-Premier League: short and alternative names
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
    # ENG-Premier League (no alias needed)
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
    # ESP-La Liga (32 teams)
    "alaves": "alaves",
    "almeria": "almeria",
    "athletic club": "athletic club",
    "atletico madrid": "atletico madrid",
    "barcelona": "barcelona",
    "cadiz": "cadiz",
    "celta vigo": "celta vigo",
    "cordoba": "cordoba",
    "deportivo la coruna": "deportivo la coruna",
    "eibar": "eibar",
    "elche": "elche",
    "espanyol": "espanyol",
    "getafe": "getafe",
    "girona": "girona",
    "granada": "granada",
    "las palmas": "las palmas",
    "leganes": "leganes",
    "levante": "levante",
    "malaga": "malaga",
    "mallorca": "mallorca",
    "osasuna": "osasuna",
    "rayo vallecano": "rayo vallecano",
    "real betis": "real betis",
    "real madrid": "real madrid",
    "real oviedo": "real oviedo",
    "real sociedad": "real sociedad",
    "real valladolid": "real valladolid",
    "sd huesca": "sd huesca",
    "sevilla": "sevilla",
    "sporting gijon": "sporting gijon",
    "valencia": "valencia",
    "villarreal": "villarreal",
    # FRA-Ligue 1 (34 teams)
    "ajaccio": "ajaccio",
    "amiens": "amiens",
    "angers": "angers",
    "auxerre": "auxerre",
    "bordeaux": "bordeaux",
    "brest": "brest",
    "caen": "caen",
    "clermont foot": "clermont foot",
    "dijon": "dijon",
    "evian thonon gaillard": "evian thonon gaillard",
    "gfc ajaccio": "gfc ajaccio",
    "guingamp": "guingamp",
    "le havre": "le havre",
    "lens": "lens",
    "lille": "lille",
    "lorient": "lorient",
    "lyon": "lyon",
    "marseille": "marseille",
    "metz": "metz",
    "monaco": "monaco",
    "montpellier": "montpellier",
    "nancy": "nancy",
    "nantes": "nantes",
    "nice": "nice",
    "nimes": "nimes",
    "paris fc": "paris fc",
    "paris saint germain": "paris saint germain",
    "reims": "reims",
    "rennes": "rennes",
    "saint-etienne": "saint-etienne",
    "sc bastia": "sc bastia",
    "strasbourg": "strasbourg",
    "toulouse": "toulouse",
    "troyes": "troyes",
    # GER-Bundesliga (30 teams)
    "arminia bielefeld": "arminia bielefeld",
    "augsburg": "augsburg",
    "bayer leverkusen": "bayer leverkusen",
    "bayern munich": "bayern munich",
    "bochum": "bochum",
    "borussia dortmund": "borussia dortmund",
    "borussia mgladbach": "borussia mgladbach",
    "darmstadt": "darmstadt",
    "eintracht frankfurt": "eintracht frankfurt",
    "fc cologne": "fc cologne",
    "fc heidenheim": "fc heidenheim",
    "fortuna duesseldorf": "fortuna duesseldorf",
    "freiburg": "freiburg",
    "greuther fuerth": "greuther fuerth",
    "hamburger sv": "hamburger sv",
    "hannover 96": "hannover 96",
    "hertha berlin": "hertha berlin",
    "hoffenheim": "hoffenheim",
    "holstein kiel": "holstein kiel",
    "ingolstadt": "ingolstadt",
    "mainz 05": "mainz 05",
    "nuernberg": "nuernberg",
    "paderborn": "paderborn",
    "rasenballsport leipzig": "rasenballsport leipzig",
    "schalke 04": "schalke 04",
    "st pauli": "st pauli",
    "union berlin": "union berlin",
    "vfb stuttgart": "vfb stuttgart",
    "werder bremen": "werder bremen",
    "wolfsburg": "wolfsburg",
    # ITA-Serie A (36 teams)
    "ac milan": "ac milan",
    "atalanta": "atalanta",
    "benevento": "benevento",
    "bologna": "bologna",
    "brescia": "brescia",
    "cagliari": "cagliari",
    "carpi": "carpi",
    "cesena": "cesena",
    "chievo": "chievo",
    "como": "como",
    "cremonese": "cremonese",
    "crotone": "crotone",
    "empoli": "empoli",
    "fiorentina": "fiorentina",
    "frosinone": "frosinone",
    "genoa": "genoa",
    "inter": "inter",
    "juventus": "juventus",
    "lazio": "lazio",
    "lecce": "lecce",
    "monza": "monza",
    "napoli": "napoli",
    "palermo": "palermo",
    # Parma went bankrupt in 2015 and was refounded; same club, same Transfermarkt id.
    "parma": "parma calcio 1913",
    "parma calcio 1913": "parma calcio 1913",
    "pescara": "pescara",
    "pisa": "pisa",
    "roma": "roma",
    "salernitana": "salernitana",
    "sampdoria": "sampdoria",
    "sassuolo": "sassuolo",
    "spal 2013": "spal 2013",
    "spezia": "spezia",
    "torino": "torino",
    "udinese": "udinese",
    "venezia": "venezia",
    "verona": "verona",
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


def alias_block(schedule: pl.DataFrame) -> str:
    """Paste-ready ALIASES lines for every schedule team not yet in ALIASES, by league."""
    teams = pl.concat(
        [
            schedule.select("league", pl.col("home_team").alias("team")),
            schedule.select("league", pl.col("away_team").alias("team")),
        ]
    ).unique()
    lines = []
    for (league,), group in teams.sort("league", "team").group_by("league", maintain_order=True):
        keys = sorted({_alias_key(t) for t in group["team"]} - ALIASES.keys())
        if keys:
            lines.append(f"    # {league}")
            lines += [f'    "{k}": "{k}",' for k in keys]
    return "\n".join(lines)
