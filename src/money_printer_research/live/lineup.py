"""From ESPN's names to the model's features: teams, players and the announced XI.

ESPN spells teams and players its own way. Teams go through the project's ALIASES, then
the closest name among the Understat teams with a fixture that day. Players are matched
within their team's squad: same name, then same surname and initial, then the closest name.

The model reads the mean form90 and prev_min of each side's starters
(feature_importance.SELECTED), so the announced XI only changes the slot columns:
predict's guessed XI (last match's starters) is swapped for the real one. A starter
matched to no Understat player (a debut, say) gets nulls, which the mean skips.
"""

import difflib
import re
import unicodedata
from typing import Any

import polars as pl

from money_printer_research.clean_data.aliases_team import ALIASES
from money_printer_research.features.player_features import FORM_WINDOW, MIN_PREV_MIN
from money_printer_research.schema import N_SLOTS, PLAYER_SLOT_FEATURES

CLUB_WORDS = {"fc", "afc", "cf", "ac", "sc", "ssc", "as", "rc", "ud", "cd", "sv", "vfb", "vfl"}
"""Words ESPN puts around club names that the Understat names leave out."""

TEAM_CUTOFF = 0.6
PLAYER_CUTOFF = 0.75


def plain(name: str) -> str:
    """Lowercase ASCII words: accents, dots and punctuation dropped."""
    ascii_name = unicodedata.normalize("NFKD", name).encode("ascii", "ignore").decode()
    return " ".join(re.sub(r"[^a-z0-9 ]", " ", ascii_name.lower().replace(".", "")).split())


def match_team(espn_name: str, candidates: list[str]) -> str | None:
    """The Understat team among `candidates` that `espn_name` names, or None."""
    key = espn_name.lower().replace(".", "").strip()
    stripped = " ".join(w for w in plain(espn_name).split() if w not in CLUB_WORDS)
    for k in (key, plain(espn_name), stripped):
        if ALIASES.get(k) in candidates:
            return ALIASES[k]
    plain_candidates = {plain(c): c for c in candidates}
    best = difflib.get_close_matches(stripped, list(plain_candidates), n=1, cutoff=TEAM_CUTOFF)
    return plain_candidates[best[0]] if best else None


def current_form(player_stat: pl.DataFrame, matches: pl.DataFrame) -> pl.DataFrame:
    """Per player, as of his last match: team, prev_min, xg90, xa90 and form90.

    The same sums as player_features.player_form over his last FORM_WINDOW matches,
    but including the last one, since the next game comes after it.
    `player_stat` has canonical team names (norm_team_names); `matches` gives the dates.
    """
    dated = player_stat.join(matches.select("game_id", "date"), on="game_id").sort(
        "player_id", "date"
    )
    last = (
        dated.group_by("player_id", maintain_order=True)
        .agg(
            pl.col("player").last(),
            pl.col("team").last(),
            pl.col("date").last().alias("last_played"),
            pl.col("minutes").tail(FORM_WINDOW).sum().alias("prev_min"),
            pl.col("xg").tail(FORM_WINDOW).sum().alias("prev_xg"),
            pl.col("xa").tail(FORM_WINDOW).sum().alias("prev_xa"),
        )
        .with_columns(pl.col("prev_min").cast(pl.Float64))
    )
    per90 = pl.when(pl.col("prev_min") >= MIN_PREV_MIN).then(90 / pl.col("prev_min"))
    return last.with_columns(
        (pl.col("prev_xg") * per90).alias("xg90"),
        (pl.col("prev_xa") * per90).alias("xa90"),
        ((pl.col("prev_xg") + pl.col("prev_xa")) * per90).alias("form90"),
    ).drop("prev_xg", "prev_xa")


def _surname_initial(name: str) -> tuple[str, str]:
    words = name.split()
    return (words[-1], words[0][0]) if words else ("", "")


def match_player(name: str, squad: pl.DataFrame) -> dict[str, Any] | None:
    """The row of `squad` (current_form rows of one team, latest first) named `name`."""
    rows = squad.to_dicts()
    target = plain(name)
    by_plain: dict[str, dict[str, Any]] = {}
    for r in rows:
        by_plain.setdefault(plain(r["player"]), r)
    if target in by_plain:
        return by_plain[target]
    si = _surname_initial(target)
    same = [
        r for p, r in by_plain.items() if len(target.split()) > 1 and _surname_initial(p) == si
    ]
    if len(same) == 1:
        return same[0]
    best = difflib.get_close_matches(target, list(by_plain), n=1, cutoff=PLAYER_CUTOFF)
    return by_plain[best[0]] if best else None


def match_starters(
    starters: list[dict[str, str]], form: pl.DataFrame, home_team: str, away_team: str
) -> list[dict[str, Any]]:
    """Each starter with his matched Understat player and form, or nulls when unmatched."""
    squads = {
        side: form.filter(pl.col("team") == team).sort("last_played", descending=True)
        for side, team in (("home", home_team), ("away", away_team))
    }
    out = []
    for s in starters:
        hit = match_player(s["name"], squads[s["side"]])
        out.append(
            {
                **s,
                "player_id": hit["player_id"] if hit else None,
                "player": hit["player"] if hit else None,
                **{f: hit[f] if hit else None for f in ("prev_min", "xg90", "xa90", "form90")},
            }
        )
    return out


def with_lineup(features: pl.DataFrame, starters: list[dict[str, Any]]) -> pl.DataFrame:
    """One fixture's feature row (predict.fixture_features) with the announced XIs' slots."""
    values: dict[str, float | None] = {}
    for prefix, side in (("h", "home"), ("a", "away")):
        xi = [s for s in starters if s["side"] == side][:N_SLOTS]
        for slot in range(1, N_SLOTS + 1):
            s = xi[slot - 1] if slot <= len(xi) else {}
            for f in PLAYER_SLOT_FEATURES:
                values[f"{prefix}_player{slot}_{f}"] = s.get(f)
    return features.with_columns(pl.lit(v, pl.Float64).alias(c) for c, v in values.items())
