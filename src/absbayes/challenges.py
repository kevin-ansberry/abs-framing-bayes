"""Parse pitch-level challenge events out of MLB's live game feed (statsapi.mlb.com).

Schema assumptions (seen in one real 2026 feed; everything else is read defensively):
  liveData.plays.allPlays[] -> about.atBatIndex (0-based), matchup.batter.id, matchup.pitcher.id,
                               playEvents[]
  playEvents[] with isPitch == True may carry reviewDetails:
      {isOverturned, inProgress, reviewType, challengeTeamId, player: {id, fullName}}
  pitchData: strikeZoneTop, strikeZoneBottom, coordinates.pX, coordinates.pZ
"""
import numpy as np
import pandas as pd

from .features import normalize_location

COLUMNS = ["game_pk", "at_bat_number", "pitch_number", "overturned", "in_progress",
           "review_type", "challenge_team_id", "challenger_id", "challenger_name",
           "challenger_role", "batter_id", "pitcher_id", "feed_call_code", "feed_call",
           "pX", "pZ", "sz_top", "sz_bot"]


def parse_feed(feed, game_pk):
    """One row per pitch-level review in a game feed (may be empty)."""
    rows = []
    plays = feed.get("liveData", {}).get("plays", {}).get("allPlays", [])
    for play in plays:
        about = play.get("about", {}) or {}
        matchup = play.get("matchup", {}) or {}
        batter = (matchup.get("batter") or {}).get("id")
        pitcher = (matchup.get("pitcher") or {}).get("id")
        for ev in play.get("playEvents", []) or []:
            rd = ev.get("reviewDetails")
            if not rd or not ev.get("isPitch"):
                continue
            pdata = ev.get("pitchData") or {}
            coords = pdata.get("coordinates") or {}
            player = rd.get("player") or {}
            pid = player.get("id")
            if pid is None:
                role = None
            elif pid == batter:
                role = "batter"
            elif pid == pitcher:
                role = "pitcher"
            else:
                role = "catcher"  # assumption; checked against Statcast fielder_2 later
            call = (ev.get("details") or {}).get("call") or {}
            idx = about.get("atBatIndex")
            rows.append({
                "game_pk": game_pk,
                "at_bat_number": None if idx is None else idx + 1,
                "pitch_number": ev.get("pitchNumber"),
                "overturned": int(bool(rd.get("isOverturned"))),
                "in_progress": int(bool(rd.get("inProgress"))),
                "review_type": rd.get("reviewType"),
                "challenge_team_id": rd.get("challengeTeamId"),
                "challenger_id": pid, "challenger_name": player.get("fullName"),
                "challenger_role": role, "batter_id": batter, "pitcher_id": pitcher,
                "feed_call_code": call.get("code"), "feed_call": call.get("description"),
                "pX": coords.get("pX"), "pZ": coords.get("pZ"),
                "sz_top": pdata.get("strikeZoneTop"), "sz_bot": pdata.get("strikeZoneBottom"),
            })
    return pd.DataFrame(rows, columns=COLUMNS)


def add_edge_dist(df):
    """Distance of the pitch from the nearest zone edge (>= 0), in the same zone-edge units
    the called-strike model uses."""
    x, z = normalize_location(df["pX"].astype(float), df["pZ"].astype(float),
                              df["sz_bot"].astype(float), df["sz_top"].astype(float))
    out = df.copy()
    out["edge_dist"] = np.abs(np.maximum(np.abs(x), np.abs(z)) - 1.0)
    return out
