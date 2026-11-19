import numpy as np

from absbayes.challenges import add_edge_dist, parse_feed


def pitch(n, review=None, pX=0.0, pZ=2.5, call="Ball"):
    ev = {"isPitch": True, "pitchNumber": n,
          "details": {"call": {"code": "B", "description": call}},
          "pitchData": {"strikeZoneTop": 3.5, "strikeZoneBottom": 1.5,
                        "coordinates": {"pX": pX, "pZ": pZ}}}
    if review:
        ev["reviewDetails"] = review
    return ev


def feed():
    review = lambda pid, over: {"isOverturned": over, "inProgress": False, "reviewType": "MJ",
                                "challengeTeamId": 133, "player": {"id": pid, "fullName": "X"}}
    return {"liveData": {"plays": {"allPlays": [
        {"about": {"atBatIndex": 0}, "matchup": {"batter": {"id": 1}, "pitcher": {"id": 2}},
         "playEvents": [pitch(1), pitch(2, review(1, True), pX=0.9), pitch(3, review(2, False)),
                        pitch(4, review(99, True), pX=1.2, pZ=3.0),
                        {"isPitch": False, "reviewDetails": review(1, True)}]},
        {"about": {"atBatIndex": 3}, "matchup": {"batter": {"id": 5}, "pitcher": {"id": 2}},
         "playEvents": [pitch(1, review(5, False))]}]}}}


def test_parse_finds_only_pitch_level_reviews():
    df = parse_feed(feed(), 123)
    assert len(df) == 4 and (df["game_pk"] == 123).all()
    assert list(df["at_bat_number"]) == [1, 1, 1, 4]
    assert list(df["pitch_number"]) == [2, 3, 4, 1]


def test_roles_and_overturn_flags():
    df = parse_feed(feed(), 1)
    assert list(df["challenger_role"]) == ["batter", "pitcher", "catcher", "batter"]
    assert list(df["overturned"]) == [1, 0, 1, 0]


def test_empty_feed_gives_empty_frame_with_columns():
    df = parse_feed({}, 7)
    assert len(df) == 0 and "challenger_role" in df.columns


def test_edge_distance_is_nonnegative_and_finite():
    df = add_edge_dist(parse_feed(feed(), 1))
    assert (df["edge_dist"] >= 0).all() and np.isfinite(df["edge_dist"]).all()
