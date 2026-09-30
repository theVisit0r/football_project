import os

from paths import RAW_DATA_DIR, FBREF_DATA_DIR, SOCCERDATA_DIR

os.environ["SOCCERDATA_DIR"] = str(SOCCERDATA_DIR)

import soccerdata as sd

# Initialise object
fbref2526 = sd.FBref(
    leagues=[
        "ENG-Premier League",
        "ENG-Championship"
    ],
    seasons="2025-2026",
    headless=True,
    data_dir=FBREF_DATA_DIR
)

# Find data for defending
opponent_shooting = fbref2526.read_team_season_stats(
    stat_type="shooting",
    opponent_stats=True
)

# Find data for attacking
shooting = fbref2526.read_team_season_stats(
    stat_type="shooting",
    opponent_stats=False
)

# Save data
opponent_shooting.to_csv(RAW_DATA_DIR / "opponent_shooting.csv")
shooting.to_csv(RAW_DATA_DIR / "shooting.csv")
