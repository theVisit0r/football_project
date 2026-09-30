from pathlib import Path

import pandas as pd
import numpy as np
import time

BASE_DIR = Path(__file__).resolve().parents[1]

DATA_DIR = BASE_DIR / "data" / "raw" / "github"

def load_football_data(columns=None):
    folder = BASE_DIR / "data" / "raw" / "football-data"

    df = combine_csv_files(folder.glob("*.csv"), folder, columns)

    mapping = {"H": "home", "D": "draw", "A": "away"}

    df["Date"] = pd.to_datetime(df["Date"], dayfirst=True)
    df["season_start"] = df["Date"].dt.year - (df["Date"].dt.month < 8)

    df["result"] = df["FTR"].map(mapping)
    df["year"] = df["Date"].dt.year
    df["month"] = df["Date"].dt.month

    col_dict = {
        "home_team_name": "HomeTeam",
        "away_team_name": "AwayTeam",
        "home_team_score": "FTHG",
        "away_team_score": "FTAG",
        "group_id": "season_start",
        "result": "result",
        "year": "year",
        "month": "month"
    }

    df = data_to_df(df, col_dict)

    return df


def load_pl_data(files, columns=None):
    df = combine_csv_files(files, BASE_DIR / "raw" / "clean" / "testNewData", columns)

    # Clean Data
    df["winner"] = df["winner"].replace({"HOME_TEAM": "home", "AWAY_TEAM": "away", "DRAW": "draw"})
    df["year_start"] = df["season_start_date"].str[0:4]
    df["year"] = df["utc_date"].str[0:4]
    df["month"] = df["utc_date"].str[5:7]

    df = df[df["status"] == "FINISHED"]

    col_dict = {
        "home_team_name":"home_team_name",
        "away_team_name":"away_team_name",
        "home_team_score":"home_goals",
        "away_team_score":"away_goals",
        "group_id":"year_start",
        "result":"winner",
        "year":"year",
        "month":"month"
    }

    df = data_to_df(df, col_dict)

    return df


def load_data(data_file, columns=None, year=None, men_only=True):
    data = pd.read_csv(DATA_DIR / data_file, usecols=columns)

    # Clean Data
    data["year"] = data["tournament_id"].str[-4:].astype("int64")
    data["month"] = data["match_time"].str[5:7]
    data["men"] = data["tournament_name"].str.contains("Men")
    data["result"] = data["result"].replace({"home team win": "home", "away team win": "away"})

    if year is not None:
        data = data[data["year"] >= year]

    if men_only:
        data = data[data["men"]]

    col_dict = {
        "home_team_name":"home_team_name",
        "away_team_name":"away_team_name",
        "home_team_score":"home_team_score",
        "away_team_score":"away_team_score",
        "group_id":"year",
        "result":"result",
        "year":"year",
        "month":"month"
    }

    df = data_to_df(data, col_dict)

    return df


def ordered_teams(df, home_col, away_col):
    """
    Creates an array with all the unique teams in alphabetical order
    Parameters
    ----------
    df: data with team names
    home_col: home team name column
    away_col: away team name column

    Returns
    -------
    An alphabetically ordered array with all the teams
    """
    ordered = np.sort(
        pd.concat([
            df[home_col],
            df[away_col]
        ]).unique()
    )

    return ordered


def data_to_df(data, col_dict):
    cols = {
        "home_team_name": col_dict["home_team_name"],
        "away_team_name": col_dict["away_team_name"],
        "home_team_score": col_dict["home_team_score"],
        "away_team_score": col_dict["away_team_score"],
        "group_id": col_dict["group_id"],
        "result": col_dict["result"],
        "year": col_dict["year"],
        "month": col_dict["month"],
    }

    df = data.loc[:, cols.values()].copy()
    df.columns = cols.keys()

    return df # todo: understand me please


def combine_csv_files(files, dir_path, columns=None):
    frames = []

    for file in files:
        df = pd.read_csv(dir_path / file, usecols=columns)
        frames.append(df)

    df = pd.concat(frames).copy()

    return df


def measure_time(func):
    def wrapper(*args, **kwargs):
        start = time.perf_counter()

        result = func(*args, **kwargs)

        elapsed = time.perf_counter() - start
        print(f"Time Taken: {elapsed:.4f} seconds")

        return result

    return wrapper