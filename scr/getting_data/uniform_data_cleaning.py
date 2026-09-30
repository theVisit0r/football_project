import pandas as pd
import numpy as np

from paths import RAW_DATA_DIR, CLEAN_DATA_DIR
from utilityFunctions import combine_csv_files

attack_stats_file = RAW_DATA_DIR / "shooting.csv"
defend_stats_file = RAW_DATA_DIR / "opponent_shooting.csv"

folder = "football-data"

# Search a three letter abbreviation for the team name
# todo: automate this, have a database which adds/stores/replaces team abbreviations
switch = {'BIR': 'Birmingham City',
 'BLA': 'Blackburn',
 'BRC': 'Bristol City',
 'CHA': 'Charlton Athletic',
 'COV': 'Coventry City',
 'DER': 'Derby County',
 'HUL': 'Hull City',
 'IPS': 'Ipswich Town',
 'LEI': 'Leicester City',
 'MID': 'Middlesbrough',
 'MIL': 'Millwall',
 'NOR': 'Norwich City',
 'OXF': 'Oxford United',
 'POR': 'Portsmouth',
 'PNE': 'Preston',
 'QPR': 'QPR',
 'SHU': 'Sheffield United',
 'SHW': 'Sheffield Weds',
 'SOU': 'Southampton',
 'STO': 'Stoke City',
 'SWA': 'Swansea City',
 'WAT': 'Watford',
 'WBA': 'West Brom',
 'WRE': 'Wrexham',
 'ARS': 'Arsenal',
 'AVL': 'Aston Villa',
 'BOU': 'Bournemouth',
 'BRE': 'Brentford',
 'BHA': 'Brighton',
 'BUR': 'Burnley',
 'CHE': 'Chelsea',
 'CRY': 'Crystal Palace',
 'EVE': 'Everton',
 'FUL': 'Fulham',
 'LEE': 'Leeds United',
 'LIV': 'Liverpool',
 'MCI': 'Manchester City',
 'MUN': 'Manchester Utd',
 'NEW': 'Newcastle',
 'NFO': 'Nottingham',
 'SUN': 'Sunderland',
 'TOT': 'Tottenham',
 'WHU': 'West Ham',
 'WOL': 'Wolves',
 'LUT': 'Luton'}

def load_football_data_1(folder, columns=None):
    """
    Takes all the data from the football-data folder and combines it into one dataset

    Parameters
    ----------
    folder: name of folder in the raw pathway
    columns: can choose specific columns to keep when importing

    Returns
    -------
    Cleaned data
    """

    team_abbreviations = {
        "Man United": "MUN",
        "Ipswich": "IPS",
        "Arsenal": "ARS",
        "Everton": "EVE",
        "Newcastle": "NEW",
        "Nott'm Forest": "NFO",
        "West Ham": "WHU",
        "Brentford": "BRE",
        "Chelsea": "CHE",
        "Leicester": "LEI",
        "Brighton": "BHA",
        "Crystal Palace": "CRY",
        "Fulham": "FUL",
        "Man City": "MCI",
        "Southampton": "SOU",
        "Tottenham": "TOT",
        "Aston Villa": "AVL",
        "Bournemouth": "BOU",
        "Wolves": "WOL",
        "Liverpool": "LIV",
        "Burnley": "BUR",
        "Sheffield United": "SHU",
        "Luton": "LUT",
        "Leeds": "LEE",
        "Sunderland": "SUN"
    }

    # Location of data
    FOLDER_DIR = RAW_DATA_DIR / folder

    # load all the data into one df
    df = combine_csv_files(FOLDER_DIR.glob("*.csv"), FOLDER_DIR, columns)

    # clean data
    # results mapping
    result_mapping = {"H": "home", "D": "draw", "A": "away"}
    df["result"] = df["FTR"].map(result_mapping)

    # date mapping
    df["Date"] = pd.to_datetime(df["Date"], dayfirst=True)
    df["year"] = df["Date"].dt.year
    df["month"] = df["Date"].dt.month

    # season spans two years, so must check if we need to go back one year
    df["season_start"] = df["year"] - (df["month"] < 8)

    df["home_team_name_short"] = df["HomeTeam"].map(team_abbreviations)
    df["away_team_name_short"] = df["AwayTeam"].map(team_abbreviations)
    # select columns to keep
    col_dict = {
        "home_team_name": "HomeTeam",
        "away_team_name": "AwayTeam",
        "home_team_name_short": "home_team_name_short",
        "away_team_name_short": "away_team_name_short",
        "home_team_score": "FTHG",
        "away_team_score": "FTAG",
        "group_id": "season_start",
        "result": "result",
        "year": "year",
        "month": "month"
    }

    # select specific columns
    clean_df = data_to_df(df, col_dict)

    return clean_df


def load_football_data_2(atk_file, def_file, save=False):
    """
    Loads data from the FBref website, considering some key team stats
    Parameters
    ----------
    atk_file: file dir for the attacking stats
    def_file: file dir for the defending stats
    save: should the output be saved into a csv file

    Returns
    -------
    Combined atk and def stats
    """

    team_abbreviations = {
        "Birmingham City": "BIR",
        "Blackburn": "BLA",
        "Bristol City": "BRC",
        "Charlton Athletic": "CHA",
        "Coventry City": "COV",
        "Derby County": "DER",
        "Hull City": "HUL",
        "Ipswich Town": "IPS",
        "Leicester City": "LEI",
        "Middlesbrough": "MID",
        "Millwall": "MIL",
        "Norwich City": "NOR",
        "Oxford United": "OXF",
        "Portsmouth": "POR",
        "Preston": "PNE",
        "QPR": "QPR",
        "Sheffield United": "SHU",
        "Sheffield Weds": "SHW",
        "Southampton": "SOU",
        "Stoke City": "STO",
        "Swansea City": "SWA",
        "Watford": "WAT",
        "West Brom": "WBA",
        "Wrexham": "WRE",

        "Arsenal": "ARS",
        "Aston Villa": "AVL",
        "Bournemouth": "BOU",
        "Brentford": "BRE",
        "Brighton": "BHA",
        "Burnley": "BUR",
        "Chelsea": "CHE",
        "Crystal Palace": "CRY",
        "Everton": "EVE",
        "Fulham": "FUL",
        "Leeds United": "LEE",
        "Liverpool": "LIV",
        "Manchester City": "MCI",
        "Manchester Utd": "MUN",
        "Newcastle": "NEW",
        "Nottingham": "NFO",
        "Sunderland": "SUN",
        "Tottenham": "TOT",
        "West Ham": "WHU",
        "Wolves": "WOL"
    }

    # Load data
    atk_df = pd.read_csv(atk_file)
    def_df = pd.read_csv(def_file)

    # Making sure the first two rows are the same
    if not atk_df.iloc[:2].equals(def_df.iloc[:2]):
        raise ValueError("The input data is not of the same format")

    # array of current column names
    new_headers = atk_df.columns.tolist()

    # Specify new headers
    new_headers[0:3] = atk_df.iloc[1, 0:3].tolist()
    new_headers[5:15] = atk_df.iloc[0, 5:15].tolist()

    # Assign new headers
    atk_df.columns = new_headers
    def_df.columns = new_headers

    # Remove rows
    atk_df = atk_df.iloc[2:].reset_index(drop=True)
    def_df = def_df.iloc[2:].reset_index(drop=True)

    # Clean column
    def_df["team"] = def_df["team"].str.removeprefix("vs ")

    atk_df["team_name_short"] = atk_df["team"].replace(team_abbreviations)
    def_df["team_name_short"] = def_df["team"].replace(team_abbreviations)

    # Sanity check for same teams
    if not atk_df["team"].equals(def_df["team"]):
        raise ValueError("The teams are not in the same order.")
    if not atk_df["team_name_short"].equals(def_df["team_name_short"]):
        raise ValueError("The teams are not in the same order.")

    # identifier
    swap_name = ['players_used', '90s', 'Gls', 'Sh', 'SoT', 'SoT%',
                 'Sh/90', 'SoT/90', 'G/Sh', 'G/SoT', 'PK', 'PKatt']
    def_df = def_df.rename(columns=dict([(i, "def_" + i) for i in swap_name]))
    atk_df = atk_df.rename(columns=dict([(i, "atk_" + i) for i in swap_name]))

    # Combine data
    combined = atk_df.merge(def_df, on=["team", "team_name_short", "url", "league", "season"], how="outer")

    # change datatype to float
    dont_change_type = ["league", "season", "team", "team_name_short", "url", "type"]
    num_col = combined.columns.difference(dont_change_type)
    combined[num_col] = combined[num_col].astype(float)

    # Add New Columns - Penalty Percentage
    combined["atk_PK%"] = np.where(combined["atk_PK"] == 0, 0, combined["atk_PK"] / combined["atk_PKatt"])
    combined["def_PK%"] = np.where(combined["def_PK"] == 0, 0, combined["def_PK"] / combined["def_PKatt"])

    combined.rename(columns={"team": "team_name"}, inplace=True)

    if save:
        combined.to_csv(CLEAN_DATA_DIR / "combined_team_stats.csv", index=False)

    return combined


def load_football_data_3():
    pass


# Utility Function

def data_to_df(data, col_dict):
    """
    Takes a whole dataframe and returns only the columns which we pick

    Parameters
    ----------
    data: all the data with the info we are trying to extract
    col_dict: labels for which columns to extract

    Returns
    -------
    dataframe with only the selected columns
    """

    # map column to data column name
    cols = {
        "home_team_name": col_dict["home_team_name"],
        "away_team_name": col_dict["away_team_name"],
        "home_team_name_short": col_dict["home_team_name_short"],
        "away_team_name_short": col_dict["away_team_name_short"],
        "home_team_score": col_dict["home_team_score"],
        "away_team_score": col_dict["away_team_score"],
        "group_id": col_dict["group_id"],
        "result": col_dict["result"],
        "year": col_dict["year"],
        "month": col_dict["month"],
    }

    # make a copy of the data and change column names
    df = data.loc[:, cols.values()].copy()
    df.columns = cols.keys()

    return df


# _____________________

# df1 = load_football_data_1(folder)
df2 = load_football_data_2(attack_stats_file, defend_stats_file)
