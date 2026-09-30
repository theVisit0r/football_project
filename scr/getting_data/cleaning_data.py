import pandas as pd
import numpy as np
from fontTools.varLib.avar import __main__

from paths import RAW_DATA_DIR, CLEAN_DATA_DIR

attack_file = RAW_DATA_DIR / "shooting.csv"
defend_file = RAW_DATA_DIR / "opponent_shooting.csv"

def fbref_clean_shooting(atk_file, def_file, save=True):
    """
    Both the files should be of the same format
    Parameters
    ----------
    atk_file: file location for the attacking data
    def_file: file location for the defending data

    Returns
    -------
    Combined data into one dataframe
    """

    # Load data
    atk_df = pd.read_csv(atk_file)
    def_df = pd.read_csv(def_file)

    # Making sure the first two rows are the same
    if not atk_df.iloc[:2].equals(def_df.iloc[:2]):
        raise ValueError("The input data is not of the same format.")

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

    # making the names match the other data
    # todo: make this better
    team_name_mapping = {
        "Leeds United": "Leeds",
        "Manchester City": "Man City",
        "Manchester Utd": "Man United",
        "Nottingham": "Nott'm Forest"
    }

    atk_df["team"] = atk_df["team"].replace(team_name_mapping)
    def_df["team"] = def_df["team"].replace(team_name_mapping)

    # Sanity check for same teams
    if not atk_df["team"].equals(def_df["team"]):
        raise ValueError("The teams are not in the same order.")

    # identifier
    swap_name = ['players_used', '90s', 'Gls', 'Sh', 'SoT', 'SoT%',
                 'Sh/90', 'SoT/90','G/Sh', 'G/SoT', 'PK', 'PKatt']
    atk_df = atk_df.rename(columns=dict([(i,"atk_"+i) for i in swap_name]))
    def_df = def_df.rename(columns=dict([(i,"def_"+i) for i in swap_name]))

    # Combine data
    combined = atk_df.merge(def_df, on=["team", "url", "league", "season"], how="outer")

    # change datatype to float
    same = ["league", "season", "team", "url", "type"]
    num_col = combined.columns.difference(same)
    combined[num_col] = combined[num_col].astype(float)

    # Add New Columns - Penalty Percentage
    combined["atk_PK%"] = np.where(combined["atk_PK"] == 0, 0, combined["atk_PK"] / combined["atk_PKatt"])
    combined["def_PK%"] = np.where(combined["def_PK"] == 0, 0, combined["def_PK"] / combined["def_PKatt"])


    if save:
        combined.to_csv(CLEAN_DATA_DIR / "combined_team_stats.csv", index=False)

    return combined

if __name__ == "__main__":
    df = fbref_clean_shooting(attack_file, defend_file)
