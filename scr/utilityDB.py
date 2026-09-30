import sqlite3
from paths import DB_PATH


def save_team_parameter(country, attack, defence, table_name):
    with sqlite3.connect(DB_PATH) as connect:
        cursor = connect.cursor()

        query = f"""
        INSERT OR REPLACE INTO {table_name} (country, attack, defence)
        VALUES (?, ?, ?)
        """

        cursor.execute(
            query,
            (country, attack, defence)
        )


def save_dixon_coles_parameter(country, h_a, h_d, a_a, a_d):
    with sqlite3.connect(DB_PATH) as connect:
        cursor = connect.cursor()

        query = """
        INSERT OR REPLACE INTO dixon_coles_parameters (country, home_atk, home_def, away_atk, away_def)
        VALUES (?, ?, ?, ?, ?)
        """

        cursor.execute(
            query,
            (country, h_a, h_d, a_a, a_d)
        )


def save_global_parameter(param_name, param_value):
    with sqlite3.connect(DB_PATH) as connect:
        cursor = connect.cursor()

        query = """
        INSERT OR REPLACE INTO global_variables (param_name, param_value)
        VALUES (?, ?)
        """

        cursor.execute(
            query,
            (param_name, param_value)
        )


if __name__ == "__main__":
    save_team_parameter("Brazil", 0.42, -0.8)
    save_team_parameter("France", 0.31, -0.1)
    save_global_parameter("home_advantage", 0.31)
    print("done")
