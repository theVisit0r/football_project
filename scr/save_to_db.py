import sqlite3
from paths import DB_PATH

def save_poisson_params(table_name, team_name, attack, defence):
    with sqlite3.connect(DB_PATH) as connect:
        cursor = connect.cursor()

        query = f"""
        INSERT OR REPLACE INTO {table_name} (team_name, attack, defence)
        VALUES (?, ?, ?)
        """

        cursor.execute(
            query,
            (team_name, float(attack), float(defence))
        )

