import sqlite3
from paths import DB_PATH, SCHEMA_PATH

with sqlite3.connect(DB_PATH) as connect:
    with open(SCHEMA_PATH, "r") as file:
        schema = file.read()

    connect.executescript(schema)

print(f"Database created at: {DB_PATH}")
