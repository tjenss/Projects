import fastf1
import os
import sqlite3
import os

# Create cache folder if it doesn't exist
if not os.path.exists('./fastf1_cache'):
    os.makedirs('./fastf1_cache')

# Enable cache
fastf1.Cache.enable_cache('./fastf1_cache')

session = fastf1.get_session(2023, 'Silverstone', 'R')
session.load()

laps = session.laps

df = laps[['Driver', 'LapNumber', 'Compound', 'TyreLife', 'TrackStatus',
           'IsAccurate', 'LapTime', 'Sector1Time', 'Sector2Time', 'Sector3Time']].copy()

time_cols = ['LapTime', 'Sector1Time', 'Sector2Time', 'Sector3Time']
for col in time_cols:
    df[col] = df[col].dt.total_seconds()

# Connect to SQLite database (creates file if it doesn't exist)
conn = sqlite3.connect('f1_race_data.db')

# Insert data (replaces table if exists)
df.to_sql('lap_data', conn, if_exists='replace', index=False)

# Confirm table structure
cursor = conn.cursor()
cursor.execute("PRAGMA table_info(lap_data);")
print(cursor.fetchall())

conn.commit()
conn.close()
