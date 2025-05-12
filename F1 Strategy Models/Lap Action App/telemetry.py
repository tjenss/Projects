import fastf1
from fastf1 import plotting
from fastf1.plotting import setup_mpl
import matplotlib.pyplot as plt
import os

# Create cache folder if it doesn't exist
# if not os.path.exists('./fastf1_cache'):
#     os.makedirs('./fastf1_cache')

# Enable cache
fastf1.Cache.enable_cache('./fastf1_cache')

# Load a race session
session = fastf1.get_session(2023, 'Silverstone', 'R')  # 'R' = race
session.load()

# Choose a driver
ver_data = session.laps.pick_driver('VER')

# Pick a specific lap
lap = ver_data.pick_fastest()
tel = lap.get_telemetry()

# Inspect telemetry (speed, throttle, gear, etc.)
print(tel.head())

# Visualize Telemetry Channel
tel.plot(x='Distance', y='Speed')
plt.title('Verstappen Fastest Lap Speed Trace')
plt.show()
