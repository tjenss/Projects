import random

# --- Configurable Parameters ---
TOTAL_LAPS = 58
PIT_STOP_LOSS = 20  # seconds
SAFETY_CAR_LAPS = [15, 36]  # laps with safety car
WEATHER_CHANGES = {
    0: 'Dry',
    23: 'Wet',
    40: 'Dry'
}

# Base lap time by condition
BASE_LAP_TIMES = {
    'Dry': 90,
    'Wet': 105
}

# Tire model: (initial pace, degradation per lap)
TIRES = {
    'Soft': (0, 0.20),
    'Medium': (0, 0.15),
    'Hard': (0, 0.10)
}

# Strategy plan: tuples of (lap to pit, tire type)
strategy_plan = [
    (0, 'Medium'),
    (25, 'Soft'),
    (41, 'Soft')
]

# Fuel impact per lap (in seconds)
FUEL_PENALTY_PER_LAP = 0.035  # roughly 2s over 58 laps

# --- Simulation Code ---
def simulate_race():
    total_time = 0
    current_tire = None
    tire_age = 0
    weather = 'Dry'
    strategy_index = 0
    lap_times = []

    for lap in range(TOTAL_LAPS):
        # Update weather if needed
        if lap in WEATHER_CHANGES:
            weather = WEATHER_CHANGES[lap]

        # Pit stop logic
        if strategy_index < len(strategy_plan) and lap == strategy_plan[strategy_index][0]:
            if current_tire is not None:
                total_time += PIT_STOP_LOSS
            current_tire = strategy_plan[strategy_index][1]
            tire_age = 0
            strategy_index += 1

        # Base lap time with condition
        base_time = BASE_LAP_TIMES[weather]

        # Apply degradation
        deg_rate = TIRES[current_tire][1]
        lap_time = base_time + (tire_age * deg_rate)

        # Apply fuel penalty (starts high and drops each lap)
        fuel_penalty = FUEL_PENALTY_PER_LAP * (TOTAL_LAPS - lap)
        lap_time += fuel_penalty

        # Apply safety car modifier
        if lap in SAFETY_CAR_LAPS:
            lap_time *= 0.6  # 40% time reduction

        lap_times.append(lap_time)
        total_time += lap_time
        tire_age += 1

    return total_time, lap_times

# --- Run and Print ---
if __name__ == '__main__':
    total, laps = simulate_race()
    print(f"Total Race Time: {total:.2f} seconds")
    print(f"Average Lap Time: {sum(laps)/len(laps):.2f} seconds")
    print("\nLap Breakdown:")
    for i, lt in enumerate(laps):
        print(f"Lap {i+1:2}: {lt:.2f}s")
