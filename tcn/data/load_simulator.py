import numpy as np


def simulate_house_day(minutes=1440):

    t = np.linspace(0, 24, minutes)

    # -------------------------
    # Base constant loads
    # -------------------------
    router = 0.05 * np.ones(minutes)

    # Fridge compressor cycling (random duty cycle)
    fridge = np.zeros(minutes)
    i = 0
    while i < minutes:
        on_duration = np.random.randint(10, 25)
        off_duration = np.random.randint(20, 40)
        fridge[i:i+on_duration] = 0.18
        i += on_duration + off_duration

    # -------------------------
    # Occupancy probability curve
    # -------------------------
    occupancy = (
        0.2
        + 0.6 * ((t >= 6) & (t <= 9))
        + 0.7 * ((t >= 18) & (t <= 23))
    )

    # -------------------------
    # Bedroom (fan + light)
    # -------------------------
    bedroom = np.zeros(minutes)
    for i in range(minutes):
        if np.random.rand() < occupancy[i] * 0.02:
            duration = np.random.randint(30, 120)
            bedroom[i:i+duration] += np.random.uniform(0.1, 0.25)

    # -------------------------
    # Hall (TV + fan)
    # -------------------------
    hall = np.zeros(minutes)
    for i in range(minutes):
        if np.random.rand() < occupancy[i] * 0.015:
            duration = np.random.randint(45, 180)
            hall[i:i+duration] += np.random.uniform(0.2, 0.4)

    # -------------------------
    # Kitchen (structured but not fixed)
    # -------------------------
    kitchen = np.zeros(minutes)
    for period in [(6, 9), (12, 14), (18, 21)]:
        mask = (t >= period[0]) & (t <= period[1])
        indices = np.where(mask)[0]
        if len(indices) > 0:
            start = np.random.choice(indices)
            duration = np.random.randint(20, 90)
            kitchen[start:start+duration] += np.random.uniform(0.5, 1.0)

    # -------------------------
    # High power spikes (correlated to kitchen times)
    # -------------------------
    spikes = np.zeros(minutes)
    for period in [(6, 9), (18, 21)]:
        mask = (t >= period[0]) & (t <= period[1])
        indices = np.where(mask)[0]
        for _ in range(np.random.randint(1, 3)):
            if len(indices) > 0:
                start = np.random.choice(indices)
                duration = np.random.randint(2, 8)
                spikes[start:start+duration] += np.random.uniform(1.0, 2.0)

    # -------------------------
    # Combine realistic total
    # -------------------------
    total = router + fridge + bedroom + hall + kitchen + spikes

    total += 0.02 * np.random.randn(minutes)
    total = np.clip(total, 0, None)

    per_room = np.vstack([
        bedroom,
        hall,
        kitchen,
        spikes
    ]).T

    return total, per_room