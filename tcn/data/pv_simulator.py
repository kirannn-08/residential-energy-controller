import numpy as np


def simulate_day(
    minutes=1440,
    peak_kw=1.0,
    cloud_chance=0.1,
    cloud_intensity=0.5
):
    """
    Simulates realistic residential PV for one day.
    """

    t = np.linspace(0, 24, minutes)

    # Solar elevation curve
    solar_curve = np.sin(np.pi * (t - 6) / 12)
    solar_curve = np.clip(solar_curve, 0, None)

    pv = peak_kw * solar_curve

    # Add cloud disturbances
    for i in range(minutes):
        if np.random.rand() < cloud_chance:
            drop = np.random.uniform(0, cloud_intensity)
            duration = np.random.randint(5, 30)
            pv[i:i+duration] *= (1 - drop)

    # Small noise
    pv += 0.02 * np.random.randn(minutes)
    pv = np.clip(pv, 0, None)

    return pv