import numpy as np
import pandas as pd
import math

from core.hybrid_controller import HybridController


# =====================================================
# REALISTIC KAGGLE HOUSE HARDWARE
# =====================================================

INVERTER_MAX_KW = 5.0
SOLAR_PEAK_KW = 4.0
BATTERY_CAPACITY_KWH = 10.0
SOC_MIN = 0.2
SOC_RESERVE_NIGHT = 0.3

STEP_MINUTES = 1
STEP_HOURS = STEP_MINUTES / 60


# =====================================================
# SOLAR MODEL
# =====================================================

def simulate_solar(minute_of_day):

    if minute_of_day < 360 or minute_of_day > 1080:
        return 0.0

    x = (minute_of_day - 360) / 720
    return SOLAR_PEAK_KW * math.sin(math.pi * x)


# =====================================================
# MAIN SIMULATION
# =====================================================

def main():

    df = pd.read_csv("data/processed/ml_household.csv")

    df = df.iloc[200000:200000+1440].reset_index(drop=True)

    controller = HybridController(
        inverter_max_kw=INVERTER_MAX_KW,
        soc_min=SOC_MIN,
        soc_reserve_night=SOC_RESERVE_NIGHT,
        spike_threshold_kw=3.0,
        spike_hold_steps=3,
    )

    battery_soc = 0.6

    total_load = 0
    total_solar_used = 0
    total_battery_used = 0
    total_grid_used = 0

    switches = 0
    previous_grid_state = None

    for minute in range(len(df)):

        load_kw = df.loc[minute, "Global_active_power"]
        solar_kw = simulate_solar(minute)
        is_night = solar_kw == 0

        decision = controller.decide(
            load_kw=load_kw,
            battery_soc=battery_soc,
            grid_available=True,
            is_night=is_night,
        )

        # -------------------------------------------------
        # HYBRID ENERGY ALLOCATION
        # -------------------------------------------------

        solar_to_load = min(load_kw, solar_kw)

        remaining_load = load_kw - solar_to_load

        battery_to_load = 0
        grid_to_load = 0

        if decision["shed_load"]:
            remaining_load = 0

        if not decision["force_grid"] and decision["allow_battery"]:
            battery_to_load = min(
                remaining_load,
                battery_soc * BATTERY_CAPACITY_KWH / STEP_HOURS
            )

        remaining_load -= battery_to_load

        grid_to_load = max(0, remaining_load)

        # -------------------------------------------------
        # UPDATE ENERGY TOTALS
        # -------------------------------------------------

        total_load += load_kw * STEP_HOURS
        total_solar_used += solar_to_load * STEP_HOURS
        total_battery_used += battery_to_load * STEP_HOURS
        total_grid_used += grid_to_load * STEP_HOURS

        # Update SOC
        battery_soc -= (battery_to_load * STEP_HOURS) / BATTERY_CAPACITY_KWH

        excess_solar = solar_kw - solar_to_load
        if excess_solar > 0:
            battery_soc += (excess_solar * STEP_HOURS) / BATTERY_CAPACITY_KWH

        battery_soc = max(0, min(1, battery_soc))

        # Switching tracking
        grid_state = grid_to_load > 0
        if grid_state != previous_grid_state:
            switches += 1
        previous_grid_state = grid_state

    # =====================================================
    # RESULTS
    # =====================================================

    print("\n=== Hybrid Simulation Results ===")
    print(f"Total Load Energy   : {total_load:.2f} kWh")
    print(f"Solar Energy Used   : {total_solar_used:.2f} kWh")
    print(f"Battery Energy Used : {total_battery_used:.2f} kWh")
    print(f"Grid Energy Used    : {total_grid_used:.2f} kWh")
    print(f"Final SOC           : {battery_soc*100:.1f}%")
    print(f"Switching Events    : {switches}")

    solar_fraction = (total_solar_used / total_load) * 100
    grid_fraction = (total_grid_used / total_load) * 100

    print(f"\nSolar Fraction      : {solar_fraction:.1f}%")
    print(f"Grid Fraction       : {grid_fraction:.1f}%")
    print("==============================================")


if __name__ == "__main__":
    main()