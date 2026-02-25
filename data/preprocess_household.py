import pandas as pd
import numpy as np


def main():

    df = pd.read_csv(
        "data/raw/household_power.csv",
        sep=",",
        na_values=["?"],
        low_memory=False
    )

    # Combine Date and Time into datetime
    df["datetime"] = pd.to_datetime(
        df["Date"] + " " + df["Time"],
        dayfirst=True
    )

    df = df.sort_values("datetime")

    # Columns we care about
    cols = [
        "Global_active_power",
        "Global_reactive_power",
        "Sub_metering_1",
        "Sub_metering_2",
        "Sub_metering_3"
    ]

    # Convert to numeric
    for col in cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # Drop rows with missing values
    df = df.dropna()

    # Keep only relevant columns
    df = df[["datetime"] + cols]

    df.to_csv("data/processed/clean_household.csv", index=False)

    print("Cleaned dataset saved.")
    print("Remaining rows:", len(df))


if __name__ == "__main__":
    main()