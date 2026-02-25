import torch
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

from tcn.datasets.load_dataset import LoadDataset
from tcn.models.load_tcn import LoadTCN


WINDOW_SIZE = 30


def main():

    df = pd.read_csv("data/processed/ml_household.csv")

    # Use last 10,000 rows for visualization
    df = df.iloc[-10000:]

    power = df["Global_active_power"].values.astype(np.float32)
    power = power.reshape(-1, 1)

    dataset = LoadDataset(power, window_size=WINDOW_SIZE)

    model = LoadTCN(
        input_channels=1,
        output_channels=1,
        channel_sizes=[16, 16, 32],
        kernel_size=3,
        dropout=0.2
    )

    model.load_state_dict(torch.load("tcn/models/load_tcn_real.pt"))
    model.eval()

    preds = []
    targets = []

    with torch.no_grad():
        for i in range(len(dataset)):
            x, y = dataset[i]
            x = x.unsqueeze(0)
            pred = model(x)
            preds.append(pred.item())
            targets.append(y.item())

    plt.figure(figsize=(12,5))
    plt.plot(targets[400:600], label="True")
    plt.plot(preds[400:600], label="Pred")
    plt.legend()
    plt.title("Real Household Load Forecasting")
    plt.show()


if __name__ == "__main__":
    main()