import torch
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
import numpy as np

from tcn.datasets.load_dataset import LoadDataset
from tcn.models.load_tcn import LoadTCN
from tcn.data.load_simulator import simulate_house_day


WINDOW_SIZE = 30


def main():

    total, _ = simulate_house_day()
    total = total.reshape(-1, 1)

    dataset = LoadDataset(total, window_size=WINDOW_SIZE)
    loader = DataLoader(dataset, batch_size=1, shuffle=False)

    model = LoadTCN(
        input_channels=1,
        channel_sizes=[16, 16, 32],
        kernel_size=3,
        dropout=0.2
    )

    model.load_state_dict(torch.load("tcn/models/load_tcn_total.pt"))
    model.eval()

    preds = []
    targets = []

    with torch.no_grad():
        for x, y in loader:
            pred = model(x)
            preds.append(pred.item())
            targets.append(y.item())

    plt.figure(figsize=(12, 5))
    plt.plot(targets[:500], label="True Load")
    plt.plot(preds[:500], label="Predicted Load")
    plt.legend()
    plt.title("Total Load Forecasting")
    plt.show()


if __name__ == "__main__":
    main()