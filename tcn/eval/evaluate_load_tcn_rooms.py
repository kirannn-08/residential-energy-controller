import torch
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
import numpy as np

from tcn.datasets.load_dataset import LoadDataset
from tcn.models.load_tcn import LoadTCN
from tcn.data.load_simulator import simulate_house_day


WINDOW_SIZE = 30


def main():

    # Generate one simulated day
    _, per_room = simulate_house_day()
    dataset = LoadDataset(per_room, window_size=WINDOW_SIZE)
    loader = DataLoader(dataset, batch_size=1, shuffle=False)

    # Load trained model
    model = LoadTCN(
        input_channels=4,
        output_channels=4,
        channel_sizes=[16, 16, 32],
        kernel_size=3,
        dropout=0.2
    )

    model.load_state_dict(torch.load("tcn/models/load_tcn_rooms.pt"))
    model.eval()

    preds = []
    targets = []

    with torch.no_grad():
        for x, y in loader:
            pred = model(x)
            preds.append(pred.numpy().flatten())
            targets.append(y.numpy().flatten())

    preds = np.array(preds)
    targets = np.array(targets)

    room_names = ["Bedroom", "Hall", "Kitchen", "Spikes"]

    plt.figure(figsize=(14, 10))

    for i in range(4):
        plt.subplot(4, 1, i+1)
        plt.plot(targets[:500, i], label="True")
        plt.plot(preds[:500, i], label="Predicted")
        plt.title(room_names[i])
        plt.legend()

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()