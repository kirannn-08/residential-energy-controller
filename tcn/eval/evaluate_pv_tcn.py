import torch
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader

from tcn.datasets.pv_dataset import PVDataset
from tcn.models.pv_tcn import PVTCN


WINDOW_SIZE = 30


def generate_dummy_pv(length=1000):
    t = torch.linspace(0, 8 * 3.1416, length)
    pv = torch.sin(t).clamp(min=0)
    noise = 0.05 * torch.randn_like(pv)
    return (pv + noise).clamp(min=0).numpy()


def main():
    # Load dataset
    pv_series = generate_dummy_pv()
    dataset = PVDataset(pv_series, window_size=WINDOW_SIZE)
    loader = DataLoader(dataset, batch_size=1, shuffle=False)

    # Load model
    model = PVTCN(
        input_channels=1,
        channel_sizes=[32, 32, 64],
        kernel_size=3,
        dropout=0.2
    )

    model.load_state_dict(torch.load("tcn/models/pv_tcn.pt"))
    model.eval()

    preds = []
    targets = []

    with torch.no_grad():
        for x, y in loader:
            pred = model(x)
            preds.append(pred.item())
            targets.append(y.item())

    # Plot
    plt.figure(figsize=(10, 5))
    plt.plot(targets[:300], label="True PV")
    plt.plot(preds[:300], label="Predicted PV")
    plt.legend()
    plt.title("PV Forecasting - True vs Predicted")
    plt.show()


if __name__ == "__main__":
    main()