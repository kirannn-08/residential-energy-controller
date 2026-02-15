import torch
from torch.utils.data import DataLoader

from tcn.datasets.pv_dataset import PVDataset
from tcn.models.pv_tcn import PVTCN


def generate_dummy_pv(length=200):
    # generating synthetic PV data
    t = torch.linspace(0, 4 * 3.1416, length)
    pv = torch.sin(t).clamp(min=0)
    return pv.numpy()


def main():
    window_size = 30
    batch_size = 8

    pv_series = generate_dummy_pv()

    dataset = PVDataset(pv_series, window_size=window_size)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    model = PVTCN(
        input_channels=1,
        channel_sizes=[16, 16, 32],
        kernel_size=3,
        dropout=0.0
    )
    model.eval()

    x, y = next(iter(loader))

    print("Input x shape:", x.shape)
    print("Target y shape:", y.shape)

    with torch.no_grad():
        y_pred = model(x)

    print("Prediction shape:", y_pred.shape)
    print("Sample predictions:", y_pred[:5].squeeze())


if __name__ == "__main__":
    main()