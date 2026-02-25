import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader, Subset

from tcn.datasets.load_dataset import LoadDataset
from tcn.models.load_tcn import LoadTCN


# -----------------------
# Configuration
# -----------------------
WINDOW_SIZE = 30
BATCH_SIZE = 64          # Safe for 8GB RAM
EPOCHS = 10
LEARNING_RATE = 1e-3
TRAIN_SPLIT = 0.8


def get_device():
    """
    Use Apple Metal (MPS) if available,
    otherwise CPU.
    """
    if torch.backends.mps.is_available():
        print("Using Apple MPS GPU")
        return torch.device("mps")
    else:
        print("Using CPU")
        return torch.device("cpu")


def main():

    print("Loading dataset...")
    df = pd.read_csv("data/processed/ml_household.csv")

    print("Total rows:", len(df))

    # Use full dataset
    power = df["Global_active_power"].values.astype(np.float32)
    power = power.reshape(-1, 1)

    dataset = LoadDataset(power, window_size=WINDOW_SIZE)

    total_samples = len(dataset)
    train_len = int(total_samples * TRAIN_SPLIT)

    # Chronological split (VERY important for time-series)
    train_ds = Subset(dataset, range(0, train_len))
    val_ds = Subset(dataset, range(train_len, total_samples))

    train_loader = DataLoader(
        train_ds,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0
    )

    model = LoadTCN(
        input_channels=1,
        output_channels=1,
        channel_sizes=[16, 16, 32],
        kernel_size=3,
        dropout=0.2
    )

    device = get_device()
    model.to(device)

    criterion = nn.SmoothL1Loss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    print("Starting training...\n")

    for epoch in range(EPOCHS):

        model.train()
        train_loss = 0.0

        for x, y in train_loader:
            x = x.to(device)
            y = y.to(device)

            optimizer.zero_grad()
            preds = model(x)
            loss = criterion(preds, y)

            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        train_loss /= len(train_loader)

        model.eval()
        val_loss = 0.0

        with torch.no_grad():
            for x, y in val_loader:
                x = x.to(device)
                y = y.to(device)

                preds = model(x)
                loss = criterion(preds, y)
                val_loss += loss.item()

        val_loss /= len(val_loader)

        print(
            f"Epoch [{epoch+1}/{EPOCHS}] "
            f"Train Loss: {train_loss:.6f} | Val Loss: {val_loss:.6f}"
        )

    torch.save(model.state_dict(), "tcn/models/load_tcn_real.pt")
    print("\nReal-data LoadTCN saved successfully.")


if __name__ == "__main__":
    main()