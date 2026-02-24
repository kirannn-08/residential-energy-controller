import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader, random_split

from tcn.datasets.load_dataset import LoadDataset
from tcn.models.load_tcn import LoadTCN
from tcn.data.load_simulator import simulate_house_day


# -----------------------------
# Configuration
# -----------------------------
WINDOW_SIZE = 30
BATCH_SIZE = 128
EPOCHS = 10
LEARNING_RATE = 1e-3
TRAIN_SPLIT = 0.8


def main():

    # -----------------------------
    # Generate multi-day dataset
    # -----------------------------
    days = 10
    total_series = []

    for _ in range(days):
        total, _ = simulate_house_day()
        total_series.append(total)

    total_series = np.concatenate(total_series)

    # reshape to (time, channels=1)
    total_series = total_series.reshape(-1, 1)

    dataset = LoadDataset(total_series, window_size=WINDOW_SIZE)

    train_len = int(len(dataset) * TRAIN_SPLIT)
    val_len = len(dataset) - train_len

    train_ds, val_ds = random_split(dataset, [train_len, val_len])

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)

    # -----------------------------
    # Model
    # -----------------------------
    model = LoadTCN(
        input_channels=1,
        channel_sizes=[16, 16, 32],
        kernel_size=3,
        dropout=0.2
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # -----------------------------
    # Training Loop
    # -----------------------------
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

        # Validation
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
            f"Train MSE: {train_loss:.6f} | Val MSE: {val_loss:.6f}"
        )

    torch.save(model.state_dict(), "tcn/models/load_tcn_total.pt")
    print("LoadTCN (Total) model saved.")


if __name__ == "__main__":
    main()