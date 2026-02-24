import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader, random_split

from tcn.datasets.pv_dataset import PVDataset
from tcn.models.pv_tcn import PVTCN


# Configuration
WINDOW_SIZE = 30
BATCH_SIZE = 32
EPOCHS = 20
LEARNING_RATE = 1e-3
TRAIN_SPLIT = 0.8


def generate_dummy_pv(length=1000):
    """
    Synthetic PV data for first training.
    """
    t = torch.linspace(0, 8 * 3.1416, length)
    pv = torch.sin(t).clamp(min=0)
    noise = 0.05 * torch.randn_like(pv)
    return (pv + noise).clamp(min=0).numpy()


def main():
   
    # Dataset
    from tcn.data.pv_simulator import simulate_day

    days = 30
    all_days = []

    for _ in range(days):
        day = simulate_day(
            peak_kw=1.0,
            cloud_chance=0.05,
            cloud_intensity=0.6
        )
        all_days.append(day)

    pv_series = np.concatenate(all_days)

    dataset = PVDataset(pv_series, window_size=WINDOW_SIZE)

    train_len = int(len(dataset) * TRAIN_SPLIT)
    val_len = len(dataset) - train_len

    train_ds, val_ds = random_split(dataset, [train_len, val_len])

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)

    # Model
    model = PVTCN(
        input_channels=1,
        channel_sizes=[32, 32, 64],
        kernel_size=3,
        dropout=0.2
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Training setup
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

   
    # Training loop
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

       
        # Validation of model
        
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

    # saving model
    torch.save(model.state_dict(), "tcn/models/pv_tcn.pt")
    print("PVTCN model saved.")


if __name__ == "__main__":
    main()