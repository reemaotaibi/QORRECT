import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader, TensorDataset, random_split
from model import QorrectTransformer
import matplotlib.pyplot as plt

# ── Load dataset ──────────────────────────────────────────
noisy = np.load("noisy_data.npy").astype(np.float32)
ideal = np.load("ideal_data.npy").astype(np.float32)
noise_levels = np.load("noise_levels.npy").astype(np.float32)

# Combine noisy probs + noise level into one input vector (size 17)
noise_levels_col = noise_levels.reshape(-1, 1)
X = np.concatenate([noisy, noise_levels_col], axis=1)  # (5000, 17)
y = ideal                                                # (5000, 16)

X_tensor = torch.tensor(X)
y_tensor = torch.tensor(y)

dataset = TensorDataset(X_tensor, y_tensor)
train_size = int(0.85 * len(dataset))
val_size = len(dataset) - train_size
train_ds, val_ds = random_split(dataset, [train_size, val_size])

train_loader = DataLoader(train_ds, batch_size=64, shuffle=True)
val_loader   = DataLoader(val_ds,   batch_size=64)

print(f"✅ Train: {train_size} samples | Val: {val_size} samples")

# ── Model, optimizer, loss ────────────────────────────────
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"🖥️  Training on: {device}")

model = QorrectTransformer().to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=50)

# KL divergence — perfect for probability distributions
criterion = nn.KLDivLoss(reduction="batchmean")

# ── Training loop ─────────────────────────────────────────
EPOCHS = 50
train_losses, val_losses = [], []

print(f"\n🚀 Training QORRECT for {EPOCHS} epochs...\n")

for epoch in range(EPOCHS):
    # Train
    model.train()
    train_loss = 0
    for X_batch, y_batch in train_loader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)
        optimizer.zero_grad()
        pred = model(X_batch)
        # KLDivLoss expects log-probabilities as input
        loss = criterion(torch.log(pred + 1e-10), y_batch)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        train_loss += loss.item()

    # Validate
    model.eval()
    val_loss = 0
    with torch.no_grad():
        for X_batch, y_batch in val_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            pred = model(X_batch)
            loss = criterion(torch.log(pred + 1e-10), y_batch)
            val_loss += loss.item()

    train_loss /= len(train_loader)
    val_loss   /= len(val_loader)
    train_losses.append(train_loss)
    val_losses.append(val_loss)
    scheduler.step()

    if (epoch + 1) % 5 == 0:
        print(f"Epoch {epoch+1:02d}/{EPOCHS} | Train Loss: {train_loss:.6f} | Val Loss: {val_loss:.6f}")

# ── Save model ────────────────────────────────────────────
torch.save(model.state_dict(), "qorrect_model.pth")
print("\n💾 Model saved to qorrect_model.pth")

# ── Plot loss curve ───────────────────────────────────────
plt.figure(figsize=(10, 5))
plt.plot(train_losses, label="Train Loss")
plt.plot(val_losses,   label="Val Loss")
plt.xlabel("Epoch")
plt.ylabel("KL Divergence Loss")
plt.title("QORRECT Training")
plt.legend()
plt.grid(True)
plt.savefig("training_loss.png")
plt.show()
print("📊 Loss curve saved to training_loss.png")