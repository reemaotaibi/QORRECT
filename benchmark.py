import torch
import numpy as np
from model import QorrectTransformer
import matplotlib.pyplot as plt

# ── Load model ─────────────────────────────────────────────
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = QorrectTransformer().to(device)
model.load_state_dict(torch.load("qorrect_model.pth", map_location=device))
model.eval()
print(f"✅ Model loaded on {device}")

# ── Load test data ─────────────────────────────────────────
noisy       = np.load("noisy_data.npy").astype(np.float32)
ideal       = np.load("ideal_data.npy").astype(np.float32)
noise_levels = np.load("noise_levels.npy").astype(np.float32)

# Use last 750 samples as test set (never seen during training)
noisy_test  = noisy[-750:]
ideal_test  = ideal[-750:]
noise_test  = noise_levels[-750:]

# ── Run QORRECT correction ─────────────────────────────────
noise_col = noise_test.reshape(-1, 1)
X_test = np.concatenate([noisy_test, noise_col], axis=1)
X_tensor = torch.tensor(X_test).to(device)

with torch.no_grad():
    corrected = model(X_tensor).cpu().numpy()

# ── Fidelity function ──────────────────────────────────────
def fidelity(p, q):
    # Quantum fidelity between two probability distributions
    return float(np.sum(np.sqrt(p * q)) ** 2)

# ── Benchmark ─────────────────────────────────────────────
fidelity_noisy     = []
fidelity_corrected = []

for i in range(len(ideal_test)):
    f_noisy = fidelity(noisy_test[i], ideal_test[i])
    f_corr  = fidelity(corrected[i],  ideal_test[i])
    fidelity_noisy.append(f_noisy)
    fidelity_corrected.append(f_corr)

fidelity_noisy     = np.array(fidelity_noisy)
fidelity_corrected = np.array(fidelity_corrected)
improvement        = fidelity_corrected - fidelity_noisy

print("\n📊 QORRECT Benchmark Results")
print("=" * 40)
print(f"Avg fidelity (noisy):     {fidelity_noisy.mean():.6f}")
print(f"Avg fidelity (QORRECT):   {fidelity_corrected.mean():.6f}")
print(f"Avg improvement:          +{improvement.mean():.6f}")
print(f"Improvement %:            {(improvement.mean() / (1 - fidelity_noisy.mean())) * 100:.2f}%")
print(f"Samples where QORRECT wins: {(improvement > 0).sum()}/{len(improvement)}")
print("=" * 40)

# ── Plot ───────────────────────────────────────────────────
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.hist(fidelity_noisy,     bins=30, alpha=0.6, label="Noisy",     color="red")
plt.hist(fidelity_corrected, bins=30, alpha=0.6, label="QORRECT",   color="green")
plt.xlabel("Fidelity")
plt.ylabel("Count")
plt.title("Fidelity Distribution")
plt.legend()
plt.grid(True)

plt.subplot(1, 2, 2)
plt.scatter(noise_test, fidelity_noisy,     alpha=0.3, s=10, label="Noisy",   color="red")
plt.scatter(noise_test, fidelity_corrected, alpha=0.3, s=10, label="QORRECT", color="green")
plt.xlabel("Noise Level")
plt.ylabel("Fidelity")
plt.title("Fidelity vs Noise Level")
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.savefig("benchmark.png")
plt.show()
print("\n📊 Benchmark plot saved to benchmark.png")