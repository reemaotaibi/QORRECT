import pennylane as qml
import numpy as np

print("🔄 Initializing devices...")
dev_ideal = qml.device("lightning.qubit", wires=4)
dev_noisy = qml.device("default.mixed", wires=4)

@qml.qnode(dev_ideal)
def ideal_circuit(params):
    qml.Hadamard(wires=0)
    qml.CNOT(wires=[0, 1])
    qml.RZ(params[0], wires=1)
    qml.RY(params[1], wires=2)
    qml.CNOT(wires=[2, 3])
    return qml.probs(wires=range(4))

@qml.qnode(dev_noisy)
def noisy_circuit(params, noise_level=0.01):
    qml.Hadamard(wires=0)
    qml.DepolarizingChannel(noise_level, wires=0)
    qml.CNOT(wires=[0, 1])
    qml.DepolarizingChannel(noise_level, wires=1)
    qml.RZ(params[0], wires=1)
    qml.DepolarizingChannel(noise_level, wires=1)
    qml.RY(params[1], wires=2)
    qml.DepolarizingChannel(noise_level, wires=2)
    qml.CNOT(wires=[2, 3])
    qml.DepolarizingChannel(noise_level, wires=3)
    return qml.probs(wires=range(4))

# Generate dataset
NUM_SAMPLES = 5000
print(f"⚙️  Generating {NUM_SAMPLES} samples...")

noisy_data = []
ideal_data = []
noise_levels = []

for i in range(NUM_SAMPLES):
    # Random circuit parameters
    params = np.random.uniform(0, 2 * np.pi, size=2)
    # Random noise level between 1% and 15%
    noise = np.random.uniform(0.01, 0.15)

    ideal = ideal_circuit(params)
    noisy = noisy_circuit(params, noise_level=noise)

    noisy_data.append(noisy)
    ideal_data.append(ideal)
    noise_levels.append(noise)

    if (i + 1) % 500 == 0:
        print(f"  ✅ {i + 1}/{NUM_SAMPLES} samples done...")

# Save to file
np.save("noisy_data.npy", np.array(noisy_data))
np.save("ideal_data.npy", np.array(ideal_data))
np.save("noise_levels.npy", np.array(noise_levels))

print("🎉 Dataset saved!")
print(f"  noisy_data.npy  → shape: {np.array(noisy_data).shape}")
print(f"  ideal_data.npy  → shape: {np.array(ideal_data).shape}")
print(f"  noise_levels.npy → shape: {np.array(noise_levels).shape}")