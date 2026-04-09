import pennylane as qml
import numpy as np

# Your first quantum circuit on QORRECT!
dev = qml.device("lightning.qubit", wires=4)

@qml.qnode(dev)
def ideal_circuit(params):
    qml.Hadamard(wires=0)
    qml.CNOT(wires=[0, 1])
    qml.RZ(params[0], wires=1)
    qml.RY(params[1], wires=2)
    qml.CNOT(wires=[2, 3])
    return qml.probs(wires=range(4))

params = np.array([0.5, 1.2])
result = ideal_circuit(params)
print("✅ Ideal circuit output:")
print(result)
print("✅ Sum of probabilities:", round(sum(result), 4))

# Noisy version of the same circuit
dev_noisy = qml.device("default.mixed", wires=4)

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

noisy_result = noisy_circuit(params, noise_level=0.05)
print("\n⚠️ Noisy circuit output:")
print(noisy_result)
print("📉 Difference (noise effect):")
print(np.abs(ideal_circuit(params) - noisy_result))