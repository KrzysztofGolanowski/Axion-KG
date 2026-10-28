import numpy as np


I = np.array([[1, 0], [0, 1]], dtype=complex)
X = np.array([[0, 1], [1, 0]], dtype=complex)
Z = np.array([[1, 0], [0, -1]], dtype=complex)
H = (1 / np.sqrt(2)) * np.array([[1, 1], [1, -1]], dtype=complex)

def Ry(theta):
    """"""
    return np.array([
        [np.cos(theta / 2), -np.sin(theta / 2)],
        [np.sin(theta / 2), np.cos(theta / 2)]
    ], dtype=complex)

CNOT = np.array([
    [1, 0, 0, 0],
    [0, 1, 0, 0],
    [0, 0, 0, 1],
    [0, 0, 1, 0]
], dtype=complex)


def measure(state):
    """Wykonuje pomiar komputacyjny - losuje stan bazowy."""
    probabilities = np.abs(state) ** 2
    probabilities /= np.sum(probabilities)
    outcome = np.random.choice(len(state), p=probabilities)
    collapsed_state = np.zeros_like(state)
    collapsed_state[outcome] = 1.0
    return outcome, collapsed_state


def main():
    print("Test 1: H * H | 0>")
    state_0 = np.array([1.0, 0.0], dtype=complex)
    state_h1 = np.dot(H, state_0)
    state_h2 = np.dot(H, state_h1)
    print("Stan po H i H:", state_h2)
    outcome_1, _ = measure(state_h2)
    print(f"Zmierzony stan: |{outcome_1}⟩\n")

    print("Test 2: Stan Bella")
    bell_state = np.array([1.0, 0.0, 0.0, 0.0], dtype=complex)

    H_tensor_I = np.kron(H, I)
    bell_state = np.dot(H_tensor_I, bell_state)

    bell_state = np.dot(CNOT, bell_state)

    print("Wynikowe prawdopodobieństwa dla |00>, |01>, |10>, |11>:",
          np.abs(bell_state) ** 2)

    results = []
    for _ in range(10):
        res, _ = measure(bell_state)
        results.append(f"{res:02b}")
    print(10, "losowych pomiarów stanu Bella:", results)

if __name__ == "__main__":
    main()