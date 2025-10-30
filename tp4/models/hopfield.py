import numpy as np
import matplotlib.pyplot as plt

class Hopfield:
    """
    Hopfield network with Hebbian training.
    - train(patterns): patterns are 1D bipolar vectors (-1, +1)
    - recall(pattern, asynchronous=True): run recall; default uses asynchronous random-sequential updates
    Notes:
    - Activation tie-break uses >= 0 -> +1 (deterministic). Change if randomized tie-breaking preferred.
    """
    def __init__(self, size):
        self.size = size
        self.weights = np.zeros((size, size))

    def train(self, patterns):
        """
        Hebb rule: W = (1/N) * sum_p p p^T, with zero diagonal (no self-connections).
        patterns: iterable of 1D bipolar arrays (values in {-1, +1}), each length == self.size
        """
        self.weights.fill(0.0)
        for p in patterns:
            p = np.asarray(p).reshape(self.size)
            self.weights += np.outer(p, p)
        # scale by number of neurons (standard)
        self.weights /= float(self.size)
        np.fill_diagonal(self.weights, 0.0)

    def recall(self, pattern, max_steps=200, record_energy=False):
        """
        Run recall from initial pattern.
        Returns list of states (each 1D length self.size). First element is the input.
        If record_energy True, returns (evolution, energy_values).
        """
        current = np.asarray(pattern).copy().reshape(self.size)
        evolution = [current.copy()]
        energy_values = [self.energy(current)] if record_energy else None

        for step in range(max_steps):
            for i in np.random.permutation(self.size):
                act = np.dot(self.weights[i, :], current)
                current[i] = 1 if act > 0 else -1
            evolution.append(current.copy())

            if record_energy:
                energy_values.append(self.energy(current))

            # convergence: no change after full update cycle
            if np.array_equal(evolution[-1], evolution[-2]):
                break

        if record_energy:
            return evolution, energy_values
        return evolution
    

    def energy(self, pattern):
        s = np.asarray(pattern).reshape(self.size)
        return -0.5 * float(s.T.dot(self.weights).dot(s))


