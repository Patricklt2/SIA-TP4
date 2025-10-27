import numpy as np
import pytest

hp_mod = pytest.importorskip("models.hopfield")
Hopfield = hp_mod.Hopfield

def flip_bits(pattern, n_flips, rng):
    p = pattern.copy()
    idx = rng.choice(p.size, n_flips, replace=False)
    p[idx] *= -1
    return p

def test_hebb_weights_symmetry_and_no_self_connections():
    rng = np.random.RandomState(0)
    # small toy patterns (3 patterns of length 9)
    patterns = [rng.choice([-1, 1], size=9) for _ in range(3)]
    net = Hopfield(size=9)
    net.train(patterns)
    W = net.weights
    # symmetry
    assert W.shape == (9, 9)
    np.testing.assert_allclose(W, W.T, atol=1e-8)
    # zero diagonal
    assert np.allclose(np.diag(W), 0.0)

def test_recall_recovers_pattern_from_low_noise_and_energy_non_increasing():
    rng = np.random.RandomState(1)
    patterns = [rng.choice([-1, 1], size=25) for _ in range(4)]
    net = Hopfield(size=25)
    net.train(patterns)
    original = patterns[0]
    noisy = flip_bits(original, max(1, int(0.2 * original.size)), rng)
    evolution = net.recall(noisy, max_steps=200, asynchronous=True, record_energy=True)
    # recall may return (evolution, energies) when record_energy True
    if isinstance(evolution, tuple):
        evolution, energies = evolution
    else:
        energies = [net.energy(s) for s in evolution]
    # Check whether any evolution state matches the original pattern
    matched = any(np.array_equal(s, original) for s in evolution)
    assert matched, "Network failed to recover pattern from low noise in this trial"
    # Energy should be non-increasing for asynchronous updates (allow tiny numerical tolerance)
    for i in range(1, len(energies)):
        assert energies[i] <= energies[i-1] + 1e-6
