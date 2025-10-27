import os
import numpy as np
import matplotlib.pyplot as plt
from models.hopfield import Hopfield

np.random.seed(42)

def train_hopfield(patterns):
    """
    Build weight matrix W using Hebb's rule (no self-connections).
    patterns: list/array of 1D bipolar vectors (values -1 or +1).
    Returns trained Hopfield instance.
    """
    size = patterns[0].size
    net = Hopfield(size=size)
    flat = [p.flatten() for p in patterns]
    net.train(flat)
    return net

def add_noise(pattern, noise_level, rng=None):
    """
    Flip a fraction of bits in a 1D bipolar pattern.
    noise_level: fraction in [0,1] of bits to flip (approx).
    Returns a new 1D array.
    """
    if rng is None:
        rng = np.random
    noisy_pattern = pattern.copy()
    num_flips = int(round(noise_level * pattern.size))
    if num_flips <= 0:
        return noisy_pattern
    flip_indices = rng.choice(pattern.size, num_flips, replace=False)
    noisy_pattern[flip_indices] *= -1
    return noisy_pattern

def update(pattern, net, max_steps=200, asynchronous=True, record_energy=False):
    """
    Run recall process. Returns evolution list (and energies if record_energy True).
    pattern: 1D bipolar vector.
    net: Hopfield instance (with trained weights).
    """
    if record_energy:
        evolution, energies = net.recall(pattern, max_steps=max_steps, asynchronous=asynchronous, record_energy=True)
        return evolution, energies
    else:
        evolution = net.recall(pattern, max_steps=max_steps, asynchronous=asynchronous, record_energy=False)
        return evolution

def plot_pattern(pattern, title=None, show=True, outpath=None):
    """
    Visualize 5x5 bipolar pattern. 1 -> black '*' style, -1 -> white background.
    pattern: 1D or 2D array with values -1/+1
    """
    p = np.asarray(pattern).reshape(int(np.sqrt(pattern.size)), int(np.sqrt(pattern.size)))
    plt.figure(figsize=(2,2))
    plt.imshow(p, cmap='gray_r', vmin=-1, vmax=1)  # gray_r so 1 is dark
    plt.axis('off')
    if title:
        plt.title(title, fontsize=10)
    if outpath:
        plt.savefig(outpath, bbox_inches='tight')
    if show:
        plt.show()
    else:
        plt.close()

def plot_energy(net, evolution, title=None, outpath=None):
    energies = [net.energy(s) for s in evolution]
    plt.figure(figsize=(4,2))
    plt.plot(energies, marker='o')
    plt.xlabel('Step'); plt.ylabel('Energy')
    if title: plt.title(title)
    if outpath: plt.savefig(outpath, bbox_inches='tight')
    plt.show()

# --- Configuration ---
OUTPUT_DIR = 'results/patterns'
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

# --- Pattern Definitions (Part A) ---
# Represent letters as 5x5 bipolar matrices (1 for pixel on, -1 for off)
A = np.array([
    [-1,  1,  1,  1, -1],
    [ 1, -1, -1, -1,  1],
    [ 1,  1,  1,  1,  1],
    [ 1, -1, -1, -1,  1],
    [ 1, -1, -1, -1,  1]
])

B = np.array([
    [ 1,  1,  1,  1, -1],
    [ 1, -1, -1, -1,  1],
    [ 1,  1,  1,  1, -1],
    [ 1, -1, -1, -1,  1],
    [ 1,  1,  1,  1, -1]
])

C = np.array([
    [-1,  1,  1,  1,  1],
    [ 1, -1, -1, -1, -1],
    [ 1, -1, -1, -1, -1],
    [ 1, -1, -1, -1, -1],
    [-1,  1,  1,  1,  1]
])

J = np.array([
    [ 1,  1,  1,  1,  1],
    [-1, -1, -1,  1, -1],
    [-1, -1, -1,  1, -1],
    [ 1, -1, -1,  1, -1],
    [-1,  1,  1, -1, -1]
])

PATTERNS = [A, B, C, J]
PATTERN_LABELS = ['A', 'B', 'C', 'J']
PATTERN_SIZE = A.size
PATTERN_SHAPE = A.shape

# --- Helper Functions ---

def add_noise(pattern, noise_level, rng=None):
    """
    Flip a fraction of bits in a 1D bipolar pattern.
    noise_level: fraction in [0,1] of bits to flip (approx).
    Returns a new 1D array.
    """
    if rng is None:
        rng = np.random
    noisy_pattern = pattern.copy()
    num_flips = int(round(noise_level * pattern.size))
    if num_flips <= 0:
        return noisy_pattern
    flip_indices = rng.choice(pattern.size, num_flips, replace=False)
    noisy_pattern[flip_indices] *= -1
    return noisy_pattern

def check_if_spurious(final_state, original_patterns):
    """Checks if a state is spurious by comparing it to all original patterns."""
    for p in original_patterns:
        if np.array_equal(final_state, p.flatten()):
            return False
    return True

def plot_pattern_evolution(evolution, title):
    """
    Visualize evolution; limit number of subplots to avoid oversized figures.
    If evolution is long, show first, a few intermediates, and last.
    """
    max_panels = 8
    num_steps = len(evolution)
    if num_steps > max_panels:
        # select indices: first, a few evenly spaced, last
        indices = [0] + list(np.linspace(1, num_steps-2, max_panels-2, dtype=int)) + [num_steps-1]
        panels = [evolution[i] for i in indices]
    else:
        indices = list(range(num_steps))
        panels = evolution

    fig, axes = plt.subplots(1, len(panels), figsize=(len(panels) * 2.5, 3))
    if len(panels) == 1:
        axes = [axes]
    fig.suptitle(title, fontsize=16)
    for i, (ax, pattern) in enumerate(zip(axes, panels)):
        step_idx = indices[i]
        ax.imshow(pattern.reshape(PATTERN_SHAPE), cmap='gray', vmin=-1, vmax=1)
        step_title = "Input" if step_idx == 0 else f"Step {step_idx}"
        if step_idx == num_steps - 1 and num_steps > 1:
            step_title = f"Final State ({num_steps-1} steps)"
        ax.set_title(step_title)
        ax.axis('off')

    plt.savefig(os.path.join(OUTPUT_DIR, f"{title.replace(' ', '_').lower()}.png"), bbox_inches='tight')
    plt.close()

def plot_energy(network, evolution, title):
    """
    Plots the energy function over iterations.
    :param network: The Hopfield network instance.
    :param evolution: A list of 1D patterns from the recall process.
    :param title: The title for the plot.
    """
    energy_values = [network.energy(p) for p in evolution]
    plt.figure(figsize=(8, 5))
    plt.plot(energy_values, marker='o', linestyle='-')
    plt.title(title)
    plt.xlabel("Iteration Step")
    plt.ylabel("Energy")
    plt.grid(True)
    plt.xticks(range(len(energy_values)))
    plt.savefig(os.path.join(OUTPUT_DIR, f"{title.replace(' ', '_').lower()}_energy.png"), bbox_inches='tight')
    plt.close()

def main():
    """Main function to run the Hopfield network simulation."""
    # --- Setup ---
    np.random.seed(42)
    hopfield_net = Hopfield(size=PATTERN_SIZE)
    flat_patterns = [p.flatten() for p in PATTERNS]
    hopfield_net.train(flat_patterns)

    print("--- Part A: Training and Recall ---")
    print(f"Hopfield network created with {PATTERN_SIZE} neurons.")
    print(f"Trained with {len(PATTERNS)} patterns: {', '.join(PATTERN_LABELS)}.\n")

    # --- Part A: Recall from noisy pattern ---
    noise_level_a = 0.20
    original_pattern_a = flat_patterns[0]
    noisy_a = add_noise(original_pattern_a, noise_level_a)
    
    print(f"Recalling pattern 'A' with {noise_level_a*100:.0f}% noise...")
    evolution_a = hopfield_net.recall(noisy_a, max_steps=200, asynchronous=True)
    # check whether any evolution step matched the original pattern (robust)
    recovered_step = next((i for i, s in enumerate(evolution_a) if np.array_equal(s, original_pattern_a)), None)
    if recovered_step is not None:
        print(f"Pattern 'A' recovered at step {recovered_step}. Plots saved to '{OUTPUT_DIR}'.\n")
    else:
        print("Failed to recover pattern 'A'.\n")

    plot_pattern_evolution(evolution_a, f"Recovery of 'A' ({noise_level_a*100:.0f}% Noise)")
    plot_energy(hopfield_net, evolution_a, f"Energy: 'A' Recovery ({noise_level_a*100:.0f}% Noise)")

    # --- Part B: High Noise Test ---
    print("--- Part B: High Noise Test ---")
    noise_level_b = 0.50
    noisy_b = add_noise(original_pattern_a, noise_level_b)
    
    print(f"Testing with a heavily noisy pattern ({noise_level_b*100:.0f}% noise)...")
    evolution_b = hopfield_net.recall(noisy_b, max_steps=200, asynchronous=True)
    final_b = evolution_b[-1]
    
    print(f"Network converged in {len(evolution_b) - 1} steps. Plots saved to '{OUTPUT_DIR}'.")
    
    is_spurious_b = True
    for i, p in enumerate(flat_patterns):
        if np.array_equal(final_b, p):
            print(f"The final state is identical to the stored pattern '{PATTERN_LABELS[i]}'.\n")
            is_spurious_b = False
            break
    if is_spurious_b:
        print("The final state is a SPURIOUS state.\n")

    plot_pattern_evolution(evolution_b, f"Convergence from High Noise ('A' + {noise_level_b*100:.0f}%)")
    plot_energy(hopfield_net, evolution_b, f"Energy: High Noise ('A' + {noise_level_b*100:.0f}%)")

    # --- Part C: Spurious State from Composite Pattern ---
    print("--- Part C: Spurious State from Composite Pattern ---")
    # Create a composite pattern by mixing A and C
    composite_pattern = (flat_patterns[0] + flat_patterns[2]) / 2
    # Binarize the composite pattern
    composite_pattern = np.where(composite_pattern >= 0, 1, -1)
    
    print("Using a composite of 'A' and 'C' as input...")
    evolution_c = hopfield_net.recall(composite_pattern, max_steps=200, asynchronous=True)
    final_c = evolution_c[-1]

    print(f"Network converged in {len(evolution_c) - 1} steps. Plots saved to '{OUTPUT_DIR}'.")
    if check_if_spurious(final_c, flat_patterns):
        print("The final state is a SPURIOUS state, different from all stored patterns.")
    else:
        for i, p in enumerate(flat_patterns):
            if np.array_equal(final_c, p):
                print(f"The final state converged to the stored pattern '{PATTERN_LABELS[i]}'.")
                break

    plot_pattern_evolution(evolution_c, "Convergence from Composite Pattern (A+C)")
    plot_energy(hopfield_net, evolution_c, "Energy: Composite Pattern (A+C)")

    print("\nSimulation finished.")


if __name__ == "__main__":
    main()
