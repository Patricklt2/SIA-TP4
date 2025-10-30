import os
import numpy as np
import matplotlib.pyplot as plt
from models.hopfield import Hopfield
from data.letters import A, B, C, J, D, X, K, M, L, I, S, U, N, O

OUTPUT_DIR = 'results/hopfield'
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)


PATTERNS = [A, B, C, J]
PATTERN_LABELS = ['A', 'B', 'C', 'J']
PATTERN_SIZE = A.size
PATTERN_SHAPE = A.shape

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

def plot_evolution_grid(evolution, energy_values=None, pattern_shape=(5, 5),
                       title="Recall Evolution", outpath=None, max_steps=12):
    """
    Plot evolution as a grid of patterns with energy value shown for each step.
    """
    n_steps = min(len(evolution), max_steps)
    n_cols = min(6, n_steps)
    n_rows = (n_steps + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(2*n_cols, 2*n_rows))

    # Asegurar estructura 2D
    if n_rows == 1 and n_cols == 1:
        axes = np.array([[axes]])
    elif n_rows == 1:
        axes = axes.reshape(1, -1)
    elif n_cols == 1:
        axes = axes.reshape(-1, 1)

    for i in range(n_steps):
        row = i // n_cols
        col = i % n_cols
        ax = axes[row, col]
        state_2d = evolution[i].reshape(pattern_shape)
        ax.imshow(state_2d, cmap='gray_r', vmin=-1, vmax=1)
        ax.axis('off')

        # título: Step + energía
        if energy_values is not None and i < len(energy_values):
            e_val = energy_values[i]
            ax.set_title(f"Step {i}\nE={e_val:.3f}", fontsize=8)
        else:
            ax.set_title(f"Step {i}", fontsize=8)

        # marcar convergencia
        if i > 0 and np.array_equal(evolution[i], evolution[i-1]):
            ax.text(0.5, -0.1, 'CONVERGED', transform=ax.transAxes,
                    ha='center', fontsize=7, color='red', weight='bold')

    # ocultar subplots vacíos
    total_slots = n_rows * n_cols
    for i in range(n_steps, total_slots):
        row = i // n_cols
        col = i % n_cols
        axes[row, col].axis('off')

    plt.suptitle(title, y=1.02, fontsize=12)
    plt.tight_layout()
    if outpath:
        plt.savefig(outpath, bbox_inches='tight', dpi=150)
    plt.close()


def flip_random_pixels(pattern, n_flips, rng=None):
    p = np.asarray(pattern).copy().ravel()
    if rng is None:
        rng = np.random.default_rng()

    n_flips = min(n_flips, p.size)  # no más de los que existen
    flip_indices = rng.choice(p.size, n_flips, replace=False)

    # invertir los valores seleccionados
    p[flip_indices] *= -1

    return p.reshape(pattern.shape)


def recall(pattern, hopfield_net, title = "Recall Evolution", outpath=None):
    evolution_A, energy_values_A = hopfield_net.recall(pattern, record_energy=True) 

    plot_evolution_grid(
        evolution_A, 
        energy_values_A,
        PATTERN_SHAPE,
        title=title,
        outpath=outpath
    )


def success_rate_vs_flips(success_counts, flips_list, n_trials=5000):
    """
    Grafica la tasa de éxito con intervalos de confianza 95% (sin texto arriba de los puntos)
    """
    success_rates = [count / n_trials for count in success_counts]
    
    # Calcular intervalo de confianza 95% (1.96 * error estándar)
    confidence_intervals = [1.96 * np.sqrt(p * (1 - p) / n_trials) for p in success_rates]
    
    rates_percent = [sr * 100 for sr in success_rates]
    ci_percent = [ci * 100 for ci in confidence_intervals]
    
    plt.figure(figsize=(10, 6))
    
    plt.errorbar(
        flips_list, rates_percent, yerr=ci_percent,
        marker='o', linestyle='-', color='blue',
        linewidth=1.5, markersize=7,
        capsize=6, capthick=1, elinewidth=1.2,
        alpha=0.9
    )
    
    plt.xlabel("Número de flips", fontsize=12)
    plt.ylabel("Tasa de éxito (%)", fontsize=12)
    plt.title(f"Recuperación de 'A' vs cantidad de flips ({n_trials} corridas)", fontsize=13)
    plt.xticks(flips_list)
    plt.ylim(0, 105)
    plt.grid(True, linestyle='--', alpha=0.6)
    
    outpath = os.path.join(OUTPUT_DIR, "success_rate_vs_flips_A.png")
    plt.savefig(outpath, bbox_inches='tight', dpi=150)
    plt.show()


def success_rate_vs_memory_size(success_counts, n_trials):
    sizes = np.arange(1, len(success_counts) + 1)  # 1..N patrones
    rates = np.array(success_counts, dtype=float) / n_trials  # proporciones
    stderr = np.sqrt(rates * (1 - rates) / n_trials)           # error estándar binomial
    z = 1.96
    ci95 = z * stderr * 100                                   # IC95 en %

    plt.figure(figsize=(8, 5))
    plt.errorbar(
        sizes, rates * 100, yerr=ci95,
        fmt='-o', color='tab:blue',
        capsize=5, elinewidth=1.5, capthick=1.5, linewidth=2, markersize=6
    )
    plt.xlabel("Cantidad de patrones almacenados", fontsize=12)
    plt.ylabel("Tasa de acierto de 'A' con 5 flips (%)", fontsize=12)
    plt.title(f"Recuperación vs tamaño de memoria (n={n_trials} corridas, IC 95%)", fontsize=13)
    plt.xticks(sizes)
    plt.ylim(-5, 105)
    plt.grid(True, linestyle='--', alpha=0.6)

    outpath = os.path.join(OUTPUT_DIR, "success_rate_vs_memory_size_A_5flips_CI95.png")
    plt.savefig(outpath, bbox_inches='tight', dpi=150)
    plt.show()

    return sizes, rates, ci95


def main():
    """Main function to run the Hopfield network simulation."""
    # --- Setup ---
    for label, pattern in zip(PATTERN_LABELS, PATTERNS):
        plot_pattern(
            pattern,
            title=f"Stored Pattern '{label}'",
            show=False,
            outpath=os.path.join(OUTPUT_DIR, f"stored_pattern_{label}.png")
        )
    np.random.seed(42)
    hopfield_net = Hopfield(size=PATTERN_SIZE)
    flat_patterns = [p.flatten() for p in PATTERNS]
    hopfield_net.train(flat_patterns)

    print("--- Part A: Training and Recall ---")
    print(f"Hopfield network created with {PATTERN_SIZE} neurons.")
    print(f"Trained with {len(PATTERNS)} patterns: {', '.join(PATTERN_LABELS)}.\n")

    recall(flat_patterns[0], hopfield_net, title="Recall Evolution for Pattern 'A'",
           outpath=os.path.join(OUTPUT_DIR, "recall_evolution_grid_A.png"))

    recall(D.flatten(), hopfield_net, title="Recall Evolution for Pattern 'D'",
           outpath=os.path.join(OUTPUT_DIR, "recall_evolution_grid_D.png"))
    
    recall(X.flatten(), hopfield_net, title="Recall Evolution for Pattern 'X'",
           outpath=os.path.join(OUTPUT_DIR, "recall_evolution_grid_X.png"))

    print("\n--- Part B: Noisy Pattern Recall ---")

    rng = np.random.default_rng(12345)  # para reproducibilidad
    n_flips_list = [2, 5, 8, 12]

    for n_flips in n_flips_list:
        noisy_A = flip_random_pixels(A, n_flips, rng=rng)

        recall(noisy_A.flatten(), hopfield_net,
               title=f"Recall Evolution for Noisy Pattern 'A' ({n_flips} Flips)",
               outpath=os.path.join(OUTPUT_DIR, f"recall_evolution_grid_noisy_A_{n_flips}_flips.png"))
        

    for n_flips in n_flips_list:
        noisy_A = flip_random_pixels(A, n_flips, rng=rng)

        recall(noisy_A.flatten(), hopfield_net,
               title=f"Recall Evolution for Noisy Pattern 'A' ({n_flips} Flips)",
               outpath=os.path.join(OUTPUT_DIR, f"recall_evolution_grid_noisy_A_{n_flips}_flips2.png"))
        
    n_trials = 1000
    success_count = []
    n_flips_list = range(13)
    for flips in n_flips_list:
        success = 0
        for i in range(n_trials):
            noisy_A = flip_random_pixels(A, flips, rng=rng)
            evolution, _ = hopfield_net.recall(noisy_A.flatten(), record_energy=True)
            final_state = evolution[-1].reshape(PATTERN_SHAPE)
            if np.array_equal(final_state, A):
                success += 1
        success_count.append(success)

    success_rate_vs_flips(success_count, list(n_flips_list), n_trials)


    success_count_size = []
    letters_train = [A, B, C, J, M, L, I, S, N, O, U, D, K]
    flat_letters = [p.flatten() for p in letters_train]

    for i in range(1, len(flat_letters) + 1):
        subset = flat_letters[:i]             
        hopfield_net = Hopfield(size=flat_letters[0].size)  
        hopfield_net.train(subset)            
        success = 0
        for i in range(n_trials):
            noisy_A = flip_random_pixels(A, 5, rng=rng)
            evolution, _ = hopfield_net.recall(noisy_A.flatten(), record_energy=True)
            final_state = evolution[-1].reshape(PATTERN_SHAPE)
            if np.array_equal(final_state, A):
                success += 1
        success_count_size.append(success)
    
    success_rate_vs_memory_size(success_count_size, n_trials)



if __name__ == "__main__":
    main()