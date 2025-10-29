import matplotlib.pyplot as plt 
import seaborn as sns
import pandas as pd
import numpy as np
import os

OUTPUT_DIR = 'results'

def ensure_output_dir():
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

# Para ver que tan similares son las neuronas
def plot_unified_matrix(u_matrix):
    ensure_output_dir()
    plt.figure(figsize=(8, 7))
    sns.heatmap(u_matrix, cmap='bone_r', annot=True, fmt='.2f', cbar = True,
                cbar_kws={'label': 'Distancia Promedio a Vecinos'})
    plt.title('Matriz U')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.savefig(os.path.join(OUTPUT_DIR, 'unified_matrix.png'), bbox_inches='tight')
    plt.close()

# Cuantos paises cayeron por neurona
def plot_activation_map(activation_map):
    ensure_output_dir()
    plt.figure(figsize=(8, 7))
    sns.heatmap(activation_map, cmap='viridis', annot=True, fmt='.0f', cbar=True,
                cbar_kws={'label': 'Número de Países Asignados'})
    plt.title('Mapa de Activacion')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.savefig(os.path.join(OUTPUT_DIR, 'activation_map.png'), bbox_inches='tight')
    plt.close()

# Paises que mejor matchean con una neurona
def plot_kohonen_org_map_results(k_org_map, mapped_data, labels):
    ensure_output_dir()
    k = k_org_map.k
    activation_map = k_org_map.get_activation_map(mapped_data)
    
    label_map = [[[] for _ in range(k)] for _ in range(k)]
    for i, (row, col) in enumerate(k_org_map.map_data(mapped_data)):
        label_map[row][col].append(labels[i])
        
    plt.figure(figsize=(k*2, k*2))
    sns.heatmap(activation_map, cmap='Pastel2', annot=False, cbar=False, square=True, linewidths=1.0, linecolor='black')
    
    for i in range(k):
        for j in range(k):
            countries = label_map[i][j]
            if countries:
                s = "\n".join(countries)
                plt.text(j + 0.5, i + 0.5, s, ha='center', va='center', fontsize=9)
                
    plt.title('Asociación de Países en Kohonen')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.savefig(os.path.join(OUTPUT_DIR, 'country_clustering.png'), bbox_inches='tight')
    plt.close()

# Indice de desarrollo socio economico
def plot_pc1_interpretation(loadings, feature_names):
    ensure_output_dir()
    plt.figure(figsize=(10, 6))
    sns.barplot(x=loadings, y=feature_names, palette='coolwarm')
    plt.title('Interpretación de la Primera Componente Principal (PC1) - Regla de Oja')
    plt.xlabel('Carga en la Componente')
    plt.ylabel('Variable Original')
    plt.axvline(0, color='black', linestyle='--', label='Carga Nula')
    plt.savefig(os.path.join(OUTPUT_DIR, 'oja_pc1_interpretation.png'), bbox_inches='tight')
    plt.close()

def plot_pc1_comparison(loadings_oja, loadings_sklearn, feature_names):
    ensure_output_dir()
    
    df_oja = pd.DataFrame({'Variable': feature_names, 'Carga': loadings_oja, 'Método': 'Regla de Oja'})
    df_sklearn = pd.DataFrame({'Variable': feature_names, 'Carga': loadings_sklearn, 'Método': 'PCA (Sklearn)'})
    
    df_combined = pd.concat([df_oja, df_sklearn])
    
    plt.figure(figsize=(12, 7))
    
    sns.barplot(x='Carga', y='Variable', hue='Método', data=df_combined, palette='tab10')
    
    plt.title('Comparación de Cargas de PC1: Oja vs. Sklearn')
    plt.xlabel('Carga en la Componente Principal 1 (PC1)')
    plt.ylabel('Variable Original')
    plt.axvline(0, color='black', linestyle='--')
    plt.legend(title='Algoritmo')
    
    plt.savefig(os.path.join(OUTPUT_DIR, 'oja_sklearn_pc1_comparison.png'), bbox_inches='tight')
    plt.close()

def plot_oja_convergence(convergence_history):
    ensure_output_dir()
    plt.figure(figsize=(8, 5))
    plt.plot(convergence_history, label='Cambio del vector de pesos (norma)', linewidth=2)
    plt.title('Convergencia de la Regla de Oja')
    plt.xlabel('Época')
    plt.ylabel('Norma de Δw')
    plt.yscale('log')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.savefig(os.path.join(OUTPUT_DIR, 'oja_convergence.png'), bbox_inches='tight')
    plt.close()

# Como se estabiliza el output
def plot_oja_output_evolution(y_history):
    ensure_output_dir()
    plt.figure(figsize=(8, 5))
    plt.plot(y_history, linewidth=2, color='tab:blue')
    plt.title('Evolución del valor medio de salida (Oja)')
    plt.xlabel('Iteración')
    plt.ylabel('Valor promedio de y')
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.savefig(os.path.join(OUTPUT_DIR, 'oja_output_evolution.png'), bbox_inches='tight')
    plt.close()

# Distribucion de datos segun PC1
def plot_oja_data_projection(data, w, feature_names):
    ensure_output_dir()
    y_proj = np.dot(data, w)
    plt.figure(figsize=(8, 5))
    sns.histplot(y_proj, kde=True, bins=20, color='tab:orange')
    plt.title('Distribución de las proyecciones sobre PC1 (Oja)')
    plt.xlabel('Proyección sobre w (PC1)')
    plt.ylabel('Frecuencia')
    plt.savefig(os.path.join(OUTPUT_DIR, 'oja_data_projection.png'), bbox_inches='tight')
    plt.close()

def plot_kohonen_k3_results(kohonen_network, data, country_labels):
    """
    Create the specific k=3 visualization with country assignments and U-matrix
    matching the style shown in the presentation slide
    """
    ensure_output_dir()
    
    k = kohonen_network.k
    assert k == 3, "This function is specifically for k=3 networks"
    
    # Country name to abbreviation mapping
    country_abbrev = {
        'Iceland': 'ICE', 'Ireland': 'IRE', 'Luxembourg': 'LUX', 'Austria': 'AUS',
        'Denmark': 'DEN', 'Belgium': 'BEL', 'Switzerland': 'SWI', 'Netherlands': 'NET',
        'Germany': 'GER', 'Sweden': 'SWE', 'Norway': 'NOR', 'Italy': 'ITA',
        'Finland': 'FIN', 'Slovenia': 'SLO', 'Czech Republic': 'CZE', 'Slovakia': 'SKO',
        'Lithuania': 'LIT', 'Latvia': 'LAT', 'Estonia': 'EST', 'United Kingdom': 'UNI',
        'Spain': 'SPA', 'Bulgaria': 'BUL', 'Ukraine': 'UKR', 'Romania': 'ROM',
        'Serbia': 'SER', 'Poland': 'POL', 'Croatia': 'CRO', 'Hungary': 'HUN',
        'Portugal': 'POR', 'Greece': 'GRE', 'France': 'FRA'
    }
    
    # Get country assignments per neuron
    activation_map = kohonen_network.get_activation_map(data)
    
    # Create country assignment map with scores
    label_map = [[[] for _ in range(k)] for _ in range(k)]
    country_mappings = kohonen_network.map_data(data)
    
    for i, (row, col) in enumerate(country_mappings):
        country_name = country_labels[i]
        # Calculate a score based on inverse distance (closer = higher score)
        winning_neuron = kohonen_network.weights[row, col]
        distance = np.linalg.norm(data[i] - winning_neuron)
        # Use a better scoring system: normalize distance and convert to 1-100 scale
        max_possible_distance = np.sqrt(len(data[i]))  # Max possible distance in standardized space
        normalized_distance = distance / max_possible_distance
        score = max(1, int(100 * (1 - normalized_distance)))  # Closer = higher score
        
        abbrev = country_abbrev.get(country_name, country_name[:3].upper())
        label_map[row][col].append(f"{abbrev}: {score}")
    
    # Get U-matrix
    u_matrix = kohonen_network.get_unified_matrix()
    
    # Create the visualization
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
    
    # Left plot: Country assignments with activation counts
    # Create a background showing activation counts
    im1 = ax1.imshow(activation_map, cmap='Greens', alpha=0.7)
    ax1.set_title('Cantidad de Países por Neurona en el tiempo', fontsize=14, pad=20)
    
    # Add country labels to each cell
    for i in range(k):
        for j in range(k):
            countries = label_map[i][j]
            count = len(countries)
            
            # Add activation count in corner
            ax1.text(j + 0.85, i + 0.15, f'{count}', 
                    ha='center', va='center', fontsize=12, fontweight='bold')
            
            if countries:
                # Create country text with abbreviations and scores
                country_text = '\n'.join(countries)
                ax1.text(j, i, country_text, 
                        ha='center', va='center', fontsize=8, color='black', fontweight='bold')
    
    # Style left plot - keep coordinate system but remove visual elements
    ax1.set_xlim(-0.5, k-0.5)
    ax1.set_ylim(-0.5, k-0.5)
    ax1.set_xticks([])
    ax1.set_yticks([])
    ax1.set_xticklabels([])
    ax1.set_yticklabels([])
    # Don't use axis('off') to maintain proper coordinate system
    
    # Add colorbar for left plot
    cbar1 = plt.colorbar(im1, ax=ax1, fraction=0.046, pad=0.04)
    cbar1.set_label('Cantidad de países', rotation=270, labelpad=15)
    
    # Right plot: U-Matrix (distances between neurons)
    im2 = ax2.imshow(u_matrix, cmap='RdYlBu_r', alpha=0.9)
    ax2.set_title('Distancias promedio entre neuronas vecinas (U-Matrix)', fontsize=14, pad=20)
    
    # Add country labels on U-matrix as well
    for i in range(k):
        for j in range(k):
            countries = label_map[i][j]
            if countries:
                # Show only first few countries to avoid overcrowding
                display_countries = countries[:3]  # Show max 3 countries
                if len(countries) > 3:
                    display_countries.append(f"... +{len(countries)-3}")
                country_text = '\n'.join(display_countries)
                ax2.text(j, i, country_text,
                        ha='center', va='center', fontsize=8,
                        color='black', fontweight='bold')
    
    # Style right plot - keep coordinate system but remove visual elements  
    ax2.set_xlim(-0.5, k-0.5)
    ax2.set_ylim(-0.5, k-0.5)
    ax2.set_xticks([])
    ax2.set_yticks([])
    ax2.set_xticklabels([])
    ax2.set_yticklabels([])
    # Don't use axis('off') to maintain proper coordinate system
    
    # Add colorbar for right plot
    cbar2 = plt.colorbar(im2, ax=ax2, fraction=0.046, pad=0.04)
    cbar2.set_label('Distancia promedio', rotation=270, labelpad=15)
    
    # Add main title
    fig.suptitle('Resultados para k=3', fontsize=18, fontweight='bold', y=0.95)
    
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'kohonen_k3_results.png'), 
                bbox_inches='tight', dpi=300)
    plt.show()
    plt.close()

def plot_kohonen_k_results(kohonen_network, data, country_labels, k_value=None):
    """
    Create visualization for any k value with country assignments and U-matrix
    Works for k=3, k=4, k=5, etc.
    """
    ensure_output_dir()
    
    k = kohonen_network.k
    if k_value is None:
        k_value = k
    
    # Country name to abbreviation mapping
    country_abbrev = {
        'Iceland': 'ICE', 'Ireland': 'IRE', 'Luxembourg': 'LUX', 'Austria': 'AUS',
        'Denmark': 'DEN', 'Belgium': 'BEL', 'Switzerland': 'SWI', 'Netherlands': 'NET',
        'Germany': 'GER', 'Sweden': 'SWE', 'Norway': 'NOR', 'Italy': 'ITA',
        'Finland': 'FIN', 'Slovenia': 'SLO', 'Czech Republic': 'CZE', 'Slovakia': 'SKO',
        'Lithuania': 'LIT', 'Latvia': 'LAT', 'Estonia': 'EST', 'United Kingdom': 'UNI',
        'Spain': 'SPA', 'Bulgaria': 'BUL', 'Ukraine': 'UKR', 'Romania': 'ROM',
        'Serbia': 'SER', 'Poland': 'POL', 'Croatia': 'CRO', 'Hungary': 'HUN',
        'Portugal': 'POR', 'Greece': 'GRE', 'France': 'FRA'
    }
    
    # Get country assignments per neuron
    activation_map = kohonen_network.get_activation_map(data)
    
    # Create country assignment map with scores
    label_map = [[[] for _ in range(k)] for _ in range(k)]
    country_mappings = kohonen_network.map_data(data)
    
    for i, (row, col) in enumerate(country_mappings):
        country_name = country_labels[i]
        # Calculate a score based on distance to winning neuron (higher = better match)
        winning_neuron = kohonen_network.weights[row, col]
        distance = np.linalg.norm(data[i] - winning_neuron)
        score = max(1, int(100 - distance * 50))  # Convert distance to score 1-100
        
        abbrev = country_abbrev.get(country_name, country_name[:3].upper())
        label_map[row][col].append(f"{abbrev}: {score}")
    
    # Get U-matrix
    u_matrix = kohonen_network.get_unified_matrix()
    
    # Adjust figure size based on k
    fig_width = max(16, k * 3)
    fig_height = max(8, k * 2)
    
    # Create the visualization
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(fig_width, fig_height))
    
    # Left plot: Country assignments with activation counts
    im1 = ax1.imshow(activation_map, cmap='Greens', alpha=0.7)
    ax1.set_title('Cantidad de Países por Neurona en el tiempo', fontsize=14, pad=20)
    
    # Adjust font sizes based on k
    count_fontsize = max(10, 16 - k)
    country_fontsize = max(8, 12 - k)
    
    # Add country labels to each cell
    for i in range(k):
        for j in range(k):
            countries = label_map[i][j]
            
            if countries:
                # Limit number of countries displayed based on k
                max_countries = max(3, 8 - k)
                display_countries = countries[:max_countries]
                if len(countries) > max_countries:
                    display_countries.append(f"... +{len(countries)-max_countries}")
                
                country_text = '\n'.join(display_countries)
                # Use exact integer coordinates for center
                ax1.text(j, i, country_text, 
                        ha='center', va='center', fontsize=country_fontsize, 
                        color='black', fontweight='bold')
    
    # Style left plot - keep coordinate system but remove visual elements
    ax1.set_xlim(-0.5, k-0.5)
    ax1.set_ylim(-0.5, k-0.5)
    ax1.set_xticks([])
    ax1.set_yticks([])
    ax1.set_xticklabels([])
    ax1.set_yticklabels([])
    # Don't use axis('off') to maintain proper coordinate system
    
    # Add colorbar for left plot
    cbar1 = plt.colorbar(im1, ax=ax1, fraction=0.046, pad=0.04)
    cbar1.set_label('Cantidad de países', rotation=270, labelpad=15)
    
    # Right plot: U-Matrix
    im2 = ax2.imshow(u_matrix, cmap='RdYlBu_r', alpha=0.9)
    ax2.set_title('Distancias promedio entre neuronas vecinas (U-Matrix)', fontsize=14, pad=20)
    
    # Add country labels on U-matrix as well
    for i in range(k):
        for j in range(k):
            countries = label_map[i][j]
            if countries:
                # Show fewer countries for larger k
                max_display = max(1, 4 - k)
                display_countries = countries[:max_display]
                if len(countries) > max_display:
                    display_countries.append(f"... +{len(countries)-max_display}")
                country_text = '\n'.join(display_countries)
                # Use exact integer coordinates for center
                ax2.text(j, i, country_text,
                        ha='center', va='center', fontsize=max(8, country_fontsize),
                        color='black', fontweight='bold')
    
    # Style right plot - keep coordinate system but remove visual elements  
    ax2.set_xlim(-0.5, k-0.5)
    ax2.set_ylim(-0.5, k-0.5)
    ax2.set_xticks([])
    ax2.set_yticks([])
    ax2.set_xticklabels([])
    ax2.set_yticklabels([])
    # Don't use axis('off') to maintain proper coordinate system
    
    # Add colorbar for right plot
    cbar2 = plt.colorbar(im2, ax=ax2, fraction=0.046, pad=0.04)
    cbar2.set_label('Distancia promedio', rotation=270, labelpad=15)
    
    # Add main title
    fig.suptitle(f'Resultados para k={k_value}', fontsize=18, fontweight='bold', y=0.95)
    
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, f'kohonen_k{k_value}_results.png'), 
                bbox_inches='tight', dpi=300)
    plt.show()
    plt.close()

def plot_kohonen_comparison(data, country_labels, feature_names, k_values=[3, 4, 5]):
    """
    Create and compare Kohonen networks for different k values
    """
    ensure_output_dir()
    
    print(f"🔄 Generando comparación de redes Kohonen para k={k_values}")
    
    for k in k_values:
        print(f"📊 Entrenando red de Kohonen k={k}...")
        
        # Import here to avoid circular imports
        from models.kohonen import Kohonen
        
        # Create and train network
        koh = Kohonen(grid_size_k=k, input_dim=data.shape[1])
        epochs = 500 * data.shape[1]  # Same heuristic as main
        koh.train(data, epochs=epochs)
        
        # Generate visualization
        plot_kohonen_k_results(koh, data, country_labels, k_value=k)
        
        # Also generate basic organization map
        plot_kohonen_org_map_results(koh, data, country_labels)
        
        print(f"✅ Guardado: kohonen_k{k}_results.png")
    
    print(f"🎯 Comparación completa para k={k_values}")

# ================= NEW HEATMAP VISUALIZATIONS =================

def plot_feature_heatmaps(kohonen_network, data, feature_names, country_labels):
    """
    Create individual heatmaps for each economic feature showing 
    how traits are distributed across SOM neurons
    """
    ensure_output_dir()
    
    k = kohonen_network.k
    
    # Create output directory for feature heatmaps
    heatmap_dir = os.path.join(OUTPUT_DIR, 'feature_heatmaps')
    if not os.path.exists(heatmap_dir):
        os.makedirs(heatmap_dir)
    
    # Map each country to its neuron
    country_mappings = kohonen_network.map_data(data)
    
    # For each feature, create a heatmap
    for feature_idx, feature_name in enumerate(feature_names):
        feature_map = np.zeros((k, k))
        feature_counts = np.zeros((k, k))
        
        # Aggregate feature values for each neuron
        for country_idx, (row, col) in enumerate(country_mappings):
            feature_value = data[country_idx, feature_idx]
            feature_map[row, col] += feature_value
            feature_counts[row, col] += 1
        
        # Calculate average feature value per neuron
        # Avoid division by zero
        avg_feature_map = np.divide(feature_map, feature_counts, 
                                  out=np.zeros_like(feature_map), 
                                  where=feature_counts!=0)
        
        # Create the heatmap
        plt.figure(figsize=(10, 8))
        
        # Use diverging colormap centered at 0 (since data is standardized)
        vmax = max(abs(avg_feature_map.min()), abs(avg_feature_map.max()))
        
        ax = sns.heatmap(avg_feature_map, 
                        cmap='RdBu_r',  # Red-Blue diverging
                        center=0,
                        vmin=-vmax, 
                        vmax=vmax,
                        annot=True, 
                        fmt='.2f',
                        square=True,
                        linewidths=0.5,
                        cbar_kws={'label': f'{feature_name} (Standardized)'})
        
        plt.title(f'Distribución de {feature_name} en el Mapa de Kohonen\n'
                 f'(Rojo = Alto, Azul = Bajo)', fontsize=14, pad=20)
        plt.xlabel('Coordenada X del Neurón')
        plt.ylabel('Coordenada Y del Neurón')
        
        # Add text annotations for empty neurons
        for i in range(k):
            for j in range(k):
                if feature_counts[i, j] == 0:
                    ax.text(j + 0.5, i + 0.5, 'Sin\nPaíses', 
                           ha='center', va='center', 
                           fontsize=8, color='gray', alpha=0.7)
        
        plt.tight_layout()
        safe_name = feature_name.replace('.', '_').replace(' ', '_')
        plt.savefig(os.path.join(heatmap_dir, f'{safe_name}_heatmap.png'), 
                   bbox_inches='tight', dpi=300)
        plt.close()

def plot_combined_feature_heatmap(kohonen_network, data, feature_names, country_labels):
    """
    Create a comprehensive dashboard showing all features in subplots
    """
    ensure_output_dir()
    
    k = kohonen_network.k
    num_features = len(feature_names)
    
    # Calculate grid dimensions for subplots
    cols = 3
    rows = (num_features + cols - 1) // cols
    
    fig, axes = plt.subplots(rows, cols, figsize=(5*cols, 4*rows))
    fig.suptitle('Panel de Características Económicas - Mapa de Kohonen', 
                fontsize=16, y=0.98)
    
    # Flatten axes for easier indexing
    if rows == 1:
        axes = axes.reshape(1, -1)
    axes_flat = axes.flatten()
    
    # Map countries to neurons
    country_mappings = kohonen_network.map_data(data)
    
    for feature_idx, feature_name in enumerate(feature_names):
        ax = axes_flat[feature_idx]
        
        # Calculate feature distribution per neuron
        feature_map = np.zeros((k, k))
        feature_counts = np.zeros((k, k))
        
        for country_idx, (row, col) in enumerate(country_mappings):
            feature_value = data[country_idx, feature_idx]
            feature_map[row, col] += feature_value
            feature_counts[row, col] += 1
        
        # Average values
        avg_feature_map = np.divide(feature_map, feature_counts, 
                                  out=np.zeros_like(feature_map), 
                                  where=feature_counts!=0)
        
        # Create heatmap
        vmax = max(abs(avg_feature_map.min()), abs(avg_feature_map.max()))
        
        sns.heatmap(avg_feature_map, 
                   cmap='RdBu_r',
                   center=0,
                   vmin=-vmax, 
                   vmax=vmax,
                   annot=True, 
                   fmt='.1f',
                   square=True,
                   cbar=True,
                   ax=ax,
                   linewidths=0.3)
        
        ax.set_title(feature_name, fontsize=12, pad=10)
        ax.set_xlabel('')
        ax.set_ylabel('')
        ax.tick_params(labelsize=8)
    
    # Hide unused subplots
    for idx in range(num_features, len(axes_flat)):
        axes_flat[idx].set_visible(False)
    
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'combined_feature_heatmap.png'), 
               bbox_inches='tight', dpi=300)
    plt.close()



def plot_neuron_specialization_heatmap(kohonen_network, data, feature_names, country_labels):
    """
    Show which neurons specialize in which economic profiles
    """
    ensure_output_dir()
    
    k = kohonen_network.k
    country_mappings = kohonen_network.map_data(data)
    
    # Create specialization matrix: neurons vs features
    specialization_matrix = np.zeros((k*k, len(feature_names)))
    neuron_labels = []
    
    # Calculate average feature values for each neuron
    for neuron_idx in range(k*k):
        row = neuron_idx // k
        col = neuron_idx % k
        neuron_labels.append(f'N({row},{col})')
        
        # Find countries assigned to this neuron
        countries_in_neuron = [i for i, (r, c) in enumerate(country_mappings) 
                              if r == row and c == col]
        
        if countries_in_neuron:
            # Average feature values for countries in this neuron
            neuron_features = np.mean(data[countries_in_neuron], axis=0)
            specialization_matrix[neuron_idx] = neuron_features
    
    # Create the heatmap
    plt.figure(figsize=(12, 16))
    
    # Filter out empty neurons for cleaner visualization
    non_empty_neurons = np.any(specialization_matrix != 0, axis=1)
    filtered_matrix = specialization_matrix[non_empty_neurons]
    filtered_labels = [neuron_labels[i] for i in range(len(neuron_labels)) if non_empty_neurons[i]]
    
    ax = sns.heatmap(filtered_matrix,
                    xticklabels=feature_names,
                    yticklabels=filtered_labels,
                    cmap='RdBu_r',
                    center=0,
                    annot=True,
                    fmt='.2f',
                    linewidths=0.5,
                    cbar_kws={'label': 'Valor Promedio de Característica (Estandarizado)'})
    
    plt.title('Especialización de Neuronas por Características Económicas\n'
             '(Rojo = Alto, Azul = Bajo)', fontsize=14, pad=20)
    plt.xlabel('Características Económicas')
    plt.ylabel('Neuronas del Mapa de Kohonen')
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'neuron_specialization_heatmap.png'), 
               bbox_inches='tight', dpi=300)
    plt.close()