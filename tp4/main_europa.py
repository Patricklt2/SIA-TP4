import pandas as pd
import numpy as np
from sklearn.decomposition import PCA

from models.kohonen import Kohonen
from models.oja_rule import Oja
from utils.preprocessing import load_and_std_europa
from utils.graphs import (
    plot_unified_matrix, 
    plot_activation_map, 
    plot_kohonen_org_map_results, 
    plot_pc1_interpretation,
    plot_pc1_comparison,
    plot_oja_convergence,
    plot_oja_output_evolution,
    plot_oja_data_projection,
)

def ejercicio_1_kohonen(scaled_data, country_labels):
    input_dim = scaled_data.shape[1]
    k_grid = 5
    epochs = 500 * input_dim
    
    koh = Kohonen(grid_size_k=k_grid, input_dim=input_dim)
    koh.train(scaled_data, epochs=epochs)

    plot_kohonen_org_map_results(koh, scaled_data, country_labels)
    
    u_matrix = koh.get_unified_matrix()
    plot_unified_matrix(u_matrix)
    
    activation_map = koh.get_activation_map(scaled_data)
    plot_activation_map(activation_map)

def ejercicio_1_oja(scaled_data, feature_names):
    input_dim = scaled_data.shape[1]
    epochs = 100
    learning_rate = 0.001
    
    oja_net = Oja(input_dim=input_dim, learning_rate=learning_rate)
    oja_net.train(scaled_data, epochs=epochs)
    
    plot_oja_convergence(oja_net.convergence_history)
    plot_oja_output_evolution(oja_net.y_mean_history)
    plot_oja_data_projection(scaled_data, oja_net.w, feature_names)
    
    pc1_loadings_oja = oja_net.get_pc1_loads()
        
    pca = PCA(n_components=1)
    pca.fit(scaled_data)
    pc1_loadings_sklearn = pca.components_[0]
    
    if np.sign(pc1_loadings_oja[0]) != np.sign(pc1_loadings_sklearn[0]):
        pc1_loadings_oja = -pc1_loadings_oja
    
    pc1_loadings_sklearn_aligned = pc1_loadings_sklearn
    if np.sign(pc1_loadings_oja[0]) != np.sign(pc1_loadings_sklearn_aligned[0]):
        pc1_loadings_sklearn_aligned = -pc1_loadings_sklearn_aligned
        
    diff = np.linalg.norm(pc1_loadings_oja - pc1_loadings_sklearn)



    plot_pc1_interpretation(pc1_loadings_oja, feature_names) 
    
    plot_pc1_comparison(pc1_loadings_oja, pc1_loadings_sklearn_aligned, feature_names)
    
    return diff


if __name__ == "__main__":
    DATA_FILE_PATH = 'data/europe.csv'
    
    try:
        scaled_data, country_labels, feature_names = load_and_std_europa(DATA_FILE_PATH)
        
        ejercicio_1_kohonen(scaled_data, country_labels)

        ejercicio_1_oja(scaled_data, feature_names)
        
    except FileNotFoundError:
        print(f"File not found :)")