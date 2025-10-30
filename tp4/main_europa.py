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
    plot_country_projections,
    # New heatmap visualizations
    plot_feature_heatmaps,
    plot_combined_feature_heatmap,
    plot_neuron_specialization_heatmap,
    # K comparison functions
    plot_kohonen_k_results,
    plot_kohonen_comparison,
    plot_kohonen_k3_results,
)

def ejercicio_1_kohonen(scaled_data, country_labels, feature_names):
    input_dim = scaled_data.shape[1]
    k_grid = 5
    epochs = 500 * input_dim
    
    print("🗺️ Entrenando red de Kohonen...")
    koh = Kohonen(grid_size_k=k_grid, input_dim=input_dim)
    koh.train(scaled_data, epochs=epochs)

    print("📊 Generando visualizaciones básicas...")
    plot_kohonen_org_map_results(koh, scaled_data, country_labels)
    
    u_matrix = koh.get_unified_matrix()
    plot_unified_matrix(u_matrix)
    
    activation_map = koh.get_activation_map(scaled_data)
    plot_activation_map(activation_map)
    
    print("🔥 Generando mapas de calor de características...")
    # NEW: Individual feature heatmaps
    plot_feature_heatmaps(koh, scaled_data, feature_names, country_labels)
    
    print("📋 Generando panel combinado de características...")
    # NEW: Combined feature dashboard
    plot_combined_feature_heatmap(koh, scaled_data, feature_names, country_labels)
    
    print("🧠 Generando mapa de especialización neuronal...")
    # NEW: Neuron specialization analysis
    plot_neuron_specialization_heatmap(koh, scaled_data, feature_names, country_labels)
    
    print("✅ Visualizaciones de Kohonen completadas!")
    return koh

def ejercicio_kohonen_k_comparison(scaled_data, country_labels, feature_names):
    """
    Compare Kohonen networks with different k values (k=3, 4, 5)
    for presentation analysis
    """
    print("\n" + "=" * 60)
    print("📊 COMPARACIÓN DE REDES KOHONEN (k=3, 4, 5)")
    print("=" * 60)
    
    k_values = [3, 4, 5]
    networks = {}
    
    for k in k_values:
        print(f"\n🗺️ Entrenando red de Kohonen k={k}...")
        
        # Create and train network
        koh = Kohonen(grid_size_k=k, input_dim=scaled_data.shape[1])
        epochs = 500 * scaled_data.shape[1]
        koh.train(scaled_data, epochs=epochs)
        
        # Store network for analysis
        networks[k] = koh
        
        # Generate k-specific visualization
        print(f"📈 Generando visualización k={k}...")
        plot_kohonen_k_results(koh, scaled_data, country_labels, k_value=k)
        
        # Basic stats
        activation_map = koh.get_activation_map(scaled_data)
        dead_neurons = np.sum(activation_map == 0)
        print(f"   📍 Neuronas muertas: {dead_neurons}/{k*k}")
        print(f"   📊 Países por neurona promedio: {len(country_labels)/(k*k-dead_neurons):.1f}")
    
    print(f"\n✅ Comparación k={k_values} completada!")
    print(f"🗂️ Resultados guardados en: results/kohonen_k*_results.png")
    
    return networks

def ejercicio_1_oja(scaled_data, feature_names, country_names):
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
    
    projections = np.dot(scaled_data, pc1_loadings_oja) 
    
    projections_df = pd.DataFrame({
        'País': country_names,
        'Proyección_PC1_Oja': projections
    }).sort_values(by='Proyección_PC1_Oja', ascending=False).reset_index(drop=True)

    plot_oja_data_projection(scaled_data, pc1_loadings_oja, feature_names)

    return diff, projections_df


if __name__ == "__main__":
    DATA_FILE_PATH = 'data/europe.csv'
    
    try:
        print("🔄 Cargando y procesando datos de Europa...")
        scaled_data, country_labels, feature_names = load_and_std_europa(DATA_FILE_PATH)
        
        print(f"📈 Datos cargados: {len(country_labels)} países, {len(feature_names)} características")
        print(f"🏷️ Características: {', '.join(feature_names)}")
        print()
        
        print("=" * 60)
        print("🗺️ EJERCICIO 1.1: RED DE KOHONEN")
        print("=" * 60)
        kohonen_network = ejercicio_1_kohonen(scaled_data, country_labels, feature_names)
        
        # NEW: K Comparison for presentation
        kohonen_networks = ejercicio_kohonen_k_comparison(scaled_data, country_labels, feature_names)
        
        print("\n" + "=" * 60)
        print("🔢 EJERCICIO 1.2: REGLA DE OJA")  
        print("=" * 60)
        oja_diff, country_projections_df = ejercicio_1_oja(scaled_data, feature_names, country_labels)
        plot_country_projections(country_projections_df)
        print(f"\n✅ Análisis completado!")
        print(f"📊 Diferencia Oja vs sklearn PCA: {oja_diff:.6f}")
        print(f"🗂️ Resultados guardados en: results/")
        print(f"🔥 Mapas de calor guardados en: results/feature_heatmaps/")
        print(f"📊 Comparaciones k=3,4,5 guardadas en: results/kohonen_k*_results.png")
        
    except FileNotFoundError:
        print(f"❌ Archivo no encontrado: {DATA_FILE_PATH}")
        print("🔍 Asegúrate de que el archivo europe.csv esté en la carpeta data/")
    except Exception as e:
        print(f"❌ Error durante la ejecución: {str(e)}")
        import traceback
        traceback.print_exc()