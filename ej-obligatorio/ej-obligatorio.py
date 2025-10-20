import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

def load_and_preprocess_data(filepath='europe.csv'):
    try:
        df = pd.read_csv(filepath)
    except FileNotFoundError:
        print(f"Error: '{filepath}' not found.")
        return None, None, None

    country_names = df['Country']
    numerical_df = df.drop('Country', axis=1)
    feature_names = numerical_df.columns

    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(numerical_df)
    
    return scaled_data, country_names, feature_names

def plot_pca_loadings(loadings, features):
    loadings_df = pd.DataFrame({'Característica': features, 'Carga_PC1': loadings})
    
    plt.figure(figsize=(10, 7))
    sns.barplot(x='Carga_PC1', y='Característica', data=loadings_df.sort_values('Carga_PC1', ascending=False), palette='viridis')
    plt.title('Carga de las variables en el primer componente principal.')
    plt.xlabel('Carga')
    plt.ylabel('Característica')
    plt.axvline(0, color='k', linestyle='--')
    plt.grid(axis='x', linestyle='--', alpha=0.6)
    plt.show()

def plot_country_rankings(scores, countries):
    pc1_index_df = pd.DataFrame({'País': countries, 'Puntuación_PC1': scores})
    pc1_index_df_sorted = pc1_index_df.sort_values('Puntuación_PC1', ascending=False)
    
    plt.figure(figsize=(12, 8))
    sns.barplot(x='Puntuación_PC1', y='País', data=pc1_index_df_sorted, palette='coolwarm')
    plt.title('Países Europeos Clasificados por el "Índice de Fuerza y Desarrollo Económico"')
    plt.xlabel('Puntuación PC1 (Mayor = Mas Desarrollado)')
    plt.ylabel('País')
    plt.show()

def plot_explained_variance(pca):
    explained_variance_ratio = pca.explained_variance_ratio_
    
    cumulative_variance = np.cumsum(explained_variance_ratio)
    
    components = range(1, len(explained_variance_ratio) + 1)
    
    plt.figure(figsize=(10, 6))
    
    sns.barplot(x=list(components), y=explained_variance_ratio, color='skyblue', label='Varianza Individual')
    
    plt.plot(components, cumulative_variance, marker='o', linestyle='-', color='red', label='Varianza Acumulada')
    
    plt.title('Varianza Explicada por Componente Principal')
    plt.xlabel('Componente Principal')
    plt.ylabel('Proporción de Varianza Explicada')
    plt.xticks(components)
    plt.legend()
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.show()

def plot_biplot(scores, loadings, countries, features):
    plt.figure(figsize=(12, 12))
    
    plt.scatter(scores[:, 0], scores[:, 1], color='grey', alpha=0.6)
    
    for i, country in enumerate(countries):
        plt.annotate(country, (scores[i, 0], scores[i, 1]), alpha=0.7, fontsize=7)

    for i, feature in enumerate(features):
        plt.arrow(0, 0, loadings[i, 0], loadings[i, 1], color='r', alpha=0.8, head_width=0.05)
        plt.text(loadings[i, 0] * 1.15, loadings[i, 1] * 1.15, feature, color='r', ha='center', va='center', fontsize=7)

    plt.title('Puntuaciones y Cargas en PC1 y PC2')
    plt.xlabel('PC1')
    plt.ylabel('PC2')
    plt.grid()
    plt.axhline(0, color='grey', linestyle='--')
    plt.axvline(0, color='grey', linestyle='--')
    plt.show()

def perform_pca(data):
    pca = PCA()
    principal_components = pca.fit_transform(data)

    pc1_loadings = pca.components_[0]
    pc2_loadings = pca.components_[1]
    
    if pc1_loadings[np.argmax(np.abs(pc1_loadings))] < 0:
         pc1_loadings = -pc1_loadings
         principal_components[:, 0] = -principal_components[:, 0]
    
    return pca, pc1_loadings, principal_components
    
def main():
    scaled_data, country_names, feature_names = load_and_preprocess_data()
    
    if scaled_data is None:
        return

    pca, pc1_loadings, principal_components = perform_pca(scaled_data) 
    pc1_scores = principal_components[:, 0]

    plot_pca_loadings(pc1_loadings, feature_names)

    plot_country_rankings(pc1_scores, country_names)
    
    plot_explained_variance(pca)

    biplot_loadings = pca.components_[[0, 1], :].T 
    if pc1_loadings[np.argmax(np.abs(pc1_loadings))] < 0:
        biplot_loadings[:, 0] = -biplot_loadings[:, 0]

    plot_biplot(principal_components, biplot_loadings, country_names, feature_names)

if __name__ == '__main__':
    main()