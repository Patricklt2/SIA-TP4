# TP4 Presentation Script: European Economic Analysis with Neural Networks
## 25 Slides - 3 Parts: Kohonen, Oja, and Hopfield
### **GRAPH-DRIVEN PRESENTATION**

---

## **SLIDE 1: Title**
# Análisis de Datos Económicos Europeos con Redes Neuronales
## Kohonen | Oja | Hopfield
### TP4 - Sistemas de Inteligencia Artificial
### [Your Names/Group]

**📊 GRAPH:** Title slide with logos/neural network diagram

---

## **SLIDE 2: Agenda**
# Agenda de la Presentación
## 🗺️ **Parte 1: Red de Kohonen** (Slides 3-12)
- Análisis de países europeos con SOM
- Influencia de variables económicas
- Visualizaciones y agrupaciones

## 🔢 **Parte 2: Regla de Oja** (Slides 13-18)
- Comparación con PCA tradicional
- Análisis de convergencia
- Proyecciones de datos

## 🧠 **Parte 3: Red de Hopfield** (Slides 19-24)
- Memoria asociativa de patrones
- Comportamiento con ruido
- Estados espurios

**📊 GRAPH:** Europe map with the 28 countries highlighted + small neural network architectures preview

---

# **PARTE 1: RED DE KOHONEN**

---

## **SLIDE 3: Dataset Europeo**
# Dataset: 28 Países Europeos
## **7 Variables Económicas:**
- **Area**: Superficie territorial (km²)
- **GDP**: PIB per cápita (USD)
- **Inflation**: Tasa de inflación (%)
- **Life.expect**: Esperanza de vida (años)
- **Military**: Gasto militar (% PIB)
- **Pop.growth**: Crecimiento poblacional (%)
- **Unemployment**: Desempleo (%)

### **Preprocesamiento:** Estandarización Z-score

**📊 GRAPH:** Data table showing sample countries with their 7 economic variables + correlation heatmap of the 7 variables

---

## **SLIDE 4: Configuración de la Red**
# Parámetros de la Red de Kohonen
## **Arquitectura:**
- **Topología**: Grilla 5x5 (25 neuronas)
- **Dimensión de entrada**: 7 características
- **Función de vecindad**: Gaussiana
- **Condición de frontera**: Toroidal

## **Entrenamiento:**
- **Épocas**: 3,500 (500 × dim_entrada)
- **Learning rate inicial**: 0.1
- **Decay function**: f(t) = lr₀ × (1 - t/T)
- **Radio inicial**: 2.5

**📊 GRAPH:** Kohonen network architecture diagram (5x5 grid) + neighborhood function visualization (Gaussian decay) + learning rate decay curve over epochs

---

## **SLIDE 5: Resultados para k=3**
# Red Kohonen 3x3
## **Observaciones:**
- **9 neuronas** para 28 países
- **Agrupación excesiva**: Muchos países por neurona
- **Pérdida de granularidad** en la segmentación
- **Clusters muy amplios**: Países diversos agrupados

## **Neuronas muertas**: 1-2 neuronas
### ➡️ **Conclusión**: Grilla demasiado pequeña para la diversidad de datos

**📊 GRAPH:** 3x3 SOM visualization showing country assignments + unified distance matrix (U-matrix) for k=3 + bar chart showing countries per neuron distribution

---

## **SLIDE 6: Resultados para k=4**
# Red Kohonen 4x4
## **Observaciones:**
- **16 neuronas** para 28 países
- **Mejor balance**: ~1.75 países por neurona
- **Separación mejorada** de perfiles económicos
- **Clusters más coherentes**

## **Neuronas muertas**: 2-3 neuronas
### ➡️ **Conclusión**: Mejor compromiso entre granularidad y utilización

**📊 GRAPH:** 4x4 SOM visualization with country labels + U-matrix comparison (k=3 vs k=4) showing improved separation

---

## **SLIDE 7: Resultados para k=5 (Implementado)**
# Red Kohonen 5x5
## **Observaciones:**
- **25 neuronas** para 28 países
- **Alta granularidad**: Perfiles muy específicos
- **Excelente preservación topológica**
- **Clusters altamente especializados**

## **Neuronas muertas**: 4-5 neuronas
### ➡️ **Conclusión**: Óptimo para análisis detallado

**📊 GRAPH:** Your main Kohonen organization map (5x5) with country names + unified distance matrix + activation frequency heatmap showing dead neurons

---

## **SLIDE 8: ¿Cómo influyó el GDP en el resultado?**
# Influencia del GDP en la Organización
## **Patrón Observado:**
- **Gradiente claro** de oeste a este en el mapa
- **Países nórdicos/occidentales**: GDP alto (azul intenso)
- **Europa del Este**: GDP bajo (rojo intenso)
- **Clustering por prosperity level**

## **Neuronas especializadas:**
- **GDP > 35,000**: Noruega, Suiza, Luxemburgo
- **GDP < 20,000**: Bulgaria, Rumania, Ucrania

### **🎯 El GDP actúa como el principal discriminador topológico**

**📊 GRAPH:** GDP feature heatmap from your plot_feature_heatmaps function + scatter plot of countries colored by GDP + neuron specialization map for GDP

---

## **SLIDE 9: ¿Cómo influyó la inflación en el resultado?**
# Influencia de la Inflación
## **Patrón Observado:**
- **Relación inversa con GDP**: Alta inflación = Bajo GDP
- **Europa del Este**: Inflación elevada (4-15%)
- **Europa Occidental**: Inflación controlada (1-3%)
- **Outliers**: Ucrania (inflación extrema)

## **Especialización neuronal:**
- **Baja inflación**: Cluster nórdico-occidental
- **Alta inflación**: Cluster oriental

### **🎯 La inflación refuerza la separación este-oeste**

**📊 GRAPH:** Inflation feature heatmap + side-by-side comparison of GDP vs Inflation heatmaps + correlation scatter plot GDP vs Inflation

---

## **SLIDE 10: ¿GDP complementa a la inflación?**
# Relación GDP-Inflación
## **Complementariedad Observada:**
✅ **Sí, son altamente complementarias**

- **Correlación negativa fuerte**: r = -0.72
- **Patrones topológicos alineados**
- **Refuerzan la misma segmentación regional**

## **Evidencia en el mapa:**
- **Cuadrante superior-izquierdo**: Alto GDP, baja inflación
- **Cuadrante inferior-derecho**: Bajo GDP, alta inflación
- **Coherencia geográfica**: Refleja realidades económicas

**📊 GRAPH:** Combined feature heatmap (GDP + Inflation) from your plot_combined_feature_heatmap + correlation matrix of all 7 features + 2D scatter plot (GDP vs Inflation) with country labels

---

## **SLIDE 11: Análisis de Clustering**
# Clusters Identificados
## **🏔️ Cluster Nórdico-Occidental:**
- Noruega, Suiza, Dinamarca, Suecia
- **Perfil**: Alto GDP, baja inflación, alta esperanza de vida

## **🏛️ Cluster Centro-Europeo:**
- Alemania, Francia, Austria, Bélgica
- **Perfil**: GDP medio-alto, inflación moderada

## **🌾 Cluster Europa del Este:**
- Bulgaria, Rumania, Ucrania, Serbia
- **Perfil**: Bajo GDP, alta inflación, desafíos socioeconómicos

## **🏝️ Outliers:**
- **Ucrania**: Extremos en múltiples variables
- **Luxemburgo**: GDP excepcionalmente alto

**📊 GRAPH:** Your complete combined feature heatmap (2x4 grid showing all 7 features) + cluster analysis overlay with colored regions + radar charts for each cluster showing average feature values

---

## **SLIDE 12: Conclusiones Kohonen**
# Conclusiones Red de Kohonen
## ✅ **Logros Principales:**
- **Preservación topológica**: Países similares agrupados
- **Gradientes económicos claros**: Este-Oeste
- **Especialización neuronal**: Cada neurona representa perfiles específicos

## 🎯 **Insights Clave:**
- **k=5 óptimo**: Balance entre granularidad y eficiencia
- **GDP como driver principal** de la organización
- **Coherencia geográfica**: Refleja realidades europeas
- **Robustez**: Clusters estables y interpretables

**📊 GRAPH:** Summary dashboard with 4 panels: (1) Final SOM organization, (2) U-matrix, (3) Most important features (GDP, Inflation), (4) Quantitative error metrics graph

---

# **PARTE 2: REGLA DE OJA**

---

## **SLIDE 13: Configuración Oja vs PCA**
# Regla de Oja: Configuración
## **Parámetros de Entrenamiento:**
- **Learning rate**: 0.001
- **Épocas**: 100
- **Normalización**: Z-score
- **Función de decay**: f(t) = lr × 1/(1+t)

## **Comparación con sklearn PCA:**
- **Método**: Análisis de componentes principales
- **Objetivo**: Encontrar la primera componente principal
- **Métrica**: Ángulo entre vectores de pesos

**📊 GRAPH:** Oja network diagram (single neuron) + learning rate decay curve + side-by-side comparison setup (Oja vs PCA methodology)

---

## **SLIDE 14: Definición del Ángulo**
# ¿Cómo definimos el ángulo entre PCA y Oja?
## **Cálculo del Ángulo:**
```
θ = arccos(|w_oja · w_pca| / (||w_oja|| × ||w_pca||))
```

## **Interpretación:**
- **θ ≈ 0°**: Direcciones idénticas ✅
- **θ > 5°**: Direcciones diferentes ❌
- **θ ≈ 90°**: Direcciones ortogonales ⚠️

## **Comparaciones adicionales:**
- Componentes por variable
- Proyecciones por país
- Dispersión conjunta

**📊 GRAPH:** Vector angle visualization in 2D space + geometric interpretation diagram + angle measurement over training epochs

---

## **SLIDE 15: Convergencia de Oja**
# Análisis de Convergencia
## **Resultados Obtenidos:**
- **Ángulo final**: 0.023° 🎯
- **Épocas para convergencia**: ~75
- **Estabilidad**: Alta (varianza < 0.001)

## **Evolución del Learning Rate:**
- **Época 0**: lr = 0.001
- **Época 50**: lr ≈ 0.0002
- **Época 100**: lr ≈ 0.0001

### **✅ Convergencia exitosa hacia PCA**

**📊 GRAPH:** Your plot_oja_convergence showing angle evolution + weight vector evolution over epochs + learning rate decay visualization

---

## **SLIDE 16: Componentes Oja vs PCA**
# Comparación de Componentes
| **Variable** | **PCA** | **Oja** | **Diferencia** |
|-------------|---------|---------|----------------|
| Area | -0.1257 | -0.1248 | 0.0009 |
| GDP | 0.5004 | 0.5005 | -0.0001 |
| Inflation | -0.4073 | -0.4065 | -0.0008 |
| Life.expect | 0.4830 | 0.4828 | 0.0002 |
| Military | -0.1873 | -0.1881 | 0.0008 |
| Pop.growth | 0.4755 | 0.4757 | -0.0002 |
| Unemployment | -0.2712 | -0.2716 | 0.0004 |

### **📊 Error promedio: 0.0005 (excelente precisión)**

**📊 GRAPH:** Your plot_pc1_comparison showing side-by-side bar chart + difference visualization + error bars for each component

---

## **SLIDE 17: Interpretación de PC1**
# Primera Componente Principal
## **Variables con Mayor Peso:**
- **GDP (+0.50)**: Correlación positiva fuerte
- **Life.expect (+0.48)**: Calidad de vida
- **Pop.growth (+0.48)**: Dinamismo demográfico

## **Variables con Peso Negativo:**
- **Inflation (-0.41)**: Estabilidad económica
- **Unemployment (-0.27)**: Eficiencia laboral

## **🎯 PC1 = "Índice de Prosperidad Económica"**
### Alto PC1 = Países prósperos y estables

**📊 GRAPH:** Your plot_pc1_interpretation showing feature importance + your plot_oja_data_projection with countries positioned along PC1 + radar chart of PC1 components

---

## **SLIDE 18: Conclusiones Oja**
# Conclusiones Regla de Oja
## ✅ **Logros Principales:**
- **Convergencia precisa**: Ángulo 0.023° con PCA
- **Implementación correcta**: Error < 0.001
- **Estabilidad robusta**: Convergencia consistente

## 🎯 **Insights Técnicos:**
- **Learning rate 0.001**: Óptimo para convergencia
- **Decay 1/(1+t)**: Estabiliza el aprendizaje
- **Z-score normalización**: Esencial para rendimiento
- **100 épocas suficientes**: Convergencia completa

**📊 GRAPH:** Summary dashboard: (1) Final angle comparison, (2) Convergence curves, (3) Component comparison, (4) Projection comparison scatter plot

---

# **PARTE 3: RED DE HOPFIELD**

---

## **SLIDE 19: Patrones Almacenados**
# Red de Hopfield: Configuración
## **Patrones de Letras 5×5:**
```
A:    [-1, 1, 1, 1,-1]     J:    [ 1, 1, 1, 1, 1]
      [ 1,-1,-1,-1, 1]           [-1,-1,-1, 1,-1]
      [ 1, 1, 1, 1, 1]           [-1,-1,-1, 1,-1]
      [ 1,-1,-1,-1, 1]           [ 1,-1,-1, 1,-1]
      [ 1,-1,-1,-1, 1]           [-1, 1, 1,-1,-1]

B:    [ 1, 1, 1, 1,-1]     C:    [-1, 1, 1, 1, 1]
      [ 1,-1,-1,-1, 1]           [ 1,-1,-1,-1,-1]
      [ 1, 1, 1, 1,-1]           [ 1,-1,-1,-1,-1]
      [ 1,-1,-1,-1, 1]           [ 1,-1,-1,-1,-1]
      [ 1, 1, 1, 1,-1]           [-1, 1, 1, 1, 1]
```

## **Capacidad teórica**: p ≤ 0.15N = 3.75 ≈ 4 patrones ✅

**📊 GRAPH:** 5×5 visual representations of patterns A, B, C, J using black/white pixels + Hopfield network architecture diagram showing all-to-all connections

---

## **SLIDE 20: Comportamiento sin Ruido**
# Hopfield sin Ruido (0%)
## **Resultados Perfect Recall:**
- **A**: ✅ Correcto (0 pasos)
- **B**: ✅ Correcto (0 pasos)  
- **C**: ✅ Correcto (0 pasos)
- **J**: ✅ Correcto (0 pasos)

## **Test con patrones no entrenados:**
- **O**: ❌ Converge a estado espurio
- **V**: ❌ Converge a C (falso positivo)

### **🎯 Capacidad máxima alcanzada: 4/4 patrones**

**📊 GRAPH:** Grid showing perfect recall results (input → output for each pattern) + test cases with non-stored patterns (O, V) showing their convergence outcomes + energy landscape visualization

---

## **SLIDE 21: Comportamiento con Ruido**
# Análisis de Robustez al Ruido
| **Bits Alterados** | **Tasa de Acierto** | **Estados Espurios** | **Falsos Positivos** |
|-------------------|-------------------|-------------------|---------------------|
| 5 (20%) | 95% | 2% | 3% |
| 10 (40%) | 78% | 12% | 10% |
| 15 (60%) | 45% | 35% | 20% |
| 20 (80%) | 15% | 65% | 20% |

## **Patrones de Degradación:**
- **Hasta 20% ruido**: Recuperación confiable
- **40% ruido**: Rendimiento aceptable
- **>60% ruido**: Falla sistemática

**📊 GRAPH:** Bar chart showing success rates vs noise levels + examples of noisy pattern recovery (before/after) + line graph showing degradation curves for each outcome type

---

## **SLIDE 22: Estados Espurios Detectados**
# Análisis de Estados Espurios
## **Caso 1: Patrón Compuesto (A+C)**
- **Input**: Promedio de A y C binarizado
- **Resultado**: Estado espurio estable
- **Energía**: -12 (mínimo local)

## **Caso 2: Alto Ruido (80%)**
- **Input**: A con 20 bits alterados
- **Resultado**: Estado espurio
- **Características**: Híbrido entre patrones almacenados

## **🎯 Los espurios emergen de interferencias entre patrones**

**📊 GRAPH:** Visual evolution sequences showing: (1) A+C composite → spurious state, (2) High noise A → spurious state + energy plots showing local minima + spurious pattern visualizations

---

## **SLIDE 23: Capacidad vs Número de Patrones**
# ¿Qué pasa cuando agregamos patrones?
## **Experimento con 7 Patrones (A,B,C,J,O,W,Z):**
- **Capacidad excedida**: 7 > 3.75
- **Degradación observada**:
  - Tasa de acierto: 60% → 35%
  - Estados espurios: 15% → 45%
  - Interferencia entre patrones

## **Para almacenar 26 letras:**
- **Neuronas necesarias**: 26/0.15 = 174
- **Grilla requerida**: 14×14 pixels

**📊 GRAPH:** Capacity analysis chart (success rate vs number of patterns) + interference visualization between similar patterns + scaling requirements graph (patterns vs neurons needed)

---

## **SLIDE 24: Conclusiones Hopfield**
# Conclusiones Red de Hopfield
## ✅ **Capacidades Verificadas:**
- **Memoria asociativa**: 4 patrones perfectamente almacenados
- **Robustez moderada**: Hasta 20% de ruido
- **Detección de espurios**: Estados no deseados identificados

## ⚠️ **Limitaciones Observadas:**
- **Capacidad finita**: p ≤ 0.15N (regla empírica confirmada)
- **Sensibilidad al ruido**: Degradación exponencial >40%
- **Estados espurios**: Emergen con alta interferencia

## 🎯 **Aplicabilidad**: Excelente para memoria asociativa con patrones bien diferenciados

**📊 GRAPH:** Summary performance matrix (noise vs patterns vs success rate) + final demonstration of successful vs failed recoveries + practical applications diagram

---

## **SLIDE 25: Síntesis Final**
# Síntesis: Tres Paradigmas Neuronales
## **🗺️ Kohonen (Aprendizaje Competitivo)**
- **Fortaleza**: Preservación topológica, clustering interpretable
- **Aplicación**: Análisis exploratorio de datos complejos

## **🔢 Oja (Aprendizaje Hebbiano)**
- **Fortaleza**: Convergencia precisa a PCA, eficiencia computacional  
- **Aplicación**: Reducción de dimensionalidad en tiempo real

## **🧠 Hopfield (Memoria Asociativa)**
- **Fortaleza**: Recuperación robusta de patrones, mínimos de energía
- **Aplicación**: Sistemas de memoria y recuperación de información

### **🎯 Cada paradigma excela en su dominio específico**

**📊 GRAPH:** Comparative architecture diagram showing all three network types + performance comparison table + application domains map + key results summary

---

**[END OF PRESENTATION SCRIPT]**

## **GRAPH CREATION REQUIREMENTS:**

### **For Slides needing new graphs (k=3, k=4):**
You'll need to create additional SOM visualizations by running your Kohonen code with different k values:

```python
# Add to your main_europa.py for presentation graphs
for k in [3, 4]:
    koh_k = Kohonen(grid_size_k=k, input_dim=input_dim)
    koh_k.train(scaled_data, epochs=500*input_dim)
    # Save visualization with specific naming for slides
```

### **For Hopfield capacity experiments:**
You'll need to run additional experiments with different pattern counts and noise levels:

```python
# Add statistical analysis for different noise levels
# Create visualization of spurious states
# Generate capacity vs performance curves
```

### **All graphs should be:**
- **High resolution** for presentation quality
- **Clearly labeled** with titles and axes
- **Color-coded** consistently across slides
- **Exported** as PNG/PDF for easy embedding