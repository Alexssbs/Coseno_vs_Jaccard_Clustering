# Pipeline de Clustering de Comentarios

Sistema completo para analizar ~237k comentarios de texto corto usando dos métodos de clustering y exportar resultados para visualización D3.js.

## 📋 Tabla de contenidos

- [Descripción general](#descripción-general)
- [Métodos implementados](#métodos-implementados)
- [Estructura del proyecto](#estructura-del-proyecto)
- [Instalación](#instalación)
- [Resultados generados](#resultados-generados)
- [Visualización interactiva](#visualización-interactiva)
- [Rendimiento y tiempos](#rendimiento-y-tiempos)

---

## Descripción general

Este pipeline procesa grandes volúmenes de comentarios de texto corto (más de 200k registros) aplicando dos enfoques complementarios de clustering:

1. **Clustering semántico** (Coseno) - Basado en embeddings de transformers
2. **Clustering literal** (Jaccard) - Basado en n-gramas de caracteres

Los resultados se exportan en formato JSON compatible con visualizaciones dendrograma interactivas en D3.js.

### Características principales

- ✅ Preprocesamiento y limpieza automática de textos
- ✅ Checkpoints intermedios para reanudar ejecuciones interrumpidas
- ✅ Optimización de memoria con procesamiento por batches
- ✅ Fallback TF-IDF + SVD si SentenceTransformers no está disponible
- ✅ Exportación a JSON para visualización en D3.js

---

## Métodos implementados

### Método 1: Similitud de Coseno (Semántico)

- **Embeddings:** `sentence-transformers/all-MiniLM-L6-v2` (384 dimensiones)
- **Búsqueda de vecinos:** FAISS con índice HNSW (búsqueda aproximada)
- **Clustering:** HDBSCAN (min_cluster_size=20, min_samples=5)
- **Ventaja:** Captura similitud semántica entre palabras diferentes

### Método 2: Similitud de Jaccard (Literal)

- **N-gramas:** caracteres, longitud 3-5
- **Firmas MinHash:** 128 permutaciones
- **LSH threshold:** 0.3 (similitud Jaccard mínima)
- **Ventaja:** Rápido, sin dependencias externas, detecta coincidencias literales

---

## Estructura del proyecto

# Coseno_vs_Jaccard_Clustering

```text
Coseno_vs_Jaccard_Clustering/
│
├── src/
│   ├── main.py                      # Pipeline principal
│   ├── preprocessing.py             # Limpieza de texto
│   ├── method_coseno.py             # Método de similitud de coseno
│   ├── method_jaccard.py            # Método de similitud de Jaccard
│   └── checkpoint_utils.py          # Guardado/carga de checkpoints
│    └── comentarios_solo_texto.csv  # Dataset de 237k de comentarios
│ 
├── results/
│   ├── clusters_coseno.json         # Asignaciones método coseno
│   ├── clusters_agregados_coseno.json # Estadísticas por cluster
│   ├── dendrograma_coseno.json      # Árbol jerárquico coseno
│   ├── clusters_jaccard.json        # Asignaciones método jaccard
│   ├── clusters_agregados_jaccard.json # Estadísticas por cluster
│   └── dendrograma_jaccard.json     # Árbol jerárquico jaccard
│
├── visualizacion/
│   └── d3_4.html                    # Visualizador dendrograma interactivo
│
├── requirements.txt                 # Dependencias del proyecto
└── README.md                        # Este archivo
```

---

## Instalación

```bash
pip install -r requirements.txt

pip install faiss-gpu  # en lugar de faiss-cpu
