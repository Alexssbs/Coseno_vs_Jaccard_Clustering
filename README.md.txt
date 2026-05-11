# Pipeline de Clustering de Comentarios

Sistema completo para analizar ~237k comentarios de texto corto usando dos métodos de clustering y exportar resultados para visualización D3.js.

## 📋 Tabla de contenidos

- [Descripción general](#descripción-general)
- [Métodos implementados](#métodos-implementados)
- [Estructura del proyecto](#estructura-del-proyecto)
- [Instalación](#instalación)
- [Ejecución del pipeline](#ejecución-del-pipeline)
- [Parámetros de configuración](#parámetros-de-configuración)
- [Resultados generados](#resultados-generados)
- [Visualización interactiva](#visualización-interactiva)
- [Rendimiento y tiempos](#rendimiento-y-tiempos)
- [Solución de problemas](#solución-de-problemas)
- [Licencia](#licencia)

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
