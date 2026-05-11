# Pipeline de Clustering de Comentarios

Sistema completo para analizar ~237k comentarios de texto corto usando dos métodos de clustering y exportar resultados para visualización D3.js.

## 📦 Instalación

```bash
pip install -r requirements.txt
```

> Si usas GPU para embeddings más rápidos:
> ```bash
> pip install faiss-gpu  # en lugar de faiss-cpu
> ```

## 🚀 Uso

### Ejecución completa (ambos métodos)
```bash
python main.py tu_dataset.csv --output-dir output/
```

### Solo método Coseno
```bash
python main.py tu_dataset.csv --method coseno
```

### Solo método Jaccard
```bash
python main.py tu_dataset.csv --method jaccard
```

### Todos los parámetros disponibles
```bash
python main.py tu_dataset.csv \
  --output-dir output/ \
  --method ambos \
  --batch-size 512 \          # batch size para embeddings
  --min-cluster-size 20 \     # HDBSCAN min_cluster_size
  --top-k 30 \                # vecinos FAISS top-k
  --num-perm 128 \            # permutaciones MinHash
  --lsh-threshold 0.3         # umbral similitud Jaccard
```

## 🧪 Test rápido

```bash
python test_pipeline.py
```

Genera 500 comentarios sintéticos, ejecuta el pipeline completo y valida los outputs.

## 📂 Estructura del proyecto

```
clustering_pipeline/
├── main.py              # Orquestador principal
├── preprocessing.py     # Limpieza de texto
├── method_coseno.py     # Método 1: Embeddings + FAISS + HDBSCAN
├── method_jaccard.py    # Método 2: MinHash + LSH
├── requirements.txt
└── README.md
```

## 📤 Archivos de salida

### Método Coseno
| Archivo | Descripción |
|---|---|
| `clusters_coseno.json` | `[{id, comentario, cluster}, ...]` — un registro por comentario |
| `clusters_agregados_coseno.json` | `{cluster_id: {size, centroid, sample_comments}}` |
| `dendrograma_coseno.json` | Árbol jerárquico D3.js sobre centroides |

### Método Jaccard
| Archivo | Descripción |
|---|---|
| `clusters_jaccard.json` | `[{id, comentario, cluster}, ...]` |
| `clusters_agregados_jaccard.json` | `{cluster_id: {size, sample_comments}}` |
| `dendrograma_jaccard.json` | Árbol jerárquico D3.js |

> **Nota:** `cluster = -1` indica ruido (puntos no asignados a ningún cluster).

## ⚙️ Arquitectura técnica

### Método Coseno
```
Texto limpio
    ↓
SentenceTransformer (all-MiniLM-L6-v2) [batches]
    ↓
Embeddings float32 normalizados (dim=384)
    ↓
FAISS IndexHNSWFlat (Inner Product ≈ cosine similarity)
    ↓
top-k vecinos (sin O(n²))
    ↓
HDBSCAN clustering (sobre embeddings)
    ↓
Centroides → scipy linkage Ward → Dendrograma D3.js
```

### Método Jaccard
```
Texto limpio
    ↓
N-gramas de caracteres (3–5)
    ↓
MinHash (128 permutaciones)
    ↓
LSH (Locality Sensitive Hashing, threshold=0.3)
    ↓
Union-Find sobre candidatos LSH → Clusters
    ↓
Representantes → distancias Jaccard → scipy linkage → Dendrograma D3.js
```

## 🔧 Ajuste de parámetros

### Para datasets muy grandes (>200k filas)

**Coseno:**
- `--batch-size 256` si hay poca RAM
- `--min-cluster-size 50` para clusters más grandes y menos ruido
- `--top-k 20` para FAISS más rápido

**Jaccard:**
- `--num-perm 64` para mayor velocidad (menor precisión)
- `--lsh-threshold 0.4` para clusters más estrictos (menos pares)
- `--lsh-threshold 0.2` para clusters más amplios (más pares)

### Para el tipo de dataset (comentarios cortos, multi-idioma)

El preprocesamiento NO hace stemming ni lematización para preservar variantes
en español, portugués, inglés y otros idiomas presentes en el dataset.

Los n-gramas de caracteres (Jaccard) son especialmente buenos para:
- Detectar duplicados y casi-duplicados
- Agrupar variantes ortográficas ("Exelente" / "Excelente")
- Funcionar bien en textos multi-idioma

## 🌐 Uso en D3.js

### dendrograma_*.json → D3.js Dendrogram

```javascript
d3.json("dendrograma_coseno.json").then(data => {
  const root = d3.hierarchy(data);
  // ... cluster dendrogram o collapsible tree
});
```

### clusters_*.json → D3.js Force Graph / Bubble Chart

```javascript
d3.json("clusters_coseno.json").then(data => {
  // data = [{id, comentario, cluster}, ...]
  const byCluster = d3.group(data, d => d.cluster);
});
```

### clusters_agregados_*.json → D3.js Treemap / Circle Pack

```javascript
d3.json("clusters_agregados_coseno.json").then(data => {
  // data = {0: {size, centroid, sample_comments}, ...}
  const nodes = Object.entries(data).map(([k, v]) => ({
    id: +k, size: v.size, comments: v.sample_comments
  }));
});
```
