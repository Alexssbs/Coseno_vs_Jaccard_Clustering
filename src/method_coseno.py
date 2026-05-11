"""
Método 1: Similitud de Coseno
Pipeline: SentenceTransformers → FAISS (HNSW) → HDBSCAN → Dendrograma por centroides
"""

import json
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Any

from checkpoint_utils import (
    save_numpy, load_numpy, save_pickle, load_pickle,
    save_faiss_index, load_faiss_index, save_metadata, load_metadata
)
# ──────────────────────────────────────────────s
# 1. EMBEDDINGS
# ──────────────────────────────────────────────

def compute_embeddings(texts: list[str], batch_size: int = 512, checkpoint_dir: Path = None) -> np.ndarray:
    """
    Genera embeddings con sentence-transformers/all-MiniLM-L6-v2.
    Procesa en batches para control de memoria.
    Retorna array float32 normalizado (norma L2 = 1 → cosine similarity = dot product).

    Si el modelo no está disponible (sin conexión / sin caché), usa TF-IDF SVD como fallback.
    """
    # --- CHECKPOINT: intentar cargar embeddings guardados ---
    if checkpoint_dir is not None:
        checkpoint_dir = Path(checkpoint_dir)
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        metadata = load_metadata(checkpoint_dir)
        if metadata and metadata.get("embeddings_done") and metadata.get("n_texts") == len(texts):
            embeddings = load_numpy(checkpoint_dir, "embeddings")
            if embeddings is not None:
                print(f"  [Embeddings] Cargados desde checkpoint. Shape: {embeddings.shape}")
                return embeddings
    
    try:
        from sentence_transformers import SentenceTransformer

        print(f"  [Embeddings] Cargando modelo sentence-transformers/all-MiniLM-L6-v2...")
        model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

        print(f"  [Embeddings] Procesando {len(texts):,} textos en batches de {batch_size}...")
        embeddings = model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=True,
            convert_to_numpy=True,
            normalize_embeddings=True,   # normaliza para que dot product = cosine similarity
        )
        embeddings = embeddings.astype(np.float32)
        print(f"  [Embeddings] Shape: {embeddings.shape}")
        
        # Al final, ANTES del return:
        if checkpoint_dir is not None:
            save_numpy(embeddings, checkpoint_dir, "embeddings")
            save_metadata(checkpoint_dir, {"embeddings_done": True, "n_texts": len(texts)})
        
        return embeddings

    except Exception as e:
        print(f"  [Embeddings] ⚠️  No se pudo cargar SentenceTransformer: {e}")
        print(f"  [Embeddings] Usando fallback TF-IDF + SVD (TruncatedSVD dim=256)...")
        embeddings = _tfidf_fallback_embeddings(texts)
        
        # Al final, ANTES del return:
        if checkpoint_dir is not None:
            save_numpy(embeddings, checkpoint_dir, "embeddings")
            save_metadata(checkpoint_dir, {"embeddings_done": True, "n_texts": len(texts)})
        
        return embeddings


def _tfidf_fallback_embeddings(texts: list[str], n_components: int = 256) -> np.ndarray:
    """
    Fallback cuando sentence-transformers no está disponible.
    Usa TF-IDF con reducción SVD como representación densa.
    Útil para pruebas offline y entornos sin GPU/HuggingFace.
    """
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.decomposition import TruncatedSVD
    from sklearn.preprocessing import normalize

    print(f"  [TF-IDF Fallback] Vectorizando {len(texts):,} textos...")
    vec = TfidfVectorizer(
        analyzer="char_wb",
        ngram_range=(3, 5),
        max_features=50000,
        sublinear_tf=True,
    )
    X = vec.fit_transform(texts)

    actual_components = min(n_components, X.shape[1] - 1, X.shape[0] - 1)
    print(f"  [TF-IDF Fallback] Reduciendo a {actual_components} dimensiones con SVD...")
    svd = TruncatedSVD(n_components=actual_components, random_state=42)
    embeddings = svd.fit_transform(X).astype(np.float32)
    embeddings = normalize(embeddings, norm="l2")
    print(f"  [TF-IDF Fallback] Shape: {embeddings.shape}")
    return embeddings


# ──────────────────────────────────────────────
# 2. FAISS – BÚSQUEDA DE VECINOS EFICIENTE
# ──────────────────────────────────────────────

def build_faiss_index(embeddings: np.ndarray, use_hnsw: bool = True, checkpoint_dir: Path = None) -> Any:
    """
    Construye índice FAISS para búsqueda aproximada de vecinos.
    - HNSW: muy rápido en búsqueda, buena calidad. Recomendado para <500k.
    - IVFFlat: más rápido en indexación para datasets muy grandes.
    Embeddings ya normalizados → usamos Inner Product (= cosine similarity).
    """
    import faiss

    # --- CHECKPOINT: intentar cargar índice FAISS ---
    if checkpoint_dir is not None:
        checkpoint_dir = Path(checkpoint_dir)
        index = load_faiss_index(checkpoint_dir, "faiss_index")
        if index is not None:
            print(f"  [FAISS] Índice cargado desde checkpoint.")
            return index

    dim = embeddings.shape[1]
    n = embeddings.shape[0]
    print(f"  [FAISS] Construyendo índice (dim={dim}, n={n:,})...")

    if use_hnsw:
        # M = conexiones por nodo, efLinks = efConstruction
        index = faiss.IndexHNSWFlat(dim, 32, faiss.METRIC_INNER_PRODUCT)
        index.hnsw.efConstruction = 200
        index.hnsw.efSearch = 64
        index.add(embeddings)
        print(f"  [FAISS] Índice HNSW construido.")
    else:
        # IVFFlat – más eficiente en memoria para datasets muy grandes
        nlist = min(4096, int(np.sqrt(n)))
        quantizer = faiss.IndexFlatIP(dim)
        index = faiss.IndexIVFFlat(quantizer, dim, nlist, faiss.METRIC_INNER_PRODUCT)
        index.train(embeddings)
        index.add(embeddings)
        index.nprobe = 64
        print(f"  [FAISS] Índice IVFFlat construido (nlist={nlist}).")

    # Al final:
    if checkpoint_dir is not None:
        save_faiss_index(index, checkpoint_dir, "faiss_index")

    return index


def search_neighbors(index, embeddings: np.ndarray, top_k: int = 30, checkpoint_dir: Path = None) -> np.ndarray:
    """
    Busca los top_k vecinos más cercanos para cada punto.
    Retorna matriz de índices (n, top_k).
    NO calcula matriz completa O(n²).
    """
    # --- CHECKPOINT: intentar cargar vecinos guardados ---
    if checkpoint_dir is not None:
        checkpoint_dir = Path(checkpoint_dir)
        neighbors = load_numpy(checkpoint_dir, f"neighbors_top{top_k}")
        if neighbors is not None:
            print(f"  [FAISS] Vecinos cargados desde checkpoint. Shape: {neighbors.shape}")
            return neighbors

    print(f"  [FAISS] Buscando top-{top_k} vecinos para {len(embeddings):,} puntos...")
    # Buscar en batches para controlar memoria
    batch_size = 4096
    all_indices = []
    for start in range(0, len(embeddings), batch_size):
        end = min(start + batch_size, len(embeddings))
        _, I = index.search(embeddings[start:end], top_k + 1)  # +1 porque incluye a sí mismo
        all_indices.append(I[:, 1:])  # excluir self (posición 0)
    neighbors = np.vstack(all_indices)
    print(f"  [FAISS] Búsqueda completada. Shape vecinos: {neighbors.shape}")
    
    # Al final:
    if checkpoint_dir is not None:
        save_numpy(neighbors, checkpoint_dir, f"neighbors_top{top_k}")
    
    return neighbors


# ──────────────────────────────────────────────
# 3. HDBSCAN CLUSTERING
# ──────────────────────────────────────────────

def run_hdbscan(embeddings: np.ndarray, min_cluster_size: int = 20, checkpoint_dir: Path = None) -> np.ndarray:
    """
    HDBSCAN directamente sobre embeddings (más robusto que sobre grafo).
    Puntos con label=-1 son ruido (no asignados a cluster).
    """
    import hdbscan

    # --- CHECKPOINT: intentar cargar labels ---
    if checkpoint_dir is not None:
        checkpoint_dir = Path(checkpoint_dir)
        labels = load_numpy(checkpoint_dir, f"hdbscan_labels_ms{min_cluster_size}")
        if labels is not None:
            print(f"  [HDBSCAN] Labels cargados desde checkpoint.")
            return labels

    print(f"  [HDBSCAN] Clustering con min_cluster_size={min_cluster_size}...")
    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=min_cluster_size,
        min_samples=5,
        metric="euclidean",        # embeddings normalizados: euclidean ≈ cosine distance
        cluster_selection_method="eom",
        prediction_data=True,
        core_dist_n_jobs=-1,
    )
    labels = clusterer.fit_predict(embeddings)
    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise = np.sum(labels == -1)
    print(f"  [HDBSCAN] Clusters encontrados: {n_clusters}")
    print(f"  [HDBSCAN] Puntos de ruido (-1): {n_noise:,} ({100*n_noise/len(labels):.1f}%)")
    
    # Al final:
    if checkpoint_dir is not None:
        save_numpy(labels, checkpoint_dir, f"hdbscan_labels_ms{min_cluster_size}")
    
    return labels


# ──────────────────────────────────────────────
# 4. DENDROGRAMA APROXIMADO (sobre centroides)
# ──────────────────────────────────────────────

def compute_centroids(embeddings: np.ndarray, labels: np.ndarray) -> dict:
    """
    Calcula el centroide de cada cluster (excluye ruido).
    Retorna {cluster_id: centroid_vector}.
    """
    unique_labels = sorted(set(labels) - {-1})
    centroids = {}
    for lbl in unique_labels:
        mask = labels == lbl
        centroids[lbl] = embeddings[mask].mean(axis=0)
    return centroids


def build_dendrogram_tree(centroids: dict) -> dict:
    """
    Aplica clustering jerárquico (scipy linkage Ward) sobre los centroides.
    Convierte la matriz de linkage al formato árbol compatible con D3.js.
    NO usa todos los puntos, solo los centroides.
    """
    from scipy.cluster.hierarchy import linkage, to_tree
    from scipy.spatial.distance import cdist

    if len(centroids) < 2:
        # Solo un cluster o ninguno
        lbl = list(centroids.keys())[0] if centroids else -1
        return {"name": f"cluster_{lbl}", "children": []}

    print(f"  [Dendrograma] Calculando jerarquía sobre {len(centroids)} centroides...")
    ids = sorted(centroids.keys())
    matrix = np.array([centroids[i] for i in ids], dtype=np.float32)

    Z = linkage(matrix, method="ward")

    def node_to_dict(node, id_map):
        if node.is_leaf():
            cluster_id = id_map[node.id]
            return {"name": f"cluster_{cluster_id}", "cluster_id": int(cluster_id)}
        return {
            "name": f"node_{node.id}",
            "children": [
                node_to_dict(node.get_left(), id_map),
                node_to_dict(node.get_right(), id_map),
            ],
            "distance": float(node.dist),
        }

    root, _ = to_tree(Z, rd=True)
    id_map = {i: ids[i] for i in range(len(ids))}
    tree = node_to_dict(root, id_map)
    tree["name"] = "root"
    print(f"  [Dendrograma] Árbol construido.")
    return tree


# ──────────────────────────────────────────────
# 5. EXPORTAR RESULTADOS
# ──────────────────────────────────────────────

def export_coseno_results(
    df: pd.DataFrame,
    labels: np.ndarray,
    embeddings: np.ndarray,
    centroids: dict,
    dendrogram_tree: dict,
    output_dir: Path,
    n_samples: int = 10,
):
    """
    Exporta los 3 archivos JSON del método coseno.
    """
    output_dir = Path(output_dir)

    # ── 1. clusters_coseno.json ──
    print("  [Export] Generando clusters_coseno.json...")
    records = []
    for idx, row in df.iterrows():
        records.append({
            "id": int(row["id"]),
            "comentario": row["comentario"],
            "cluster": int(labels[idx]),
        })
    with open(output_dir / "clusters_coseno.json", "w", encoding="utf-8") as f:
        json.dump(records, f, ensure_ascii=False, separators=(",", ":"))
    print(f"    Guardado: clusters_coseno.json ({len(records):,} registros)")

    # ── 2. clusters_agregados_coseno.json ──
    print("  [Export] Generando clusters_agregados_coseno.json...")
    agregados = {}
    unique_labels = sorted(set(labels))
    for lbl in unique_labels:
        mask = labels == lbl
        subset = df[mask]
        sample = subset["comentario"].head(n_samples).tolist()
        entry = {
            "size": int(mask.sum()),
            "sample_comments": sample,
        }
        if lbl in centroids:
            entry["centroid"] = centroids[lbl].tolist()
        agregados[int(lbl)] = entry
    with open(output_dir / "clusters_agregados_coseno.json", "w", encoding="utf-8") as f:
        json.dump(agregados, f, ensure_ascii=False, indent=2)
    print(f"    Guardado: clusters_agregados_coseno.json")

    # ── 3. dendrograma_coseno.json ──
    print("  [Export] Generando dendrograma_coseno.json...")
    with open(output_dir / "dendrograma_coseno.json", "w", encoding="utf-8") as f:
        json.dump(dendrogram_tree, f, ensure_ascii=False, indent=2)
    print(f"    Guardado: dendrograma_coseno.json")


# ──────────────────────────────────────────────
# PIPELINE PRINCIPAL
# ──────────────────────────────────────────────

def run_coseno_pipeline(
    df: pd.DataFrame,
    output_dir: Path,
    batch_size: int = 512,
    top_k: int = 30,
    min_cluster_size: int = 20,
    checkpoint_dir: Path = None,  # NUEVO
):
    # Si no se especifica checkpoint_dir, usar output_dir/checkpoints_coseno
    if checkpoint_dir is None:
        checkpoint_dir = output_dir / "checkpoints_coseno"
    checkpoint_dir = Path(checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    texts = df["comentario_clean"].tolist()

    # 1. Embeddings (pasar checkpoint_dir)
    embeddings = compute_embeddings(texts, batch_size=batch_size, checkpoint_dir=checkpoint_dir)

    # 2. FAISS index + búsqueda de vecinos (pasar checkpoint_dir)
    index = build_faiss_index(embeddings, use_hnsw=True, checkpoint_dir=checkpoint_dir)
    neighbors = search_neighbors(index, embeddings, top_k=top_k, checkpoint_dir=checkpoint_dir)

    # 3. HDBSCAN clustering (pasar checkpoint_dir)
    labels = run_hdbscan(embeddings, min_cluster_size=min_cluster_size, checkpoint_dir=checkpoint_dir)

    # 4. Centroides + dendrograma
    centroids = compute_centroids(embeddings, labels)
    dendrogram_tree = build_dendrogram_tree(centroids)

    # 5. Exportar
    export_coseno_results(
        df=df,
        labels=labels,
        embeddings=embeddings,
        centroids=centroids,
        dendrogram_tree=dendrogram_tree,
        output_dir=output_dir,
    )

    print("\n  ✅ Método Coseno completado.")
    return labels