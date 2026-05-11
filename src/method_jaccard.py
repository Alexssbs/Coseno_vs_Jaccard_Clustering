"""
Método 2: Similitud de Jaccard
Pipeline: N-gramas de caracteres → MinHash → LSH → Clustering → Dendrograma
"""

import json
import numpy as np
import pandas as pd
from pathlib import Path
from collections import defaultdict

from checkpoint_utils import (
    save_pickle, load_pickle, save_metadata, load_metadata, save_numpy, load_numpy
)

# ──────────────────────────────────────────────
# 1. N-GRAMAS DE CARACTERES
# ──────────────────────────────────────────────

def get_char_ngrams(text: str, n_min: int = 3, n_max: int = 5) -> set:
    """
    Genera n-gramas de caracteres de longitud n_min a n_max.
    Ej: "hola" con n=3 → {"hol", "ola"}
    """
    ngrams = set()
    for n in range(n_min, n_max + 1):
        for i in range(len(text) - n + 1):
            ngrams.add(text[i:i + n])
    return ngrams


# ──────────────────────────────────────────────
# 2. MINHASH + LSH
# ──────────────────────────────────────────────

def build_minhash_lsh(
    texts: list[str],
    num_perm: int = 128,
    threshold: float = 0.3,
    n_min: int = 3,
    n_max: int = 5,
    batch_size: int = 5000,
    checkpoint_dir: Path = None,  # NUEVO
):
    """
    Construye firmas MinHash para todos los textos y usa LSH para encontrar
    pares similares sin comparar todo contra todo (NO O(n²)).
    
    Retorna:
        - minhashes: lista de objetos MinHash
        - lsh: índice LSH con todos los documentos
    """
    from datasketch import MinHash, MinHashLSH

    # --- CHECKPOINT: intentar cargar minhashes y lsh ---
    if checkpoint_dir is not None:
        checkpoint_dir = Path(checkpoint_dir)
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        metadata = load_metadata(checkpoint_dir)
        if metadata and metadata.get("lsh_done") and metadata.get("n_texts") == len(texts):
            minhashes = load_pickle(checkpoint_dir, "minhashes")
            lsh = load_pickle(checkpoint_dir, "lsh")
            if minhashes is not None and lsh is not None:
                print(f"  [MinHash+LSH] Cargados desde checkpoint.")
                return minhashes, lsh

    print(f"  [MinHash] Construyendo firmas para {len(texts):,} textos (num_perm={num_perm})...")
    print(f"  [LSH] Umbral de similitud Jaccard: {threshold}")

    lsh = MinHashLSH(threshold=threshold, num_perm=num_perm)
    minhashes = []

    for i, text in enumerate(texts):
        m = MinHash(num_perm=num_perm)
        ngrams = get_char_ngrams(text, n_min=n_min, n_max=n_max)
        if not ngrams:
            # texto muy corto sin n-gramas: usar caracteres individuales
            ngrams = set(text)
        for gram in ngrams:
            m.update(gram.encode("utf-8"))
        minhashes.append(m)
        lsh.insert(str(i), m)

        if (i + 1) % batch_size == 0:
            pct = 100 * (i + 1) / len(texts)
            print(f"    Procesados: {i+1:,}/{len(texts):,} ({pct:.1f}%)")

    print(f"  [MinHash+LSH] Índice construido.")
    
    # Al final, antes del return:
    if checkpoint_dir is not None:
        save_pickle(minhashes, checkpoint_dir, "minhashes")
        save_pickle(lsh, checkpoint_dir, "lsh")
        save_metadata(checkpoint_dir, {"lsh_done": True, "n_texts": len(texts)})
    
    return minhashes, lsh


# ──────────────────────────────────────────────
# 3. EXTRAER CANDIDATOS Y FORMAR CLUSTERS
# ──────────────────────────────────────────────

def cluster_from_lsh(
    lsh,
    minhashes: list,
    n: int,
    batch_size: int = 2000,
    checkpoint_dir: Path = None,  # NUEVO
) -> np.ndarray:
    """
    Usa Union-Find para agrupar todos los pares similares encontrados por LSH.
    Evita comparar todo contra todo: solo evalúa candidatos LSH.
    
    Retorna array de labels (int) de tamaño n.
    -1 = singleton (no tiene vecinos similares).
    """
    
    # --- CHECKPOINT: intentar cargar labels ---
    if checkpoint_dir is not None:
        checkpoint_dir = Path(checkpoint_dir)
        labels = load_numpy(checkpoint_dir, "jaccard_labels")
        if labels is not None:
            print(f"  [LSH Clustering] Labels cargados desde checkpoint.")
            return labels

    print(f"  [LSH Clustering] Agrupando candidatos con Union-Find...")

    # Union-Find
    parent = list(range(n))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]  # path compression
            x = parent[x]
        return x

    def union(x, y):
        rx, ry = find(x), find(y)
        if rx != ry:
            parent[rx] = ry

    total_pairs = 0
    for i in range(0, n, batch_size):
        end = min(i + batch_size, n)
        for j in range(i, end):
            candidates = lsh.query(minhashes[j])
            for c in candidates:
                c_idx = int(c)
                if c_idx != j:
                    union(j, c_idx)
                    total_pairs += 1
        if (end) % 20000 == 0:
            pct = 100 * end / n
            print(f"    Procesados: {end:,}/{n:,} ({pct:.1f}%)")

    print(f"  [LSH Clustering] Pares similares encontrados: {total_pairs:,}")

    # Asignar labels por componente conectada
    root_to_label = {}
    labels = np.full(n, -1, dtype=int)
    label_counter = 0

    for i in range(n):
        root = find(i)
        if root not in root_to_label:
            root_to_label[root] = label_counter
            label_counter += 1
        labels[i] = root_to_label[root]

    # Identificar clusters de tamaño 1 (singletons) → marcarlos como -1
    from collections import Counter
    counts = Counter(labels)
    singleton_labels = {lbl for lbl, cnt in counts.items() if cnt == 1}

    # Re-mapear: singletons → -1, resto → IDs consecutivos
    valid_labels = sorted(set(labels) - singleton_labels)
    remap = {old: new for new, old in enumerate(valid_labels)}

    final_labels = np.where(
        np.isin(labels, list(singleton_labels)),
        -1,
        np.vectorize(lambda x: remap.get(x, -1))(labels)
    )

    n_clusters = len(valid_labels)
    n_noise = np.sum(final_labels == -1)
    print(f"  [LSH Clustering] Clusters formados: {n_clusters}")
    print(f"  [LSH Clustering] Singletons (ruido): {n_noise:,} ({100*n_noise/n:.1f}%)")

    # Al final, antes del return (después de tener final_labels):
    if checkpoint_dir is not None:
        save_numpy(final_labels, checkpoint_dir, "jaccard_labels")
    
    return final_labels


# ──────────────────────────────────────────────
# 4. DENDROGRAMA APROXIMADO (sobre representantes de cluster)
# ──────────────────────────────────────────────

def build_jaccard_dendrogram(
    labels: np.ndarray,
    minhashes: list,
    max_clusters_for_dendro: int = 500,
) -> dict:
    """
    Construye dendrograma jerárquico sobre los clusters Jaccard.
    Usa estimaciones de Jaccard entre representantes (MinHash) como distancias.
    Solo usa hasta max_clusters_for_dendro clusters para evitar explosión de memoria.
    """
    from scipy.cluster.hierarchy import linkage, to_tree

    unique_labels = sorted(set(labels) - {-1})
    n_clusters = len(unique_labels)

    if n_clusters < 2:
        lbl = unique_labels[0] if unique_labels else -1
        return {"name": f"cluster_{lbl}", "children": []}

    # Si hay demasiados clusters, tomar muestra representativa
    if n_clusters > max_clusters_for_dendro:
        print(f"  [Dendrograma Jaccard] Demasiados clusters ({n_clusters}), usando top-{max_clusters_for_dendro} por tamaño...")
        from collections import Counter
        counts = Counter(labels[labels >= 0])
        top_labels = [lbl for lbl, _ in counts.most_common(max_clusters_for_dendro)]
        unique_labels = sorted(top_labels)
        n_clusters = len(unique_labels)

    print(f"  [Dendrograma Jaccard] Calculando distancias entre {n_clusters} clusters...")

    # Para cada cluster, tomar el primer elemento como representante
    representatives = {}
    for lbl in unique_labels:
        idxs = np.where(labels == lbl)[0]
        representatives[lbl] = minhashes[idxs[0]]

    # Calcular matriz de distancias Jaccard entre representantes
    # Aquí sí calculamos O(k²) pero k es máximo max_clusters_for_dendro
    ids = sorted(representatives.keys())
    k = len(ids)
    dist_matrix = np.zeros((k, k), dtype=np.float32)

    for i in range(k):
        for j in range(i + 1, k):
            jaccard = representatives[ids[i]].jaccard(representatives[ids[j]])
            dist = 1.0 - jaccard  # convertir similitud a distancia
            dist_matrix[i, j] = dist
            dist_matrix[j, i] = dist

    from scipy.spatial.distance import squareform
    condensed = squareform(dist_matrix, checks=False)
    Z = linkage(condensed, method="average")

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
    print(f"  [Dendrograma Jaccard] Árbol construido.")
    return tree


# ──────────────────────────────────────────────
# 5. EXPORTAR RESULTADOS
# ──────────────────────────────────────────────

def export_jaccard_results(
    df: pd.DataFrame,
    labels: np.ndarray,
    dendrogram_tree: dict,
    output_dir: Path,
    n_samples: int = 10,
):
    output_dir = Path(output_dir)

    # ── 1. clusters_jaccard.json ──
    print("  [Export] Generando clusters_jaccard.json...")
    records = []
    for idx, row in df.iterrows():
        records.append({
            "id": int(row["id"]),
            "comentario": row["comentario"],
            "cluster": int(labels[idx]),
        })
    with open(output_dir / "clusters_jaccard.json", "w", encoding="utf-8") as f:
        json.dump(records, f, ensure_ascii=False, separators=(",", ":"))
    print(f"    Guardado: clusters_jaccard.json ({len(records):,} registros)")

    # ── 2. clusters_agregados_jaccard.json ──
    print("  [Export] Generando clusters_agregados_jaccard.json...")
    agregados = {}
    unique_labels = sorted(set(labels))
    for lbl in unique_labels:
        mask = labels == lbl
        subset = df[mask]
        sample = subset["comentario"].head(n_samples).tolist()
        agregados[int(lbl)] = {
            "size": int(mask.sum()),
            "sample_comments": sample,
        }
    with open(output_dir / "clusters_agregados_jaccard.json", "w", encoding="utf-8") as f:
        json.dump(agregados, f, ensure_ascii=False, indent=2)
    print(f"    Guardado: clusters_agregados_jaccard.json")

    # ── 3. dendrograma_jaccard.json ──
    print("  [Export] Generando dendrograma_jaccard.json...")
    with open(output_dir / "dendrograma_jaccard.json", "w", encoding="utf-8") as f:
        json.dump(dendrogram_tree, f, ensure_ascii=False, indent=2)
    print(f"    Guardado: dendrograma_jaccard.json")


# ──────────────────────────────────────────────
# PIPELINE PRINCIPAL
# ──────────────────────────────────────────────

def run_jaccard_pipeline(
    df: pd.DataFrame,
    output_dir: Path,
    num_perm: int = 128,
    threshold: float = 0.3,
    checkpoint_dir: Path = None,  # NUEVO
):
    # Si no se especifica checkpoint_dir, usar output_dir/checkpoints_jaccard
    if checkpoint_dir is None:
        checkpoint_dir = output_dir / "checkpoints_jaccard"
    checkpoint_dir = Path(checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    texts = df["comentario_clean"].tolist()
    n = len(texts)

    # 1. MinHash + LSH (pasar checkpoint_dir)
    minhashes, lsh = build_minhash_lsh(
        texts,
        num_perm=num_perm,
        threshold=threshold,
        checkpoint_dir=checkpoint_dir,
    )

    # 2. Clustering (pasar checkpoint_dir)
    labels = cluster_from_lsh(lsh, minhashes, n, checkpoint_dir=checkpoint_dir)

    # 3. Dendrograma
    dendrogram_tree = build_jaccard_dendrogram(labels, minhashes)

    # 4. Exportar
    export_jaccard_results(
        df=df,
        labels=labels,
        dendrogram_tree=dendrogram_tree,
        output_dir=output_dir,
    )

    print("\n  ✅ Método Jaccard completado.")
    return labels