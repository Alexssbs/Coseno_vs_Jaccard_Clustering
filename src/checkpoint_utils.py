"""
Utilidades para guardar/cargar estado intermedio en el pipeline.
"""

import pickle
import json
from pathlib import Path
from typing import Any, Optional

import numpy as np


def save_numpy(data: np.ndarray, path: Path, name: str) -> None:
    """Guarda array numpy con .npy"""
    np.save(path / f"{name}.npy", data)
    print(f"  [Checkpoint] Guardado: {name}.npy")


def load_numpy(path: Path, name: str) -> Optional[np.ndarray]:
    """Carga array numpy si existe"""
    filepath = path / f"{name}.npy"
    if filepath.exists():
        data = np.load(filepath)
        print(f"  [Checkpoint] Cargado: {name}.npy (shape={data.shape})")
        return data
    return None


def save_pickle(obj: Any, path: Path, name: str) -> None:
    """Guarda objeto Python con pickle"""
    with open(path / f"{name}.pkl", "wb") as f:
        pickle.dump(obj, f, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"  [Checkpoint] Guardado: {name}.pkl")


def load_pickle(path: Path, name: str) -> Optional[Any]:
    """Carga objeto Python si existe"""
    filepath = path / f"{name}.pkl"
    if filepath.exists():
        with open(filepath, "rb") as f:
            obj = pickle.load(f)
        print(f"  [Checkpoint] Cargado: {name}.pkl")
        return obj
    return None


def save_faiss_index(index, path: Path, name: str) -> None:
    """Guarda índice FAISS"""
    import faiss
    faiss.write_index(index, str(path / f"{name}.faiss"))
    print(f"  [Checkpoint] Guardado: {name}.faiss")


def load_faiss_index(path: Path, name: str, dim: int = 384):
    """Carga índice FAISS si existe"""
    filepath = path / f"{name}.faiss"
    if filepath.exists():
        import faiss
        index = faiss.read_index(str(filepath))
        print(f"  [Checkpoint] Cargado: {name}.faiss")
        return index
    return None


def save_metadata(path: Path, metadata: dict) -> None:
    """Guarda metadatos del progreso"""
    with open(path / "checkpoint_metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)
    print(f"  [Checkpoint] Metadatos guardados")


def load_metadata(path: Path) -> Optional[dict]:
    """Carga metadatos si existen"""
    filepath = path / "checkpoint_metadata.json"
    if filepath.exists():
        with open(filepath, "r") as f:
            return json.load(f)
    return None