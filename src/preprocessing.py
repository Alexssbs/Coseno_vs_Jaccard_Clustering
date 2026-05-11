"""
Módulo de limpieza y preprocesamiento de texto.
Aplicado de igual manera en ambos métodos.
"""

import re
import pandas as pd


def clean_text(text: str) -> str:
    """
    Limpieza básica de texto:
    - lowercase
    - eliminar espacios extra
    Sin stemming ni lematización.
    """
    if not isinstance(text, str):
        return ""
    text = text.lower().strip()
    text = re.sub(r'\s+', ' ', text)
    return text


def is_valid(text: str) -> bool:
    """
    Filtra textos basura:
    - longitud < 2
    - solo números
    - solo espacios/vacíos
    """
    if len(text) < 2:
        return False
    if re.fullmatch(r'\d+', text):
        return False
    return True


def load_and_clean(csv_path: str) -> pd.DataFrame:
    """
    Carga el CSV, limpia la columna 'comentario' y filtra filas inválidas.
    Devuelve DataFrame con índice original preservado y columna 'comentario_clean'.
    """
    print(f"  Leyendo archivo CSV...")
    df = pd.read_csv(
        csv_path,
        usecols=["comentario"],
        dtype={"comentario": str},
        keep_default_na=False,
        low_memory=False,
    )

    total_original = len(df)
    print(f"  Filas leídas: {total_original:,}")

    df["comentario_clean"] = df["comentario"].apply(clean_text)
    mask_valid = df["comentario_clean"].apply(is_valid)
    df = df[mask_valid].copy()
    df = df.reset_index(drop=True)
    df["id"] = df.index

    n_filtradas = total_original - len(df)
    print(f"  Filas filtradas (basura): {n_filtradas:,}")
    print(f"  Filas válidas para clustering: {len(df):,}")

    return df
