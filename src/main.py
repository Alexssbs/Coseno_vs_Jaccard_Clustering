#!/usr/bin/env python3
"""
Pipeline principal de clustering de comentarios.
Ejecuta ambos métodos (Coseno + Jaccard) y exporta JSONs para D3.js
"""

import sys
from pathlib import Path
from google.colab import drive  # <-- AGREGAR ESTA LÍNEA

def main():

    # 🔧 CONFIGURACIÓN FIJA (modifica aquí tus rutas y parámetros)
    class Args:
        input_csv = "comentarios_solo_texto.csv"   # ← CAMBIA ESTA RUTA
        output_dir = "output"
        method = "ambos"  # "coseno", "jaccard", "ambos"
        batch_size = 512
        min_cluster_size = 20   
        top_k = 30
        num_perm = 128
        lsh_threshold = 0.3

    args = Args()

    output_dir = Path(args.output_dir)
    
    # ========== AGREGAR ESTAS 6 LÍNEAS ==========
    # Montar Google Drive (solo pregunta contraseña la primera vez)
    drive.mount('/content/drive')
    
    # Redirigir todo a Drive (cambia "MiCarpeta" por lo que quieras)
    drive_base = Path("/content/drive/MyDrive/clustering_output")
    output_dir = drive_base / args.output_dir
    # ============================================

    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("  PIPELINE DE CLUSTERING DE COMENTARIOS")
    print("=" * 60)

    # Cargar y limpiar datos
    from preprocessing import load_and_clean
    print(f"\n[1/5] Cargando dataset: {args.input_csv}")
    df = load_and_clean(args.input_csv)
    print(f"      Filas originales totales procesadas: {len(df)}")

    if args.method in ("coseno", "ambos"):
        print("\n" + "=" * 60)
        print("  MÉTODO 1: SIMILITUD DE COSENO (Embeddings + FAISS + HDBSCAN)")
        print("=" * 60)
        from method_coseno import run_coseno_pipeline
        run_coseno_pipeline(
            df=df,
            output_dir=output_dir,
            batch_size=args.batch_size,
            top_k=args.top_k,
            min_cluster_size=args.min_cluster_size,
            checkpoint_dir=output_dir / "checkpoints_coseno",  # NUEVO
        )

    if args.method in ("jaccard", "ambos"):
        print("\n" + "=" * 60)
        print("  MÉTODO 2: SIMILITUD DE JACCARD (MinHash + LSH)")
        print("=" * 60)
        from method_jaccard import run_jaccard_pipeline
        run_jaccard_pipeline(
            df=df,
            output_dir=output_dir,
            num_perm=args.num_perm,
            threshold=args.lsh_threshold,
            checkpoint_dir=output_dir / "checkpoints_jaccard",  # NUEVO
        )

    print("\n" + "=" * 60)
    print("  ✅ PIPELINE COMPLETADO")
    print(f"  Archivos exportados en: {output_dir.resolve()}")
    print("=" * 60)


if __name__ == "__main__":
    main()