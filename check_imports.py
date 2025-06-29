print("--- Starting import check ---")

try:
    print("Importing torch...")
    import torch
    print("torch OK")

    print("Importing sentence_transformers...")
    from sentence_transformers import SentenceTransformer
    print("sentence_transformers OK")

    print("Importing faiss...")
    import faiss
    print("faiss OK")

    print("Importing numpy...")
    import numpy
    print("numpy OK")

    print("Importing jedi...")
    import jedi
    print("jedi OK")

    print("\n--- All imports successful ---")

except Exception as e:
    print(f"\n--- An error occurred ---")
    print(e)
    import traceback
    traceback.print_exc()
