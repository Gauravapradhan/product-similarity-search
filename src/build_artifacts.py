import pandas as pd

from config import DATA_PATH
from similarity_engine import (
    ProductSimilarityEngine
)

df = pd.read_json(
    DATA_PATH,
    lines=True
)

engine = (
    ProductSimilarityEngine()
    .fit(df)
    .build_faiss_index()
)

engine.save("artifacts")

print("Artifacts generated successfully")