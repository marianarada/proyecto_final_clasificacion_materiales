"""Preprocesador de features para la clasificacion de materiales.
- texto_completo -> TF-IDF
- marca, um -> OneHotEncoder
- precio_usd, filas_me2m, items_ariba -> StandardScaler
'grupo_sap' se excluye (noisy_label en config.yaml).
"""
from __future__ import annotations

from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

TEXT_COL = "texto_completo"
CATEGORICAL_COLS = ["marca", "um"]
NUMERIC_COLS = ["precio_usd", "filas_me2m", "items_ariba"]


def build_preprocessor(config: dict | None = None) -> ColumnTransformer:
    tfidf_cfg = (config or {}).get("tfidf", {})
    tfidf = TfidfVectorizer(
        max_features=tfidf_cfg.get("max_features", 5000),
        ngram_range=tuple(tfidf_cfg.get("ngram_range", [1, 2])),
        min_df=tfidf_cfg.get("min_df", 2),
        sublinear_tf=tfidf_cfg.get("sublinear_tf", True),
    )
    return ColumnTransformer(
        transformers=[
            ("text", tfidf, TEXT_COL),
            ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL_COLS),
            ("num", StandardScaler(), NUMERIC_COLS),
        ],
        remainder="drop",
        sparse_threshold=1.0,
    )
