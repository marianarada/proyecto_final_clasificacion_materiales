"""Graficas de evaluacion del modelo campeon (Fase 4, paso 5).

Lee el campeon de reports/metrics_summary.json, lo reentrena con los mismos
splits y semilla que src.models.train y guarda en reports/:
  - confusion_matrix_champion.png
  - f1_por_clase_champion.png
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import ConfusionMatrixDisplay, f1_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder

from src.data.load_data import load_maestro
from src.features.build_features import build_preprocessor
from src.models.train import _get_candidates, make_splits
from src.utils.config_loader import load_config

REPORTS = Path("reports")


def main():
    summary_path = REPORTS / "metrics_summary.json"
    if not summary_path.exists():
        raise FileNotFoundError("Primero ejecuta: python -m src.models.train")
    champion = json.loads(summary_path.read_text(encoding="utf-8"))["champion"]
    print(f"Campeon: {champion}")

    config = load_config()
    df = load_maestro(config)
    encoder = LabelEncoder()
    df["familia_cod"] = encoder.fit_transform(df["familia_cod"])
    classes = list(encoder.classes_)

    df_train, _, df_test, y_train, _, y_test = make_splits(df, "familia_cod", config)
    seed = config.get("split", {}).get("random_state", 42)
    estimator = _get_candidates(seed)[champion]
    pipe = Pipeline([("preprocessor", build_preprocessor(config)), ("clf", estimator)])
    pipe.fit(df_train, y_train)
    y_pred = pipe.predict(df_test)

    fig, ax = plt.subplots(figsize=(8, 7))
    ConfusionMatrixDisplay.from_predictions(
        y_test, y_pred, display_labels=classes, cmap="Blues", ax=ax, colorbar=False
    )
    ax.set_title(f"Matriz de confusion - {champion} (test)")
    fig.tight_layout()
    cm_path = REPORTS / "confusion_matrix_champion.png"
    fig.savefig(cm_path, dpi=150)
    plt.close(fig)

    f1s = f1_score(y_test, y_pred, average=None)
    order = np.argsort(f1s)
    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.barh([classes[i] for i in order], f1s[order], color="#2a6fdb")
    ax.bar_label(bars, fmt="%.3f", padding=3)
    ax.set_xlim(0, 1.08)
    ax.set_xlabel("F1-score")
    ax.set_title(f"F1 por clase - {champion} (test)")
    fig.tight_layout()
    f1_path = REPORTS / "f1_por_clase_champion.png"
    fig.savefig(f1_path, dpi=150)
    plt.close(fig)

    print(f"F1-macro test: {f1s.mean():.4f}")
    print(f"Guardado: {cm_path}")
    print(f"Guardado: {f1_path}")


if __name__ == "__main__":
    main()
