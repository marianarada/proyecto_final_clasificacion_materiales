"""Entrenamiento sin MLflow - compatible con restricciones AppLocker."""
from __future__ import annotations

import json
import time
import warnings
from pathlib import Path

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, f1_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier

from src.data.load_data import load_maestro
from src.features.build_features import build_preprocessor
from src.utils.config_loader import load_config

warnings.filterwarnings("ignore", category=UserWarning)


def _get_candidates(random_state: int = 42) -> dict:
    return {
        "LogisticRegression": LogisticRegression(
            max_iter=1000, C=1.0, class_weight="balanced",
            random_state=random_state, n_jobs=-1,
        ),
        "RandomForest": RandomForestClassifier(
            n_estimators=200, class_weight="balanced",
            random_state=random_state, n_jobs=-1,
        ),
        "XGBoost": XGBClassifier(
            n_estimators=200, max_depth=6, learning_rate=0.1,
            eval_metric="mlogloss", random_state=random_state,
            n_jobs=-1, verbosity=0,
        ),
    }


def make_splits(df, target_col, config):
    split_cfg = config.get("split", {})
    test_size    = split_cfg.get("test_size", 0.20)
    val_size     = split_cfg.get("val_size", 0.10)
    random_state = split_cfg.get("random_state", 42)

    y = df[target_col]
    df_tv, df_test, y_tv, y_test = train_test_split(
        df, y, test_size=test_size, random_state=random_state, stratify=y
    )
    val_ratio = val_size / (1 - test_size)
    df_train, df_val, y_train, y_val = train_test_split(
        df_tv, y_tv, test_size=val_ratio, random_state=random_state, stratify=y_tv
    )
    print(f"  Train: {len(df_train)}  Val: {len(df_val)}  Test: {len(df_test)}")
    return df_train, df_val, df_test, y_train, y_val, y_test


def train_one(name, estimator, df_train, df_val, y_train, y_val, config):
    preprocessor = build_preprocessor(config)
    pipeline = Pipeline([("preprocessor", preprocessor), ("clf", estimator)])

    t0 = time.time()
    pipeline.fit(df_train, y_train)
    train_time = time.time() - t0

    y_pred = pipeline.predict(df_val)
    f1_macro    = f1_score(y_val, y_pred, average="macro")
    f1_weighted = f1_score(y_val, y_pred, average="weighted")

    reports_dir = Path("reports")
    reports_dir.mkdir(exist_ok=True)
    report_path = reports_dir / f"val_report_{name}.txt"
    report_path.write_text(classification_report(y_val, y_pred), encoding="utf-8")

    print(f"  [{name}] F1-macro={f1_macro:.4f} | F1-weighted={f1_weighted:.4f} | {train_time:.1f}s")
    return pipeline, f1_macro, f1_weighted, train_time


def run_training(config=None):
    if config is None:
        config = load_config()

    print("Cargando datos...")
    df = load_maestro(config)

    print("\nGenerando splits...")
    df_train, df_val, df_test, y_train, y_val, y_test = make_splits(
        df, "familia_cod", config
    )

    candidates = _get_candidates(config.get("split", {}).get("random_state", 42))
    results = {}

    print(f"\nEntrenando {len(candidates)} modelos...\n")
    for name, estimator in candidates.items():
        print(f"--- {name} ---")
        pipeline, f1_val, f1_w, t = train_one(
            name, estimator, df_train, df_val, y_train, y_val, config
        )
        results[name] = {"pipeline": pipeline, "f1_val": f1_val,
                         "f1_weighted": f1_w, "train_time": t}

    # Campeon
    champion_name = max(results, key=lambda k: results[k]["f1_val"])
    champion_pipe = results[champion_name]["pipeline"]
    champion_f1   = results[champion_name]["f1_val"]

    print(f"\n{'='*40}")
    print(f"CAMPEON: {champion_name}  (F1-macro val = {champion_f1:.4f})")
    print(f"{'='*40}\n")

    # Evaluar en test
    y_pred_test = champion_pipe.predict(df_test)
    f1_test = f1_score(y_test, y_pred_test, average="macro")
    print(f"F1-macro TEST: {f1_test:.4f}")
    print("\nReporte test:\n")
    print(classification_report(y_test, y_pred_test))

    # Guardar reporte y metricas
    Path("reports").mkdir(exist_ok=True)
    Path("reports/test_report_champion.txt").write_text(
        classification_report(y_test, y_pred_test), encoding="utf-8"
    )

    metrics = {
        "champion": champion_name,
        "f1_macro_val":  round(champion_f1, 4),
        "f1_macro_test": round(f1_test, 4),
        "all_models": {
            k: {"f1_val": round(v["f1_val"], 4), "f1_weighted": round(v["f1_weighted"], 4)}
            for k, v in results.items()
        }
    }
    Path("reports/metrics_summary.json").write_text(
        json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"\nMetricas guardadas en reports/metrics_summary.json")
    return results


if __name__ == "__main__":
    run_training()