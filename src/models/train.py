"""Entrenamiento y seleccion del modelo campeon (familia de material).

Flujo: carga -> validacion -> split estratificado por grupos -> entrenamiento
de candidatos -> seleccion por F1-macro en validacion -> evaluacion en test.
"""
from __future__ import annotations

import json
import time
import warnings
from pathlib import Path

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, f1_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier

from src.data.load_data import load_maestro
from src.data.validate import validate_dataset
from src.features.build_features import build_preprocessor
from src.utils.config_loader import load_config

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

TARGET = "familia_cod"
REPORTS = Path("reports")


def _get_candidates(random_state: int = 42) -> dict:
    return {
        "LogisticRegression": LogisticRegression(
            max_iter=1000, C=1.0, class_weight="balanced", random_state=random_state,
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


def _group_fold(df, y, groups, n_splits, random_state):
    """Devuelve (idx_resto, idx_fold) del primer fold estratificado por grupos."""
    sgkf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    return next(sgkf.split(df, y, groups))


def make_splits(df, target_col, config):
    """Split train/val/test estratificado y sin compartir grupos entre conjuntos.

    Si existe `grupo_id`, un material y sus duplicados quedan siempre en el
    mismo conjunto. Proporciones aproximadas: test_size y val_size del config.
    """
    split_cfg = config.get("split", {})
    test_size = split_cfg.get("test_size", 0.20)
    val_size = split_cfg.get("val_size", 0.10)
    random_state = split_cfg.get("random_state", 42)

    df = df.reset_index(drop=True)
    groups = df["grupo_id"] if "grupo_id" in df.columns else df.index.to_series()
    y = df[target_col]

    idx_tv, idx_test = _group_fold(df, y, groups, round(1 / test_size), random_state)
    df_tv, df_test = df.iloc[idx_tv].reset_index(drop=True), df.iloc[idx_test]

    n_val = round((1 - test_size) / val_size)
    idx_train, idx_val = _group_fold(
        df_tv, df_tv[target_col], groups.iloc[idx_tv].reset_index(drop=True), n_val, random_state
    )
    df_train, df_val = df_tv.iloc[idx_train], df_tv.iloc[idx_val]

    if "grupo_id" in df.columns:
        g_tr, g_va, g_te = (set(d["grupo_id"]) for d in (df_train, df_val, df_test))
        assert not (g_tr & g_te or g_tr & g_va or g_va & g_te), "Grupos compartidos entre splits"

    print(f"  Train: {len(df_train)}  Val: {len(df_val)}  Test: {len(df_test)}"
          f"  (sin grupos compartidos)")
    return (df_train, df_val, df_test,
            df_train[target_col], df_val[target_col], df_test[target_col])


def train_one(name, estimator, df_train, df_val, y_train, y_val, config, class_names):
    pipeline = Pipeline([("preprocessor", build_preprocessor(config)), ("clf", estimator)])
    t0 = time.time()
    pipeline.fit(df_train, y_train)
    train_time = time.time() - t0

    y_pred = pipeline.predict(df_val)
    f1_macro = f1_score(y_val, y_pred, average="macro")
    f1_weighted = f1_score(y_val, y_pred, average="weighted")

    REPORTS.mkdir(exist_ok=True)
    (REPORTS / f"val_report_{name}.txt").write_text(
        classification_report(y_val, y_pred, target_names=class_names), encoding="utf-8"
    )
    print(f"  [{name}] F1-macro={f1_macro:.4f} | F1-weighted={f1_weighted:.4f} | {train_time:.1f}s")
    return pipeline, f1_macro, f1_weighted, train_time


def run_training(config=None):
    config = config or load_config()

    print("Cargando datos...")
    df = load_maestro(config)

    print("\nValidando datos...")
    report = validate_dataset(df, config)
    print(report.summary())
    if not report.passed:
        raise ValueError("La validacion de datos fallo; revisa los errores arriba.")

    # XGBoost exige etiquetas numericas: se codifican para todos los modelos
    encoder = LabelEncoder()
    df[TARGET] = encoder.fit_transform(df[TARGET])
    class_names = list(encoder.classes_)

    print("\nGenerando splits...")
    df_train, df_val, df_test, y_train, y_val, y_test = make_splits(df, TARGET, config)

    candidates = _get_candidates(config.get("split", {}).get("random_state", 42))
    results = {}
    print(f"\nEntrenando {len(candidates)} modelos...\n")
    for name, estimator in candidates.items():
        print(f"--- {name} ---")
        pipe, f1_val, f1_w, t = train_one(
            name, estimator, df_train, df_val, y_train, y_val, config, class_names
        )
        results[name] = {"pipeline": pipe, "f1_val": f1_val, "f1_weighted": f1_w, "train_time": t}

    champion = max(results, key=lambda k: results[k]["f1_val"])
    champion_pipe = results[champion]["pipeline"]
    print(f"\n{'=' * 40}\nCAMPEON: {champion}  (F1-macro val = {results[champion]['f1_val']:.4f})"
          f"\n{'=' * 40}\n")

    y_pred_test = champion_pipe.predict(df_test)
    f1_test = f1_score(y_test, y_pred_test, average="macro")
    report_test = classification_report(y_test, y_pred_test, target_names=class_names)
    print(f"F1-macro TEST: {f1_test:.4f}\n\nReporte test:\n\n{report_test}")

    (REPORTS / "test_report_champion.txt").write_text(report_test, encoding="utf-8")
    metrics = {
        "champion": champion,
        "f1_macro_val": round(results[champion]["f1_val"], 4),
        "f1_macro_test": round(f1_test, 4),
        "classes": class_names,
        "split": {"train": len(df_train), "val": len(df_val), "test": len(df_test),
                  "agrupado_por": "grupo_id"},
        "all_models": {
            k: {"f1_val": round(v["f1_val"], 4), "f1_weighted": round(v["f1_weighted"], 4)}
            for k, v in results.items()
        },
    }
    (REPORTS / "metrics_summary.json").write_text(
        json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print("\nMetricas guardadas en reports/metrics_summary.json")
    return results


if __name__ == "__main__":
    run_training()
