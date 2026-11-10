"""Lineas base para contextualizar el desempeno del modelo campeon.

Compara el campeon contra modelos simples sobre el mismo split de test y
evalua el campeon en un test "limpio" (sin texto_breve repetido en train).
Guarda reports/baselines.json y reports/baselines.md.
"""
from __future__ import annotations

import json
import warnings
from pathlib import Path

from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler

from src.data.load_data import load_maestro
from src.features.build_features import build_preprocessor
from src.models.train import _get_candidates, make_splits
from src.utils.config_loader import load_config

warnings.filterwarnings("ignore")
REPORTS = Path("reports")
NUM = ["precio_usd", "filas_me2m", "items_ariba"]


def _f1(pipe, df_train, y_train, df_eval, y_eval):
    pipe.fit(df_train, y_train)
    return round(f1_score(y_eval, pipe.predict(df_eval), average="macro"), 4)


def main():
    config = load_config()
    seed = config.get("split", {}).get("random_state", 42)
    df = load_maestro(config)
    df["familia_cod"] = LabelEncoder().fit_transform(df["familia_cod"])
    df_train, _, df_test, y_train, _, y_test = make_splits(df, "familia_cod", config)

    summary_path = REPORTS / "metrics_summary.json"
    champion = (
        json.loads(summary_path.read_text(encoding="utf-8"))["champion"]
        if summary_path.exists() else "RandomForest"
    )

    baselines = {
        "Azar (clase mas frecuente)": Pipeline([
            ("p", ColumnTransformer([("n", "passthrough", ["precio_usd"])])),
            ("c", DummyClassifier(strategy="most_frequent")),
        ]),
        "Agrupacion SAP actual (grupo_sap)": Pipeline([
            ("p", ColumnTransformer([("g", OneHotEncoder(handle_unknown="ignore"), ["grupo_sap"])])),
            ("c", LogisticRegression(max_iter=1000)),
        ]),
        "Sin texto (precio, marca, UM)": Pipeline([
            ("p", ColumnTransformer([
                ("cat", OneHotEncoder(handle_unknown="ignore"), ["marca", "um"]),
                ("num", StandardScaler(), NUM),
            ])),
            ("c", RandomForestClassifier(n_estimators=200, random_state=seed, n_jobs=-1)),
        ]),
        "Solo texto_breve (TF-IDF + LogReg)": Pipeline([
            ("p", ColumnTransformer([("t", TfidfVectorizer(), "texto_breve")])),
            ("c", LogisticRegression(max_iter=1000)),
        ]),
    }

    results = {}
    for name, pipe in baselines.items():
        results[name] = _f1(pipe, df_train, y_train, df_test, y_test)
        print(f"  {name:<40} F1-macro test = {results[name]:.4f}")

    def champion_pipe():
        return Pipeline([
            ("preprocessor", build_preprocessor(config)),
            ("clf", _get_candidates(seed)[champion]),
        ])

    champ_name = f"Modelo del proyecto ({champion})"
    results[champ_name] = _f1(champion_pipe(), df_train, y_train, df_test, y_test)
    print(f"  {champ_name:<40} F1-macro test = {results[champ_name]:.4f}")

    mask = ~df_test["texto_breve"].isin(set(df_train["texto_breve"]))
    n_dup = int((~mask).sum())
    clean_f1 = _f1(champion_pipe(), df_train, y_train, df_test[mask], y_test[mask])
    print(f"\n  Test sin duplicados de train ({int(mask.sum())} de {len(df_test)} filas, "
          f"{n_dup} quitadas): F1-macro = {clean_f1:.4f}")

    REPORTS.mkdir(exist_ok=True)
    out = {
        "f1_macro_test": results,
        "test_sin_duplicados": {
            "filas": int(mask.sum()), "duplicados_quitados": n_dup, "f1_macro": clean_f1,
        },
    }
    (REPORTS / "baselines.json").write_text(
        json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    lines = ["# Lineas base - F1-macro en test", "", "| Enfoque | F1-macro |", "|---|---|"]
    lines += [f"| {k} | {v:.4f} |" for k, v in results.items()]
    lines += [
        "", f"Test sin texto_breve repetido en train: {int(mask.sum())} filas "
        f"({n_dup} quitadas), F1-macro = {clean_f1:.4f}", "",
    ]
    (REPORTS / "baselines.md").write_text("\n".join(lines), encoding="utf-8")
    print("\nGuardado: reports\\baselines.json y reports\\baselines.md")


if __name__ == "__main__":
    main()
