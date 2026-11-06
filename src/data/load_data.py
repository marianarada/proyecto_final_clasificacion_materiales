from pathlib import Path
import pandas as pd
from unidecode import unidecode
from src.utils.config_loader import load_config

def _normalize_text(text) -> str:
    if not isinstance(text, str):
        return ""
    return " ".join(unidecode(text).lower().split())

def _combine_text(row, text_cols):
    parts = [_normalize_text(str(row[c])) for c in text_cols if c in row.index]
    return " | ".join(p for p in parts if p)

def load_maestro(config):
    raw_dir = Path(config["data"]["raw_dir"])
    filename = config["data"]["files"]["maestro"]
    path = raw_dir / filename
    sheet = config["data"]["sheets"]["maestro_materiales"]
    df = pd.read_excel(path, sheet_name=sheet)
    rename_map = {
        "Familia (c\u00f3digo)": "familia_cod",
        "Familia (codigo)": "familia_cod",
        "Categor\u00eda (c\u00f3digo)": "categoria_cod",
        "Subcategor\u00eda (c\u00f3digo)": "subcategoria_cod",
        "Texto breve": "texto_breve",
        "Texto largo": "texto_largo",
        "Marca": "marca",
        "UM": "um",
        "Precio base (USD)": "precio_usd",
        "Filas ME2M": "filas_me2m",
        "\u00cdtems en eventos Ariba": "items_ariba",
        "Grupo de art\u00edculos SAP actual": "grupo_sap",
        "Material": "material_id",
    }
    df = df.rename(columns=rename_map)
    keep = ["material_id","familia_cod","categoria_cod","subcategoria_cod",
            "texto_breve","texto_largo","marca","um",
            "precio_usd","filas_me2m","items_ariba","grupo_sap"]
    df = df[[c for c in keep if c in df.columns]].copy()
    df["texto_completo"] = df.apply(lambda r: _combine_text(r, ["texto_breve","texto_largo"]), axis=1)
    df["marca"] = df["marca"].fillna("SIN_MARCA").apply(_normalize_text)
    df["um"] = df["um"].apply(_normalize_text)
    df["grupo_sap"] = df["grupo_sap"].astype(str).replace("nan", "").apply(_normalize_text)
    df["precio_usd"] = pd.to_numeric(df["precio_usd"], errors="coerce").fillna(0.0)
    df["filas_me2m"] = pd.to_numeric(df["filas_me2m"], errors="coerce").fillna(0).astype(int)
    df["items_ariba"] = pd.to_numeric(df["items_ariba"], errors="coerce").fillna(0).astype(int)
    return df

def load_dataset(config_path=None):
    config = load_config(config_path)
    return load_maestro(config)

if __name__ == "__main__":
    df = load_dataset()
    print(f"Dataset cargado: {df.shape}")
    print(df["familia_cod"].value_counts())
    print(f"\nNulos: {df.isnull().sum().to_dict()}")
