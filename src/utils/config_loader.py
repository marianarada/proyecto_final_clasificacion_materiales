from pathlib import Path
import yaml

def _find_project_root():
    current = Path.cwd()
    for parent in [current, *current.parents]:
        if (parent / "pyproject.toml").exists():
            return parent
    return current

def load_config(config_path=None):
    if config_path is None:
        config_path = _find_project_root() / "configs" / "config.yaml"
    config_path = Path(config_path)
    if not config_path.exists():
        raise FileNotFoundError(f"No se encontro config en: {config_path}")
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)