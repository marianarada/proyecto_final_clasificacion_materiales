import logging
from dataclasses import dataclass, field
import pandas as pd

logger = logging.getLogger(__name__)

@dataclass
class ValidationReport:
    passed: bool = True
    errors: list = field(default_factory=list)
    warnings: list = field(default_factory=list)

    def fail(self, msg):
        self.passed = False
        self.errors.append(msg)

    def warn(self, msg):
        self.warnings.append(msg)

    def summary(self):
        status = "OK" if self.passed else "FALLIDO"
        lines = [f"=== Validacion: {status} ==="]
        for e in self.errors:
            lines.append(f"  ERROR: {e}")
        for w in self.warnings:
            lines.append(f"  WARN:  {w}")
        return "\n".join(lines)

def validate_dataset(df, config):
    report = ValidationReport()
    target_col = "familia_cod"
    if len(df) < 100:
        report.fail(f"Solo {len(df)} filas")
    required = ["texto_completo","marca","um","precio_usd","filas_me2m","items_ariba",target_col]
    missing = [c for c in required if c not in df.columns]
    if missing:
        report.fail(f"Columnas faltantes: {missing}")
    if target_col in df.columns:
        nulls = df[target_col].isnull().sum()
        if nulls > 0:
            report.fail(f"Target tiene {nulls} nulos")
        n = df[target_col].nunique()
        exp = config["target"]["n_classes"]
        if n != exp:
            report.warn(f"Se esperaban {exp} clases, hay {n}")
        for cls, pct in df[target_col].value_counts(normalize=True).items():
            if pct < 0.05:
                report.warn(f"Clase {cls!r} tiene solo {pct:.1%}")
    return report