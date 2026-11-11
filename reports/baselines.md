# Lineas base - F1-macro en test

| Enfoque | F1-macro |
|---|---|
| Azar (clase mas frecuente) | 0.0315 |
| Agrupacion SAP actual (grupo_sap) | 0.7215 |
| Sin texto (precio, marca, UM) | 0.7370 |
| Solo texto_breve (TF-IDF + LogReg) | 0.9979 |
| Modelo del proyecto (RandomForest) | 0.9893 |

Test sin texto_breve repetido en train: 451 filas (49 quitadas), F1-macro = 0.9885
