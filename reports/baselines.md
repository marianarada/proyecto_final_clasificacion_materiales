# Lineas base - F1-macro en test

| Enfoque | F1-macro |
|---|---|
| Azar (clase mas frecuente) | 0.0315 |
| Agrupacion SAP actual (grupo_sap) | 0.7244 |
| Sin texto (precio, marca, UM) | 0.7289 |
| Solo texto_breve (TF-IDF + LogReg) | 0.9940 |
| Modelo del proyecto (RandomForest) | 0.9958 |

Test sin texto_breve repetido en train: 436 filas (64 quitadas), F1-macro = 0.9953
