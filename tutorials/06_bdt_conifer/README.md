# Tutorial 06: Boosted Decision Trees (BDTs) com Conifer

Conversão de um modelo `GradientBoostingClassifier` (`scikit-learn`) para C++/HLS em ponto fixo (`ap_fixed<18,8>`) utilizando o compilador `conifer`.

---

## Arquivos

- `6_bdt_conifer_tutorial.ipynb`: Notebook interativo para Jupyter Lab.
- `run_tutorial_06.py`: Script para execução direta via terminal.
- `curva_roc_bdt.png`: Curvas ROC comparando `scikit-learn` (`float64`) e `conifer` (`ap_fixed<18,8>`).

---

## Como Executar

### Via Terminal
```bash
source .venv/bin/activate
python tutorials/06_bdt_conifer/run_tutorial_06.py
```

### Via Jupyter Lab
```bash
source .venv/bin/activate
jupyter lab tutorials/06_bdt_conifer/6_bdt_conifer_tutorial.ipynb
```

---

## Resultados de Referência

- **Acurácia `scikit-learn` (`float64`)**: 91.87%
- **Acurácia `conifer` (`ap_fixed<18,8>`)**: 91.87%
- **AUC `scikit-learn` (`float64`)**: 0.9432
- **AUC `conifer` (`ap_fixed<18,8>`)**: 0.9432
