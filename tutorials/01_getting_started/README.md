# Tutorial 01: Getting Started com hls4ml e PyTorch

Conversão de uma rede MLP (`JetTagger`) treinada em PyTorch para C++/HLS utilizando o dataset `hls4ml_lhc_jets_hlf`.

---

## Arquivos

- `1_getting_started_tutorial.ipynb`: Notebook interativo para Jupyter Lab.
- `run_tutorial_01.py`: Script para execução direta via terminal.
- `models.py` e `plotting.py`: Definição do modelo PyTorch e função de plotagem ROC.
- `curva_roc_comparativa.png`: Curvas ROC comparando PyTorch (`float32`) e `hls4ml` (`ap_fixed<16,6>`).

---

## Como Executar

### Via Terminal
```bash
source .venv/bin/activate
python tutorials/01_getting_started/run_tutorial_01.py
```

### Via Jupyter Lab
```bash
source .venv/bin/activate
jupyter lab tutorials/01_getting_started/1_getting_started_tutorial.ipynb
```

---

## Resultados de Referência

- **Acurácia PyTorch (`float32`)**: ~72.6% – 73.4%
- **Acurácia `hls4ml` (`ap_fixed<16,6>`)**: ~72.5% – 73.3%
- **Diferença de Acurácia**: < 0.2%
