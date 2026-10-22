# Tutorial 05: Autoencoders para Detecção de Anomalias

Conversão de um Autoencoder denso (`16 -> 8 -> 3 -> 8 -> 16`) em PyTorch para C++/HLS voltado à detecção não-supervisionada de anomalias por Erro Médio Quadrático (MSE) de reconstrução.

---

## Arquivos

- `5_autoencoders_tutorial.ipynb`: Notebook interativo para Jupyter Lab.
- `run_tutorial_05.py`: Script para execução direta via terminal.
- `distribuicao_erro_reconstrucao.png`: Histograma do erro de reconstrução (MSE) em `ap_fixed<16,6>`.
- `curva_roc_autoencoder.png`: Curva ROC comparando PyTorch (`float32`) e `hls4ml` (`ap_fixed<16,6>`).

---

## Como Executar

### Via Terminal
```bash
source .venv/bin/activate
python tutorials/05_autoencoders/run_tutorial_05.py
```

### Via Jupyter Lab
```bash
source .venv/bin/activate
jupyter lab tutorials/05_autoencoders/5_autoencoders_tutorial.ipynb
```

---

## Resultados de Referência

- **AUC PyTorch (`float32`)**: 0.6782
- **AUC `hls4ml` (`ap_fixed<16,6>`)**: 0.6788
- **Diferença de AUC**: 0.0006
- **Erro Médio Absoluto de Reconstrução**: 0.01489
