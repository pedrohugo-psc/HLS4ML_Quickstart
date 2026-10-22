# Tutorial 04: Redes Neurais Convolucionais (CNNs) e `io_stream`

Conversão de uma arquitetura convolucional 1D (`JetCNN1D`) em PyTorch para C++/HLS utilizando interface `io_stream` (buffers FIFO / AXI4-Stream e *Line Buffers*).

---

## Arquivos

- `4_cnns_tutorial.ipynb`: Notebook interativo para Jupyter Lab.
- `run_tutorial_04.py`: Script para execução direta via terminal.
- `curva_roc_cnn.png`: Curvas ROC comparando PyTorch (`float32`) e `hls4ml` (`ap_fixed<16,8>`).

---

## Como Executar

### Via Terminal
```bash
source .venv/bin/activate
python tutorials/04_cnns/run_tutorial_04.py
```

### Via Jupyter Lab
```bash
source .venv/bin/activate
jupyter lab tutorials/04_cnns/4_cnns_tutorial.ipynb
```

---

## Resultados de Referência

- **Acurácia PyTorch (`float32`)**: ~71.22%
- **Acurácia `hls4ml` (`ap_fixed<16,8>`)**: ~71.04%
- **Diferença de Acurácia**: 0.18%
