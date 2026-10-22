# Tutorial 02: Quantização (PTQ vs QAT com Brevitas)

Comparação entre *Post-Training Quantization* (PTQ) direto no `hls4ml` e *Quantization-Aware Training* (QAT) em 6 bits utilizando Brevitas e QONNX.

---

## Arquivos

- `2_quantization_tutorial.ipynb`: Notebook interativo para Jupyter Lab.
- `run_tutorial_02.py`: Script para execução direta via terminal.
- `curva_roc_quantizacao.png`: Curvas ROC comparando `float32`, PTQ (`ap_fixed<6,2>`) e QAT (6 bits).

---

## Como Executar

### Via Terminal
```bash
source .venv/bin/activate
python tutorials/02_quantization/run_tutorial_02.py
```

### Via Jupyter Lab
```bash
source .venv/bin/activate
jupyter lab tutorials/02_quantization/2_quantization_tutorial.ipynb
```

---

## Resultados de Referência

| Configuração | Largura de Bits | Acurácia |
| :--- | :--- | :--- |
| PyTorch Baseline | `float32` (32 bits) | ~72.9% |
| PTQ `ap_fixed<16,6>` | 16 bits | ~72.7% |
| PTQ `ap_fixed<8,3>` | 8 bits | ~44.0% |
| PTQ `ap_fixed<6,2>` | 6 bits (PTQ) | ~19.6% |
| QAT Brevitas (`hls4ml`) | 6 bits (QAT) | ~71.8% |
