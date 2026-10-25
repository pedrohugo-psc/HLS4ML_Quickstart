<p align="center">
  <img src="https://raw.githubusercontent.com/fastmachinelearning/hls4ml/refs/heads/main/docs/img/hls4ml_logo.svg" alt="hls4ml logo" width="320"/>
</p>

# hls4ml Quickstart: Ambiente e Tutoriais Oficiais

Repositório de automação de ambiente Linux para síntese de *Machine Learning* em FPGAs, configurado diretamente sobre os clones oficiais e intocados de [`fastmachinelearning/hls4ml`](https://github.com/fastmachinelearning/hls4ml) e [`fastmachinelearning/hls4ml-tutorial`](https://github.com/fastmachinelearning/hls4ml-tutorial), além de um template independente para criação de novos projetos.

---

## 1. Visão Geral

- **`hls4ml/` (Clone oficial):** Biblioteca Python que traduz modelos treinados em **Keras/QKeras, PyTorch, Brevitas/QONNX, snnTorch, ONNX e Regressão Simbólica (PySR)** (além de árvores de decisão via **Conifer**) em projetos **C++/HLS** prontos para simulação bit-accurate e síntese RTL (Verilog/VHDL).
- **`hls4ml-tutorial/` (Clone oficial):** Conjunto original de **11 notebooks interativos** mantidos pela equipe do `fastmachinelearning`, cobrindo desde redes densas básicas até quantização em 6 bits, *profiling* numérico, CNNs com `io_stream`, redes *spiking* (SNN), *Boosted Decision Trees* e regressão simbólica.
- **Isolamento na `.venv`:** O script `scripts/setup_env.sh` provisiona automaticamente **Python 3.12** isolado, instala todas as dependências em modo compatível, pré-baixa o dataset `hls4ml_lhc_jets_hlf` com retomada automática de rede e configura shims de compatibilidade exclusivamente dentro da `.venv` — sem modificar nenhum arquivo dos repositórios oficiais clonados.

---

## 2. Instalação (Linux — Ubuntu/Debian)

A compilação e a simulação funcional em ponto fixo (*C-Simulation*, `hls_model.compile()` + `predict()`) utilizam apenas o compilador `g++`. O *AMD Vitis HLS / Vivado* só é necessário caso deseje executar as células finais opcionais de síntese RTL (`hls_model.build()`).

### 2.1 Dependências do Sistema
```bash
sudo apt update
sudo apt install -y build-essential g++ python3 git curl graphviz
```

### 2.2 Clonar e Configurar o Ambiente
```bash
git clone https://github.com/pedrohugo-psc/HSL4ML_Quickstart.git
cd HSL4ML_Quickstart
./scripts/setup_env.sh
```
O script `setup_env.sh` realiza automaticamente:
1. Clonagem dos repositórios oficiais `./hls4ml` e `./hls4ml-tutorial`.
2. Criação do ambiente `.venv` (Python 3.12) e instalação editável do `./hls4ml` junto a TensorFlow/Keras, QKerasV3, PyTorch, Brevitas, QONNX, snnTorch, Conifer, XGBoost e PySR/Julia.
3. Download resiliente e preparação do dataset `hls4ml_lhc_jets_hlf` em `hls4ml-tutorial/data/jet-tagging/`.

### 2.3 Ativação e Verificação
```bash
source .venv/bin/activate
python scripts/verify_install.py
```

---

## 3. Execução dos Tutoriais Oficiais (`hls4ml-tutorial/`)

Com o ambiente ativo (`source .venv/bin/activate`), inicie o Jupyter Lab:
```bash
jupyter lab
```
No navegador, abra a pasta **`hls4ml-tutorial/`** para acessar os 11 notebooks oficiais exatamente como no repositório original:

| Notebook Oficial (`hls4ml-tutorial/`) | Escopo Técnico | Pré-requisito |
| :--- | :--- | :--- |
| `1_getting_started/1a_train_keras.ipynb` | Treinamento do classificador MLP em Keras/TensorFlow (`keras_model_part1.h5`). | — |
| `1_getting_started/1b_train_pytorch.ipynb` | Treinamento do classificador MLP em PyTorch (`pytorch_weights_part1.pt`). | — |
| `1_getting_started/1c_hls4ml_synth.ipynb` | Conversão do MLP (`MODEL_TYPE = 'keras'` ou `'pytorch'`) para C++/HLS e validação `ap_fixed<16,6>`. | `1a` ou `1b` |
| `2_quantization/2a_qkeras.ipynb` | *Quantization-Aware Training* (QAT) em 6 bits com **QKeras** e conversão HLS. | `1a` |
| `2_quantization/2b_brevitas.ipynb` | *Quantization-Aware Training* (QAT) em 6 bits com **Brevitas** exportado via **QONNX**. | `1b` |
| `3_advanced_config/3a_reuse_factor.ipynb` | Configuração de `ReuseFactor` e impacto de paralelismo vs área (`keras` ou `pytorch`). | `1a`/`1b` (+ `1c` p/ relatório RTL) |
| `3_advanced_config/3b_profiling.ipynb` | *Profiling* numérico de pesos/ativações, `Trace=True` e precisão heterogênea por camada. | `1a` ou `1b` |
| `4_advanced_models/4a_qkeras_cnn_svhn.ipynb` | CNN 2D quantizada (QKeras) no dataset **SVHN** com interface `io_stream`. | — |
| `4_advanced_models/4b_snn.ipynb` | Rede Neural *Spiking* (**snnTorch**) com `io_stream` e janela temporal em ponto fixo. | — |
| `6_more_models/6a_bdt.ipynb` | *Boosted Decision Trees* (**XGBoost**) convertidas para C++/HLS via **Conifer**. | `1a` ou `1b` |
| `6_more_models/6b_symbolic_regression.ipynb` | Descoberta de expressões analíticas (**PySR**) e síntese C++/HLS direta e via LUTs. | — |

> **Observações importantes ao executar os notebooks:**
> - **Escolha de framework (`MODEL_TYPE`):** Nos notebooks `1c`, `3a`, `3b` e `6a`, a primeira célula define `MODEL_TYPE = 'keras'` (requer ter rodado `1a`) ou `MODEL_TYPE = 'pytorch'` (requer ter rodado `1b`).
> - **Simulação C++ (`g++`) vs Síntese RTL (`Vitis HLS`):** Todas as células até `hls_model.compile()`, `hls_model.predict()` e gráficos ROC funcionam apenas com o compilador livre `g++`. As células finais da seção *Synthesize* (`hls_model.build(...)` e `read_vivado_report(...)`) invocam o sintetizador proprietário *AMD Vitis HLS* (`vitis-run`) caso esteja instalado no `PATH`. Em computadores pessoais e ambientes de estudo sem o Vitis HLS instalado (~100 GB), é padrão e esperado rodar até as células de predição e curvas ROC (que já validam o circuito bit-accurate em ponto fixo) e pular a célula final de síntese RTL.

---

## 4. Como Criar um Novo Projeto Independente

Para converter um modelo próprio fora da pasta `hls4ml-tutorial/`, utilize o script [`novo_projeto_template.py`](novo_projeto_template.py) na raiz do repositório:

```bash
source .venv/bin/activate
python novo_projeto_template.py
```

Fluxo mínimo de conversão em Python:
```python
import numpy as np
import torch
import hls4ml

modelo.eval()

config = hls4ml.utils.config_from_pytorch_model(
    modelo,
    input_shape=(10,),
    granularity='model',
    backend='Vitis',                    # 'Vitis' ou 'Vivado'
    default_precision='ap_fixed<16,6>',
    default_reuse_factor=1,
)

hls_model = hls4ml.converters.convert_from_pytorch_model(
    modelo,
    hls_config=config,
    backend='Vitis',
    output_dir='meu_projeto_hls',
    part='xcu200-fsgd2104-2-e',
    io_type='io_parallel',              # 'io_parallel' ou 'io_stream'
)
hls_model.write()

# Compilação C++ com g++ e simulação bit-accurate em software
hls_model.compile()
y_hls = hls_model.predict(np.ascontiguousarray(X_teste, dtype=np.float32))

# Síntese RTL para geração de Verilog/VHDL (requer Vitis/Vivado HLS instalado)
# hls_model.build(csim=False, synth=True, export=True)
```

---

## 5. Estrutura dos Arquivos Gerados (C/C++ e Verilog/VHDL)

```text
<output_dir>/
├── firmware/                              # [Gerado pelo Python em write() / compile()]
│   ├── myproject.cpp / myproject.h        # Função top-level C++/HLS com diretivas #pragma HLS
│   ├── defines.h / parameters.h           # Tipos ap_fixed e parâmetros das camadas
│   ├── weights/                           # Pesos e biases constantes (.h)
│   └── nnet_utils/ & ap_types/            # Templates C++ HLS e tipos de ponto fixo
├── myproject_test.cpp / myproject_bridge.cpp
├── build_prj.tcl / project.tcl            # Scripts TCL de síntese para Vitis/Vivado HLS
│
└── myproject_prj/solution1/               # [Gerado pelo Vitis/Vivado HLS em build(synth=True)]
    ├── syn/verilog/                       # Arquivos Verilog (.v)
    ├── syn/vhdl/                          # Arquivos VHDL (.vhd)
    ├── syn/report/                        # Relatórios de latência e área (LUT, FF, DSP, BRAM)
    └── impl/ip/                           # Pacote IP Core exportado
```

---

## 6. Estrutura do Repositório

```text
.
├── README.md                    # Documentação principal
├── requirements.txt             # Dependências Python para os 11 tutoriais oficiais
├── novo_projeto_template.py     # Template independente para criação de novos projetos
├── scripts/
│   ├── setup_env.sh             # Clona hls4ml + hls4ml-tutorial, cria .venv (Python 3.12) e baixa dados
│   └── verify_install.py        # Verifica integridade dos clones, pacotes e compilação C++
├── hls4ml/                      # [Clonado por setup_env.sh] Repositório oficial fastmachinelearning/hls4ml
└── hls4ml-tutorial/             # [Clonado por setup_env.sh] Repositório oficial fastmachinelearning/hls4ml-tutorial
```
