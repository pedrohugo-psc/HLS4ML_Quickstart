<p align="center">
  <img src="https://raw.githubusercontent.com/fastmachinelearning/hls4ml/refs/heads/main/docs/img/hls4ml_logo.svg" alt="hls4ml logo" width="320"/>
</p>

# hls4ml: Síntese de Machine Learning para FPGAs

Repositório de configuração de ambiente, exemplos de referência (baseados no [hls4ml-tutorial](https://github.com/fastmachinelearning/hls4ml-tutorial)) e template para criação de projetos de inferência em FPGAs utilizando [`hls4ml`](https://github.com/fastmachinelearning/hls4ml).

---

## 1. Visão Geral: HLS e `hls4ml`

- **HLS (*High-Level Synthesis*):** Ferramentas de síntese (como *AMD/Xilinx Vivado HLS* e *Vitis HLS*) que traduzem especificações algorítmicas em **C/C++** para descrição de hardware em nível de transferência de registradores (**RTL: Verilog e VHDL**).
- **`hls4ml` (*High-Level Synthesis for Machine Learning*):** Pacote Python que converte modelos treinados em **PyTorch, Keras/TensorFlow, ONNX, Brevitas/QKeras** (e modelos baseados em árvores via **Conifer**) em projetos **C++/HLS** prontos para simulação e síntese em FPGA.

### Parâmetros Principais de Configuração
1. **`Precision` (Aritmética de Ponto Fixo — ex: `ap_fixed<16,6>`):** Define a representação numérica utilizada nas matrizes de pesos, acumuladores e ativações. Em `ap_fixed<W, I>`, `W` é a largura total em bits e `I` é o número de bits da parte inteira (incluindo o sinal). Exemplo: `ap_fixed<16,6>` aloca 6 bits inteiros e 10 bits fracionários.
2. **`ReuseFactor` (Fator de Reuso):** Controla a paralelização das operações aritméticas no hardware:
   - `ReuseFactor = 1`: Paralelismo máximo (cada multiplicação utiliza um bloco multiplicador dedicado; menor latência e maior consumo de DSPs/LUTs).
   - `ReuseFactor > 1` (ex: `4`, `8`, `16`): Compartilha cada multiplicador físico ao longo de `R` ciclos de clock, reduzindo a área ocupada no FPGA em troca de maior latência e intervalo de iniciação (*Initiation Interval*).
3. **`io_type` (Interface de Entrada/Saída):**
   - `io_parallel`: Entradas e saídas mapeadas em portas paralelas (indicado para redes densas/MLPs).
   - `io_stream`: Processamento em fluxo contínuo baseado em canais FIFO `hls::stream` / AXI4-Stream (indicado para CNNs e entradas de maior dimensão).

---

## 2. Instalação

A compilação e a simulação funcional de ponto fixo (*C-Simulation*) utilizam apenas o compilador `g++` no Linux (ou WSL2 no Windows). O *Vivado / Vitis HLS* só é necessário na etapa opcional de síntese RTL final.

### 2.1 Dependências do Sistema (Ubuntu/Debian)
```bash
sudo apt update
sudo apt install -y build-essential g++ python3 python3-venv python3-dev git
```

### 2.2 Configuração do Ambiente Virtual (`.venv`)
Na raiz do repositório, execute:
```bash
chmod +x scripts/setup_env.sh
./scripts/setup_env.sh
```
O script cria o ambiente `.venv` e instala os pacotes listados em `requirements.txt` (`hls4ml[profiling]`, `torch` CPU, `onnx`, `onnxoptimizer`, `brevitas`, `qonnx`, `conifer`, `scikit-learn` e `jupyterlab`).

### 2.3 Ativação e Verificação
Ative o ambiente virtual:
```bash
source .venv/bin/activate
```
Execute o script de verificação para testar a conversão e compilação C++ com `g++`:
```bash
python scripts/verify_install.py
```

---

## 3. Execução dos Tutoriais

Com o ambiente ativo (`source .venv/bin/activate`), os tutoriais podem ser executados diretamente via terminal ou pelo Jupyter Lab.

### Via Terminal
```bash
python tutorials/01_getting_started/run_tutorial_01.py
python tutorials/02_quantization/run_tutorial_02.py
python tutorials/03_advanced_config/run_tutorial_03.py
python tutorials/04_cnns/run_tutorial_04.py
python tutorials/05_autoencoders/run_tutorial_05.py
python tutorials/06_bdt_conifer/run_tutorial_06.py
```

### Via Jupyter Lab
```bash
jupyter lab
```

| Diretório | Escopo Técnico |
| :--- | :--- |
| `tutorials/01_getting_started/` | Conversão de MLP em PyTorch para HLS, compilação C++ com `g++` e validação em `ap_fixed<16,6>`. |
| `tutorials/02_quantization/` | Comparação entre *Post-Training Quantization* (PTQ) e *Quantization-Aware Training* (QAT) em 6 bits com Brevitas/QONNX. |
| `tutorials/03_advanced_config/` | Configuração de `ReuseFactor` (1, 4, 8), *profiling* numérico de tensores e precisão customizada por camada. |
| `tutorials/04_cnns/` | Síntese de CNN 1D com interface `io_stream` (*Line Buffers* e canais FIFO). |
| `tutorials/05_autoencoders/` | Autoencoder denso para detecção não-supervisionada de anomalias via erro de reconstrução (MSE). |
| `tutorials/06_bdt_conifer/` | Conversão de *Boosted Decision Trees* (`scikit-learn`) para hardware utilizando `conifer`. |

---

## 4. Como Criar um Novo Projeto

Para converter um modelo próprio fora dos tutoriais, utilize o script [`novo_projeto_template.py`](novo_projeto_template.py) na raiz do repositório como base:

```bash
source .venv/bin/activate
python novo_projeto_template.py
```

Fluxo padrão de conversão em Python:

```python
import numpy as np
import torch
import hls4ml

# 1. Modelo PyTorch em modo de avaliação
modelo.eval()

# 2. Geração do dicionário de configuração HLS (input_shape sem a dimensão de batch)
config = hls4ml.utils.config_from_pytorch_model(
    modelo,
    input_shape=(10,),
    granularity='model',
    backend='Vivado',                   # 'Vivado' ou 'Vitis'
    default_precision='ap_fixed<16,6>',
    default_reuse_factor=1
)

# 3. Conversão e escrita do projeto C++/HLS em disco
hls_model = hls4ml.converters.convert_from_pytorch_model(
    modelo,
    hls_config=config,
    output_dir='meu_projeto_hls',
    part='xcu200-fsgd2104-2-e',         # Part Number do FPGA alvo
    io_type='io_parallel'               # 'io_parallel' ou 'io_stream'
)
hls_model.write()

# 4. Compilação C++ com g++ e simulação bit-accurate em software
hls_model.compile()
y_hls = hls_model.predict(np.ascontiguousarray(X_teste, dtype=np.float32))

# 5. Síntese RTL para geração de Verilog/VHDL (requer Vivado/Vitis HLS instalado)
# hls_model.build(csim=False, synth=True, export=True)
```
*(Para modelos Keras ou ONNX, utilize `config_from_keras_model` / `convert_from_keras_model` ou `config_from_onnx_model` / `convert_from_onnx_model`).*

---

## 5. Estrutura dos Arquivos Gerados (C/C++ e Verilog/VHDL)

Os arquivos do projeto de hardware são gerados dentro do diretório configurado em `output_dir`:

### 5.1 Arquivos C/C++ (Gerados pelo Python em `hls_model.write()` / `compile()`)
- **`<output_dir>/firmware/myproject.cpp` e `myproject.h`**: Função *top-level* do circuito em C++ com as chamadas das camadas (`nnet::dense`, `nnet::conv_1d_cl`, `nnet::relu`, `nnet::softmax`) e diretivas `#pragma HLS`.
- **`<output_dir>/firmware/defines.h`**: Definições das dimensões de entrada/saída e dos tipos `ap_fixed` de cada camada.
- **`<output_dir>/firmware/parameters.h`**: Estruturas de configuração das camadas (`ReuseFactor`, tamanhos de tabelas de ativação e parâmetros de subcamadas).
- **`<output_dir>/firmware/weights/`**: Cabeçalhos `.h` (`w2.h`, `b2.h`, etc.) com os pesos e *biases* exportados como arrays constantes.
- **`<output_dir>/firmware/nnet_utils/` e `ap_types/`**: Templates C++ de camadas do `hls4ml` e tipos de ponto fixo (`ap_int.h`, `ap_fixed.h`).
- **`<output_dir>/myproject_test.cpp` e `myproject_bridge.cpp`**: *Testbench* C++ para o Vivado/Vitis HLS e interface C/C++ para chamadas via Python (`hls_model.predict()`).
- **`<output_dir>/build_prj.tcl` e `project.tcl`**: Scripts TCL para execução automatizada da síntese no Vivado/Vitis HLS.

### 5.2 Arquivos Verilog e VHDL (Gerados pelo Vivado / Vitis HLS em `hls_model.build()`)
Ao executar a síntese de hardware (`hls_model.build(synth=True)` no Python ou `vitis_hls -f build_prj.tcl` no terminal), o compilador da AMD/Xilinx gera a pasta `<output_dir>/myproject_prj/solution1/`:

- **Código RTL em Verilog (`.v`):** `<output_dir>/myproject_prj/solution1/syn/verilog/`
- **Código RTL em VHDL (`.vhd`):** `<output_dir>/myproject_prj/solution1/syn/vhdl/`
- **Relatórios de Síntese (`.rpt` / `.xml`):** `<output_dir>/myproject_prj/solution1/syn/report/myproject_csynth.rpt` (latência em ciclos de clock e utilização de BRAM, DSP, FF e LUT).
- **IP Core Exportado (`export=True`):** `<output_dir>/myproject_prj/solution1/impl/ip/`

```text
<output_dir>/
├── firmware/                              # [Gerado pelo Python - C/C++]
│   ├── myproject.cpp / myproject.h        # Função top-level C++/HLS
│   ├── defines.h / parameters.h           # Tipos ap_fixed e parâmetros das camadas
│   ├── weights/                           # Pesos e biases constantes (.h)
│   └── nnet_utils/ & ap_types/            # Bibliotecas C++ HLS e tipos de ponto fixo
├── myproject_test.cpp                     # Testbench C++
├── myproject_bridge.cpp                   # Wrapper C/Python para simulação C
├── build_prj.tcl / project.tcl            # Scripts TCL de síntese para Vivado/Vitis HLS
│
└── myproject_prj/solution1/               # [Gerado pelo Vivado/Vitis HLS após a síntese]
    ├── syn/verilog/                       # Arquivos Verilog (.v) gerados
    ├── syn/vhdl/                          # Arquivos VHDL (.vhd) gerados
    ├── syn/report/                        # Relatórios de latência e área (LUT, FF, DSP, BRAM)
    └── impl/ip/                           # Pacote IP Core exportado
```

---

## 6. Estrutura do Repositório

```text
.
├── README.md                    # Documentação principal
├── novo_projeto_template.py     # Script base independente para criação de novos projetos
├── requirements.txt             # Dependências Python do ambiente
├── scripts/
│   ├── setup_env.sh             # Script de criação do ambiente virtual (.venv) e instalação
│   └── verify_install.py        # Script de verificação da instalação e compilação C++
├── tutorials/                   # Scripts e notebooks dos 6 tutoriais
├── models.py                    # Módulos PyTorch utilizados exclusivamente pelos tutoriais
├── plotting.py                  # Funções de plotagem utilizadas exclusivamente pelos tutoriais
├── data/                        # Dataset (hls4ml_lhc_jets_hlf) utilizado nos tutoriais
└── models/                      # Pesos pré-treinados (.pt / .joblib) utilizados nos tutoriais
```

> **Nota:** Os arquivos `models.py`, `plotting.py` e as pastas `data/` e `models/` são módulos auxiliares do benchmark dos tutoriais (`hls4ml-tutorial`). Para criar um projeto próprio (como em `novo_projeto_template.py`), não é necessário utilizá-los.
