# Tutoriais de Referência (`hls4ml`)

Diretório com scripts executáveis e notebooks Jupyter baseados no repositório oficial [`hls4ml-tutorial`](https://github.com/fastmachinelearning/hls4ml-tutorial).

---

## Lista de Tutoriais

### [Tutorial 01: Getting Started](01_getting_started/README.md)
- **Modelo**: MLP de 3 camadas (`JetTagger`) em PyTorch para classificação do dataset `hls4ml_lhc_jets_hlf`.
- **Escopo**: Treinamento, conversão para C++/HLS, compilação com `g++`, avaliação da precisão `ap_fixed<16,6>` e curvas ROC.

### [Tutorial 02: Quantização (PTQ vs QAT)](02_quantization/README.md)
- **Modelo**: `JetTagger` (PTQ) e `JetTaggerBrevitas` (QAT em 6 bits).
- **Escopo**: Avaliação de *Post-Training Quantization* (`ap_fixed<16,6>`, `ap_fixed<8,3>`, `ap_fixed<6,2>`) vs *Quantization-Aware Training* com Brevitas e exportação QONNX.

### [Tutorial 03: Configuração Avançada e Profiling](03_advanced_config/README.md)
- **Modelo**: `JetTagger` em PyTorch.
- **Escopo**: Configuração de `ReuseFactor` (1, 4, 8), *profiling* numérico de pesos/ativações (`hls4ml.model.profiling.numerical`) e precisão heterogênea por camada (`granularity='name'`).

### [Tutorial 04: Redes Neurais Convolucionais (CNNs)](04_cnns/README.md)
- **Modelo**: `JetCNN1D` (`Conv1d` + `ReLU` + `MaxPool1d` + `Linear`) em PyTorch.
- **Escopo**: Síntese com interface `io_stream` (buffers FIFO / AXI4-Stream e *Line Buffers*) e dimensionamento em `ap_fixed<16,8>`.

### [Tutorial 05: Autoencoders](05_autoencoders/README.md)
- **Modelo**: `JetAutoencoder` ($16 \to 8 \to 3 \to 8 \to 16$) em PyTorch.
- **Escopo**: Treinamento não-supervisionado e detecção de anomalias via Erro Médio Quadrático (MSE) de reconstrução em `ap_fixed<16,6>`.

### [Tutorial 06: Boosted Decision Trees com Conifer](06_bdt_conifer/README.md)
- **Modelo**: `GradientBoostingClassifier` (`scikit-learn`).
- **Escopo**: Conversão de ensembles de árvores de decisão para C++/HLS em `ap_fixed<18,8>` utilizando `conifer`.
