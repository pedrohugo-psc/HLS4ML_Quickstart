# Tutorial 03: Configuração Avançada (`ReuseFactor` e Profiling)

Configuração do parâmetro `ReuseFactor`, análise de distribuição numérica de tensores (`hls4ml.model.profiling.numerical`) e definição de precisão heterogênea por camada (`granularity='name'`).

---

## Arquivos

- `3_advanced_config_tutorial.ipynb`: Notebook interativo para Jupyter Lab.
- `run_tutorial_03.py`: Script para execução direta via terminal.
- `profiling_pesos_ativacoes.png`: Distribuição de pesos e ativações por camada.

---

## Como Executar

### Via Terminal
```bash
source .venv/bin/activate
python tutorials/03_advanced_config/run_tutorial_03.py
```

### Via Jupyter Lab
```bash
source .venv/bin/activate
jupyter lab tutorials/03_advanced_config/3_advanced_config_tutorial.ipynb
```

---

## Resultados de Referência

- `ReuseFactor = 1`, `4` e `8` produzem exatamente a mesma acurácia numérica (~72.7%), alterando apenas o número de multiplicadores instanciados vs ciclos de clock na síntese HLS.
- Configuração heterogênea (`granularity='name'`) permite definir larguras de bits específicas (`weight`, `bias`, `result`) individualmente para cada camada da rede.
