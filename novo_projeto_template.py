#!/usr/bin/env python3
"""
Template base para conversão de modelos PyTorch no hls4ml.

Substitua a definição em `MeuModelo` e os dados em `X_teste` pelo modelo
e conjunto de dados do seu projeto.
"""

import os
import numpy as np
import torch
import torch.nn as nn
import hls4ml


class MeuModelo(nn.Module):
    """Exemplo de rede MLP: entrada de 10 variáveis, 2 camadas ocultas e 3 saídas."""

    def __init__(self):
        super().__init__()
        self.rede = nn.Sequential(
            nn.Linear(10, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 3),
            nn.Softmax(dim=-1),
        )

    def forward(self, x):
        return self.rede(x)


def main():
    # 1. Instanciar modelo e carregar pesos (se aplicável)
    modelo = MeuModelo()
    # modelo.load_state_dict(torch.load("meus_pesos.pt", map_location="cpu"))
    modelo.eval()

    # 2. Configurar parâmetros de síntese HLS
    # input_shape deve informar o formato de 1 amostra (sem dimensão de batch)
    config = hls4ml.utils.config_from_pytorch_model(
        modelo,
        input_shape=(10,),
        granularity='model',
        backend='Vivado',
        default_precision='ap_fixed<16,6>',
        default_reuse_factor=1,
    )

    # 3. Converter modelo PyTorch para projeto C++/HLS
    pasta_saida = os.path.join(os.path.dirname(os.path.abspath(__file__)), "meu_projeto_hls")

    hls_model = hls4ml.converters.convert_from_pytorch_model(
        modelo,
        hls_config=config,
        output_dir=pasta_saida,
        part='xcu200-fsgd2104-2-e',
        io_type='io_parallel',
    )

    hls_model.write()
    print(f"[OK] Arquivos C/C++ gerados em: {pasta_saida}/firmware/")

    # 4. Compilar C++ com g++ e executar C-Simulation bit-accurate
    print("[INFO] Compilando projeto C++ com g++...")
    hls_model.compile()

    X_teste = np.ascontiguousarray(np.random.randn(20, 10), dtype=np.float32)

    with torch.no_grad():
        y_pytorch = modelo(torch.from_numpy(X_teste)).numpy()

    y_hls = hls_model.predict(X_teste)

    erro_medio = np.mean(np.abs(y_pytorch - y_hls))
    print(f"[OK] Simulação concluída. Erro médio absoluto (Float32 vs ap_fixed<16,6>): {erro_medio:.6f}")

    # 5. Síntese de hardware no Vivado/Vitis HLS para gerar Verilog/VHDL (opcional)
    # hls_model.build(csim=False, synth=True, export=True)
    # hls4ml.report.read_vivado_report(pasta_saida)


if __name__ == "__main__":
    main()
