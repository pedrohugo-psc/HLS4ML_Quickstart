"""Definições de modelos PyTorch para o benchmark Jet Tagging (hls4ml tutorial)."""

import torch
import torch.nn as nn


class JetTagger(nn.Module):
    """
    Classificador MLP de 3 camadas ocultas (Float32 baseline):
    Entrada: 16 variáveis físicas de jatos de partículas
    Camadas: Linear(16 -> 64) -> ReLU -> Linear(64 -> 32) -> ReLU -> Linear(32 -> 32) -> ReLU -> Linear(32 -> 5)
    Saída: 5 classes (gluon, quark leve, top, W, Z)
    """

    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(16, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 32)
        self.output = nn.Linear(32, 5)

    def logits(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc3(x))
        return self.output(x)

    def forward(self, x):
        return torch.softmax(self.logits(x), dim=1)


class JetTaggerBrevitas(nn.Module):
    """
    Classificador MLP com Quantization-Aware Training (QAT) usando Brevitas:
    Pesos quantizados para 6 bits em ponto fixo (Int8WeightPerTensorFixedPoint)
    Bias quantizados para 6 bits em ponto fixo
    Ativações ReLU quantizadas para 6 bits (Uint8ActPerTensorFixedPoint)
    """

    def __init__(self):
        super().__init__()
        import brevitas.nn as qnn
        from brevitas.quant import (
            Int8WeightPerTensorFixedPoint,
            Int8BiasPerTensorFixedPointInternalScaling,
            Uint8ActPerTensorFixedPoint,
        )

        self.fc1 = qnn.QuantLinear(
            16, 64,
            weight_quant=Int8WeightPerTensorFixedPoint,
            weight_bit_width=6,
            bias_quant=Int8BiasPerTensorFixedPointInternalScaling,
            bias_bit_width=6,
        )
        self.relu1 = qnn.QuantReLU(act_quant=Uint8ActPerTensorFixedPoint, bit_width=6)

        self.fc2 = qnn.QuantLinear(
            64, 32,
            weight_quant=Int8WeightPerTensorFixedPoint,
            weight_bit_width=6,
            bias_quant=Int8BiasPerTensorFixedPointInternalScaling,
            bias_bit_width=6,
        )
        self.relu2 = qnn.QuantReLU(act_quant=Uint8ActPerTensorFixedPoint, bit_width=6)

        self.fc3 = qnn.QuantLinear(
            32, 32,
            weight_quant=Int8WeightPerTensorFixedPoint,
            weight_bit_width=6,
            bias_quant=Int8BiasPerTensorFixedPointInternalScaling,
            bias_bit_width=6,
        )
        self.relu3 = qnn.QuantReLU(act_quant=Uint8ActPerTensorFixedPoint, bit_width=6)

        self.output = qnn.QuantLinear(
            32, 5,
            weight_quant=Int8WeightPerTensorFixedPoint,
            weight_bit_width=6,
            bias_quant=Int8BiasPerTensorFixedPointInternalScaling,
            bias_bit_width=6,
        )

    def logits(self, x):
        x = self.relu1(self.fc1(x))
        x = self.relu2(self.fc2(x))
        x = self.relu3(self.fc3(x))
        return self.output(x)

    def forward(self, x):
        return torch.softmax(self.logits(x), dim=1)


class JetCNN1D(nn.Module):
    """
    Classificador Convolucional 1D (CNN) para jatos do CERN LHC:
    Entrada: (batch, 1, 16) - 1 canal com 16 features
    Camadas: Conv1D(1 -> 8, k=3, pad=1) -> ReLU -> MaxPool1D(k=2) -> Flatten -> Linear(64 -> 32) -> ReLU -> Linear(32 -> 5)
    Saída: 5 classes (gluon, quark leve, top, W, Z)
    """

    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=8, kernel_size=3, stride=1, padding=1)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool1d(kernel_size=2)
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(8 * 8, 32)
        self.relu2 = nn.ReLU()
        self.fc2 = nn.Linear(32, 5)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        x = self.relu1(self.conv1(x))
        x = self.pool1(x)
        x = self.flatten(x)
        x = self.relu2(self.fc1(x))
        x = self.fc2(x)
        return self.softmax(x)


class JetAutoencoder(nn.Module):
    """
    Autoencoder denso para Detecção de Anomalias (não-supervisionada) no CERN LHC:
    Entrada: (batch, 16)
    Encoder: Linear(16 -> 8) -> ReLU -> Linear(8 -> 3) -> ReLU (Bottleneck de 3 dimensões)
    Decoder: Linear(3 -> 8) -> ReLU -> Linear(8 -> 16)
    Saída: Reconstrução das 16 features originais
    Score de Anomalia: Erro Médio Quadrático (MSE) entre Entrada e Saída
    """

    def __init__(self):
        super().__init__()
        # Encoder
        self.enc1 = nn.Linear(16, 8)
        self.relu1 = nn.ReLU()
        self.enc2 = nn.Linear(8, 3)
        self.relu2 = nn.ReLU()
        # Decoder
        self.dec1 = nn.Linear(3, 8)
        self.relu3 = nn.ReLU()
        self.dec2 = nn.Linear(8, 16)

    def encode(self, x):
        return self.relu2(self.enc2(self.relu1(self.enc1(x))))

    def decode(self, z):
        return self.dec2(self.relu3(self.dec1(z)))

    def forward(self, x):
        return self.decode(self.encode(x))

