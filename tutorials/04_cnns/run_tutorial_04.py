#!/usr/bin/env python3
"""
Tutorial 04: Redes Neurais Convolucionais (CNNs) e Streaming I/O em FPGAs com hls4ml.

Este script demonstra:
1. Treinamento de uma CNN 1D (Conv1D + ReLU + MaxPool1D + Flatten + Linear + Softmax) em PyTorch.
2. Arquitetura de hardware streaming (io_stream / AXI4-Stream) e Line Buffers / Shift Registers.
3. Conversão para HLS C++ em ponto fixo ap_fixed<16,6>.
4. Compilação e C-Simulation nativa com g++ (sem necessidade de licenças externas).
5. Avaliação de acurácia e geração das Curvas ROC comparativas.
"""

import os
import sys
import time
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc

# Adicionar diretório raiz ao PYTHONPATH
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
sys.path.append(ROOT_DIR)

from models import JetCNN1D
import hls4ml


def main():
    print("=" * 80)
    print("TUTORIAL 04: REDES NEURAIS CONVOLUCIONAIS (CNNs) & STREAMING I/O NO HLS4ML")
    print("=" * 80)

    # 1. Carregar Dados de Jatos do CERN LHC
    data_dir = os.path.join(ROOT_DIR, "data", "jet-tagging")
    print("\n[Passo 1/6] Carregando conjunto de dados do CERN LHC...")
    X_train = np.load(os.path.join(data_dir, "X_train.npy"))
    y_train = np.load(os.path.join(data_dir, "y_train.npy"))
    X_test = np.load(os.path.join(data_dir, "X_test.npy"))
    y_test = np.load(os.path.join(data_dir, "y_test.npy"))
    classes = np.load(os.path.join(data_dir, "classes.npy"), allow_pickle=True)

    print(f"  Shape original X_train: {X_train.shape}")
    print(f"  Shape original X_test : {X_test.shape}")
    print(f"  Classes ({len(classes)}): {list(classes)}")

    # Ajustar dimensão para entrada convolucional 1D: (Batch, Canais=1, Comprimento=16)
    X_train_cnn = np.ascontiguousarray(X_train[:, np.newaxis, :], dtype=np.float32)
    X_test_cnn = np.ascontiguousarray(X_test[:, np.newaxis, :], dtype=np.float32)
    print(f"  Shape Convolucional 1D: {X_train_cnn.shape}")

    # 2. Instanciar e Treinar ou Carregar Modelo PyTorch
    print("\n[Passo 2/6] Preparando modelo Convolucional (JetCNN1D)...")
    device = torch.device("cpu")
    model = JetCNN1D().to(device)

    weights_dir = os.path.join(ROOT_DIR, "models")
    os.makedirs(weights_dir, exist_ok=True)
    weights_path = os.path.join(weights_dir, "pytorch_weights_cnn.pt")

    if os.path.exists(weights_path):
        print(f"  Carregando pesos salvos de: {weights_path}")
        model.load_state_dict(torch.load(weights_path, map_location=device))
    else:
        print("  Treinando modelo CNN em PyTorch (5 épocas)...")
        # Subconjunto balanceado para treino rápido
        n_train = 100000
        train_ds = TensorDataset(
            torch.from_numpy(X_train_cnn[:n_train]),
            torch.from_numpy(y_train[:n_train].astype(np.float32))
        )
        train_loader = DataLoader(train_ds, batch_size=256, shuffle=True)

        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.002)

        t_start = time.time()
        for epoch in range(1, 6):
            model.train()
            running_loss = 0.0
            for x_batch, y_batch in train_loader:
                optimizer.zero_grad()
                outputs = model(x_batch)
                loss = criterion(outputs, y_batch)
                loss.backward()
                optimizer.step()
                running_loss += loss.item()
            print(f"    Época {epoch}/5 | Loss Média: {running_loss / len(train_loader):.4f}")

        print(f"  Treinamento concluído em {time.time() - t_start:.2f}s!")
        torch.save(model.state_dict(), weights_path)
        print(f"  Pesos salvos em: {weights_path}")

    # Avaliação do Modelo PyTorch
    model.eval()
    eval_n = 10000
    with torch.no_grad():
        y_pred_pytorch = model(torch.from_numpy(X_test_cnn[:eval_n])).cpu().numpy()

    acc_pt = (np.argmax(y_pred_pytorch, axis=1) == np.argmax(y_test[:eval_n], axis=1)).mean()
    print(f"  Acurácia PyTorch (Float32): {acc_pt * 100:.2f}%")

    # 3. Configurar Hardware no hls4ml
    print("\n[Passo 3/6] Configurando tradução de hardware para CNN (io_stream)...")
    config = hls4ml.utils.config_from_pytorch_model(model, input_shape=(1, 16))
    # Para CNNs com acúmulo de convolução e camadas densas, 8 bits inteiros garantem
    # faixa de [-128, 127.99], evitando wrap-around de overflow nas ativações.
    config["Model"]["Precision"]["default"] = "ap_fixed<16,8>"
    config["Model"]["Strategy"] = "Latency"
    config["Model"]["ReuseFactor"] = 1

    print("  Estratégia de E/S: io_stream (AXI4-Stream)")
    print("  Precisão Padrão:   ap_fixed<16,8> (8 bits inteiros para prevenir overflow)")
    print("  Fator de Reuso:    1 (Paralelismo Total)")

    # 4. Gerar Firmware C++/HLS
    output_dir = os.path.join(os.path.dirname(__file__), "hls4ml_prj_cnn")
    print(f"\n[Passo 4/6] Convertendo para projeto de hardware HLS em: {output_dir}")
    hls_model = hls4ml.converters.convert_from_pytorch_model(
        model,
        input_shape=(1, 16),
        hls_config=config,
        output_dir=output_dir,
        part="xc7z020clg400-1",
        io_type="io_stream"
    )

    # 5. Compilar com g++ nativo
    print("\n[Passo 5/6] Compilando emulador de C-Simulation nativo com g++...")
    t_comp = time.time()
    hls_model.compile()
    print(f"  Compilação g++ concluída com sucesso em {time.time() - t_comp:.2f}s!")

    # 6. Executar Simulação de Hardware
    print(f"\n[Passo 6/6] Executando C-Simulation em {eval_n} eventos de teste...")
    t_sim = time.time()
    y_pred_hls = hls_model.predict(X_test_cnn[:eval_n])
    print(f"  Simulação concluída em {time.time() - t_sim:.2f}s!")

    acc_hls = (np.argmax(y_pred_hls, axis=1) == np.argmax(y_test[:eval_n], axis=1)).mean()
    delta_acc = (acc_pt - acc_hls) * 100

    print("\n" + "=" * 50)
    print("           RESULTADOS COMPARATIVOS")
    print("=" * 50)
    print(f"Acurácia PyTorch (Float32)    : {acc_pt * 100:.2f}%")
    print(f"Acurácia hls4ml (ap_fixed<16,8>): {acc_hls * 100:.2f}%")
    print(f"Diferença de Acurácia         : {delta_acc:.2f}%")
    print("=" * 50)

    # 7. Gerar Gráficos e Curvas ROC
    plot_path = os.path.join(os.path.dirname(__file__), "curva_roc_cnn.png")
    print(f"\nGerando curvas ROC comparativas em: {plot_path}")
    plt.figure(figsize=(9, 7))

    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd"]
    for i, label in enumerate(classes):
        fpr_pt, tpr_pt, _ = roc_curve(y_test[:eval_n, i], y_pred_pytorch[:, i])
        fpr_hls, tpr_hls, _ = roc_curve(y_test[:eval_n, i], y_pred_hls[:, i])
        auc_pt = auc(fpr_pt, tpr_pt)
        auc_hls = auc(fpr_hls, tpr_hls)

        plt.plot(fpr_pt, tpr_pt, color=colors[i], linestyle="--", alpha=0.7,
                 label=f"{label} - PyTorch (AUC = {auc_pt:.3f})")
        plt.plot(fpr_hls, tpr_hls, color=colors[i], linestyle="-", lw=1.8,
                 label=f"{label} - hls4ml (AUC = {auc_hls:.3f})")

    plt.plot([0, 1], [0, 1], "k--", alpha=0.5, label="Baseline Aleatório")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("Taxa de Falso Positivo (FPR)", fontsize=12)
    plt.ylabel("Taxa de Verdadeiro Positivo (TPR)", fontsize=12)
    plt.title("Curvas ROC: PyTorch Float32 vs hls4ml ap_fixed<16,8> (CNN 1D)", fontsize=13, fontweight="bold")
    plt.legend(loc="lower right", fontsize=9)
    plt.grid(True, linestyle=":", alpha=0.6)
    plt.tight_layout()
    plt.savefig(plot_path, dpi=300)
    plt.close()
    print("Gráfico salvo com sucesso!")
    print("\nExecução do Tutorial 04 finalizada com sucesso!")


if __name__ == "__main__":
    main()
