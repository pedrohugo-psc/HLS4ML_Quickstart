#!/usr/bin/env python3
"""
Tutorial 01: Getting Started com hls4ml e PyTorch.

Fluxo de execucao:
1. Carregamento do dataset hls4ml_lhc_jets_hlf (Jet Tagging)
2. Treinamento da rede MLP (JetTagger) em PyTorch
3. Configuracao de parametros HLS (ap_fixed<16,6>, ReuseFactor = 1)
4. Conversao do modelo PyTorch para projeto C++/HLS
5. Compilacao C-Simulation com g++ e inferencia em ponto fixo
6. Comparacao de acuracia e geracao das curvas ROC
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.abspath(os.path.join(CURRENT_DIR, "../.."))
sys.path.append(ROOT_DIR)
sys.path.append(CURRENT_DIR)

import models
import plotting
import hls4ml

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset


def main():
    print("=" * 80)
    print("TUTORIAL 01: GETTING STARTED COM HLS4ML E PYTORCH")
    print("=" * 80)

    data_dir = os.path.join(ROOT_DIR, "data", "jet-tagging")
    models_dir = os.path.join(ROOT_DIR, "models")
    os.makedirs(data_dir, exist_ok=True)
    os.makedirs(models_dir, exist_ok=True)

    # 1. Carregamento dos Dados
    print("\n[Passo 1/6] Carregando conjunto de dados (hls4ml_lhc_jets_hlf)...")
    x_test_path = os.path.join(data_dir, "X_test.npy")
    y_test_path = os.path.join(data_dir, "y_test.npy")
    classes_path = os.path.join(data_dir, "classes.npy")

    if os.path.exists(x_test_path) and os.path.exists(y_test_path):
        X_train = np.load(os.path.join(data_dir, "X_train.npy"))
        y_train = np.load(os.path.join(data_dir, "y_train.npy"))
        X_test = np.load(x_test_path)
        y_test = np.load(y_test_path)
        classes = np.load(classes_path, allow_pickle=True)
    else:
        print("  Baixando dataset 'hls4ml_lhc_jets_hlf' via OpenML...")
        from sklearn.datasets import fetch_openml
        data = fetch_openml('hls4ml_lhc_jets_hlf', version=1, as_frame=True)
        X = data['data'].to_numpy(dtype=np.float32)
        y_raw = data['target'].to_numpy()

        le = LabelEncoder()
        y_int = le.fit_transform(y_raw)
        classes = le.classes_
        y_onehot = np.eye(len(classes))[y_int]

        scaler = StandardScaler()
        X = scaler.fit_transform(X)

        X_train, X_test, y_train, y_test = train_test_split(
            X, y_onehot, test_size=0.2, random_state=42, stratify=y_int
        )

        np.save(os.path.join(data_dir, "X_train.npy"), X_train)
        np.save(os.path.join(data_dir, "y_train.npy"), y_train)
        np.save(x_test_path, X_test)
        np.save(y_test_path, y_test)
        np.save(classes_path, classes)

    print(f"  Shape X_train: {X_train.shape} | Shape X_test: {X_test.shape}")
    print(f"  Classes: {list(classes)}")

    # 2. Treinamento da Rede em PyTorch
    print("\n[Passo 2/6] Treinando modelo PyTorch (JetTagger)...")
    torch_weights_path = os.path.join(models_dir, "pytorch_weights_part1.pt")
    model = models.JetTagger()

    n_train_samples = min(50000, len(X_train))
    train_tensor_x = torch.tensor(X_train[:n_train_samples], dtype=torch.float32)
    train_tensor_y = torch.tensor(np.argmax(y_train[:n_train_samples], axis=1), dtype=torch.long)
    dataset = TensorDataset(train_tensor_x, train_tensor_y)
    dataloader = DataLoader(dataset, batch_size=256, shuffle=True)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()

    model.train()
    for epoch in range(1, 6):
        total_loss = 0.0
        for batch_x, batch_y in dataloader:
            optimizer.zero_grad()
            logits = model.logits(batch_x)
            loss = criterion(logits, batch_y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"  Epoca {epoch}/5 | Loss: {total_loss / len(dataloader):.4f}")

    torch.save(model.state_dict(), torch_weights_path)
    print(f"  Pesos salvos em: {torch_weights_path}")

    model.eval()
    with torch.no_grad():
        test_tensor_x = torch.tensor(X_test, dtype=torch.float32)
        y_torch_pred = model(test_tensor_x).numpy()

    acc_torch = accuracy_score(np.argmax(y_test, axis=1), np.argmax(y_torch_pred, axis=1))
    print(f"  Acuracia PyTorch (Float32): {acc_torch * 100:.2f}%")

    # 3. Conversao para HLS com hls4ml
    print("\n[Passo 3/6] Gerando configuracao e projeto C++/HLS...")
    hls_output_dir = os.path.join(CURRENT_DIR, "hls4ml_prj_base_part1")

    config = hls4ml.utils.config_from_pytorch_model(
        model,
        input_shape=(16,),
        granularity='model',
        backend='Vivado',
    )
    plotting.print_dict(config)

    hls_model = hls4ml.converters.convert_from_pytorch_model(
        model,
        hls_config=config,
        output_dir=hls_output_dir,
        part='xcu200-fsgd2104-2-e',
    )
    hls_model.write()
    print(f"  Projeto C++/HLS salvo em: {hls_output_dir}")

    # 4. Compilacao C++ (C-Simulation com g++)
    print("\n[Passo 4/6] Compilando projeto C++ com g++...")
    hls_model.compile()
    print("  Compilacao concluida.")

    # 5. Inferencia em Ponto Fixo
    eval_n = min(5000, len(X_test))
    print(f"\n[Passo 5/6] Executando C-Simulation em {eval_n} amostras de teste...")
    y_hls_pred = hls_model.predict(X_test[:eval_n])

    acc_hls = accuracy_score(np.argmax(y_test[:eval_n], axis=1), np.argmax(y_hls_pred, axis=1))
    acc_torch_eval = accuracy_score(np.argmax(y_test[:eval_n], axis=1), np.argmax(y_torch_pred[:eval_n], axis=1))

    print("\n" + "=" * 50)
    print("           RESULTADOS COMPARATIVOS")
    print("=" * 50)
    print(f"Acuracia PyTorch (Float32)   : {acc_torch_eval * 100:.2f}%")
    print(f"Acuracia hls4ml (ap_fixed)   : {acc_hls * 100:.2f}%")
    print(f"Diferenca de Acuracia        : {abs(acc_torch_eval - acc_hls) * 100:.2f}%")
    print("=" * 50)

    # 6. Geracao de Curva ROC
    roc_img_path = os.path.join(CURRENT_DIR, "curva_roc_comparativa.png")
    print(f"\n[Passo 6/6] Gerando curvas ROC em: {roc_img_path}")
    fig, ax = plt.subplots(figsize=(8, 7))
    plotting.makeRoc(y_test[:eval_n], y_torch_pred[:eval_n], list(classes), linestyle='-')
    plt.gca().set_prop_cycle(None)
    plotting.makeRoc(y_test[:eval_n], y_hls_pred, list(classes), linestyle='--')

    from matplotlib.lines import Line2D
    lines = [
        Line2D([0], [0], color='black', linestyle='-', label='PyTorch (Float32)'),
        Line2D([0], [0], color='black', linestyle='--', label='hls4ml (ap_fixed<16,6>)')
    ]
    ax.legend(handles=lines, loc='lower right')

    plt.tight_layout()
    plt.savefig(roc_img_path, dpi=150)
    plt.close()
    print("Execucao do Tutorial 01 finalizada.")


if __name__ == "__main__":
    main()
