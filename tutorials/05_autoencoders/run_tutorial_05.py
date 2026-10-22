#!/usr/bin/env python3
"""
Tutorial 05: Autoencoders e Detecção de Anomalias Não-Supervisionada no FPGA com hls4ml.

Este script demonstra:
1. Treinamento de um Autoencoder denso (16 -> 8 -> 3 -> 8 -> 16) em PyTorch com jatos de fundo (QCD: gluons e quarks leves).
2. Detecção de anomalias por Erro Médio Quadrático (MSE) de reconstrução para novas partículas (Top quarks, W, Z).
3. Conversão para firmware HLS em ponto fixo ap_fixed<16,6>.
4. Compilação e C-Simulation nativa com g++ (sem necessidade de licenças proprietárias).
5. Comparação da perda de reconstrução e Curvas ROC de detecção de anomalias (PyTorch vs hls4ml).
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

from models import JetAutoencoder
import hls4ml


def main():
    print("=" * 80)
    print("TUTORIAL 05: AUTOENCODERS E DETECÇÃO DE ANOMALIAS NO FPGA COM HLS4ML")
    print("=" * 80)

    # 1. Carregar Dados de Jatos do CERN LHC
    data_dir = os.path.join(ROOT_DIR, "data", "jet-tagging")
    print("\n[Passo 1/6] Carregando conjunto de dados do CERN LHC...")
    X_train = np.load(os.path.join(data_dir, "X_train.npy"))
    y_train = np.load(os.path.join(data_dir, "y_train.npy"))
    X_test = np.load(os.path.join(data_dir, "X_test.npy"))
    y_test = np.load(os.path.join(data_dir, "y_test.npy"))
    classes = list(np.load(os.path.join(data_dir, "classes.npy"), allow_pickle=True))

    print(f"  Shape X_train: {X_train.shape}")
    print(f"  Shape X_test : {X_test.shape}")
    print(f"  Classes ({len(classes)}): {classes}")

    # Separar dados de fundo (Background Standard Model: 'g' e 'q') para treino não-supervisionado
    idx_g = classes.index("g")
    idx_q = classes.index("q")
    idx_t = classes.index("t")

    is_bkg_train = (y_train[:, idx_g] == 1) | (y_train[:, idx_q] == 1)
    X_train_bkg = X_train[is_bkg_train][:100000].astype(np.float32)
    print(f"  Amostras de Background selecionadas para treino (gluons + quarks): {len(X_train_bkg)}")

    # 2. Instanciar e Treinar ou Carregar Modelo PyTorch
    print("\n[Passo 2/6] Preparando Autoencoder (JetAutoencoder: 16 -> 8 -> 3 -> 8 -> 16)...")
    device = torch.device("cpu")
    model = JetAutoencoder().to(device)

    weights_dir = os.path.join(ROOT_DIR, "models")
    os.makedirs(weights_dir, exist_ok=True)
    weights_path = os.path.join(weights_dir, "pytorch_weights_autoencoder.pt")

    if os.path.exists(weights_path):
        print(f"  Carregando pesos pré-treinados de: {weights_path}")
        model.load_state_dict(torch.load(weights_path, map_location=device))
    else:
        print("  Treinando Autoencoder não-supervisionado (5 épocas)...")
        train_ds = TensorDataset(torch.from_numpy(X_train_bkg), torch.from_numpy(X_train_bkg))
        train_loader = DataLoader(train_ds, batch_size=256, shuffle=True)

        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.005)

        t_start = time.time()
        for epoch in range(1, 6):
            model.train()
            running_loss = 0.0
            for xb, yb in train_loader:
                optimizer.zero_grad()
                out = model(xb)
                loss = criterion(out, yb)
                loss.backward()
                optimizer.step()
                running_loss += loss.item()
            print(f"    Época {epoch}/5 | MSE Médio de Reconstrução: {running_loss / len(train_loader):.4f}")

        print(f"  Treinamento concluído em {time.time() - t_start:.2f}s!")
        torch.save(model.state_dict(), weights_path)
        print(f"  Pesos salvos em: {weights_path}")

    # Avaliação do Modelo PyTorch no Test Set
    eval_n = 10000
    X_eval = X_test[:eval_n].astype(np.float32)
    y_eval = y_test[:eval_n]

    model.eval()
    with torch.no_grad():
        recon_pt = model(torch.from_numpy(X_eval)).cpu().numpy()

    mse_pt = np.mean((X_eval - recon_pt) ** 2, axis=1)

    is_bkg_eval = (y_eval[:, idx_g] == 1) | (y_eval[:, idx_q] == 1)
    is_anomaly_eval = (y_eval[:, idx_t] == 1)  # Top quark como sinal anômalo
    mask = is_bkg_eval | is_anomaly_eval

    y_true_binary = is_anomaly_eval[mask].astype(int)
    scores_pt = mse_pt[mask]

    fpr_pt, tpr_pt, _ = roc_curve(y_true_binary, scores_pt)
    auc_pt = auc(fpr_pt, tpr_pt)
    print(f"  AUC de Detecção de Anomalia (PyTorch Float32 - Top vs Background): {auc_pt:.4f}")

    # 3. Configurar Hardware no hls4ml
    print("\n[Passo 3/6] Configurando conversão de hardware para o Autoencoder...")
    config = hls4ml.utils.config_from_pytorch_model(model, input_shape=(16,))
    config["Model"]["Precision"]["default"] = "ap_fixed<16,6>"
    config["Model"]["Strategy"] = "Latency"
    config["Model"]["ReuseFactor"] = 1

    print("  Precisão Padrão: ap_fixed<16,6>")
    print("  Fator de Reuso : 1 (Paralelismo Total)")
    print("  Estratégia     : Latency")

    # 4. Gerar Firmware C++/HLS
    output_dir = os.path.join(os.path.dirname(__file__), "hls4ml_prj_autoencoder")
    print(f"\n[Passo 4/6] Convertendo para projeto de hardware HLS em: {output_dir}")
    hls_model = hls4ml.converters.convert_from_pytorch_model(
        model,
        input_shape=(16,),
        hls_config=config,
        output_dir=output_dir,
        part="xc7z020clg400-1",
        io_type="io_parallel"
    )

    # 5. Compilar com g++ nativo
    print("\n[Passo 5/6] Compilando emulador de C-Simulation nativo com g++...")
    t_comp = time.time()
    hls_model.compile()
    print(f"  Compilação g++ concluída com sucesso em {time.time() - t_comp:.2f}s!")

    # 6. Executar Simulação de Hardware
    print(f"\n[Passo 6/6] Executando C-Simulation em {eval_n} eventos de teste...")
    t_sim = time.time()
    recon_hls = hls_model.predict(X_eval)
    print(f"  Simulação concluída em {time.time() - t_sim:.2f}s!")

    mse_hls = np.mean((X_eval - recon_hls) ** 2, axis=1)
    scores_hls = mse_hls[mask]

    fpr_hls, tpr_hls, _ = roc_curve(y_true_binary, scores_hls)
    auc_hls = auc(fpr_hls, tpr_hls)

    diff_recon = np.mean(np.abs(recon_pt - recon_hls))

    print("\n" + "=" * 50)
    print("           RESULTADOS COMPARATIVOS")
    print("=" * 50)
    print(f"AUC PyTorch (Float32)         : {auc_pt:.4f}")
    print(f"AUC hls4ml (ap_fixed<16,6>)   : {auc_hls:.4f}")
    print(f"Diferença Absoluta de AUC     : {abs(auc_pt - auc_hls):.4f}")
    print(f"Erro Médio Absoluto Reconstrução: {diff_recon:.5f}")
    print("=" * 50)

    # 7. Gerar Gráficos: Distribuição do Erro e Curva ROC
    dist_plot_path = os.path.join(os.path.dirname(__file__), "distribuicao_erro_reconstrucao.png")
    print(f"\nGerando histograma de erro de reconstrução em: {dist_plot_path}")
    plt.figure(figsize=(9, 5))
    bins = np.linspace(0, 0.4, 60)
    plt.hist(mse_hls[is_bkg_eval], bins=bins, alpha=0.6, color="#1f77b4", label="Fundo Conhecido (Gluon + Quark)", density=True)
    plt.hist(mse_hls[is_anomaly_eval], bins=bins, alpha=0.6, color="#d62728", label="Anomalia (Top Quarks)", density=True)
    plt.xlabel("Erro Médio Quadrático de Reconstrução (MSE)", fontsize=11)
    plt.ylabel("Densidade de Probabilidade", fontsize=11)
    plt.title("Distribuição do Erro de Reconstrução no FPGA (hls4ml ap_fixed<16,6>)", fontsize=12, fontweight="bold")
    plt.legend(loc="upper right", fontsize=10)
    plt.grid(True, linestyle=":", alpha=0.6)
    plt.tight_layout()
    plt.savefig(dist_plot_path, dpi=300)
    plt.close()

    roc_plot_path = os.path.join(os.path.dirname(__file__), "curva_roc_autoencoder.png")
    print(f"Gerando curva ROC comparativa em: {roc_plot_path}")
    plt.figure(figsize=(8, 6))
    plt.plot(fpr_pt, tpr_pt, "b--", lw=1.8, label=f"PyTorch Float32 (AUC = {auc_pt:.3f})")
    plt.plot(fpr_hls, tpr_hls, "r-", lw=1.8, label=f"hls4ml ap_fixed<16,6> (AUC = {auc_hls:.3f})")
    plt.plot([0, 1], [0, 1], "k--", alpha=0.5, label="Baseline Aleatório (AUC = 0.500)")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("Taxa de Falso Positivo (FPR)", fontsize=11)
    plt.ylabel("Taxa de Verdadeiro Positivo (TPR)", fontsize=11)
    plt.title("Detecção Não-Supervisionada de Anomalias: Curva ROC (Top Quark vs QCD)", fontsize=12, fontweight="bold")
    plt.legend(loc="lower right", fontsize=10)
    plt.grid(True, linestyle=":", alpha=0.6)
    plt.tight_layout()
    plt.savefig(roc_plot_path, dpi=300)
    plt.close()

    print("Gráficos salvos com sucesso!")
    print("\nExecução do Tutorial 05 finalizada com sucesso!")


if __name__ == "__main__":
    main()
