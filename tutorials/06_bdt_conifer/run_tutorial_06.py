#!/usr/bin/env python3
"""
Tutorial 06: Boosted Decision Trees (BDTs) em FPGA com Conifer.

Este script demonstra:
1. Treinamento de um ensemble de árvores de decisão (Gradient Boosting do scikit-learn) no conjunto de jatos do CERN LHC.
2. Como árvores de decisão operam em FPGA: mapeamento de comparações em LUTs sem necessidade de multiplicadores DSP.
3. Conversão de modelo scikit-learn para hardware em ponto fixo ap_fixed<18,8> via Conifer.
4. Compilação e simulação nativa C++ com g++.
5. Avaliação de acurácia e Curvas ROC (scikit-learn Float64 vs Conifer Hardware em Ponto Fixo).
"""

import os
import sys
import time
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import roc_curve, auc, accuracy_score
import joblib

# Adicionar diretório raiz ao PYTHONPATH
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
sys.path.append(ROOT_DIR)

import conifer
import hls4ml
import nlohmann_json


def setup_conifer_headers():
    """Garante que os cabeçalhos C++ necessários estejam vinculados ao Conifer."""
    c_path = conifer.__path__[0]
    h_path = os.path.dirname(hls4ml.__file__)
    n_path = os.path.dirname(nlohmann_json.__file__)

    os.makedirs(f"{c_path}/external/json", exist_ok=True)
    if not os.path.exists(f"{c_path}/external/json/include"):
        try:
            os.symlink(f"{n_path}/include", f"{c_path}/external/json/include")
        except OSError:
            pass

    os.makedirs(f"{c_path}/external/ap_types", exist_ok=True)
    if not os.path.exists(f"{c_path}/external/ap_types/include"):
        try:
            os.symlink(f"{h_path}/templates/vivado/ap_types", f"{c_path}/external/ap_types/include")
        except OSError:
            pass


def main():
    print("=" * 80)
    print("TUTORIAL 06: BOOSTED DECISION TREES (BDTs) EM FPGA COM CONIFER")
    print("=" * 80)

    setup_conifer_headers()

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

    # Tarefa de Classificação de Gatilho: Identificar Quarks Top ('t') vs Fundo QCD
    idx_t = classes.index("t")
    y_train_bin = (y_train[:, idx_t] == 1).astype(int)
    y_test_bin = (y_test[:, idx_t] == 1).astype(int)

    n_train = 50000
    X_train_sub = X_train[:n_train]
    y_train_sub = y_train_bin[:n_train]

    eval_n = 10000
    X_eval = X_test[:eval_n]
    y_eval = y_test_bin[:eval_n]

    # 2. Treinar ou Carregar Modelo scikit-learn
    print("\n[Passo 2/6] Preparando modelo de Boosted Decision Trees (scikit-learn)...")
    models_dir = os.path.join(ROOT_DIR, "models")
    os.makedirs(models_dir, exist_ok=True)
    model_path = os.path.join(models_dir, "sklearn_bdt_model.joblib")

    if os.path.exists(model_path):
        print(f"  Carregando modelo salvo de: {model_path}")
        clf = joblib.load(model_path)
    else:
        print("  Treinando GradientBoostingClassifier (n_estimators=30, max_depth=3)...")
        t_start = time.time()
        clf = GradientBoostingClassifier(n_estimators=30, max_depth=3, learning_rate=0.2, random_state=42)
        clf.fit(X_train_sub, y_train_sub)
        print(f"  Treinamento concluído em {time.time() - t_start:.2f}s!")
        joblib.dump(clf, model_path)
        print(f"  Modelo salvo em: {model_path}")

    # Avaliação do Modelo scikit-learn
    scores_sklearn = clf.decision_function(X_eval)
    preds_sklearn = (scores_sklearn > 0).astype(int)
    acc_sklearn = accuracy_score(y_eval, preds_sklearn)
    fpr_sk, tpr_sk, _ = roc_curve(y_eval, scores_sklearn)
    auc_sk = auc(fpr_sk, tpr_sk)
    print(f"  Acurácia scikit-learn (Float64): {acc_sklearn * 100:.2f}%")
    print(f"  AUC scikit-learn (Float64)     : {auc_sk:.4f}")

    # 3. Configurar Hardware no Conifer
    print("\n[Passo 3/6] Configurando tradução de hardware para BDTs no Conifer...")
    output_dir = os.path.join(os.path.dirname(__file__), "conifer_prj_bdt")
    cfg = conifer.backends.cpp.auto_config()
    cfg["OutputDir"] = output_dir
    cfg["Precision"] = "ap_fixed<18,8>"

    print(f"  Backend Selecionado: {cfg['Backend']}")
    print(f"  Precisão de Ponto Fixo: {cfg['Precision']}")
    print(f"  Diretório de Saída: {cfg['OutputDir']}")

    # 4. Converter BDT para Projeto de Hardware
    print(f"\n[Passo 4/6] Convertendo árvores de decisão para C++ no Conifer...")
    model_conifer = conifer.converters.convert_from_sklearn(clf, cfg)

    # 5. Compilar com g++ nativo
    print("\n[Passo 5/6] Compilando emulador de hardware nativo C++ com g++...")
    t_comp = time.time()
    model_conifer.compile()
    print(f"  Compilação g++ concluída com sucesso em {time.time() - t_comp:.2f}s!")

    # 6. Executar Simulação de Hardware
    print(f"\n[Passo 6/6] Executando simulação de hardware em {eval_n} eventos de teste...")
    t_sim = time.time()
    scores_conifer = model_conifer.decision_function(X_eval).ravel()
    print(f"  Simulação concluída em {time.time() - t_sim:.2f}s!")

    preds_conifer = (scores_conifer > 0).astype(int)
    acc_conifer = accuracy_score(y_eval, preds_conifer)
    fpr_cn, tpr_cn, _ = roc_curve(y_eval, scores_conifer)
    auc_cn = auc(fpr_cn, tpr_cn)

    print("\n" + "=" * 50)
    print("           RESULTADOS COMPARATIVOS")
    print("=" * 50)
    print(f"Acurácia scikit-learn (Float64) : {acc_sklearn * 100:.2f}%")
    print(f"Acurácia Conifer (ap_fixed<18,8>): {acc_conifer * 100:.2f}%")
    print(f"AUC scikit-learn (Float64)      : {auc_sk:.4f}")
    print(f"AUC Conifer (ap_fixed<18,8>)    : {auc_cn:.4f}")
    print(f"Diferença de Acurácia           : {abs(acc_sklearn - acc_conifer) * 100:.3f}%")
    print(f"Diferença Absoluta de AUC       : {abs(auc_sk - auc_cn):.5f}")
    print("=" * 50)

    # 7. Gerar Gráficos: Curvas ROC Comparativas
    plot_path = os.path.join(os.path.dirname(__file__), "curva_roc_bdt.png")
    print(f"\nGerando curvas ROC comparativas em: {plot_path}")
    plt.figure(figsize=(8, 6))
    plt.plot(fpr_sk, tpr_sk, "b--", lw=1.8, label=f"scikit-learn Float64 (AUC = {auc_sk:.3f})")
    plt.plot(fpr_cn, tpr_cn, "r-", lw=1.8, label=f"Conifer ap_fixed<18,8> (AUC = {auc_cn:.3f})")
    plt.plot([0, 1], [0, 1], "k--", alpha=0.5, label="Baseline Aleatório (AUC = 0.500)")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("Taxa de Falso Positivo (FPR)", fontsize=11)
    plt.ylabel("Taxa de Verdadeiro Positivo (TPR)", fontsize=11)
    plt.title("Classificação de Top Quarks com BDT em FPGA (Conifer vs scikit-learn)", fontsize=12, fontweight="bold")
    plt.legend(loc="lower right", fontsize=10)
    plt.grid(True, linestyle=":", alpha=0.6)
    plt.tight_layout()
    plt.savefig(plot_path, dpi=300)
    plt.close()

    print("Gráfico salvo com sucesso!")
    print("\nExecução do Tutorial 06 finalizada com sucesso!")


if __name__ == "__main__":
    main()
