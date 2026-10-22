#!/usr/bin/env python3
"""
Tutorial 02: Quantizacao em FPGA (PTQ vs QAT com Brevitas e hls4ml).

Fluxo de execucao:
1. Carregamento dos dados do benchmark Jet Tagging
2. Avaliacao do modelo baseline em PyTorch (Float32)
3. Post-Training Quantization (PTQ): ap_fixed<16,6>, ap_fixed<8,3> e ap_fixed<6,2>
4. Quantization-Aware Training (QAT): treinamento de JetTaggerBrevitas em 6 bits
5. Exportacao QONNX, conversao para HLS e compilacao C++ com g++
6. Comparacao de acuracia e geracao das curvas ROC
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
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

from brevitas.export import export_qonnx
import qonnx.util.cleanup
from qonnx.core.modelwrapper import ModelWrapper
from qonnx.transformation.gemm_to_matmul import GemmToMatMul


def main():
    print("=" * 80)
    print("TUTORIAL 02: QUANTIZACAO (PTQ VS QAT COM BREVITAS E HLS4ML)")
    print("=" * 80)

    data_dir = os.path.join(ROOT_DIR, "data", "jet-tagging")
    models_dir = os.path.join(ROOT_DIR, "models")
    os.makedirs(models_dir, exist_ok=True)

    # 1. Carregando dados
    print("\n[Passo 1/6] Carregando dados de teste...")
    X_train = np.load(os.path.join(data_dir, "X_train.npy"))
    y_train = np.load(os.path.join(data_dir, "y_train.npy"))
    X_test = np.load(os.path.join(data_dir, "X_test.npy"))
    y_test = np.load(os.path.join(data_dir, "y_test.npy"))
    classes = np.load(os.path.join(data_dir, "classes.npy"), allow_pickle=True)

    eval_n = min(5000, len(X_test))
    print(f"  Amostras de avaliacao: {eval_n}")

    # 2. Carregando Modelo Baseline (Float32)
    print("\n[Passo 2/6] Avaliando modelo baseline (PyTorch Float32)...")
    model_baseline = models.JetTagger()
    weights_path = os.path.join(models_dir, "pytorch_weights_part1.pt")
    if os.path.exists(weights_path):
        model_baseline.load_state_dict(torch.load(weights_path))
    else:
        print("  Pesos do Tutorial 01 nao encontrados. Executando treinamento baseline...")
        train_x = torch.tensor(X_train[:30000], dtype=torch.float32)
        train_y = torch.tensor(np.argmax(y_train[:30000], axis=1), dtype=torch.long)
        loader = DataLoader(TensorDataset(train_x, train_y), batch_size=256, shuffle=True)
        opt = torch.optim.Adam(model_baseline.parameters(), lr=1e-3)
        crit = nn.CrossEntropyLoss()
        model_baseline.train()
        for _ in range(3):
            for bx, by in loader:
                opt.zero_grad()
                crit(model_baseline.logits(bx), by).backward()
                opt.step()
        torch.save(model_baseline.state_dict(), weights_path)

    model_baseline.eval()
    with torch.no_grad():
        y_baseline_pred = model_baseline(torch.tensor(X_test[:eval_n], dtype=torch.float32)).numpy()
    acc_baseline = accuracy_score(np.argmax(y_test[:eval_n], axis=1), np.argmax(y_baseline_pred, axis=1))
    print(f"  Acuracia Baseline (PyTorch Float32): {acc_baseline * 100:.2f}%")

    # 3. Post-Training Quantization (PTQ)
    print("\n[Passo 3/6] Executando Post-Training Quantization (PTQ)...")
    precisions_to_test = ['ap_fixed<16,6>', 'ap_fixed<8,3>', 'ap_fixed<6,2>']
    ptq_results = {}

    for prec in precisions_to_test:
        print(f"\n  Convertendo e compilando com precisao: {prec}...")
        cfg = hls4ml.utils.config_from_pytorch_model(
            model_baseline,
            input_shape=(16,),
            granularity='model',
            backend='Vivado',
        )
        cfg['Model']['Precision']['default'] = prec
        cfg['Model']['ReuseFactor'] = 1

        out_dir = os.path.join(CURRENT_DIR, f"hls_prj_ptq_{prec.replace('<','_').replace('>','').replace(',','_')}")
        h_model = hls4ml.converters.convert_from_pytorch_model(
            model_baseline,
            hls_config=cfg,
            output_dir=out_dir,
            part='xcu200-fsgd2104-2-e'
        )
        h_model.compile()
        y_ptq = h_model.predict(X_test[:eval_n])
        acc_ptq = accuracy_score(np.argmax(y_test[:eval_n], axis=1), np.argmax(y_ptq, axis=1))
        ptq_results[prec] = (acc_ptq, y_ptq)
        print(f"  Acuracia PTQ ({prec}): {acc_ptq * 100:.2f}% (Diferenca: {abs(acc_baseline - acc_ptq)*100:.2f}%)")

    # 4. Quantization-Aware Training (QAT) com Brevitas
    print("\n[Passo 4/6] Treinando modelo com Quantization-Aware Training (Brevitas 6-bit)...")
    model_brevitas = models.JetTaggerBrevitas()
    brevitas_weights_path = os.path.join(models_dir, "brevitas_weights_part2.pt")

    n_samples = 50000
    train_x = torch.tensor(X_train[:n_samples], dtype=torch.float32)
    train_y = torch.tensor(np.argmax(y_train[:n_samples], axis=1), dtype=torch.long)
    loader = DataLoader(TensorDataset(train_x, train_y), batch_size=256, shuffle=True)

    opt = torch.optim.Adam(model_brevitas.parameters(), lr=1e-3)
    crit = nn.CrossEntropyLoss()

    model_brevitas.train()
    for ep in range(1, 6):
        tot_loss = 0.0
        for bx, by in loader:
            opt.zero_grad()
            logits = model_brevitas.logits(bx)
            loss = crit(logits, by)
            loss.backward()
            opt.step()
            tot_loss += loss.item()
        print(f"  Epoca {ep}/5 | Loss QAT: {tot_loss / len(loader):.4f}")

    torch.save(model_brevitas.state_dict(), brevitas_weights_path)
    print(f"  Pesos QAT salvos em: {brevitas_weights_path}")

    model_brevitas.eval()
    with torch.no_grad():
        y_brevitas_float = model_brevitas(torch.tensor(X_test[:eval_n], dtype=torch.float32)).numpy()
    acc_brevitas_float = accuracy_score(np.argmax(y_test[:eval_n], axis=1), np.argmax(y_brevitas_float, axis=1))
    print(f"  Acuracia Brevitas (6-bit em software): {acc_brevitas_float * 100:.2f}%")

    # 5. Exportacao para QONNX e Conversao no hls4ml
    print("\n[Passo 5/6] Exportando grafo QONNX e compilando C++/HLS...")
    raw_onnx = os.path.join(CURRENT_DIR, "brevitas_raw.onnx")
    clean_onnx = os.path.join(CURRENT_DIR, "brevitas_clean.onnx")

    export_qonnx(model_brevitas, args=torch.randn(1, 16), export_path=raw_onnx, opset_version=14, dynamo=False)
    qonnx.util.cleanup.cleanup(raw_onnx, out_file=clean_onnx)
    m_wrap = ModelWrapper(clean_onnx)
    m_wrap = m_wrap.transform(GemmToMatMul())
    m_wrap = qonnx.util.cleanup.cleanup_model(m_wrap)
    m_wrap.save(clean_onnx)

    onnx_cfg = hls4ml.utils.config_from_onnx_model(m_wrap, granularity='name', backend='Vivado')
    onnx_out_dir = os.path.join(CURRENT_DIR, "hls4ml_prj_brevitas_6bit")

    hls_brevitas = hls4ml.converters.convert_from_onnx_model(
        m_wrap,
        output_dir=onnx_out_dir,
        backend='Vivado',
        hls_config=onnx_cfg,
        part='xcu200-fsgd2104-2-e'
    )
    hls_brevitas.compile()

    y_hls_brevitas = hls_brevitas.predict(np.ascontiguousarray(X_test[:eval_n], dtype=np.float32))
    acc_hls_brevitas = accuracy_score(np.argmax(y_test[:eval_n], axis=1), np.argmax(y_hls_brevitas, axis=1))

    if os.path.exists(raw_onnx):
        os.remove(raw_onnx)
    if os.path.exists(clean_onnx):
        os.remove(clean_onnx)

    # 6. Comparativo e Curvas ROC
    print("\n[Passo 6/6] Resultados comparativos e geracao das curvas ROC...")
    print("-" * 65)
    print(f"{'Configuracao':<30} | {'Largura de Bits':<15} | {'Acuracia':<10}")
    print("-" * 65)
    print(f"{'PyTorch Baseline':<30} | {'Float32 (32 bits)':<15} | {acc_baseline*100:.2f}%")
    print(f"{'PTQ ap_fixed<16,6>':<30} | {'16 bits':<15} | {ptq_results['ap_fixed<16,6>'][0]*100:.2f}%")
    print(f"{'PTQ ap_fixed<8,3>':<30} | {'8 bits':<15} | {ptq_results['ap_fixed<8,3>'][0]*100:.2f}%")
    print(f"{'PTQ ap_fixed<6,2>':<30} | {'6 bits (PTQ)':<15} | {ptq_results['ap_fixed<6,2>'][0]*100:.2f}%")
    print(f"{'QAT Brevitas (hls4ml)':<30} | {'6 bits (QAT)':<15} | {acc_hls_brevitas*100:.2f}%")
    print("-" * 65)

    fig, ax = plt.subplots(figsize=(8, 7))
    plotting.makeRoc(y_test[:eval_n], y_baseline_pred, list(classes), linestyle='-')
    plt.gca().set_prop_cycle(None)
    plotting.makeRoc(y_test[:eval_n], ptq_results['ap_fixed<6,2>'][1], list(classes), linestyle=':')
    plt.gca().set_prop_cycle(None)
    plotting.makeRoc(y_test[:eval_n], y_hls_brevitas, list(classes), linestyle='--')

    from matplotlib.lines import Line2D
    lines = [
        Line2D([0], [0], color='black', linestyle='-', label='PyTorch Baseline (Float32)'),
        Line2D([0], [0], color='black', linestyle=':', label='PTQ (ap_fixed<6,2>)'),
        Line2D([0], [0], color='black', linestyle='--', label='QAT Brevitas (6 bits)')
    ]
    ax.legend(handles=lines, loc='lower right')
    plt.title("Comparativo de Quantizacao: Float32 vs PTQ (6-bit) vs QAT (6-bit)")
    roc_path = os.path.join(CURRENT_DIR, "curva_roc_quantizacao.png")
    plt.tight_layout()
    plt.savefig(roc_path, dpi=150)
    plt.close()
    print(f"Grafico salvo em: {roc_path}")
    print("Execucao do Tutorial 02 finalizada.")


if __name__ == "__main__":
    main()
