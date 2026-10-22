#!/usr/bin/env python3
"""
Tutorial 03: Configuracoes Avancadas de Hardware (ReuseFactor e Profiling Numerico).

Fluxo de execucao:
1. Carregamento dos dados e do modelo PyTorch (JetTagger)
2. Avaliacao do ReuseFactor (R = 1, 4, 8)
3. Profiling numerico de pesos e ativacoes (hls4ml.model.profiling.numerical)
4. Configuracao de precisao heterogenea por camada (granularity='name')
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
import hls4ml
from hls4ml.model.profiling import numerical

import torch


def main():
    print("=" * 80)
    print("TUTORIAL 03: REUSE FACTOR E PROFILING NUMERICO COM HLS4ML")
    print("=" * 80)

    data_dir = os.path.join(ROOT_DIR, "data", "jet-tagging")
    models_dir = os.path.join(ROOT_DIR, "models")

    # 1. Carregamento dos dados e modelo
    print("\n[Passo 1/4] Carregando modelo PyTorch e conjunto de teste...")
    X_test = np.load(os.path.join(data_dir, "X_test.npy"))
    y_test = np.load(os.path.join(data_dir, "y_test.npy"))

    eval_n = min(5000, len(X_test))

    model = models.JetTagger()
    os.makedirs(models_dir, exist_ok=True)
    weights_path = os.path.join(models_dir, "pytorch_weights_part1.pt")
    if os.path.exists(weights_path):
        model.load_state_dict(torch.load(weights_path))
    else:
        print("  Pesos baseline nao encontrados. Treinando modelo base (3 epocas)...")
        X_train = np.load(os.path.join(data_dir, "X_train.npy"))
        y_train = np.load(os.path.join(data_dir, "y_train.npy"))
        from torch.utils.data import DataLoader, TensorDataset
        loader = DataLoader(
            TensorDataset(
                torch.tensor(X_train[:30000], dtype=torch.float32),
                torch.tensor(np.argmax(y_train[:30000], axis=1), dtype=torch.long),
            ),
            batch_size=256,
            shuffle=True,
        )
        opt = torch.optim.Adam(model.parameters(), lr=1e-3)
        crit = torch.nn.CrossEntropyLoss()
        model.train()
        for _ in range(3):
            for bx, by in loader:
                opt.zero_grad()
                crit(model.logits(bx), by).backward()
                opt.step()
        torch.save(model.state_dict(), weights_path)
    model.eval()

    with torch.no_grad():
        y_torch = model(torch.tensor(X_test[:eval_n], dtype=torch.float32)).numpy()
    acc_torch = accuracy_score(np.argmax(y_test[:eval_n], axis=1), np.argmax(y_torch, axis=1))
    print(f"  Acuracia PyTorch Baseline (Float32): {acc_torch * 100:.2f}%")

    # 2. Avaliacao de ReuseFactor (1, 4, 8)
    print("\n[Passo 2/4] Compilando e simulando para ReuseFactor = 1, 4 e 8...")
    reuse_factors = [1, 4, 8]
    rf_results = {}

    for rf in reuse_factors:
        cfg = hls4ml.utils.config_from_pytorch_model(
            model,
            input_shape=(16,),
            granularity='model',
            backend='Vivado',
        )
        cfg['Model']['Precision']['default'] = 'ap_fixed<16,6>'
        cfg['Model']['ReuseFactor'] = rf

        out_dir = os.path.join(CURRENT_DIR, f"hls_prj_reuse_{rf}")
        h_model = hls4ml.converters.convert_from_pytorch_model(
            model,
            hls_config=cfg,
            output_dir=out_dir,
            part='xcu200-fsgd2104-2-e'
        )
        h_model.compile()
        y_rf = h_model.predict(X_test[:eval_n])
        acc_rf = accuracy_score(np.argmax(y_test[:eval_n], axis=1), np.argmax(y_rf, axis=1))
        rf_results[rf] = acc_rf
        print(f"  ReuseFactor = {rf:<2} | Acuracia: {acc_rf * 100:.2f}%")

    # 3. Profiling Numerico
    print("\n[Passo 3/4] Executando profiling numerico de pesos e ativacoes (Trace=True)...")
    prof_cfg = hls4ml.utils.config_from_pytorch_model(
        model,
        input_shape=(16,),
        granularity='name',
        backend='Vivado',
    )
    for layer in prof_cfg['LayerName'].keys():
        prof_cfg['LayerName'][layer]['Trace'] = True

    prof_dir = os.path.join(CURRENT_DIR, "hls_prj_profiling")
    hls_prof = hls4ml.converters.convert_from_pytorch_model(
        model,
        hls_config=prof_cfg,
        output_dir=prof_dir,
        part='xcu200-fsgd2104-2-e'
    )
    hls_prof.compile()

    plt.figure()
    numerical(model=model, hls_model=hls_prof, X=X_test[:500])
    prof_img_path = os.path.join(CURRENT_DIR, "profiling_pesos_ativacoes.png")
    plt.savefig(prof_img_path, dpi=150, bbox_inches='tight')
    plt.close('all')
    print(f"  Grafico de profiling salvo em: {prof_img_path}")

    # 4. Configuracao Heterogenea por Camada
    print("\n[Passo 4/4] Avaliando precisao customizada por camada (granularity='name')...")
    custom_cfg = hls4ml.utils.config_from_pytorch_model(
        model,
        input_shape=(16,),
        granularity='name',
        backend='Vivado',
    )
    custom_cfg['LayerName']['fc1']['Precision']['weight'] = 'ap_fixed<8,2>'
    custom_cfg['LayerName']['fc1']['Precision']['result'] = 'ap_fixed<8,2>'
    custom_cfg['LayerName']['output']['Precision']['weight'] = 'ap_fixed<16,6>'
    custom_cfg['LayerName']['output']['Precision']['result'] = 'ap_fixed<16,6>'

    custom_dir = os.path.join(CURRENT_DIR, "hls_prj_custom_precision")
    hls_custom = hls4ml.converters.convert_from_pytorch_model(
        model,
        hls_config=custom_cfg,
        output_dir=custom_dir,
        part='xcu200-fsgd2104-2-e'
    )
    hls_custom.compile()
    y_custom = hls_custom.predict(X_test[:eval_n])
    acc_custom = accuracy_score(np.argmax(y_test[:eval_n], axis=1), np.argmax(y_custom, axis=1))

    print("\n" + "=" * 55)
    print("               RESULTADOS COMPARATIVOS")
    print("=" * 55)
    print(f"PyTorch Float32 Baseline             : {acc_torch*100:.2f}%")
    print(f"Homogeneo 16-bit (ap_fixed<16,6>)    : {rf_results[1]*100:.2f}%")
    print(f"Heterogeneo (fc1: 8-bit, demais: 16b): {acc_custom*100:.2f}%")
    print("=" * 55)
    print("Execucao do Tutorial 03 finalizada.")


if __name__ == "__main__":
    main()
