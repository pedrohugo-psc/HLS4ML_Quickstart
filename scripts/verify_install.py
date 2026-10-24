#!/usr/bin/env python3
"""
Script de verificacao da instalacao do ambiente hls4ml e dos clones oficiais.

Verifica:
1. Ambiente Python (3.12 na .venv)
2. Clones oficiais intactos (./hls4ml e ./hls4ml-tutorial com os 11 notebooks)
3. Pacotes Python requeridos pelos 11 tutoriais
4. Compiladores e ferramentas (g++, dot/graphviz e Vitis HLS opcional)
5. Teste funcional de conversao e compilacao C++ (C-Simulation)
"""

import os
import shutil
import subprocess
import sys
import tempfile
import numpy as np

PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

EXPECTED_NOTEBOOKS = [
    "1_getting_started/1a_train_keras.ipynb",
    "1_getting_started/1b_train_pytorch.ipynb",
    "1_getting_started/1c_hls4ml_synth.ipynb",
    "2_quantization/2a_qkeras.ipynb",
    "2_quantization/2b_brevitas.ipynb",
    "3_advanced_config/3a_reuse_factor.ipynb",
    "3_advanced_config/3b_profiling.ipynb",
    "4_advanced_models/4a_qkeras_cnn_svhn.ipynb",
    "4_advanced_models/4b_snn.ipynb",
    "6_more_models/6a_bdt.ipynb",
    "6_more_models/6b_symbolic_regression.ipynb",
]


def print_header(title):
    print("\n" + "=" * 68)
    print(f"  {title}")
    print("=" * 68)


def check_command(cmd):
    path = shutil.which(cmd)
    if path:
        return f"OK ({path})"
    return "Nao encontrado no PATH"


def main():
    ok = True

    print_header("1. Ambiente Python")
    print(f"  Interpretador : {sys.executable}")
    print(f"  Versao Python : {sys.version.split()[0]}")

    print_header("2. Clones Oficiais (fastmachinelearning)")
    for repo_name in ("hls4ml", "hls4ml-tutorial"):
        repo_path = os.path.join(PROJECT_DIR, repo_name)
        git_dir = os.path.join(repo_path, ".git")
        if os.path.isdir(git_dir):
            diff_out = subprocess.check_output(
                ["git", "-C", repo_path, "diff", "--name-only"], text=True
            ).strip()
            clean_str = "intacto (0 arquivos modificados)" if not diff_out else f"MODIFICADO ({diff_out})"
            if diff_out:
                ok = False
            print(f"  {repo_name:18} : OK — {clean_str}")
        else:
            print(f"  {repo_name:18} : AUSENTE ({repo_path})")
            ok = False

    tut_dir = os.path.join(PROJECT_DIR, "hls4ml-tutorial")
    missing_nb = [nb for nb in EXPECTED_NOTEBOOKS if not os.path.isfile(os.path.join(tut_dir, nb))]
    if not missing_nb:
        print(f"  Notebooks oficiais : OK ({len(EXPECTED_NOTEBOOKS)}/11 presentes)")
    else:
        print(f"  Notebooks ausentes : {missing_nb}")
        ok = False

    jet_dir = os.path.join(tut_dir, "data", "jet-tagging")
    jet_files = ["X_train_val.npy", "X_test.npy", "y_train_val.npy", "y_test.npy", "classes.npy"]
    if all(os.path.isfile(os.path.join(jet_dir, f)) for f in jet_files):
        print("  Dataset JetTagging : OK (hls4ml-tutorial/data/jet-tagging/)")
    else:
        print("  Dataset JetTagging : Pendente (sera baixado na execucao)")

    print_header("3. Pacotes Python")
    # Nota: juliacall/pysr deve inicializar o LLVM do Julia antes do import de torch/hls4ml
    packages = [
        ("pysr", "pysr"),
        ("hls4ml", "hls4ml"),
        ("tensorflow", "tensorflow"),
        ("keras", "keras"),
        ("qkeras", "qkeras"),
        ("torch", "torch"),
        ("brevitas", "brevitas"),
        ("qonnx", "qonnx"),
        ("snntorch", "snntorch"),
        ("onnx", "onnx"),
        ("conifer", "conifer"),
        ("xgboost", "xgboost"),
        ("sklearn", "scikit-learn"),
        ("pydot", "pydot"),
        ("jupyterlab", "jupyterlab"),
    ]
    for mod_name, label in packages:
        try:
            mod = __import__(mod_name)
            version = getattr(mod, "__version__", None)
            ver_str = f"v{version}" if version else "instalado"
            print(f"  {label:16} : OK ({ver_str})")
        except ImportError as exc:
            print(f"  {label:16} : AUSENTE ({exc})")
            ok = False

    print_header("4. Compiladores C/C++ e Ferramentas Auxiliares")
    print(f"  g++        : {check_command('g++')}")
    print(f"  gcc        : {check_command('gcc')}")
    print(f"  dot        : {check_command('dot')} (Graphviz para plot_model)")
    print(f"  vitis_hls  : {check_command('vitis_hls')} (opcional — apenas sintese RTL)")
    print(f"  vivado     : {check_command('vivado')} (opcional — apenas sintese RTL)")

    print_header("5. Teste de Conversao e Compilacao C++ (hls4ml)")
    temp_dir = tempfile.mkdtemp(prefix="hls4ml_verify_")
    original_cwd = os.getcwd()
    try:
        import torch
        import torch.nn as nn
        import hls4ml

        print("  [1/3] Instanciando rede de teste (PyTorch MLP)...")
        net = nn.Sequential(nn.Linear(16, 16), nn.ReLU(), nn.Linear(16, 5), nn.Softmax(dim=-1)).eval()
        cfg = hls4ml.utils.config_from_pytorch_model(net, input_shape=(16,), granularity="model", backend="Vitis")

        print("  [2/3] Convertendo modelo para C++/HLS e compilando com g++...")
        hls_model = hls4ml.converters.convert_from_pytorch_model(
            net, hls_config=cfg, backend="Vitis", output_dir=os.path.join(temp_dir, "hls_prj")
        )
        hls_model.compile()

        print("  [3/3] Executando inferencia em ponto fixo...")
        dummy_input = np.ones((1, 16), dtype=np.float32)
        prediction = hls_model.predict(dummy_input)
        print(f"  [OK] Inferencia executada com sucesso. Saida: {np.round(prediction, 4)}")
    except Exception as exc:
        print(f"\n[ERRO] Falha durante a verificacao C++: {exc}")
        ok = False
    finally:
        os.chdir(original_cwd)
        shutil.rmtree(temp_dir, ignore_errors=True)

    if not ok:
        print("\n[ERRO] Verificacao encontrou pendencias acima.")
        sys.exit(1)

    print("\n" + "=" * 68)
    print("  Verificacao concluida com 100% de sucesso.")
    print("=" * 68)


if __name__ == "__main__":
    main()
