#!/usr/bin/env python3
"""
Script de verificação da instalação do ambiente hls4ml.

Verifica pacotes Python instalados, presença do compilador C++ (g++)
e executa um teste de conversão e compilação C-Simulation.
"""

import os
import sys
import shutil
import tempfile
import numpy as np


def print_header(title):
    print("\n" + "=" * 65)
    print(f"  {title}")
    print("=" * 65)


def check_command(cmd):
    path = shutil.which(cmd)
    if path:
        return f"OK ({path})"
    return "Nao encontrado no PATH"


def main():
    print_header("1. Ambiente Python")
    print(f"  Interpretador : {sys.executable}")
    print(f"  Versao Python : {sys.version.split()[0]}")

    print_header("2. Pacotes Python")
    packages = ["hls4ml", "torch", "onnx", "onnxoptimizer", "brevitas", "qonnx", "conifer", "numpy", "pandas", "matplotlib", "sklearn", "jupyterlab"]
    for pkg in packages:
        try:
            mod = __import__(pkg)
            version = getattr(mod, "__version__", "instalado")
            print(f"  {pkg:15} : OK (v{version})")
        except ImportError:
            print(f"  {pkg:15} : AUSENTE")

    print_header("3. Compiladores C/C++ e Ferramentas HLS")
    print(f"  g++        : {check_command('g++')}")
    print(f"  gcc        : {check_command('gcc')}")
    print(f"  vivado     : {check_command('vivado')} (opcional para C-Simulation)")
    print(f"  vitis_hls  : {check_command('vitis_hls')} (opcional para C-Simulation)")

    print_header("4. Teste de Conversao e Compilacao C++ (hls4ml)")
    temp_dir = tempfile.mkdtemp(prefix="hls4ml_verify_")
    try:
        import hls4ml
        original_cwd = os.getcwd()
        os.chdir(temp_dir)

        print("  [1/3] Carregando modelo de teste (KERAS_3layer)...")
        config = hls4ml.utils.fetch_example_model('KERAS_3layer.json')
        config['OutputDir'] = os.path.join(temp_dir, 'hls_prj')

        print("  [2/3] Convertendo modelo para C++/HLS e compilando com g++...")
        hls_model = hls4ml.converters.keras_v2_to_hls(config)
        hls_model.compile()

        print("  [3/3] Executando inferencia em ponto fixo...")
        dummy_input = np.ones((1, 16), dtype=np.float32)
        prediction = hls_model.predict(dummy_input)

        os.chdir(original_cwd)
        print(f"  [OK] Inferencia executada com sucesso. Saida: {prediction}")

    except Exception as e:
        print(f"\n[ERRO] Falha durante a verificacao: {e}")
        sys.exit(1)
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)

    print("\n" + "=" * 65)
    print("  Verificacao concluida com sucesso.")
    print("=" * 65)


if __name__ == "__main__":
    main()
