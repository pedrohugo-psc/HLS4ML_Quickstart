#!/usr/bin/env bash
# Script de configuração do ambiente virtual Python (.venv) para o hls4ml

set -e

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_DIR"

echo "=========================================================="
echo "  Configuracao do Ambiente Virtual Python (hls4ml)"
echo "=========================================================="

# 1. Verificar Python 3
if ! command -v python3 &>/dev/null; then
    echo "[ERRO] python3 nao encontrado no sistema."
    exit 1
fi

echo "[INFO] Python detectado: $(python3 --version)"

# 2. Criar ambiente virtual caso nao exista
VENV_DIR="$PROJECT_DIR/.venv"
if [ ! -d "$VENV_DIR" ]; then
    echo "[INFO] Criando ambiente virtual em .venv..."
    python3 -m venv "$VENV_DIR"
else
    echo "[INFO] Ambiente virtual (.venv) ja existente."
fi

# 3. Atualizar ferramentas de empacotamento
echo "[INFO] Atualizando pip, setuptools e wheel..."
"$VENV_DIR/bin/pip" install --upgrade pip setuptools wheel

# 4. Instalar dependencias do requirements.txt
if [ -f "$PROJECT_DIR/requirements.txt" ]; then
    echo "[INFO] Instalando pacotes a partir de requirements.txt..."
    "$VENV_DIR/bin/pip" install -r "$PROJECT_DIR/requirements.txt"
else
    echo "[INFO] Instalando hls4ml[profiling]..."
    "$VENV_DIR/bin/pip" install "hls4ml[profiling]"
fi

# 5. Configurar cabecalhos C++ do Conifer (ap_types e nlohmann-json)
"$VENV_DIR/bin/python" -c "
import os, conifer, hls4ml, nlohmann_json
c_path = conifer.__path__[0]
h_path = os.path.dirname(hls4ml.__file__)
n_path = os.path.dirname(nlohmann_json.__file__)
os.makedirs(f'{c_path}/external/json', exist_ok=True)
if not os.path.exists(f'{c_path}/external/json/include'):
    os.symlink(f'{n_path}/include', f'{c_path}/external/json/include')
os.makedirs(f'{c_path}/external/ap_types', exist_ok=True)
if not os.path.exists(f'{c_path}/external/ap_types/include'):
    os.symlink(f'{h_path}/templates/vivado/ap_types', f'{c_path}/external/ap_types/include')
" 2>/dev/null || true

# 6. Verificar instalacao do CLI
if "$VENV_DIR/bin/hls4ml" --version &>/dev/null; then
    echo "[OK] $($VENV_DIR/bin/hls4ml --version) instalado com sucesso."
fi

echo "=========================================================="
echo "Configuracao finalizada."
echo "Para ativar o ambiente:  source .venv/bin/activate"
echo "Para testar o ambiente:  python scripts/verify_install.py"
echo "=========================================================="
