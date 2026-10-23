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
"$VENV_DIR/bin/pip" install --upgrade pip "setuptools>=77.0.3,<81" wheel

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

# 6. Preparar diretorios e dataset dos tutoriais (hls4ml_lhc_jets_hlf)
echo "[INFO] Verificando conjunto de dados dos tutoriais (hls4ml_lhc_jets_hlf)..."
PROJECT_DIR="$PROJECT_DIR" "$VENV_DIR/bin/python" - <<'EOF' || echo "[AVISO] Nao foi possivel preparar o dataset agora; ele sera baixado na execucao do tutorial."
import hashlib
import json
import os
import shutil
import sys
import tempfile
import time
import urllib.error
import urllib.request

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

project_dir = os.environ["PROJECT_DIR"]
data_dir = os.path.join(project_dir, "data", "jet-tagging")
models_dir = os.path.join(project_dir, "models")
os.makedirs(data_dir, exist_ok=True)
os.makedirs(models_dir, exist_ok=True)

required = ["X_train.npy", "y_train.npy", "X_test.npy", "y_test.npy", "classes.npy"]
if all(os.path.exists(os.path.join(data_dir, f)) for f in required):
    print("[OK] Dataset ja disponivel em data/jet-tagging/.")
    sys.exit(0)


def download_with_resume(url, dest_path, expected_md5=None, max_retries=10):
    part_path = dest_path + ".part"
    headers_base = {"User-Agent": "Mozilla/5.0 (hls4ml-quickstart)"}
    for attempt in range(1, max_retries + 1):
        try:
            existing = os.path.getsize(part_path) if os.path.exists(part_path) else 0
            headers = dict(headers_base)
            if existing > 0:
                headers["Range"] = f"bytes={existing}-"
            req = urllib.request.Request(url, headers=headers)
            try:
                resp = urllib.request.urlopen(req, timeout=30)
            except urllib.error.HTTPError as e:
                if e.code == 416 and existing > 0:
                    resp = None
                else:
                    raise
            if resp is not None:
                with resp:
                    status = getattr(resp, "status", resp.getcode())
                    if status == 206:
                        mode = "ab"
                        cr = resp.headers.get("Content-Range", "")
                        total = int(cr.split("/")[-1]) if "/" in cr and cr.split("/")[-1].isdigit() else None
                        downloaded = existing
                    else:
                        mode = "wb"
                        cl = resp.headers.get("Content-Length")
                        total = int(cl) if cl and cl.isdigit() else None
                        downloaded = 0
                    with open(part_path, mode) as out:
                        while True:
                            chunk = resp.read(256 * 1024)
                            if not chunk:
                                break
                            out.write(chunk)
                            downloaded += len(chunk)
                            if total:
                                pct = downloaded * 100.0 / total
                                sys.stdout.write(f"\r  Baixando: {downloaded / 1e6:.1f} / {total / 1e6:.1f} MB ({pct:.0f}%)")
                            else:
                                sys.stdout.write(f"\r  Baixando: {downloaded / 1e6:.1f} MB")
                            sys.stdout.flush()
                    sys.stdout.write("\n")
                    if total and os.path.getsize(part_path) < total:
                        raise IOError(f"Conexao encerrada prematuramente ({os.path.getsize(part_path)}/{total} bytes)")
            if expected_md5:
                md5 = hashlib.md5()
                with open(part_path, "rb") as fh:
                    for blk in iter(lambda: fh.read(1024 * 1024), b""):
                        md5.update(blk)
                if md5.hexdigest() != expected_md5:
                    os.remove(part_path)
                    raise ValueError(f"MD5 divergente: obtido {md5.hexdigest()}, esperado {expected_md5}")
            os.replace(part_path, dest_path)
            return
        except Exception as exc:
            if attempt == max_retries:
                raise
            wait_s = min(2 * attempt, 10)
            print(f"  [Tentativa {attempt}/{max_retries}] Oscilacao de rede ({exc}). Retomando em {wait_s}s...")
            time.sleep(wait_s)


tmp_dir = tempfile.mkdtemp(prefix="hls4ml_data_")
try:
    df = None
    pq_urls = []
    try:
        req = urllib.request.Request(
            "https://www.openml.org/api/v1/json/data/42468",
            headers={"User-Agent": "Mozilla/5.0 (hls4ml-quickstart)"},
        )
        with urllib.request.urlopen(req, timeout=10) as r:
            meta = json.loads(r.read().decode("utf-8")).get("data_set_description", {})
            for key in ("parquet_url", "minio_url"):
                u = meta.get(key)
                if u and u.endswith(".pq") and u not in pq_urls:
                    pq_urls.append(u)
    except Exception:
        pass

    for fallback_pq in (
        "https://data.openml.org/datasets/0004/42468/dataset_42468.pq",
        "https://openml1.win.tue.nl/datasets/0004/42468/dataset_42468.pq",
    ):
        if fallback_pq not in pq_urls:
            pq_urls.append(fallback_pq)

    pq_file = os.path.join(tmp_dir, "dataset_42468.pq")
    for url in pq_urls:
        try:
            print(f"  Obtendo dataset oficial em Parquet ({url})...")
            download_with_resume(url, pq_file, max_retries=5)
            df = pd.read_parquet(pq_file)
            break
        except Exception as e:
            print(f"  [Aviso] Espelho Parquet indisponivel ({e}).")
            if os.path.exists(pq_file):
                os.remove(pq_file)

    if df is not None:
        target_col = "class" if "class" in df.columns else df.columns[-1]
        X = df.drop(columns=[target_col]).to_numpy(dtype=np.float32)
        s = df[target_col]
        y_raw = (
            s.str.decode("utf-8").to_numpy()
            if len(s) > 0 and isinstance(s.iloc[0], (bytes, bytearray))
            else s.astype(str).to_numpy()
        )
    else:
        arff_url = "https://www.openml.org/data/download/21854360/hls4ml_HLF.arff"
        arff_file = os.path.join(tmp_dir, "hls4ml_HLF.arff")
        print(f"  Baixando ARFF oficial com retomada automatica ({arff_url})...")
        download_with_resume(arff_url, arff_file, expected_md5="b0cade1d843feb6199c746918453c378", max_retries=15)

        with open(arff_file, "r", encoding="utf-8") as fh:
            for line_idx, line in enumerate(fh):
                if line.strip().upper() == "@DATA":
                    data_start = line_idx + 1
                    break
        raw_df = pd.read_csv(arff_file, skiprows=data_start, header=None, comment="%")
        X = raw_df.iloc[:, :-1].to_numpy(dtype=np.float32)
        y_raw = raw_df.iloc[:, -1].astype(str).str.strip().str.strip("'\"").to_numpy()

    le = LabelEncoder()
    y_int = le.fit_transform(y_raw)
    classes = le.classes_
    y_onehot = np.eye(len(classes))[y_int]

    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y_onehot, test_size=0.2, random_state=42, stratify=y_int
    )

    for fname, arr in (
        ("X_train.npy", X_train),
        ("y_train.npy", y_train),
        ("X_test.npy", X_test),
        ("y_test.npy", y_test),
        ("classes.npy", classes),
    ):
        final_p = os.path.join(data_dir, fname)
        tmp_p = final_p + ".tmp"
        with open(tmp_p, "wb") as f_np:
            np.save(f_np, arr)
        os.replace(tmp_p, final_p)

    print(f"[OK] Dataset preparado em data/jet-tagging/ (Treino: {X_train.shape}, Teste: {X_test.shape}).")
finally:
    shutil.rmtree(tmp_dir, ignore_errors=True)
EOF

# 7. Verificar instalacao do CLI
if "$VENV_DIR/bin/hls4ml" --version &>/dev/null; then
    echo "[OK] $($VENV_DIR/bin/hls4ml --version) instalado com sucesso."
fi

echo "=========================================================="
echo "Configuracao finalizada."
echo "Para ativar o ambiente:  source .venv/bin/activate"
echo "Para testar o ambiente:  python scripts/verify_install.py"
echo "=========================================================="
