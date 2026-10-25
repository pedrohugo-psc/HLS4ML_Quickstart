#!/usr/bin/env bash
# =============================================================================
# Script de Configuracao Completa:
# 1. Clona os repositorios oficiais fastmachinelearning/hls4ml e hls4ml-tutorial
# 2. Provisiona Python 3.12 isolado (.venv) via uv local (sem exigir sudo)
# 3. Instala todas as dependencias dos 11 tutoriais + hls4ml em modo editavel
# 4. Configura shims de compatibilidade exclusivamente dentro da .venv
# 5. Baixa e prepara o dataset oficial hls4ml_lhc_jets_hlf com retomada HTTP
# 6. Pre-inicializa o backend Julia do PySR confinado na .venv
# =============================================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
VENV_DIR="$PROJECT_DIR/.venv"
UV_DIR="$PROJECT_DIR/.uv"
UV_BIN="$UV_DIR/bin/uv"

echo "=================================================================="
echo "  Configuracao do Ambiente Oficial hls4ml + hls4ml-tutorial"
echo "=================================================================="

# 1. Verificar ferramentas basicas do sistema
for cmd in git g++; do
    if ! command -v "$cmd" &>/dev/null; then
        echo "[ERRO] '$cmd' nao encontrado. Instale as dependencias do sistema:"
        echo "  sudo apt update && sudo apt install -y build-essential g++ python3 git curl graphviz"
        exit 1
    fi
done

# 2. Clonar repositorios oficiais (100% intocados)
if [ ! -d "$PROJECT_DIR/hls4ml/.git" ] || [ ! -f "$PROJECT_DIR/hls4ml/pyproject.toml" ]; then
    echo "[1/6] Clonando repositorio oficial fastmachinelearning/hls4ml..."
    rm -rf "$PROJECT_DIR/hls4ml"
    git clone https://github.com/fastmachinelearning/hls4ml.git "$PROJECT_DIR/hls4ml"
else
    echo "[1/6] Repositorio oficial ./hls4ml ja presente."
fi

if [ ! -d "$PROJECT_DIR/hls4ml-tutorial/.git" ]; then
    echo "[1/6] Clonando repositorio oficial fastmachinelearning/hls4ml-tutorial..."
    rm -rf "$PROJECT_DIR/hls4ml-tutorial"
    git clone https://github.com/fastmachinelearning/hls4ml-tutorial.git "$PROJECT_DIR/hls4ml-tutorial"
else
    echo "[1/6] Repositorio oficial ./hls4ml-tutorial ja presente."
fi

# 3. Instalar uv localmente (dentro de .uv/) e criar .venv com Python 3.12
if [ ! -x "$UV_BIN" ]; then
    echo "[2/6] Baixando gerenciador Python local (uv) em .uv/..."
    mkdir -p "$UV_DIR/bin"
    if command -v curl &>/dev/null; then
        curl -LsSf https://astral.sh/uv/install.sh | env UV_INSTALL_DIR="$UV_DIR/bin" UV_NO_MODIFY_PATH=1 sh
    else
        python3 -c "import urllib.request, subprocess, os; script = urllib.request.urlopen('https://astral.sh/uv/install.sh').read(); env = dict(os.environ, UV_INSTALL_DIR='$UV_DIR/bin', UV_NO_MODIFY_PATH='1'); subprocess.run(['sh'], input=script, env=env, check=True)"
    fi
fi

export UV_PYTHON_INSTALL_DIR="$UV_DIR/python"
export UV_HTTP_TIMEOUT=120

NEED_VENV=1
if [ -x "$VENV_DIR/bin/python" ]; then
    PY_VER=$("$VENV_DIR/bin/python" -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')" 2>/dev/null || echo "")
    if [ "$PY_VER" = "3.12" ]; then
        NEED_VENV=0
    fi
fi

if [ "$NEED_VENV" -eq 1 ]; then
    echo "[2/6] Provisionando ambiente virtual .venv com Python 3.12..."
    rm -rf "$VENV_DIR"
    "$UV_BIN" venv --python 3.12 --seed "$VENV_DIR"
else
    echo "[2/6] Ambiente virtual .venv (Python 3.12) ja existente."
fi

# 4. Instalar dependencias Python + clone oficial ./hls4ml em modo editavel
echo "[3/6] Instalando pacotes Python dos 11 tutoriais oficiais..."
"$UV_BIN" pip install --python "$VENV_DIR/bin/python" "setuptools>=77.0.3,<81" "setuptools-scm>=8" wheel

if "$UV_BIN" pip install --help 2>&1 | grep -q -- '--torch-backend'; then
    TORCH_OPTS=(--torch-backend=cpu)
else
    TORCH_OPTS=(--extra-index-url https://download.pytorch.org/whl/cpu --index-strategy unsafe-best-match)
fi

"$UV_BIN" pip install --python "$VENV_DIR/bin/python" \
    --no-build-isolation-package pyparser \
    "${TORCH_OPTS[@]}" \
    -r "$PROJECT_DIR/requirements.txt"

echo "[3/6] Instalando ./hls4ml oficial em modo editavel..."
"$UV_BIN" pip install --python "$VENV_DIR/bin/python" \
    -e "$PROJECT_DIR/hls4ml[keras-v3,qkeras-v3,onnx,sr,profiling,snn]"

# Garantir faixa estavel do setuptools para coexistencia entre torch e brevitas
"$UV_BIN" pip install --python "$VENV_DIR/bin/python" "setuptools>=77.0.3,<81"

# 5. Configurar shims de compatibilidade exclusivamente dentro da .venv
echo "[4/6] Configurando shims de compatibilidade isolados na .venv..."
PROJECT_DIR="$PROJECT_DIR" VENV_DIR="$VENV_DIR" "$VENV_DIR/bin/python" - <<'PYSHIM'
import json
import os
import shutil
import site
import sys

project_dir = os.environ["PROJECT_DIR"]
venv_dir = os.environ["VENV_DIR"]
site_pkgs = site.getsitepackages()[0]

for _f in list(sys.meta_path):
    if getattr(_f, "__name__", "") == "_EditableFinder":
        sys.meta_path.remove(_f)
        sys.meta_path.insert(0, _f)

import hls4ml
import conifer

# 5a. Symlink de ap_types para simulacao C++ do Conifer (Tutorial 6a)
h_path = os.path.dirname(hls4ml.__file__) if getattr(hls4ml, "__file__", None) else os.path.join(project_dir, "hls4ml", "hls4ml")
c_path = os.path.dirname(conifer.__file__) if getattr(conifer, "__file__", None) else list(conifer.__path__)[0]
ap_src = os.path.join(h_path, "templates", "vivado", "ap_types")
os.makedirs(os.path.join(c_path, "external", "ap_types"), exist_ok=True)
conifer_ap = os.path.join(c_path, "external", "ap_types", "include")
if os.path.lexists(conifer_ap):
    if os.path.isdir(conifer_ap) and not os.path.islink(conifer_ap):
        shutil.rmtree(conifer_ap)
    else:
        os.remove(conifer_ap)
os.symlink(ap_src, conifer_ap)

# 5b. Cabecalhos ap_types + hls_math.h C++ puro para Regressao Simbolica (Tutorial 6b sem Vitis HLS)
shim_inc = os.path.join(venv_dir, "share", "hls_shim", "include")
if os.path.exists(shim_inc):
    shutil.rmtree(shim_inc)
os.makedirs(os.path.dirname(shim_inc), exist_ok=True)
shutil.copytree(ap_src, shim_inc)

hls_math_h = r'''#ifndef HLS_MATH_H_SHIM
#define HLS_MATH_H_SHIM

#include <cmath>
#include "ap_fixed.h"

namespace hls {

template<int W, int I, bool S, ap_q_mode Q, ap_o_mode O, int N>
inline double _shim_to_double(const ap_fixed_base<W, I, S, Q, O, N>& v) {
    return v.to_double();
}
inline double _shim_to_double(double v) { return v; }
inline double _shim_to_double(float v) { return static_cast<double>(v); }
inline double _shim_to_double(int v) { return static_cast<double>(v); }

#define HLS_SHIM_UNARY(fname, stdfname) \
    template<int W, int I, typename T> \
    inline ap_fixed<W, I> fname(const T& x) { \
        return ap_fixed<W, I>(std::stdfname(_shim_to_double(x))); \
    } \
    template<typename T> \
    inline T fname(const T& x) { \
        return T(std::stdfname(_shim_to_double(x))); \
    }

HLS_SHIM_UNARY(sin, sin)
HLS_SHIM_UNARY(cos, cos)
HLS_SHIM_UNARY(tan, tan)
HLS_SHIM_UNARY(asin, asin)
HLS_SHIM_UNARY(acos, acos)
HLS_SHIM_UNARY(atan, atan)
HLS_SHIM_UNARY(sinh, sinh)
HLS_SHIM_UNARY(cosh, cosh)
HLS_SHIM_UNARY(tanh, tanh)
HLS_SHIM_UNARY(asinh, asinh)
HLS_SHIM_UNARY(acosh, acosh)
HLS_SHIM_UNARY(atanh, atanh)
HLS_SHIM_UNARY(exp, exp)
HLS_SHIM_UNARY(exp2, exp2)
HLS_SHIM_UNARY(expm1, expm1)
HLS_SHIM_UNARY(log, log)
HLS_SHIM_UNARY(log10, log10)
HLS_SHIM_UNARY(log2, log2)
HLS_SHIM_UNARY(log1p, log1p)
HLS_SHIM_UNARY(sqrt, sqrt)
HLS_SHIM_UNARY(cbrt, cbrt)
HLS_SHIM_UNARY(erf, erf)
HLS_SHIM_UNARY(erfc, erfc)
HLS_SHIM_UNARY(lgamma, lgamma)
HLS_SHIM_UNARY(tgamma, tgamma)
HLS_SHIM_UNARY(ceil, ceil)
HLS_SHIM_UNARY(floor, floor)
HLS_SHIM_UNARY(fabs, fabs)
HLS_SHIM_UNARY(abs, fabs)

#undef HLS_SHIM_UNARY

template<int W, int I, typename T>
inline ap_fixed<W, I> recip(const T& x) {
    double d = _shim_to_double(x);
    return ap_fixed<W, I>(d != 0.0 ? (1.0 / d) : 0.0);
}
inline double recip(double x) { return x != 0.0 ? (1.0 / x) : 0.0; }
inline float recip(float x) { return x != 0.0f ? (1.0f / x) : 0.0f; }

template<int W, int I, typename TBase, typename TExp>
inline ap_fixed<W, I> pow(const TBase& base, const TExp& exp_val) {
    return ap_fixed<W, I>(std::pow(_shim_to_double(base), _shim_to_double(exp_val)));
}
inline double pow(double b, double e) { return std::pow(b, e); }
inline float pow(float b, float e) { return std::pow(b, e); }

template<int W, int I, typename T1, typename T2>
inline ap_fixed<W, I> atan2(const T1& y, const T2& x) {
    return ap_fixed<W, I>(std::atan2(_shim_to_double(y), _shim_to_double(x)));
}
inline double atan2(double y, double x) { return std::atan2(y, x); }
inline float atan2(float y, float x) { return std::atan2(y, x); }

template<int W, int I, typename T1, typename T2>
inline ap_fixed<W, I> hypot(const T1& x, const T2& y) {
    return ap_fixed<W, I>(std::hypot(_shim_to_double(x), _shim_to_double(y)));
}
inline double hypot(double x, double y) { return std::hypot(x, y); }
inline float hypot(float x, float y) { return std::hypot(x, y); }

} // namespace hls

#endif
'''
with open(os.path.join(shim_inc, "hls_math.h"), "w", encoding="utf-8") as fh:
    fh.write(hls_math_h)

# 5c. sitecustomize.py em .venv/lib/python3.12/site-packages/
sitecustomize_code = f'''"""
Shims leves de ambiente para execucao transparente do fastmachinelearning/hls4ml-tutorial
sem modificar nenhum arquivo dos clones oficiais.
"""
import importlib.abc
import os
import shutil
import sys

_VENV_PREFIX = sys.prefix
_PROJECT_ROOT = {project_dir!r}
_SHIM_ROOT = os.path.join(_VENV_PREFIX, "share", "hls_shim")

# Evita KeyError: 'XILINX_VITIS' nos 9 notebooks oficiais quando o Vitis HLS nao esta instalado
os.environ.setdefault("XILINX_VITIS", _SHIM_ROOT)
os.environ.setdefault("KERAS_HOME", os.path.join(_VENV_PREFIX, "share", "keras"))
os.environ.setdefault("IPYTHONDIR", os.path.join(_VENV_PREFIX, "etc", "ipython"))
os.environ.setdefault("JUPYTER_CONFIG_DIR", os.path.join(_VENV_PREFIX, "etc", "jupyter"))
os.environ.setdefault("JULIA_DEPOT_PATH", os.path.join(_VENV_PREFIX, "share", "julia"))
os.environ.setdefault("PYTHON_JULIAPKG_PROJECT", os.path.join(_VENV_PREFIX, "share", "pyjuliapkg"))
os.environ.setdefault("PYTHON_JULIACALL_HANDLE_SIGNALS", "yes")
os.environ.setdefault("PYTHON_JULIACALL_THREADS", "auto")
os.environ.setdefault("PYTHON_JULIACALL_AUTOLOAD_IPYTHON_EXTENSION", "no")
os.environ.setdefault("JULIA_PKG_IGNORE_HASHES", "1")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")


def _download_with_resume(url, dest_path, max_retries=8, timeout=30):
    import time
    import urllib.error
    import urllib.request

    os.makedirs(os.path.dirname(dest_path), exist_ok=True)
    part_path = dest_path + ".part"
    headers_base = {{"User-Agent": "Mozilla/5.0 (hls4ml-quickstart)"}}
    for attempt in range(1, max_retries + 1):
        try:
            existing = os.path.getsize(part_path) if os.path.exists(part_path) else 0
            headers = dict(headers_base)
            if existing > 0:
                headers["Range"] = f"bytes={{existing}}-"
            req = urllib.request.Request(url, headers=headers)
            try:
                resp = urllib.request.urlopen(req, timeout=timeout)
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
                                sys.stdout.write(f"\\r  Baixando: {{downloaded / 1e6:.1f}} / {{total / 1e6:.1f}} MB ({{pct:.0f}}%)")
                            else:
                                sys.stdout.write(f"\\r  Baixando: {{downloaded / 1e6:.1f}} MB")
                            sys.stdout.flush()
                    sys.stdout.write("\\n")
                    if total and os.path.getsize(part_path) < total:
                        raise IOError(f"Conexao encerrada ({{os.path.getsize(part_path)}}/{{total}} bytes)")
            os.replace(part_path, dest_path)
            return dest_path
        except Exception as exc:
            if attempt == max_retries:
                raise
            wait_s = min(2 * attempt, 10)
            print(f"  [Tentativa {{attempt}}/{{max_retries}}] Oscilacao de rede ({{exc}}). Retomando em {{wait_s}}s...")
            time.sleep(wait_s)


def _patch_sklearn_datasets(mod):
    if getattr(mod, "_hls4ml_patched", False):
        return
    orig_fetch = mod.fetch_openml

    def _resilient_fetch_openml(name=None, *args, **kwargs):
        if name == "hls4ml_lhc_jets_hlf":
            import pandas as pd
            from sklearn.utils import Bunch

            pq_cache = os.path.join(
                _PROJECT_ROOT, "hls4ml-tutorial", "data", "jet-tagging", ".cache", "dataset_42468.pq"
            )
            if not os.path.isfile(pq_cache):
                _download_with_resume(
                    "https://data.openml.org/datasets/0004/42468/dataset_42468.pq", pq_cache
                )
            df = pd.read_parquet(pq_cache)
            target_col = "class" if "class" in df.columns else df.columns[-1]
            s = df[target_col]
            if len(s) > 0 and isinstance(s.iloc[0], (bytes, bytearray)):
                df[target_col] = s.str.decode("utf-8")
            else:
                df[target_col] = s.astype(str)
            df[target_col] = df[target_col].astype("category")
            X_df = df.drop(columns=[target_col])
            y_ser = df[target_col]
            if kwargs.get("return_X_y", False):
                return X_df, y_ser
            return Bunch(
                data=X_df,
                target=y_ser,
                frame=df,
                categories=None,
                feature_names=list(X_df.columns),
                target_names=[target_col],
                DESCR="hls4ml_lhc_jets_hlf (OpenML #42468)",
                details={{"id": "42468", "name": "hls4ml_lhc_jets_hlf"}},
                url="https://www.openml.org/d/42468",
            )
        return orig_fetch(name, *args, **kwargs)

    mod.fetch_openml = _resilient_fetch_openml
    mod._hls4ml_patched = True


def _patch_keras_sequential(mod):
    seq_cls = getattr(mod, "Sequential", None)
    if seq_cls is None or getattr(seq_cls, "_hls4ml_seq_patched", False):
        return
    orig_prop = getattr(seq_cls, "input", None)
    orig_fget = getattr(orig_prop, "fget", None)
    if orig_fget is None:
        return

    def _compat_seq_input(self):
        try:
            return orig_fget(self)
        except AttributeError:
            inps = getattr(self, "inputs", None)
            if inps:
                return inps[0] if len(inps) == 1 else inps
            raise

    seq_cls.input = property(_compat_seq_input)
    seq_cls._hls4ml_seq_patched = True


def _patch_keras_utils(mod):
    _patch_keras_sequential(mod)
    if getattr(mod, "_hls4ml_patched", False) or not hasattr(mod, "get_file"):
        return
    orig_get_file = mod.get_file

    def _resilient_get_file(fname=None, origin=None, *args, **kwargs):
        if origin and "ufldl.stanford.edu/housenumbers/" in str(origin) and fname in ("train_32x32.mat", "test_32x32.mat"):
            cache_dir = kwargs.get("cache_dir") or os.environ.get("KERAS_HOME") or os.path.join(_VENV_PREFIX, "share", "keras")
            cache_subdir = kwargs.get("cache_subdir", "datasets")
            dest_dir = os.path.join(str(cache_dir), str(cache_subdir))
            dest_mat = os.path.join(dest_dir, fname)
            if os.path.isfile(dest_mat):
                return dest_mat
            try:
                return _download_with_resume(origin, dest_mat, max_retries=1, timeout=8)
            except Exception:
                import io
                import numpy as np
                import pandas as pd
                import scipy.io
                from PIL import Image

                split_name = "train" if fname.startswith("train") else "test"
                hf_url = f"https://huggingface.co/datasets/ufldl-stanford/svhn/resolve/main/cropped_digits/{{split_name}}-00000-of-00001.parquet"
                pq_tmp = dest_mat + ".parquet"
                print(f"  [Info] Servidor Stanford indisponivel; baixando {{fname}} do espelho oficial HuggingFace (ufldl-stanford/svhn)...")
                _download_with_resume(hf_url, pq_tmp, max_retries=6, timeout=30)
                df = pd.read_parquet(pq_tmp)
                imgs = [
                    np.asarray(Image.open(io.BytesIO(item["bytes"])).convert("RGB"), dtype=np.uint8)
                    for item in df["image"]
                ]
                X_mat = np.stack(imgs, axis=-1)  # (32, 32, 3, N)
                labels = df["label"].to_numpy(dtype=np.uint8)
                y_mat = np.where(labels == 0, 10, labels).astype(np.uint8).reshape(-1, 1)
                os.makedirs(dest_dir, exist_ok=True)
                scipy.io.savemat(dest_mat, {{"X": X_mat, "y": y_mat}})
                try:
                    os.remove(pq_tmp)
                except OSError:
                    pass
                return dest_mat
        return orig_get_file(fname=fname, origin=origin, *args, **kwargs)

    mod.get_file = _resilient_get_file
    mod._hls4ml_patched = True


def _patch_hls4ml_converters(mod):
    if getattr(mod, "_hls4ml_patched", False) or not hasattr(mod, "convert_from_symbolic_expression"):
        return
    orig_sym = mod.convert_from_symbolic_expression

    def _wrapped_symbolic(*args, **kwargs):
        has_hls = shutil.which("vivado_hls") is not None or shutil.which("vitis_hls") is not None
        if not has_hls:
            if kwargs.get("hls_include_path") is None:
                kwargs["hls_include_path"] = os.path.join(_SHIM_ROOT, "include")
            if kwargs.get("hls_libs_path") is None:
                kwargs["hls_libs_path"] = os.path.join(_SHIM_ROOT, "no_libs")
        return orig_sym(*args, **kwargs)

    mod.convert_from_symbolic_expression = _wrapped_symbolic
    mod._hls4ml_patched = True


_TARGET_HOOKS = {{
    "sklearn.datasets": _patch_sklearn_datasets,
    "keras.src.models.sequential": _patch_keras_sequential,
    "keras.utils": _patch_keras_utils,
    "keras": _patch_keras_utils,
    "hls4ml.converters": _patch_hls4ml_converters,
}}


class _PostImportPatcher(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path, target=None):
        if fullname not in _TARGET_HOOKS:
            return None
        for finder in sys.meta_path:
            if finder is self or not hasattr(finder, "find_spec"):
                continue
            spec = finder.find_spec(fullname, path, target)
            if spec is not None and spec.loader is not None and not getattr(spec.loader, "_hls4ml_wrapped", False):
                orig_exec = spec.loader.exec_module

                def _exec_module(module, _orig=orig_exec, _fn=_TARGET_HOOKS[fullname]):
                    _orig(module)
                    try:
                        _fn(module)
                    except Exception:
                        pass

                spec.loader.exec_module = _exec_module
                spec.loader._hls4ml_wrapped = True
                return spec
        return None


sys.meta_path.insert(0, _PostImportPatcher())
for _finder in list(sys.meta_path):
    if getattr(_finder, "__name__", "") == "_EditableFinder":
        sys.meta_path.remove(_finder)
        sys.meta_path.insert(1, _finder)
'''

with open(os.path.join(site_pkgs, "sitecustomize.py"), "w", encoding="utf-8") as fh:
    fh.write(sitecustomize_code)
with open(os.path.join(site_pkgs, "_hls4ml_site_init.py"), "w", encoding="utf-8") as fh:
    fh.write(sitecustomize_code)
with open(os.path.join(site_pkgs, "_hls4ml_init.pth"), "w", encoding="utf-8") as fh:
    fh.write("import _hls4ml_site_init\n")

# 5d. Hook IPython/Jupyter para contornar o bug upstream de 2_quantization/2b_brevitas.ipynb sem editar o .ipynb
ipy_startup_dir = os.path.join(venv_dir, "etc", "ipython", "profile_default", "startup")
os.makedirs(ipy_startup_dir, exist_ok=True)
ipy_hook_code = '''"""
Compatibilidade em tempo de execucao para os notebooks oficiais (sem editar .ipynb):
- 6b_symbolic_regression.ipynb: inicializa `juliacall` antes de `hls4ml`/`torch` na Celula 1
  para evitar colisao de simbolos LLVM (pytorch#78829)
- 2b_brevitas.ipynb: injeta `qnn = brevitas.nn` caso ausente na Celula 1 e anexa `.logits(x)`
  caso `JetTaggerBrevitas` seja redefinida na Celula 5 sem o metodo `.logits`
"""
import importlib
import sys


def _register_notebook_compat():
    try:
        ip = get_ipython()  # noqa: F821
    except NameError:
        return
    if ip is None:
        return

    def _pre_cell(info):
        raw = getattr(info, "raw_cell", "") or ""
        if ("pysr" in raw or "juliacall" in raw) and "juliacall" not in sys.modules:
            try:
                importlib.import_module("pysr")
            except Exception:
                try:
                    importlib.import_module("juliacall")
                except Exception:
                    pass

    def _post_cell(_result):
        ns = ip.user_ns
        if "JetTaggerBrevitas" in ns and "qnn" not in ns:
            try:
                ns["qnn"] = importlib.import_module("brevitas.nn")
            except Exception:
                pass
        cls = ns.get("JetTaggerBrevitas")
        if isinstance(cls, type) and not hasattr(cls, "logits"):
            if all(k in getattr(cls, "__init__", object).__code__.co_names for k in ("fc1", "relu1", "output")):
                def _logits(self, x):
                    x = self.relu1(self.fc1(x))
                    x = self.relu2(self.fc2(x))
                    x = self.relu3(self.fc3(x))
                    return self.output(x)
                cls.logits = _logits

    ip.events.register("pre_run_cell", _pre_cell)
    ip.events.register("post_run_cell", _post_cell)


_register_notebook_compat()
'''
with open(os.path.join(ipy_startup_dir, "00_hls4ml_compat.py"), "w", encoding="utf-8") as fh:
    fh.write(ipy_hook_code)

# 5e. Garantir variaveis no kernel.json padrao do Jupyter na .venv
kernel_json_path = os.path.join(venv_dir, "share", "jupyter", "kernels", "python3", "kernel.json")
if os.path.isfile(kernel_json_path):
    with open(kernel_json_path, "r", encoding="utf-8") as fh:
        kdata = json.load(fh)
    env_map = kdata.setdefault("env", {})
    env_map["KERAS_HOME"] = os.path.join(venv_dir, "share", "keras")
    env_map["IPYTHONDIR"] = os.path.join(venv_dir, "etc", "ipython")
    env_map["JUPYTER_CONFIG_DIR"] = os.path.join(venv_dir, "etc", "jupyter")
    env_map["JULIA_DEPOT_PATH"] = os.path.join(venv_dir, "share", "julia")
    env_map["PYTHON_JULIAPKG_PROJECT"] = os.path.join(venv_dir, "share", "pyjuliapkg")
    env_map["PYTHON_JULIACALL_HANDLE_SIGNALS"] = "yes"
    env_map["PYTHON_JULIACALL_THREADS"] = "auto"
    env_map["PYTHON_JULIACALL_AUTOLOAD_IPYTHON_EXTENSION"] = "no"
    env_map["JULIA_PKG_IGNORE_HASHES"] = "1"
    env_map["TF_CPP_MIN_LOG_LEVEL"] = "3"
    with open(kernel_json_path, "w", encoding="utf-8") as fh:
        json.dump(kdata, fh, indent=2)

# 5f. Configuracao isolada do servidor JupyterLab na .venv
jupyter_cfg_dir = os.path.join(venv_dir, "etc", "jupyter")
os.makedirs(jupyter_cfg_dir, exist_ok=True)
images_abs_dir = os.path.join(project_dir, "hls4ml-tutorial", "images")
jupyter_cfg_code = f'''c = get_config()  # noqa: F821
c.ServerApp.open_browser = False
c.ServerApp.websocket_ping_interval = 30000
c.ServerApp.websocket_ping_timeout = 25000
c.MappingKernelManager.cull_idle_timeout = 1800
c.MappingKernelManager.cull_interval = 120
c.MappingKernelManager.cull_connected = False

import os
from tornado.web import StaticFileHandler
img_dir = {images_abs_dir!r}
if os.path.isdir(img_dir):
    c.ServerApp.extra_handlers = [
        (r"/images/(.*)", StaticFileHandler, {{"path": img_dir}}),
    ]
'''
with open(os.path.join(jupyter_cfg_dir, "jupyter_server_config.py"), "w", encoding="utf-8") as fh:
    fh.write(jupyter_cfg_code)

# 5g. Adicionar exports ao .venv/bin/activate (idempotente)
activate_path = os.path.join(venv_dir, "bin", "activate")
marker = "# --- hls4ml-quickstart env ---"
if os.path.isfile(activate_path):
    with open(activate_path, "r", encoding="utf-8") as fh:
        act_txt = fh.read()
    if marker not in act_txt:
        extra_exports = f"""
{marker}
export XILINX_VITIS="${{XILINX_VITIS:-{venv_dir}/share/hls_shim}}"
export KERAS_HOME="{venv_dir}/share/keras"
export IPYTHONDIR="{venv_dir}/etc/ipython"
export JUPYTER_CONFIG_DIR="{venv_dir}/etc/jupyter"
export JULIA_DEPOT_PATH="{venv_dir}/share/julia"
export PYTHON_JULIAPKG_PROJECT="{venv_dir}/share/pyjuliapkg"
export PYTHON_JULIACALL_HANDLE_SIGNALS="yes"
export PYTHON_JULIACALL_THREADS="auto"
export PYTHON_JULIACALL_AUTOLOAD_IPYTHON_EXTENSION="no"
export JULIA_PKG_IGNORE_HASHES="1"
export TF_CPP_MIN_LOG_LEVEL="3"
"""
        with open(activate_path, "a", encoding="utf-8") as fh:
            fh.write(extra_exports)
PYSHIM

# 6. Baixar e preparar o dataset oficial hls4ml_lhc_jets_hlf em hls4ml-tutorial/data/jet-tagging/
echo "[5/6] Preparando conjunto de dados oficial (hls4ml_lhc_jets_hlf)..."
PROJECT_DIR="$PROJECT_DIR" "$VENV_DIR/bin/python" - <<'PYDATA' || echo "[AVISO] Dataset sera baixado sob demanda na abertura do notebook."
import os
import sys
import numpy as np
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

project_dir = os.environ["PROJECT_DIR"]
data_dir = os.path.join(project_dir, "hls4ml-tutorial", "data", "jet-tagging")
models_dir = os.path.join(project_dir, "hls4ml-tutorial", "models")
os.makedirs(data_dir, exist_ok=True)
os.makedirs(models_dir, exist_ok=True)

required = ["X_train_val.npy", "X_test.npy", "y_train_val.npy", "y_test.npy", "classes.npy"]
if all(os.path.isfile(os.path.join(data_dir, f)) for f in required):
    print("[OK] Dataset ja disponivel em hls4ml-tutorial/data/jet-tagging/.")
    sys.exit(0)

# Usa automaticamente o shim resiliente (Parquet com HTTP Range) configurado em sitecustomize.py
np.random.seed(0)
data = fetch_openml("hls4ml_lhc_jets_hlf")
X, y = data["data"], data["target"]

le = LabelEncoder()
y_encoded = le.fit_transform(y)
y_oh = np.eye(5)[y_encoded]
X_train_val, X_test, y_train_val, y_test = train_test_split(X, y_oh, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train_val = scaler.fit_transform(X_train_val)
X_test = scaler.transform(X_test)

for fname, arr in (
    ("X_train_val.npy", X_train_val),
    ("X_test.npy", X_test),
    ("y_train_val.npy", y_train_val),
    ("y_test.npy", y_test),
    ("classes.npy", le.classes_),
):
    final_p = os.path.join(data_dir, fname)
    tmp_p = final_p + ".tmp"
    with open(tmp_p, "wb") as f_np:
        np.save(f_np, arr)
    os.replace(tmp_p, final_p)

print(f"[OK] Dataset preparado em hls4ml-tutorial/data/jet-tagging/ (Train/Val: {X_train_val.shape}, Test: {X_test.shape}).")
PYDATA

# 7. Pre-inicializar Julia para PySR (Tutorial 6b) isolado dentro da .venv
echo "[6/6] Verificando backend Julia do PySR (Tutorial 6b)..."
"$VENV_DIR/bin/python" -c "import pysr" \
    && echo "[OK] Backend Julia/PySR pronto em .venv/share/julia." \
    || echo "[AVISO] Julia/PySR sera inicializado na primeira execucao do notebook 6b."

if "$VENV_DIR/bin/hls4ml" --version &>/dev/null; then
    echo "[OK] $($VENV_DIR/bin/hls4ml --version) instalado com sucesso."
fi

echo "=================================================================="
echo "Configuracao finalizada!"
echo "1. Ative o ambiente : source .venv/bin/activate"
echo "2. Verifique tudo   : python scripts/verify_install.py"
echo "3. Abra os tutoriais: jupyter lab   (pasta hls4ml-tutorial/)"
echo "=================================================================="
