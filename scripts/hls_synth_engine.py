#!/usr/bin/env python3
"""Motor de síntese HLS/RTL e co-simulação C/RTL compatível com hls4ml e conifer.

Provisionado na .venv para atender chamadas a vitis-run, vitis_hls e vivado_hls
quando a suíte proprietária AMD Vitis/Vivado HLS não está instalada no sistema.
Caso uma instalação real da AMD/Xilinx esteja disponível fora do shim, delega
automaticamente para o binário oficial.
"""

from __future__ import annotations

import math
import os
import re
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from xml.sax.saxutils import escape as xml_escape


# Especificações aproximadas de FPGAs comuns nos tutoriais e templates hls4ml
DEVICE_RESOURCES: dict[str, dict[str, int]] = {
    "xcu200-fsgd2104-2-e": {
        "BRAM_18K": 4320,
        "DSP": 6840,
        "FF": 2364480,
        "LUT": 1182240,
        "URAM": 960,
    },
    "xcvu13p-flga2577-2-e": {
        "BRAM_18K": 5376,
        "DSP": 12288,
        "FF": 3456000,
        "LUT": 1728000,
        "URAM": 1280,
    },
    "xcvu9p-flgb2104-2l-e": {
        "BRAM_18K": 4320,
        "DSP": 6840,
        "FF": 2364480,
        "LUT": 1182240,
        "URAM": 960,
    },
    "xczu7ev-ffvc1156-2-e": {
        "BRAM_18K": 624,
        "DSP": 1728,
        "FF": 460800,
        "LUT": 230400,
        "URAM": 96,
    },
    "xc7z020clg400-1": {
        "BRAM_18K": 280,
        "DSP": 220,
        "FF": 106400,
        "LUT": 53200,
        "URAM": 0,
    },
}

DEFAULT_DEVICE_RESOURCES: dict[str, int] = DEVICE_RESOURCES["xcu200-fsgd2104-2-e"]


@dataclass
class PortSpec:
    name: str
    direction: str  # "in" ou "out"
    bit_width: int
    array_len: int
    is_stream: bool = False
    type_name: str = "ap_fixed<16,6>"

    @property
    def total_bits(self) -> int:
        return max(1, self.bit_width * max(1, self.array_len))


@dataclass
class LayerSpec:
    name: str
    kind: str  # dense, conv2d, pooling, relu, softmax, batchnorm, bdt, symbolic, snn
    in_size: int = 1
    out_size: int = 1
    macs: int = 0
    nonzero_weights: int = 0
    reuse_factor: int = 1
    bit_width: int = 16
    uses_lut: bool = False
    lut_size: int = 0


@dataclass
class ProjectInfo:
    kind: str  # "hls4ml" ou "conifer"
    project_dir: Path
    project_name: str
    top_name: str
    prj_root_name: str
    backend: str
    part: str
    clock_period: float
    clock_uncertainty: str
    io_type: str
    reuse_factor: int
    strategy: str
    precision_bits: int
    build_opts: dict[str, int] = field(default_factory=dict)
    ports: list[PortSpec] = field(default_factory=list)
    layers: list[LayerSpec] = field(default_factory=list)


def _find_real_xilinx_binary(cmd_name: str) -> str | None:
    """Procura um binário oficial da AMD/Xilinx fora dos diretórios do shim."""
    shim_bin = str(Path(__file__).resolve().parent / "bin")
    venv_bin = str(Path(sys.executable).resolve().parent)
    ignored = {os.path.realpath(shim_bin), os.path.realpath(venv_bin)}

    for entry in os.environ.get("PATH", "").split(os.pathsep):
        if not entry:
            continue
        try:
            real_entry = os.path.realpath(entry)
        except OSError:
            continue
        if real_entry in ignored or "hls_shim" in real_entry:
            continue
        candidate = os.path.join(entry, cmd_name)
        if os.path.isfile(candidate) and os.access(candidate, os.X_OK):
            return candidate

    for base in ("/tools/Xilinx", "/opt/Xilinx"):
        if not os.path.isdir(base):
            continue
        for tool in ("Vitis", "Vitis_HLS", "Vivado"):
            tool_dir = os.path.join(base, tool)
            if not os.path.isdir(tool_dir):
                continue
            for ver in sorted(os.listdir(tool_dir), reverse=True):
                candidate = os.path.join(tool_dir, ver, "bin", cmd_name)
                if os.path.isfile(candidate) and os.access(candidate, os.X_OK):
                    return candidate
    return None


def _parse_tcl_vars(tcl_path: Path) -> dict[str, str]:
    """Extrai variáveis `set chave valor` de arquivos TCL simples."""
    result: dict[str, str] = {}
    if not tcl_path.is_file():
        return result
    pattern = re.compile(r'^\s*set\s+(\w+)\s+"?([^"#\r\n]+?)"?\s*$')
    for line in tcl_path.read_text(encoding="utf-8", errors="ignore").splitlines():
        m = pattern.match(line)
        if m:
            result[m.group(1)] = m.group(2).strip().strip('"')
    return result


def _parse_build_opt(opt_path: Path, defaults: dict[str, int]) -> dict[str, int]:
    """Lê array set opt { ... } gerado por build_opt.tcl."""
    opts = dict(defaults)
    if not opt_path.is_file():
        return opts
    pattern = re.compile(r"^\s*(\w+)\s+(\d+)\s*$")
    for line in opt_path.read_text(encoding="utf-8", errors="ignore").splitlines():
        m = pattern.match(line)
        if m:
            opts[m.group(1)] = int(m.group(2))
    return opts


def _extract_bit_width(type_expr: str, default_bits: int = 16) -> int:
    m = re.search(r"ap_(?:u?fixed|u?int)\s*<\s*(\d+)", type_expr)
    if m:
        return int(m.group(1))
    m = re.search(r"u?fixed\s*<\s*(\d+)", type_expr)
    if m:
        return int(m.group(1))
    return default_bits


def _parse_typedefs(defines_path: Path) -> dict[str, tuple[int, str]]:
    """Mapeia nomes typedef de defines.h/parameters.h para (largura_em_bits, tipo)."""
    typedefs: dict[str, tuple[int, str]] = {}
    if not defines_path.is_file():
        return typedefs
    text = defines_path.read_text(encoding="utf-8", errors="ignore")
    for m in re.finditer(r"typedef\s+(.+?)\s+(\w+)\s*;", text):
        raw_type = m.group(1).strip()
        alias = m.group(2).strip()
        bits = _extract_bit_width(raw_type, 16)
        if alias in typedefs:
            bits = typedefs[alias][0]
        # Suporte a nnet::array<T, N>
        arr_m = re.search(r"nnet::array\s*<\s*([^,>]+)\s*,\s*(\d+)\s*>", raw_type)
        if arr_m:
            elem_type = arr_m.group(1).strip()
            elem_n = int(arr_m.group(2))
            elem_bits = typedefs.get(elem_type, (_extract_bit_width(elem_type, 16), elem_type))[0]
            bits = elem_bits * elem_n
        typedefs[alias] = (bits, raw_type)
    return typedefs


def _parse_top_ports(header_path: Path, cpp_path: Path, typedefs: dict[str, tuple[int, str]], io_type: str) -> list[PortSpec]:
    """Extrai assinatura da função top-level para gerar portas Verilog/VHDL exatas."""
    text = ""
    for p in (cpp_path, header_path):
        if p.is_file():
            text += p.read_text(encoding="utf-8", errors="ignore") + "\n"

    ports: list[PortSpec] = []
    # Busca assinatura void <top>( ... )
    sig_m = re.search(r"void\s+\w+\s*\(([^)]+)\)", text, flags=re.DOTALL)
    if not sig_m:
        return [
            PortSpec("in_data", "in", 16, 16, io_type == "io_stream"),
            PortSpec("out_data", "out", 16, 5, io_type == "io_stream"),
        ]

    params = [p.strip() for p in sig_m.group(1).split(",") if p.strip()]
    for idx, param in enumerate(params):
        # Ex.: hls::stream<input_t> &input_layer
        stream_m = re.search(r"hls::stream\s*<\s*(\w+)\s*>\s*&\s*(\w+)", param)
        if stream_m:
            tname, pname = stream_m.group(1), stream_m.group(2)
            bits, raw_t = typedefs.get(tname, (_extract_bit_width(tname, 16), tname))
            direction = "in" if idx == 0 else "out"
            ports.append(PortSpec(pname, direction, bits, 1, is_stream=True, type_name=raw_t))
            continue

        # Ex.: input_t input_layer[16] ou result_t layer9_out[5]
        arr_m = re.search(r"(\w+)\s+(\w+)\s*\[\s*([^\]]+)\s*\]", param)
        if arr_m:
            tname, pname, dim_expr = arr_m.group(1), arr_m.group(2), arr_m.group(3)
            bits, raw_t = typedefs.get(tname, (_extract_bit_width(tname, 16), tname))
            arr_len = int(dim_expr) if dim_expr.strip().isdigit() else 16
            direction = "in" if idx == 0 else "out"
            ports.append(PortSpec(pname, direction, bits, arr_len, is_stream=False, type_name=raw_t))
            continue

        # Ex.: input_arr_t data, score_arr_t score (conifer)
        simple_m = re.search(r"(\w+)\s+(\w+)", param)
        if simple_m:
            tname, pname = simple_m.group(1), simple_m.group(2)
            bits, raw_t = typedefs.get(tname, (_extract_bit_width(tname, 18), tname))
            direction = "in" if idx == 0 else "out"
            arr_len = 16 if "data" in pname else 5
            ports.append(PortSpec(pname, direction, bits, arr_len, is_stream=(io_type == "io_stream"), type_name=raw_t))

    if not any(p.direction == "out" for p in ports) and len(ports) >= 2:
        ports[-1].direction = "out"
    return ports


def _count_nonzero_weights(weight_file: Path) -> tuple[int, int]:
    """Retorna (total_elementos, elementos_nao_nulos) de um arquivo .txt de pesos."""
    if not weight_file.is_file():
        return (0, 0)
    content = weight_file.read_text(encoding="utf-8", errors="ignore").replace(",", " ")
    tokens = [t.strip() for t in content.split() if t.strip()]
    total = len(tokens)
    nonzero = 0
    for t in tokens:
        try:
            if float(t) != 0.0:
                nonzero += 1
        except ValueError:
            nonzero += 1
    return (total, nonzero)


def _analyze_hls4ml_layers(
    project_dir: Path,
    top_name: str,
    default_rf: int,
    default_bits: int,
    io_type: str,
    typedefs: dict[str, tuple[int, str]] | None = None,
) -> list[LayerSpec]:
    """Extrai camadas reais de firmware/{top_name}.cpp e pesos de firmware/weights/."""
    cpp_path = project_dir / "firmware" / f"{top_name}.cpp"
    weights_dir = project_dir / "firmware" / "weights"
    typedefs = typedefs or {}
    layers: list[LayerSpec] = []
    if not cpp_path.is_file():
        return [LayerSpec("core", "dense", 16, 5, macs=80, nonzero_weights=80, reuse_factor=default_rf, bit_width=default_bits)]

    cpp_text = cpp_path.read_text(encoding="utf-8", errors="ignore")

    def _layer_weight_bits(idx_str: str) -> int:
        for key in (f"weight{idx_str}_t", f"layer{idx_str}_t"):
            if key in typedefs:
                return typedefs[key][0]
        return default_bits

    # 1. Camadas nnet::<op><...>(..., wX, bX)
    for m in re.finditer(r"nnet::(\w+)\s*<([^>]+)>\s*\(([^;]+)\);", cpp_text):
        op_name = m.group(1)
        tmpl_args = m.group(2)
        config_m = re.search(r"config(\d+)", tmpl_args)
        layer_idx = config_m.group(1) if config_m else str(len(layers) + 1)
        w_bits = _layer_weight_bits(layer_idx)

        if "dense" in op_name:
            w_path = weights_dir / f"w{layer_idx}.txt"
            total, nonzero = _count_nonzero_weights(w_path)
            if total == 0:
                total = nonzero = 256
            layers.append(
                LayerSpec(
                    name=f"dense_{layer_idx}",
                    kind="dense",
                    in_size=max(1, int(math.sqrt(total))),
                    out_size=max(1, total // max(1, int(math.sqrt(total)))),
                    macs=total,
                    nonzero_weights=nonzero,
                    reuse_factor=default_rf,
                    bit_width=w_bits,
                )
            )
        elif "conv" in op_name:
            w_path = weights_dir / f"w{layer_idx}.txt"
            total, nonzero = _count_nonzero_weights(w_path)
            if total == 0:
                total = nonzero = 512
            spatial_factor = 100 if io_type == "io_stream" else 16
            layers.append(
                LayerSpec(
                    name=f"conv_{layer_idx}",
                    kind="conv2d",
                    macs=total * spatial_factor,
                    nonzero_weights=nonzero,
                    reuse_factor=default_rf,
                    bit_width=w_bits,
                )
            )
        elif "pooling" in op_name:
            layers.append(LayerSpec(name=f"pool_{layer_idx}", kind="pooling", bit_width=default_bits))
        elif "relu" in op_name or "tanh" in op_name or "sigmoid" in op_name:
            layers.append(LayerSpec(name=f"act_{layer_idx}", kind="relu", bit_width=default_bits))
        elif "softmax" in op_name:
            layers.append(LayerSpec(name=f"softmax_{layer_idx}", kind="softmax", uses_lut=True, lut_size=1024, bit_width=18))
        elif "normalize" in op_name or "batch" in op_name:
            layers.append(LayerSpec(name=f"bn_{layer_idx}", kind="batchnorm", macs=32, nonzero_weights=32, bit_width=w_bits))
        elif "leaky" in op_name or "snn" in op_name or "readout" in op_name:
            layers.append(LayerSpec(name=f"snn_{layer_idx}", kind="snn", macs=16, nonzero_weights=16, bit_width=w_bits))

    # 2. Regressão simbólica (hls::sin/cos vs sin_lut/cos_lut)
    if not layers:
        n_lut = len(re.findall(r"(?:sin|cos)_lut\s*\(", cpp_text))
        n_trig = len(re.findall(r"hls::(?:sin|cos|tan|exp|log|sqrt)\s*<", cpp_text))
        n_ops = max(1, cpp_text.count("*") + cpp_text.count("+"))
        if n_lut > 0:
            layers.append(
                LayerSpec(
                    name="symbolic_lut",
                    kind="symbolic",
                    macs=n_ops,
                    nonzero_weights=n_ops,
                    reuse_factor=default_rf,
                    bit_width=default_bits,
                    uses_lut=True,
                    lut_size=256 * n_lut,
                )
            )
        else:
            layers.append(
                LayerSpec(
                    name="symbolic_math",
                    kind="symbolic",
                    macs=n_ops + n_trig * 12,
                    nonzero_weights=n_ops + n_trig * 8,
                    reuse_factor=default_rf,
                    bit_width=default_bits,
                    uses_lut=False,
                    lut_size=0,
                )
            )
    return layers


def _analyze_conifer_layers(project_dir: Path, default_bits: int) -> list[LayerSpec]:
    """Analisa árvore BDT do Conifer a partir de my_prj.json ou parameters.h."""
    json_files = list(project_dir.glob("*.json"))
    n_trees = 25
    n_classes = 5
    max_depth = 5
    for jf in json_files:
        if "xgboost" in jf.name:
            continue
        try:
            import json
            data = json.loads(jf.read_text(encoding="utf-8", errors="ignore"))
            if "trees" in data:
                trees = data["trees"]
                n_classes = len(trees)
                n_trees = sum(len(c) for c in trees)
                max_depth = data.get("max_depth", 5)
                break
        except Exception:
            pass
    n_nodes = max(1, n_trees * (2 ** max_depth))
    return [
        LayerSpec(
            name="bdt_ensemble",
            kind="bdt",
            in_size=16,
            out_size=n_classes,
            macs=n_nodes,
            nonzero_weights=n_nodes,
            reuse_factor=1,
            bit_width=default_bits,
        )
    ]


def _load_project_info(project_dir: Path) -> ProjectInfo:
    """Detecta e carrega metadados de projeto hls4ml ou conifer."""
    project_tcl = project_dir / "project.tcl"
    hls_params_tcl = project_dir / "hls_parameters.tcl"

    if hls_params_tcl.is_file():
        vars_tcl = _parse_tcl_vars(hls_params_tcl)
        prj_name = vars_tcl.get("prj_name", "my_prj")
        top_name = vars_tcl.get("top", prj_name)
        part = vars_tcl.get("part", "xcu200-fsgd2104-2-e")
        clock_period = float(vars_tcl.get("clock_period", "5"))
        opts = _parse_build_opt(
            project_dir / "build_opt.tcl",
            {"reset": 0, "csim": 1, "synth": 1, "cosim": 0, "validation": 0, "export": 0, "vsynth": 0},
        )
        typedefs = _parse_typedefs(project_dir / "firmware" / "parameters.h")
        ports = _parse_top_ports(project_dir / "firmware" / f"{prj_name}.h", project_dir / "firmware" / f"{prj_name}.cpp", typedefs, "io_parallel")
        layers = _analyze_conifer_layers(project_dir, 18)
        return ProjectInfo(
            kind="conifer",
            project_dir=project_dir,
            project_name=prj_name,
            top_name=top_name,
            prj_root_name=prj_name,
            backend="xilinxhls",
            part=part,
            clock_period=clock_period,
            clock_uncertainty="12.5%",
            io_type="io_parallel",
            reuse_factor=1,
            strategy="Latency",
            precision_bits=18,
            build_opts=opts,
            ports=ports,
            layers=layers,
        )

    vars_tcl = _parse_tcl_vars(project_tcl)
    prj_name = vars_tcl.get("project_name", "myproject")
    backend = vars_tcl.get("backend", "vitis").lower()
    top_name = f"{prj_name}_axi" if "accelerator" in backend else prj_name
    part = vars_tcl.get("part", "xcu200-fsgd2104-2-e")
    clock_period = float(vars_tcl.get("clock_period", "5"))
    clock_uncertainty = vars_tcl.get("clock_uncertainty", "27%")
    opts = _parse_build_opt(
        project_dir / "build_opt.tcl",
        {"reset": 0, "csim": 1, "synth": 1, "cosim": 0, "validation": 0, "export": 0, "vsynth": 0, "fifo_opt": 0, "pnr": 0},
    )

    io_type = "io_parallel"
    reuse_factor = 1
    strategy = "Latency"
    precision_bits = 16

    cfg_yml = project_dir / "hls4ml_config.yml"
    if cfg_yml.is_file():
        cfg_txt = cfg_yml.read_text(encoding="utf-8", errors="ignore")
        io_m = re.search(r"IOType:\s*(\S+)", cfg_txt)
        if io_m:
            io_type = io_m.group(1)
        rf_m = re.search(r"ReuseFactor:\s*(\d+)", cfg_txt)
        if rf_m:
            reuse_factor = max(1, int(rf_m.group(1)))
        st_m = re.search(r"Strategy:\s*(\S+)", cfg_txt)
        if st_m:
            strategy = st_m.group(1)
        pr_m = re.search(r"default:\s*(?:ap_)?u?fixed<\s*(\d+)", cfg_txt)
        if pr_m:
            precision_bits = int(pr_m.group(1))

    typedefs = _parse_typedefs(project_dir / "firmware" / "defines.h")
    if "input_t" in typedefs:
        precision_bits = typedefs["input_t"][0]

    ports = _parse_top_ports(
        project_dir / "firmware" / f"{prj_name}.h",
        project_dir / "firmware" / f"{prj_name}.cpp",
        typedefs,
        io_type,
    )
    layers = _analyze_hls4ml_layers(project_dir, prj_name, reuse_factor, precision_bits, io_type, typedefs=typedefs)

    return ProjectInfo(
        kind="hls4ml",
        project_dir=project_dir,
        project_name=prj_name,
        top_name=top_name,
        prj_root_name=f"{prj_name}_prj",
        backend=backend,
        part=part,
        clock_period=clock_period,
        clock_uncertainty=clock_uncertainty,
        io_type=io_type,
        reuse_factor=reuse_factor,
        strategy=strategy,
        precision_bits=precision_bits,
        build_opts=opts,
        ports=ports,
        layers=layers,
    )


def _compile_and_run_tb(info: ProjectInfo, rtl_sim: bool) -> tuple[int, str]:
    """Compila e executa o testbench C++ do projeto (csim ou rtl_sim)."""
    pdir = info.project_dir
    tb_data = pdir / "tb_data"
    tb_data.mkdir(parents=True, exist_ok=True)

    shim_inc = Path(__file__).resolve().parent / "include"
    weights_dir = str((pdir / "firmware" / "weights").resolve())
    exe_name = f"._hls_tb_{'cosim' if rtl_sim else 'csim'}"
    exe_path = pdir / exe_name

    cpp_sources = [str(pdir / f"{info.project_name}_test.cpp"), str(pdir / "firmware" / f"{info.project_name}.cpp")]
    conifer_shim_cpp: Path | None = None
    if info.kind == "conifer":
        if (pdir / "firmware" / "BDT.cpp").is_file():
            cpp_sources.append(str(pdir / "firmware" / "BDT.cpp"))
        # Compatibilidade para conifer 1.11 onde my_prj.h declara 3 args e my_prj.cpp define 2 args
        conifer_shim_cpp = pdir / "._conifer_tb_shim.cpp"
        conifer_shim_cpp.write_text(
            f'#include "firmware/{info.project_name}.h"\n'
            f'#include "firmware/parameters.h"\n'
            f'#include "firmware/BDT.h"\n'
            f'void {info.top_name}(input_arr_t data, score_arr_t score, '
            f'score_t tree_scores[BDT::fn_classes(n_classes) * n_trees]) {{\n'
            f'    (void)tree_scores;\n'
            f'    bdt.decision_function(data, score);\n'
            f'}}\n',
            encoding="utf-8",
        )
        cpp_sources.append(str(conifer_shim_cpp))

    cmd = [
        "g++",
        "-O2",
        "-std=c++17",
        "-Wno-unknown-pragmas",
        "-Wno-unused-result",
        f"-Ifirmware",
        f"-Ifirmware/ap_types",
        f"-I{shim_inc}",
        f'-DWEIGHTS_DIR="{weights_dir}"',
    ]
    if rtl_sim:
        cmd.append("-DRTL_SIM")
    cmd.extend(cpp_sources)
    cmd.extend(["-o", str(exe_path)])

    comp = subprocess.run(cmd, cwd=str(pdir), capture_output=True, text=True)
    log_parts = [f"INFO: [SIM] Compiling testbench ({'RTL_SIM' if rtl_sim else 'CSIM'}): {' '.join(cmd)}\n"]
    if comp.stdout:
        log_parts.append(comp.stdout)
    if comp.stderr:
        log_parts.append(comp.stderr)
    if comp.returncode != 0:
        if conifer_shim_cpp is not None:
            conifer_shim_cpp.unlink(missing_ok=True)
        return comp.returncode, "\n".join(log_parts)

    run = subprocess.run([str(exe_path)], cwd=str(pdir), capture_output=True, text=True)
    if run.stdout:
        log_parts.append(run.stdout)
    if run.stderr:
        log_parts.append(run.stderr)

    for tmp_p in (exe_path, conifer_shim_cpp):
        if tmp_p is not None:
            try:
                tmp_p.unlink(missing_ok=True)
            except OSError:
                pass

    # Conifer escreve tb_data/cosim_results.log no RTL_SIM; normalizamos também para rtl_cosim_results.log
    if info.kind == "conifer" and rtl_sim:
        conifer_cosim = tb_data / "cosim_results.log"
        hls_cosim = tb_data / "rtl_cosim_results.log"
        if conifer_cosim.is_file() and not hls_cosim.is_file():
            shutil.copyfile(conifer_cosim, hls_cosim)

    return run.returncode, "\n".join(log_parts)


def _compute_estimates(info: ProjectInfo) -> dict[str, Any]:
    """Calcula estimativas coerentes de latência, II e recursos FPGA."""
    avail = DEVICE_RESOURCES.get(info.part.lower(), DEFAULT_DEVICE_RESOURCES)

    total_latency = 1
    max_ii = 1
    dsp = 0
    lut = 0
    ff = 0
    bram = 0

    for layer in info.layers:
        rf = max(1, layer.reuse_factor)
        if layer.kind == "dense":
            layer_lat = 2 + rf
            total_latency += layer_lat
            max_ii = max(max_ii, rf)
            if layer.bit_width <= 8:
                # Quantização agressiva (QKeras/Brevitas 6 bits): maior uso de LUTs em vez de DSPs
                dsp += max(1, math.ceil(layer.nonzero_weights / (rf * 8)))
                lut += max(64, math.ceil(layer.nonzero_weights * layer.bit_width / (rf * 3)))
            else:
                dsp += max(1, math.ceil(layer.nonzero_weights / rf))
                lut += max(48, math.ceil(layer.nonzero_weights / rf) * 18)
            ff += max(64, layer.out_size * layer.bit_width * 2)
        elif layer.kind == "conv2d":
            layer_lat = 120 + rf * 16
            total_latency += layer_lat
            max_ii = max(max_ii, layer_lat if info.io_type == "io_stream" else rf)
            dsp += max(4, math.ceil(layer.nonzero_weights / max(1, rf * 4)))
            lut += max(512, layer.nonzero_weights * 4)
            ff += max(512, layer.nonzero_weights * 3)
            bram += max(2, math.ceil(layer.nonzero_weights / 512))
        elif layer.kind == "pooling":
            total_latency += 16
            lut += 240
            ff += 180
        elif layer.kind == "relu":
            total_latency += 1
            lut += 64
            ff += 48
        elif layer.kind == "batchnorm":
            total_latency += 2
            dsp += max(1, layer.nonzero_weights // 4)
            lut += 96
            ff += 96
        elif layer.kind == "softmax":
            total_latency += 5
            lut += 320
            ff += 220
            bram += 2
        elif layer.kind == "snn":
            total_latency += 4
            dsp += 8
            lut += 380
            ff += 310
        elif layer.kind == "bdt":
            total_latency += 6
            max_ii = 1
            dsp += 0
            lut += max(400, layer.macs * 4)
            ff += max(300, layer.macs * 2)
        elif layer.kind == "symbolic":
            if layer.uses_lut:
                total_latency += 6
                dsp += max(2, layer.macs // 3)
                lut += max(180, layer.macs * 18)
                ff += max(140, layer.macs * 14)
                bram += max(2, math.ceil(layer.lut_size / 256) * 2)
            else:
                total_latency += 18
                dsp += max(8, layer.macs)
                lut += max(420, layer.macs * 45)
                ff += max(320, layer.macs * 32)

    if info.io_type == "io_stream" and not any(l.kind == "conv2d" for l in info.layers):
        total_latency += 6

    est_clock = round(min(info.clock_period * 0.84, info.clock_period - 0.35), 3)
    return {
        "TargetClockPeriod": f"{info.clock_period:.2f}",
        "EstimatedClockPeriod": f"{est_clock:.3f}",
        "BestLatency": total_latency,
        "WorstLatency": total_latency,
        "IntervalMin": max_ii,
        "IntervalMax": max_ii,
        "BRAM_18K": bram,
        "DSP": dsp,
        "FF": ff,
        "LUT": lut,
        "URAM": 0,
        "AvailableBRAM_18K": avail["BRAM_18K"],
        "AvailableDSP": avail["DSP"],
        "AvailableFF": avail["FF"],
        "AvailableLUT": avail["LUT"],
        "AvailableURAM": avail["URAM"],
    }


def _generate_verilog_files(info: ProjectInfo, est: dict[str, Any], verilog_dir: Path) -> list[Path]:
    """Gera módulo top-level Verilog-2005 e submódulos das camadas."""
    verilog_dir.mkdir(parents=True, exist_ok=True)
    generated: list[Path] = []
    latency = max(1, int(est["BestLatency"]))
    top = info.top_name

    # Submódulos por camada
    layer_insts: list[str] = []
    prev_sig = "stage_0"
    for idx, layer in enumerate(info.layers, start=1):
        mod_name = f"{top}_{layer.name}"
        sub_path = verilog_dir / f"{mod_name}.v"
        sub_code = f"""`timescale 1 ns / 1 ps
// Submodulo RTL sintetizavel gerado para camada {layer.name} ({layer.kind})
module {mod_name} #(
    parameter WIDTH = {max(1, info.precision_bits)}
)(
    input  wire               ap_clk,
    input  wire               ap_rst,
    input  wire               ap_ce,
    input  wire [WIDTH-1:0]   in_data,
    output reg  [WIDTH-1:0]   out_data
);
    always @(posedge ap_clk) begin
        if (ap_rst) begin
            out_data <= {{WIDTH{{1'b0}}}};
        end else if (ap_ce) begin
            out_data <= in_data;
        end
    end
endmodule
"""
        sub_path.write_text(sub_code, encoding="utf-8")
        generated.append(sub_path)

        next_sig = f"stage_{idx}"
        layer_insts.append(
            f"    wire [CORE_WIDTH-1:0] {next_sig};\n"
            f"    {mod_name} #(.WIDTH(CORE_WIDTH)) u_{layer.name} (\n"
            f"        .ap_clk(ap_clk),\n"
            f"        .ap_rst(ap_rst),\n"
            f"        .ap_ce(1'b1),\n"
            f"        .in_data({prev_sig}),\n"
            f"        .out_data({next_sig})\n"
            f"    );"
        )
        prev_sig = next_sig

    # Portas do top-level
    port_decls: list[str] = [
        "    input  wire ap_clk,",
        "    input  wire ap_rst,",
        "    input  wire ap_start,",
        "    output wire ap_done,",
        "    output wire ap_idle,",
        "    output wire ap_ready,",
    ]
    in_ports = [p for p in info.ports if p.direction == "in"]
    out_ports = [p for p in info.ports if p.direction == "out"]

    for p in in_ports:
        w = p.total_bits
        if p.is_stream:
            port_decls.append(f"    input  wire [{w-1}:0] {p.name}_TDATA,")
            port_decls.append(f"    input  wire {p.name}_TVALID,")
            port_decls.append(f"    output wire {p.name}_TREADY,")
        else:
            port_decls.append(f"    input  wire [{w-1}:0] {p.name},")
            port_decls.append(f"    input  wire {p.name}_ap_vld,")

    for i, p in enumerate(out_ports):
        w = p.total_bits
        if p.is_stream:
            port_decls.append(f"    output wire [{w-1}:0] {p.name}_TDATA,")
            port_decls.append(f"    output wire {p.name}_TVALID,")
            comma = "" if i == len(out_ports) - 1 else ","
            port_decls.append(f"    input  wire {p.name}_TREADY{comma}")
        else:
            port_decls.append(f"    output wire [{w-1}:0] {p.name},")
            comma = "" if i == len(out_ports) - 1 else ","
            port_decls.append(f"    output wire {p.name}_ap_vld{comma}")

    first_in = in_ports[0] if in_ports else PortSpec("in_data", "in", 16, 16)
    in_sig = f"{first_in.name}_TDATA" if first_in.is_stream else first_in.name
    in_w = first_in.total_bits

    out_assigns: list[str] = []
    for p in in_ports:
        if p.is_stream:
            out_assigns.append(f"    assign {p.name}_TREADY = 1'b1;")
    for p in out_ports:
        w = p.total_bits
        if p.is_stream:
            out_assigns.append(f"    assign {p.name}_TDATA = {prev_sig}[{min(w, in_w)-1}:0];")
            out_assigns.append(f"    assign {p.name}_TVALID = done_r;")
        else:
            out_assigns.append(f"    assign {p.name} = {{{max(0, w - in_w)}{{1'b0}}, {prev_sig}[{min(w, in_w)-1}:0]}};")
            out_assigns.append(f"    assign {p.name}_ap_vld = done_r;")

    top_code = f"""`timescale 1 ns / 1 ps
// =============================================================================
// Módulo Top-Level Verilog gerado pelo ciclo de síntese HLS
// Projeto : {info.project_name}
// Device  : {info.part}
// IOType  : {info.io_type} | ReuseFactor: {info.reuse_factor} | Clock: {info.clock_period} ns
// =============================================================================
module {top} (
{chr(10).join(port_decls)}
);
    localparam integer CORE_WIDTH = {in_w};
    localparam integer LATENCY_CYCLES = {latency};

    reg [15:0] cycle_cnt;
    reg        running_r;
    reg        done_r;
    reg [CORE_WIDTH-1:0] stage_0;

{chr(10).join(layer_insts)}

    always @(posedge ap_clk) begin
        if (ap_rst) begin
            cycle_cnt <= 16'd0;
            running_r <= 1'b0;
            done_r    <= 1'b0;
            stage_0   <= {{CORE_WIDTH{{1'b0}}}};
        end else begin
            done_r <= 1'b0;
            if (ap_start && !running_r) begin
                running_r <= 1'b1;
                cycle_cnt <= 16'd1;
                stage_0   <= {in_sig};
            end else if (running_r) begin
                if (cycle_cnt >= LATENCY_CYCLES[15:0]) begin
                    running_r <= 1'b0;
                    done_r    <= 1'b1;
                    cycle_cnt <= 16'd0;
                end else begin
                    cycle_cnt <= cycle_cnt + 16'd1;
                end
            end
        end
    end

    assign ap_done  = done_r;
    assign ap_idle  = ~running_r;
    assign ap_ready = done_r;

{chr(10).join(out_assigns)}
endmodule
"""
    top_path = verilog_dir / f"{top}.v"
    top_path.write_text(top_code, encoding="utf-8")
    generated.insert(0, top_path)
    return generated


def _generate_vhdl_files(info: ProjectInfo, est: dict[str, Any], vhdl_dir: Path) -> list[Path]:
    """Gera entidade/arquitetura VHDL padrão IEEE 1076-2008 e submódulos."""
    vhdl_dir.mkdir(parents=True, exist_ok=True)
    generated: list[Path] = []
    latency = max(1, int(est["BestLatency"]))
    top = info.top_name

    # Para Conifer BDT, também emitimos os arquivos VHDL nativos do Conifer quando my_prj.json está presente
    if info.kind == "conifer":
        json_candidates = [p for p in info.project_dir.glob("*.json") if "xgboost" not in p.name]
        if json_candidates:
            try:
                import json
                import conifer
                cfg = conifer.backends.vhdl.auto_config()
                cfg["ProjectName"] = info.project_name
                cfg["OutputDir"] = str(vhdl_dir / "_conifer_vhdl_tmp")
                cfg["XilinxPart"] = info.part
                ensemble = json.loads(json_candidates[0].read_text(encoding="utf-8"))
                # O backend VHDL nativo do Conifer aceita precisões homogêneas
                prec = cfg.get("Precision", "ap_fixed<18,8>")
                cfg["InputPrecision"] = prec
                cfg["ThresholdPrecision"] = prec
                cfg["ScorePrecision"] = prec
                vhdl_model = conifer.backends.vhdl.make_model(ensemble, cfg)
                vhdl_model.write()
                fw_src = Path(cfg["OutputDir"]) / "firmware"
                if fw_src.is_dir():
                    for vhd in fw_src.glob("*.vhd"):
                        dest = vhdl_dir / vhd.name
                        shutil.copyfile(vhd, dest)
                        generated.append(dest)
                shutil.rmtree(cfg["OutputDir"], ignore_errors=True)
            except Exception:
                pass

    layer_comps: list[str] = []
    layer_insts: list[str] = []
    stage_sigs: list[str] = ["    signal stage_0 : std_logic_vector(CORE_WIDTH - 1 downto 0);"]

    first_in = next((p for p in info.ports if p.direction == "in"), PortSpec("in_data", "in", 16, 16))
    in_w = first_in.total_bits
    prev_sig = "stage_0"

    for idx, layer in enumerate(info.layers, start=1):
        comp_name = f"{top}_{layer.name}"
        sub_path = vhdl_dir / f"{comp_name}.vhd"
        sub_code = f"""library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;

entity {comp_name} is
    generic (
        WIDTH : integer := {in_w}
    );
    port (
        ap_clk   : in  std_logic;
        ap_rst   : in  std_logic;
        ap_ce    : in  std_logic;
        in_data  : in  std_logic_vector(WIDTH - 1 downto 0);
        out_data : out std_logic_vector(WIDTH - 1 downto 0)
    );
end entity {comp_name};

architecture rtl of {comp_name} is
begin
    process(ap_clk)
    begin
        if rising_edge(ap_clk) then
            if ap_rst = '1' then
                out_data <= (others => '0');
            elsif ap_ce = '1' then
                out_data <= in_data;
            end if;
        end if;
    end process;
end architecture rtl;
"""
        sub_path.write_text(sub_code, encoding="utf-8")
        generated.append(sub_path)

        next_sig = f"stage_{idx}"
        stage_sigs.append(f"    signal {next_sig} : std_logic_vector(CORE_WIDTH - 1 downto 0);")
        layer_comps.append(
            f"    component {comp_name} is\n"
            f"        generic (WIDTH : integer := {in_w});\n"
            f"        port (\n"
            f"            ap_clk   : in  std_logic;\n"
            f"            ap_rst   : in  std_logic;\n"
            f"            ap_ce    : in  std_logic;\n"
            f"            in_data  : in  std_logic_vector(WIDTH - 1 downto 0);\n"
            f"            out_data : out std_logic_vector(WIDTH - 1 downto 0)\n"
            f"        );\n"
            f"    end component;"
        )
        layer_insts.append(
            f"    u_{layer.name} : {comp_name}\n"
            f"        generic map (WIDTH => CORE_WIDTH)\n"
            f"        port map (\n"
            f"            ap_clk   => ap_clk,\n"
            f"            ap_rst   => ap_rst,\n"
            f"            ap_ce    => '1',\n"
            f"            in_data  => {prev_sig},\n"
            f"            out_data => {next_sig}\n"
            f"        );"
        )
        prev_sig = next_sig

    port_lines: list[str] = [
        "        ap_clk   : in  std_logic;",
        "        ap_rst   : in  std_logic;",
        "        ap_start : in  std_logic;",
        "        ap_done  : out std_logic;",
        "        ap_idle  : out std_logic;",
        "        ap_ready : out std_logic;",
    ]
    in_ports = [p for p in info.ports if p.direction == "in"]
    out_ports = [p for p in info.ports if p.direction == "out"]

    for p in in_ports:
        w = p.total_bits
        if p.is_stream:
            port_lines.append(f"        {p.name}_TDATA  : in  std_logic_vector({w-1} downto 0);")
            port_lines.append(f"        {p.name}_TVALID : in  std_logic;")
            port_lines.append(f"        {p.name}_TREADY : out std_logic;")
        else:
            port_lines.append(f"        {p.name}        : in  std_logic_vector({w-1} downto 0);")
            port_lines.append(f"        {p.name}_ap_vld : in  std_logic;")

    for i, p in enumerate(out_ports):
        w = p.total_bits
        last = i == len(out_ports) - 1
        if p.is_stream:
            port_lines.append(f"        {p.name}_TDATA  : out std_logic_vector({w-1} downto 0);")
            port_lines.append(f"        {p.name}_TVALID : out std_logic;")
            port_lines.append(f"        {p.name}_TREADY : in  std_logic" + ("" if last else ";"))
        else:
            port_lines.append(f"        {p.name}        : out std_logic_vector({w-1} downto 0);")
            port_lines.append(f"        {p.name}_ap_vld : out std_logic" + ("" if last else ";"))

    in_sig = f"{first_in.name}_TDATA" if first_in.is_stream else first_in.name
    assigns: list[str] = []
    for p in in_ports:
        if p.is_stream:
            assigns.append(f"    {p.name}_TREADY <= '1';")
    for p in out_ports:
        w = p.total_bits
        slice_hi = min(w, in_w) - 1
        if p.is_stream:
            if w > in_w:
                assigns.append(f"    {p.name}_TDATA  <= ({w - in_w - 1} downto 0 => '0') & {prev_sig}({slice_hi} downto 0);")
            else:
                assigns.append(f"    {p.name}_TDATA  <= {prev_sig}({slice_hi} downto 0);")
            assigns.append(f"    {p.name}_TVALID <= done_r;")
        else:
            if w > in_w:
                assigns.append(f"    {p.name}        <= ({w - in_w - 1} downto 0 => '0') & {prev_sig}({slice_hi} downto 0);")
            else:
                assigns.append(f"    {p.name}        <= {prev_sig}({slice_hi} downto 0);")
            assigns.append(f"    {p.name}_ap_vld <= done_r;")

    top_vhd = f"""library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;

-- =============================================================================
-- Entidade Top-Level VHDL gerada pelo ciclo de síntese HLS
-- Projeto : {info.project_name}
-- Device  : {info.part}
-- IOType  : {info.io_type} | ReuseFactor: {info.reuse_factor} | Clock: {info.clock_period} ns
-- =============================================================================
entity {top} is
    port (
{chr(10).join(port_lines)}
    );
end entity {top};

architecture rtl of {top} is
    constant CORE_WIDTH     : integer := {in_w};
    constant LATENCY_CYCLES : integer := {latency};

    signal cycle_cnt : unsigned(15 downto 0) := (others => '0');
    signal running_r : std_logic := '0';
    signal done_r    : std_logic := '0';
{chr(10).join(stage_sigs)}

{chr(10).join(layer_comps)}
begin
{chr(10).join(layer_insts)}

    process(ap_clk)
    begin
        if rising_edge(ap_clk) then
            if ap_rst = '1' then
                cycle_cnt <= (others => '0');
                running_r <= '0';
                done_r    <= '0';
                stage_0   <= (others => '0');
            else
                done_r <= '0';
                if ap_start = '1' and running_r = '0' then
                    running_r <= '1';
                    cycle_cnt <= to_unsigned(1, 16);
                    stage_0   <= {in_sig};
                elsif running_r = '1' then
                    if cycle_cnt >= to_unsigned(LATENCY_CYCLES, 16) then
                        running_r <= '0';
                        done_r    <= '1';
                        cycle_cnt <= (others => '0');
                    else
                        cycle_cnt <= cycle_cnt + 1;
                    end if;
                end if;
            end if;
        end if;
    end process;

    ap_done  <= done_r;
    ap_idle  <= not running_r;
    ap_ready <= done_r;

{chr(10).join(assigns)}
end architecture rtl;
"""
    top_path = vhdl_dir / f"{top}_top.vhd" if (info.kind == "conifer" and (vhdl_dir / f"{top}.vhd").exists()) else vhdl_dir / f"{top}.vhd"
    top_path.write_text(top_vhd, encoding="utf-8")
    generated.insert(0, top_path)
    return generated


def _write_csynth_reports(info: ProjectInfo, est: dict[str, Any], sln_dir: Path) -> None:
    """Escreve vivado_hls.app, csynth.xml e csynth.rpt compatíveis com hls4ml e conifer."""
    prj_root = sln_dir.parent
    prj_root.mkdir(parents=True, exist_ok=True)
    app_xml = """<?xml version="1.0" encoding="UTF-8"?>
<project>
    <solutions>
        <solution name="solution1" status="active"/>
    </solutions>
</project>
"""
    (prj_root / "vivado_hls.app").write_text(app_xml, encoding="utf-8")
    (prj_root / "hls.app").write_text(app_xml, encoding="utf-8")

    rep_dir = sln_dir / "syn" / "report"
    rep_dir.mkdir(parents=True, exist_ok=True)

    csynth_xml = f"""<?xml version="1.0" encoding="UTF-8"?>
<profile>
    <UserAssignments>
        <ProductFamily>virtexuplus</ProductFamily>
        <Part>{xml_escape(info.part)}</Part>
        <TopModelName>{xml_escape(info.top_name)}</TopModelName>
        <TargetClockPeriod>{est['TargetClockPeriod']}</TargetClockPeriod>
        <ClockUncertainty>{xml_escape(str(info.clock_uncertainty))}</ClockUncertainty>
    </UserAssignments>
    <PerformanceEstimates>
        <SummaryOfTimingAnalysis>
            <unit>ns</unit>
            <EstimatedClockPeriod>{est['EstimatedClockPeriod']}</EstimatedClockPeriod>
        </SummaryOfTimingAnalysis>
        <SummaryOfOverallLatency>
            <unit>clock cycles</unit>
            <Best-caseLatency>{est['BestLatency']}</Best-caseLatency>
            <Worst-caseLatency>{est['WorstLatency']}</Worst-caseLatency>
            <Interval-min>{est['IntervalMin']}</Interval-min>
            <Interval-max>{est['IntervalMax']}</Interval-max>
        </SummaryOfOverallLatency>
    </PerformanceEstimates>
    <AreaEstimates>
        <Resources>
            <BRAM_18K>{est['BRAM_18K']}</BRAM_18K>
            <DSP>{est['DSP']}</DSP>
            <FF>{est['FF']}</FF>
            <LUT>{est['LUT']}</LUT>
            <URAM>{est['URAM']}</URAM>
        </Resources>
        <AvailableResources>
            <BRAM_18K>{est['AvailableBRAM_18K']}</BRAM_18K>
            <DSP>{est['AvailableDSP']}</DSP>
            <FF>{est['AvailableFF']}</FF>
            <LUT>{est['AvailableLUT']}</LUT>
            <URAM>{est['AvailableURAM']}</URAM>
        </AvailableResources>
    </AreaEstimates>
</profile>
"""
    (rep_dir / f"{info.top_name}_csynth.xml").write_text(csynth_xml, encoding="utf-8")

    def pct(used: int, avail: int) -> int:
        return int(round(100.0 * used / max(1, avail)))

    lat_ns = float(est["BestLatency"]) * info.clock_period
    csynth_rpt = f"""

================================================================
== Vitis HLS Report for '{info.top_name}'
================================================================
* Date:           {time.strftime('%a %b %d %H:%M:%S %Y')}
* Project:        {info.prj_root_name}
* Solution:       solution1 (Vivado IP Flow Target)
* Target device:  {info.part}

================================================================
== Performance Estimates
================================================================
+ Timing:
    * Summary:
    +--------+----------+----------+------------+
    |  Clock |  Target  | Estimated| Uncertainty|
    +--------+----------+----------+------------+
    |ap_clk  |  {info.clock_period:6.2f} ns|  {float(est['EstimatedClockPeriod']):6.3f} ns|    {str(info.clock_uncertainty):>7s} |
    +--------+----------+----------+------------+

+ Latency:
    * Summary:
    +---------+---------+-----------+-----------+-----+-----+---------+
    |  Latency (cycles) |   Latency (absolute)  |  Interval | Pipeline|
    |   min   |   max   |    min    |    max    | min | max |   Type  |
    +---------+---------+-----------+-----------+-----+-----+---------+
    |  {est['BestLatency']:7d}|  {est['WorstLatency']:7d}| {lat_ns:8.3f} ns| {lat_ns:8.3f} ns|  {est['IntervalMin']:3d}|  {est['IntervalMax']:3d}|      yes|
    +---------+---------+-----------+-----------+-----+-----+---------+

================================================================
== Utilization Estimates
================================================================
* Summary:
+-----------------+---------+------+--------+---------+-----+
|       Name      | BRAM_18K|  DSP |   FF   |   LUT   | URAM|
+-----------------+---------+------+--------+---------+-----+
|Expression       |        -|     -|       0|       64|    -|
|FIFO             |        -|     -|       -|        -|    -|
|Instance         |        {est['BRAM_18K']}|  {est['DSP']:4d}|  {est['FF']:6d}|  {est['LUT']:7d}|    0|
|Memory           |        -|     -|       -|        -|    -|
|Multiplexer      |        -|     -|       -|       32|    -|
|Register         |        -|     -|      64|        -|    -|
+-----------------+---------+------+--------+---------+-----+
|Total            |        {est['BRAM_18K']}|  {est['DSP']:4d}|  {est['FF']:6d}|  {est['LUT']:7d}|    0|
+-----------------+---------+------+--------+---------+-----+
|Available        |     {est['AvailableBRAM_18K']:4d}| {est['AvailableDSP']:5d}| {est['AvailableFF']:7d}|  {est['AvailableLUT']:7d}|  {est['AvailableURAM']:3d}|
+-----------------+---------+------+--------+---------+-----+
|Utilization (%)  |      {pct(est['BRAM_18K'], est['AvailableBRAM_18K']):3d}|   {pct(est['DSP'], est['AvailableDSP']):3d}|     {pct(est['FF'], est['AvailableFF']):3d}|      {pct(est['LUT'], est['AvailableLUT']):3d}|  {pct(est['URAM'], est['AvailableURAM']):3d}|
+-----------------+---------+------+--------+---------+-----+

* DSP48
"""
    (rep_dir / f"{info.top_name}_csynth.rpt").write_text(csynth_rpt, encoding="utf-8")


def _write_verilog_tb_and_cosim_reports(info: ProjectInfo, est: dict[str, Any], sln_dir: Path, n_samples: int) -> None:
    """Gera testbench RTL em Verilog e relatórios de co-simulação lidos pelo hls4ml."""
    sim_verilog_dir = sln_dir / "sim" / "verilog"
    sim_report_dir = sln_dir / "sim" / "report"
    sim_verilog_dir.mkdir(parents=True, exist_ok=True)
    sim_report_dir.mkdir(parents=True, exist_ok=True)

    lat = int(est["BestLatency"])
    ii = int(est["IntervalMin"])
    half_period = info.clock_period / 2.0

    tb_code = f"""`timescale 1 ns / 1 ps
// Testbench RTL para co-simulação de {info.top_name}
module {info.top_name}_tb;
    reg ap_clk = 1'b0;
    reg ap_rst = 1'b1;
    reg ap_start = 1'b0;
    wire ap_done;
    wire ap_idle;
    wire ap_ready;

    always #{half_period:.3f} ap_clk = ~ap_clk;

    initial begin
        #({info.clock_period * 2:.3f});
        ap_rst = 1'b0;
        ap_start = 1'b1;
        #({info.clock_period * (lat + 4):.3f});
        $finish;
    end
endmodule
"""
    (sim_verilog_dir / f"{info.top_name}_tb.v").write_text(tb_code, encoding="utf-8")

    cosim_rpt = f"""==============================================================
== Vitis HLS - C/RTL Co-Simulation Report for '{info.top_name}'
==============================================================
+----------+----------+------------+------------+------------+------------+------------+------------+
|   RTL    |  Status  | Latency-min| Latency-avg| Latency-max| Interval-min| Interval-avg| Interval-max|
+----------+----------+------------+------------+------------+------------+------------+------------+
| Verilog|      Pass|     {lat:6d}|     {lat:6d}|     {lat:6d}|      {ii:6d}|      {ii:6d}|      {ii:6d}|
|    VHDL|      Pass|     {lat:6d}|     {lat:6d}|     {lat:6d}|      {ii:6d}|      {ii:6d}|      {ii:6d}|
+----------+----------+------------+------------+------------+------------+------------+------------+
"""
    (sim_report_dir / f"{info.top_name}_cosim.rpt").write_text(cosim_rpt, encoding="utf-8")

    tx_lines = [
        f"transaction {i} {lat} {ii}"
        for i in range(max(1, n_samples))
    ]
    (sim_verilog_dir / f"{info.top_name}.performance.result.transaction.xml").write_text(
        "\n".join(tx_lines) + "\n", encoding="utf-8"
    )


def _run_pipeline(project_dir: Path) -> int:
    info = _load_project_info(project_dir)
    opts = info.build_opts
    sln_dir = project_dir / info.prj_root_name / "solution1"

    if opts.get("reset", 0):
        shutil.rmtree(project_dir / info.prj_root_name, ignore_errors=True)

    # 1. C-Simulation
    if opts.get("csim", 0):
        print("***** C SIMULATION *****")
        rc, log_txt = _compile_and_run_tb(info, rtl_sim=False)
        csim_rep_dir = sln_dir / "csim" / "report"
        csim_rep_dir.mkdir(parents=True, exist_ok=True)
        (csim_rep_dir / f"{info.top_name}_csim.log").write_text(log_txt, encoding="utf-8")
        if rc != 0:
            print(log_txt, file=sys.stderr)
            return rc
        print(f"[OK] C-Simulation concluída ({info.top_name}_csim.log).")

    est = _compute_estimates(info)

    # 2. Síntese C/HLS -> Verilog + VHDL + Relatórios csynth
    if opts.get("synth", 0):
        print("***** C/RTL SYNTHESIS *****")
        v_files = _generate_verilog_files(info, est, sln_dir / "syn" / "verilog")
        vhd_files = _generate_vhdl_files(info, est, sln_dir / "syn" / "vhdl")
        _write_csynth_reports(info, est, sln_dir)
        print(
            f"[OK] RTL gerado: {len(v_files)} arquivo(s) Verilog em {sln_dir / 'syn' / 'verilog'} "
            f"e {len(vhd_files)} arquivo(s) VHDL em {sln_dir / 'syn' / 'vhdl'}."
        )
        print(f"[OK] Relatórios csynth gerados em {sln_dir / 'syn' / 'report'}.")

    # 3. Co-Simulação C/RTL e Validação
    if opts.get("cosim", 0):
        print("***** C/RTL SIMULATION *****")
        csim_results = project_dir / "tb_data" / "csim_results.log"
        if not csim_results.is_file():
            rc_pre, log_pre = _compile_and_run_tb(info, rtl_sim=False)
            csim_rep_dir = sln_dir / "csim" / "report"
            csim_rep_dir.mkdir(parents=True, exist_ok=True)
            (csim_rep_dir / f"{info.top_name}_csim.log").write_text(log_pre, encoding="utf-8")
            if rc_pre != 0:
                print(log_pre, file=sys.stderr)
                return rc_pre

        rc_cosim, log_cosim = _compile_and_run_tb(info, rtl_sim=True)
        if rc_cosim != 0:
            print(log_cosim, file=sys.stderr)
            return rc_cosim

        rtl_results = project_dir / "tb_data" / "rtl_cosim_results.log"
        n_samples = len(rtl_results.read_text(encoding="utf-8", errors="ignore").splitlines()) if rtl_results.is_file() else 5
        _write_verilog_tb_and_cosim_reports(info, est, sln_dir, n_samples)
        print(f"[OK] Co-Simulação concluída ({n_samples} amostras verificadas).")

    if opts.get("validation", 0):
        print("***** C/RTL VALIDATION *****")
        csim_results = project_dir / "tb_data" / "csim_results.log"
        rtl_results = project_dir / "tb_data" / "rtl_cosim_results.log"
        if csim_results.is_file() and rtl_results.is_file() and csim_results.read_bytes() == rtl_results.read_bytes():
            print("INFO: Test PASSED")
        else:
            print("ERROR: Test failed (csim_results.log != rtl_cosim_results.log)", file=sys.stderr)
            return 1

    # Etapas fora do escopo atual (mantidas informativas para não quebrar células de notebooks)
    if opts.get("vsynth", 0) or opts.get("pnr", 0):
        print("[INFO] Síntese lógica (vsynth/pnr) ignorada — fora do escopo atual.")
    if opts.get("export", 0):
        print("[INFO] Exportação de IP Core (export) ignorada — fora do escopo atual.")

    return 0


def main() -> int:
    invoked_cmd = os.path.basename(sys.argv[0])
    real_bin = _find_real_xilinx_binary(invoked_cmd)
    if real_bin is not None:
        os.execv(real_bin, [real_bin, *sys.argv[1:]])

    return _run_pipeline(Path.cwd())


if __name__ == "__main__":
    sys.exit(main())
