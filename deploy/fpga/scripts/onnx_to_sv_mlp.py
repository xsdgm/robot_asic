#!/usr/bin/env python3
"""Convert a feed-forward ONNX MLP policy to synthesizable Verilog-2001.

Supported graph subset:
- Linear layers exported as Gemm
- Optional activation after each Gemm: Relu or Tanh

Output files:
- generated/policy_dims.vh
- generated/policy_mlp_core.v

Notes:
- Tanh is implemented as hard-tanh clip for hardware friendliness.
- This script is intentionally strict and fails fast on unsupported ops.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import onnx
from onnx import numpy_helper


@dataclass
class Layer:
    weight: np.ndarray  # [out_dim, in_dim]
    bias: np.ndarray  # [out_dim]
    act: str  # relu | tanh | linear


def _to_initializer_map(model: onnx.ModelProto) -> Dict[str, np.ndarray]:
    out: Dict[str, np.ndarray] = {}
    for init in model.graph.initializer:
        out[init.name] = numpy_helper.to_array(init).astype(np.float64)
    return out


def _get_attr_int(node: onnx.NodeProto, name: str, default: int) -> int:
    for a in node.attribute:
        if a.name == name:
            return int(a.i)
    return default


def _find_successor_act(nodes: List[onnx.NodeProto], output_name: str) -> Tuple[str, int]:
    for idx, n in enumerate(nodes):
        if not n.input:
            continue
        if n.input[0] == output_name and n.op_type in {"Relu", "Tanh"}:
            return n.op_type.lower(), idx
    return "linear", -1


def _extract_layers(model: onnx.ModelProto) -> List[Layer]:
    inits = _to_initializer_map(model)
    nodes = list(model.graph.node)
    layers: List[Layer] = []
    consumed = set()

    for i, node in enumerate(nodes):
        if i in consumed:
            continue
        if node.op_type != "Gemm":
            continue

        if len(node.input) < 2:
            raise ValueError(f"Gemm node {node.name or i} missing required inputs")

        a_name = node.input[0]
        b_name = node.input[1]
        c_name = node.input[2] if len(node.input) > 2 else None

        if b_name not in inits:
            raise ValueError(f"Gemm weight {b_name} is not an initializer")

        b = inits[b_name]
        trans_b = _get_attr_int(node, "transB", 0)

        if trans_b == 1:
            # PyTorch Linear typically exports [out, in] with transB=1.
            w = b.astype(np.float64)
        else:
            # If B is [in, out], convert to [out, in].
            w = b.T.astype(np.float64)

        if w.ndim != 2:
            raise ValueError(f"Weight for node {node.name or i} is not rank-2")

        out_dim, in_dim = int(w.shape[0]), int(w.shape[1])

        if c_name is not None and c_name in inits:
            bias = inits[c_name].astype(np.float64).reshape(-1)
        else:
            bias = np.zeros((out_dim,), dtype=np.float64)

        if bias.shape[0] != out_dim:
            raise ValueError(
                f"Bias shape mismatch for node {node.name or i}: {bias.shape[0]} vs {out_dim}"
            )

        act, act_idx = _find_successor_act(nodes, node.output[0])
        if act_idx >= 0:
            consumed.add(act_idx)

        layers.append(Layer(weight=w, bias=bias, act=act))

        # Keep some unsupported-op checks strict for easier debugging.
        if i + 1 < len(nodes):
            n2 = nodes[i + 1]
            if n2.op_type in {"MatMul", "Conv", "LayerNormalization"}:
                raise ValueError(
                    f"Unsupported op sequence near node {n2.name or (i+1)}: {n2.op_type}. "
                    "This converter currently supports Gemm-based MLP only."
                )

    if not layers:
        ops = sorted({n.op_type for n in nodes})
        raise ValueError(
            "No Gemm layers found in ONNX graph. Supported model type is MLP exported with Gemm. "
            f"Graph ops: {ops}"
        )

    for i in range(1, len(layers)):
        if layers[i - 1].weight.shape[0] != layers[i].weight.shape[1]:
            raise ValueError(
                "Layer dimension mismatch: "
                f"L{i-1} out={layers[i-1].weight.shape[0]}, L{i} in={layers[i].weight.shape[1]}"
            )

    return layers


def _quantize(arr: np.ndarray, data_w: int, frac_w: int) -> np.ndarray:
    scale = 1 << frac_w
    qmax = (1 << (data_w - 1)) - 1
    qmin = -(1 << (data_w - 1))
    q = np.round(arr * scale).astype(np.int64)
    q = np.clip(q, qmin, qmax)
    return q


def _v_int(v: int) -> str:
    return str(int(v))


def _emit_dims(path: Path, layers: List[Layer], data_w: int, frac_w: int) -> None:
    input_dim = layers[0].weight.shape[1]
    output_dim = layers[-1].weight.shape[0]

    lines: List[str] = []
    lines.append("`ifndef POLICY_DIMS_VH")
    lines.append("`define POLICY_DIMS_VH")
    lines.append(f"`define POLICY_DATA_W {data_w}")
    lines.append(f"`define POLICY_FRAC_W {frac_w}")
    lines.append(f"`define POLICY_NUM_LAYERS {len(layers)}")
    lines.append(f"`define POLICY_INPUT_DIM {input_dim}")
    lines.append(f"`define POLICY_OUTPUT_DIM {output_dim}")
    lines.append("`endif")
    path.write_text("\n".join(lines), encoding="utf-8")


def _emit_core(path: Path, layers: List[Layer], data_w: int, frac_w: int) -> None:
    max_dim = max(max(layer.weight.shape[0], layer.weight.shape[1]) for layer in layers)
    cnt_w = max(1, (max_dim - 1).bit_length())

    lines: List[str] = []
    lines.append("`include \"policy_dims.vh\"")
    lines.append("")
    lines.append("module policy_mlp_core #(")
    lines.append("  parameter ACC_W = 40,")
    lines.append(f"  parameter DATA_W = {data_w},")
    lines.append(f"  parameter FRAC_W = {frac_w},")
    lines.append("  parameter INPUT_DIM = `POLICY_INPUT_DIM,")
    lines.append("  parameter OUTPUT_DIM = `POLICY_OUTPUT_DIM")
    lines.append(") (")
    lines.append("  input clk,")
    lines.append("  input rst_n,")
    lines.append("  input in_valid,")
    lines.append("  input signed [DATA_W*INPUT_DIM-1:0] in_vec_flat,")
    lines.append("  output reg out_valid,")
    lines.append("  output reg signed [DATA_W*OUTPUT_DIM-1:0] out_vec_flat")
    lines.append(");")
    lines.append("")
    lines.append(f"  localparam NUM_LAYERS = {len(layers)};")
    lines.append(f"  localparam MAX_DIM = {max_dim};")
    lines.append(f"  localparam CNT_W = {cnt_w};")
    lines.append("")
    lines.append("  localparam ST_IDLE  = 2'd0;")
    lines.append("  localparam ST_MAC   = 2'd1;")
    lines.append("  localparam ST_STORE = 2'd2;")
    lines.append("  localparam ST_DONE  = 2'd3;")
    lines.append("")
    lines.append("  integer i;")
    lines.append("  reg [1:0] state;")
    lines.append("  reg src_sel;")
    lines.append("  reg [CNT_W-1:0] layer_idx;")
    lines.append("  reg [CNT_W-1:0] out_idx;")
    lines.append("  reg [CNT_W-1:0] in_idx;")
    lines.append("  reg signed [ACC_W-1:0] acc;")
    lines.append("  reg signed [DATA_W-1:0] cur_in;")
    lines.append("  reg signed [DATA_W-1:0] act_val;")
    lines.append("  reg signed [DATA_W-1:0] buf_a [0:MAX_DIM-1];")
    lines.append("  reg signed [DATA_W-1:0] buf_b [0:MAX_DIM-1];")
    lines.append("")
    lines.append("  function signed [DATA_W-1:0] sat_data_w;")
    lines.append("    input signed [ACC_W-1:0] x;")
    lines.append("    reg signed [ACC_W-1:0] max_v;")
    lines.append("    reg signed [ACC_W-1:0] min_v;")
    lines.append("    begin")
    lines.append("      max_v = (1 <<< (DATA_W-1)) - 1;")
    lines.append("      min_v = -(1 <<< (DATA_W-1));")
    lines.append("      if (x > max_v) sat_data_w = max_v[DATA_W-1:0];")
    lines.append("      else if (x < min_v) sat_data_w = min_v[DATA_W-1:0];")
    lines.append("      else sat_data_w = x[DATA_W-1:0];")
    lines.append("    end")
    lines.append("  endfunction")
    lines.append("")

    lines.append("  function integer layer_in_dim;")
    lines.append("    input integer l;")
    lines.append("    begin")
    lines.append("      case (l)")
    for li, layer in enumerate(layers):
        lines.append(f"        {li}: layer_in_dim = {layer.weight.shape[1]};")
    lines.append("        default: layer_in_dim = 1;")
    lines.append("      endcase")
    lines.append("    end")
    lines.append("  endfunction")
    lines.append("")

    lines.append("  function integer layer_out_dim;")
    lines.append("    input integer l;")
    lines.append("    begin")
    lines.append("      case (l)")
    for li, layer in enumerate(layers):
        lines.append(f"        {li}: layer_out_dim = {layer.weight.shape[0]};")
    lines.append("        default: layer_out_dim = 1;")
    lines.append("      endcase")
    lines.append("    end")
    lines.append("  endfunction")
    lines.append("")

    lines.append("  function integer layer_act;")
    lines.append("    input integer l;")
    lines.append("    begin")
    lines.append("      case (l)")
    for li, layer in enumerate(layers):
        act_mode = 1 if layer.act == "relu" else 2 if layer.act == "tanh" else 0
        lines.append(f"        {li}: layer_act = {act_mode};")
    lines.append("        default: layer_act = 0;")
    lines.append("      endcase")
    lines.append("    end")
    lines.append("  endfunction")
    lines.append("")

    lines.append("  function signed [DATA_W-1:0] apply_act;")
    lines.append("    input integer act_mode;")
    lines.append("    input signed [DATA_W-1:0] x;")
    lines.append("    begin")
    lines.append("      case (act_mode)")
    lines.append("        1: apply_act = (x < 0) ? 0 : x;")
    lines.append("        2: apply_act = hard_tanh(x);")
    lines.append("        default: apply_act = x;")
    lines.append("      endcase")
    lines.append("    end")
    lines.append("  endfunction")
    lines.append("")
    lines.append("  function signed [DATA_W-1:0] fx_mul;")
    lines.append("    input signed [DATA_W-1:0] a;")
    lines.append("    input signed [DATA_W-1:0] b;")
    lines.append("    reg signed [2*DATA_W-1:0] p;")
    lines.append("    reg signed [ACC_W-1:0] s;")
    lines.append("    begin")
    lines.append("      p = a * b;")
    lines.append("      s = p >>> FRAC_W;")
    lines.append("      fx_mul = sat_data_w(s);")
    lines.append("    end")
    lines.append("  endfunction")
    lines.append("")
    lines.append("  function signed [DATA_W-1:0] hard_tanh;")
    lines.append("    input signed [DATA_W-1:0] x;")
    lines.append("    reg signed [DATA_W-1:0] one_q;")
    lines.append("    begin")
    lines.append("      one_q = (1 <<< FRAC_W);")
    lines.append("      if (x > one_q) hard_tanh = one_q;")
    lines.append("      else if (x < -one_q) hard_tanh = -one_q;")
    lines.append("      else hard_tanh = x;")
    lines.append("    end")
    lines.append("  endfunction")
    lines.append("")

    lines.append("  function signed [DATA_W-1:0] in_at;")
    lines.append("    input integer idx;")
    lines.append("    begin")
    lines.append("      in_at = in_vec_flat[idx*DATA_W +: DATA_W];")
    lines.append("    end")
    lines.append("  endfunction")
    lines.append("")

    for li, layer in enumerate(layers):
        in_dim = layer.weight.shape[1]
        out_dim = layer.weight.shape[0]
        lines.append(f"  localparam L{li}_IN_DIM = {in_dim};")
        lines.append(f"  localparam L{li}_OUT_DIM = {out_dim};")
        lines.append(f"  reg signed [DATA_W-1:0] W{li} [0:{out_dim*in_dim-1}];")
        lines.append(f"  reg signed [DATA_W-1:0] B{li} [0:{out_dim-1}];")
    lines.append("")

    lines.append("  function signed [DATA_W-1:0] w_at;")
    lines.append("    input integer l;")
    lines.append("    input integer o;")
    lines.append("    input integer ii;")
    lines.append("    begin")
    lines.append("      case (l)")
    for li, _ in enumerate(layers):
        lines.append(f"        {li}: w_at = W{li}[o*L{li}_IN_DIM + ii];")
    lines.append("        default: w_at = 0;")
    lines.append("      endcase")
    lines.append("    end")
    lines.append("  endfunction")
    lines.append("")

    lines.append("  function signed [DATA_W-1:0] b_at;")
    lines.append("    input integer l;")
    lines.append("    input integer o;")
    lines.append("    begin")
    lines.append("      case (l)")
    for li, _ in enumerate(layers):
        lines.append(f"        {li}: b_at = B{li}[o];")
    lines.append("        default: b_at = 0;")
    lines.append("      endcase")
    lines.append("    end")
    lines.append("  endfunction")

    lines.append("")
    lines.append("  initial begin")
    lines.append("    state = ST_IDLE;")
    lines.append("    src_sel = 1'b0;")
    lines.append("    layer_idx = 0;")
    lines.append("    out_idx = 0;")
    lines.append("    in_idx = 0;")
    lines.append("    acc = 0;")
    lines.append("    out_valid = 1'b0;")
    lines.append("    out_vec_flat = 0;")
    lines.append("    for (i = 0; i < MAX_DIM; i = i + 1) begin")
    lines.append("      buf_a[i] = 0;")
    lines.append("      buf_b[i] = 0;")
    lines.append("    end")
    for li, layer in enumerate(layers):
        out_dim, in_dim = layer.weight.shape
        wq = _quantize(layer.weight, data_w, frac_w)
        bq = _quantize(layer.bias, data_w, frac_w)
        idx = 0
        for o in range(out_dim):
            for j in range(in_dim):
                lines.append(f"    W{li}[{idx}] = {_v_int(wq[o, j])};")
                idx += 1
        for o in range(out_dim):
            lines.append(f"    B{li}[{o}] = {_v_int(bq[o])};")
    lines.append("  end")
    lines.append("")

    last_out = layers[-1].weight.shape[0]
    lines.append("  always @(posedge clk or negedge rst_n) begin")
    lines.append("    if (!rst_n) begin")
    lines.append("      state <= ST_IDLE;")
    lines.append("      src_sel <= 1'b0;")
    lines.append("      layer_idx <= 0;")
    lines.append("      out_idx <= 0;")
    lines.append("      in_idx <= 0;")
    lines.append("      acc <= 0;")
    lines.append("      cur_in <= 0;")
    lines.append("      act_val <= 0;")
    lines.append("      out_valid <= 1'b0;")
    lines.append("      for (i = 0; i < MAX_DIM; i = i + 1) begin")
    lines.append("        buf_a[i] <= 0;")
    lines.append("        buf_b[i] <= 0;")
    lines.append("      end")
    lines.append(f"      for (i = 0; i < {last_out}; i = i + 1) begin")
    lines.append("        out_vec_flat[i*DATA_W +: DATA_W] <= 0;")
    lines.append("      end")
    lines.append("    end else begin")
    lines.append("      out_valid <= 1'b0;")
    lines.append("      case (state)")
    lines.append("        ST_IDLE: begin")
    lines.append("          if (in_valid) begin")
    lines.append("            for (i = 0; i < INPUT_DIM; i = i + 1) begin")
    lines.append("              buf_a[i] <= in_vec_flat[i*DATA_W +: DATA_W];")
    lines.append("            end")
    lines.append("            src_sel <= 1'b0;")
    lines.append("            layer_idx <= 0;")
    lines.append("            out_idx <= 0;")
    lines.append("            in_idx <= 0;")
    lines.append("            acc <= b_at(0, 0);")
    lines.append("            state <= ST_MAC;")
    lines.append("          end")
    lines.append("        end")
    lines.append("")
    lines.append("        ST_MAC: begin")
    lines.append("          if (src_sel == 1'b0) cur_in <= buf_a[in_idx];")
    lines.append("          else cur_in <= buf_b[in_idx];")
    lines.append("          acc <= acc + fx_mul((src_sel == 1'b0) ? buf_a[in_idx] : buf_b[in_idx], w_at(layer_idx, out_idx, in_idx));")
    lines.append("          if (in_idx == layer_in_dim(layer_idx) - 1) begin")
    lines.append("            state <= ST_STORE;")
    lines.append("          end else begin")
    lines.append("            in_idx <= in_idx + 1'b1;")
    lines.append("          end")
    lines.append("        end")
    lines.append("")
    lines.append("        ST_STORE: begin")
    lines.append("          act_val <= apply_act(layer_act(layer_idx), sat_data_w(acc));")
    lines.append("          if (src_sel == 1'b0) buf_b[out_idx] <= apply_act(layer_act(layer_idx), sat_data_w(acc));")
    lines.append("          else buf_a[out_idx] <= apply_act(layer_act(layer_idx), sat_data_w(acc));")
    lines.append("          if (out_idx == layer_out_dim(layer_idx) - 1) begin")
    lines.append("            if (layer_idx == NUM_LAYERS - 1) begin")
    lines.append("              state <= ST_DONE;")
    lines.append("            end else begin")
    lines.append("              src_sel <= ~src_sel;")
    lines.append("              layer_idx <= layer_idx + 1'b1;")
    lines.append("              out_idx <= 0;")
    lines.append("              in_idx <= 0;")
    lines.append("              acc <= b_at(layer_idx + 1'b1, 0);")
    lines.append("              state <= ST_MAC;")
    lines.append("            end")
    lines.append("          end else begin")
    lines.append("            out_idx <= out_idx + 1'b1;")
    lines.append("            in_idx <= 0;")
    lines.append("            acc <= b_at(layer_idx, out_idx + 1'b1);")
    lines.append("            state <= ST_MAC;")
    lines.append("          end")
    lines.append("        end")
    lines.append("")
    lines.append("        ST_DONE: begin")
    lines.append("          out_valid <= 1'b1;")
    lines.append(f"          for (i = 0; i < {last_out}; i = i + 1) begin")
    lines.append("            if (src_sel == 1'b0) out_vec_flat[i*DATA_W +: DATA_W] <= buf_b[i];")
    lines.append("            else out_vec_flat[i*DATA_W +: DATA_W] <= buf_a[i];")
    lines.append("          end")
    lines.append("          state <= ST_IDLE;")
    lines.append("        end")
    lines.append("")
    lines.append("        default: begin")
    lines.append("          state <= ST_IDLE;")
    lines.append("        end")
    lines.append("      endcase")
    lines.append("    end")
    lines.append("  end")
    lines.append("endmodule")

    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert ONNX MLP policy to Verilog-2001")
    parser.add_argument("--onnx", required=True, type=Path, help="Input ONNX file")
    parser.add_argument("--out-dir", required=True, type=Path, help="Output directory")
    parser.add_argument("--data-width", type=int, default=16, help="Fixed-point data width")
    parser.add_argument("--frac-width", type=int, default=12, help="Fixed-point fractional bits")
    args = parser.parse_args()

    if args.data_width < 8:
        raise ValueError("--data-width must be >= 8")
    if args.frac_width < 1 or args.frac_width >= args.data_width - 1:
        raise ValueError("--frac-width must satisfy 1 <= frac < data_width-1")

    model = onnx.load(str(args.onnx))
    layers = _extract_layers(model)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    dims_path = args.out_dir / "policy_dims.vh"
    core_path = args.out_dir / "policy_mlp_core.v"

    _emit_dims(dims_path, layers, data_w=args.data_width, frac_w=args.frac_width)
    _emit_core(core_path, layers, data_w=args.data_width, frac_w=args.frac_width)

    print(f"[OK] Generated: {dims_path}")
    print(f"[OK] Generated: {core_path}")
    print(
        "[INFO] Network dims: "
        f"input={layers[0].weight.shape[1]}, output={layers[-1].weight.shape[0]}, layers={len(layers)}"
    )


if __name__ == "__main__":
    main()
