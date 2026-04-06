# FPGA 策略 IP 生成与 LabVIEW 对接

本目录提供一条从训练得到的 ONNX 策略到 Verilog-2001 IP 的落地流程，目标用于 NI 机箱中的 LabVIEW FPGA 集成。

## 1. 当前能力

- 将 Gemm 型 MLP ONNX（常见于 RL policy）转换为可综合的 Verilog-2001：
  - `deploy/fpga/generated/policy_dims.vh`
  - `deploy/fpga/generated/policy_mlp_core.v`
- 提供 AXI4-Stream 包装顶层：
  - `deploy/fpga/rtl/policy_ip_top.v`
  - `deploy/fpga/rtl/axis_feature_ingress.v`
  - `deploy/fpga/rtl/axis_action_egress.v`
  - `deploy/fpga/rtl/udp_payload_to_axis_stub.v`（网口 UDP 载荷桥接骨架）
- 提供 Vivado IP 打包脚本：
  - `deploy/fpga/tcl/package_ip.tcl`

## 2. 一键生成 Verilog-2001

输入 ONNX 示例（你当前训练结果）：
- `logs/rsl_rl/g1_velocity/2026-04-05_17-31-28/policy.onnx`

执行：

```bash
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate g1_vel_gpu
python deploy/fpga/scripts/onnx_to_sv_mlp.py \
  --onnx logs/rsl_rl/g1_velocity/2026-04-05_17-31-28/policy.onnx \
  --out-dir deploy/fpga/generated \
  --data-width 16 \
  --frac-width 12
```

## 3. 打包为 Vivado IP

```bash
vivado -mode batch -source deploy/fpga/tcl/package_ip.tcl -tclargs \
  ./deploy/fpga/.vivado_proj \
  ./deploy/fpga/ip_repo \
  your.company user policy_ip 1.0 xc5vsx95tff1136-1
```

生成后可将 `deploy/fpga/ip_repo` 加入工程 IP Repository。

脚本默认目标器件也已设为 `xc5vsx95tff1136-1`（Virtex-5 XC5VSX95T）。

## 4. LabVIEW/NI 机箱侧接入建议

建议采用以下分层：

- 网口数据接收：PC 仿真软件通过以太网发送状态向量。
- 协议解析层：在 RT 控制器或 FPGA 前级逻辑把 UDP/TCP payload 转成定长特征向量。
- 可直接复用 `udp_payload_to_axis_stub.v` 作为起点，把 UDP 字节流打包为 AXI4-Stream。
- IP 推理层：本 IP 只处理定长向量推理，不直接解析 UDP/TCP 报文。
- 输出层：将动作向量回传到上位机或后级控制逻辑。

数据约定（建议）：

- 输入：AXI4-Stream，每拍一个定点数（`DATA_W` 位，低位有效）。
- 输入帧长：`INPUT_DIM` 拍，最后一拍 `tlast=1`。
- 输出：AXI4-Stream，每拍一个动作定点数，帧长 `OUTPUT_DIM`。

## 5. 约束与后续增强

### 5.1 Virtex-5 资源友好架构（已启用）

当前生成的 `policy_mlp_core.v` 使用单 MAC 串行累加状态机（而非全并行层计算）：

- 优点：DSP48/LUT 占用显著降低，更适配 XC5VSX95T。
- 代价：单次推理延迟上升（按神经元逐项累加）。

粗略估算：

- 每层时钟周期约为 `out_dim * in_dim`（外加少量状态切换开销）。
- 总推理周期约为各层 `out_dim * in_dim` 之和。

如果后续需要提高吞吐，可在此基础上扩展为 2/4/8 路并行 MAC（资源与吞吐线性权衡）。

当前转换器仅支持 Gemm 型 MLP（Relu/Tanh/Linear 激活）。若你的 ONNX 包含 Conv、LayerNorm、Attention 等结构，需要扩展转换器或切换到 HLS/FINN/hls4ml 流程。

Tanh 当前使用 hard-tanh 近似（裁剪到 [-1, 1]），便于综合与时序。

## 6. 建议的下一步

- 增加固定点误差评估脚本（Python 对比 ONNX 与 Verilog 定点输出）。
- 增加 UDP 包解析 RTL（或在 LabVIEW RT 层完成解析）。
- 根据目标 NI FPGA 型号做时钟域和 AXI 频率约束。
