`include "policy_dims.vh"

module policy_ip_top #(
  parameter AXIS_W = 32,
  parameter DATA_W = `POLICY_DATA_W,
  parameter INPUT_DIM = `POLICY_INPUT_DIM,
  parameter OUTPUT_DIM = `POLICY_OUTPUT_DIM
) (
  input clk,
  input rst_n,
  input [AXIS_W-1:0] s_axis_tdata,
  input s_axis_tvalid,
  output s_axis_tready,
  input s_axis_tlast,
  output [AXIS_W-1:0] m_axis_tdata,
  output m_axis_tvalid,
  input m_axis_tready,
  output m_axis_tlast
);

  wire in_vec_valid;
  wire out_vec_valid;
  wire signed [DATA_W*INPUT_DIM-1:0] in_vec_flat;
  wire signed [DATA_W*OUTPUT_DIM-1:0] out_vec_flat;

  axis_feature_ingress #(
    .DATA_W(DATA_W),
    .INPUT_DIM(INPUT_DIM),
    .AXIS_W(AXIS_W)
  ) u_ingress (
    .clk(clk),
    .rst_n(rst_n),
    .s_axis_tdata(s_axis_tdata),
    .s_axis_tvalid(s_axis_tvalid),
    .s_axis_tready(s_axis_tready),
    .s_axis_tlast(s_axis_tlast),
    .vec_valid(in_vec_valid),
    .vec_data_flat(in_vec_flat)
  );

  policy_mlp_core #(
    .DATA_W(DATA_W),
    .FRAC_W(`POLICY_FRAC_W),
    .INPUT_DIM(INPUT_DIM),
    .OUTPUT_DIM(OUTPUT_DIM)
  ) u_core (
    .clk(clk),
    .rst_n(rst_n),
    .in_valid(in_vec_valid),
    .in_vec_flat(in_vec_flat),
    .out_valid(out_vec_valid),
    .out_vec_flat(out_vec_flat)
  );

  axis_action_egress #(
    .DATA_W(DATA_W),
    .OUTPUT_DIM(OUTPUT_DIM),
    .AXIS_W(AXIS_W)
  ) u_egress (
    .clk(clk),
    .rst_n(rst_n),
    .out_valid(out_vec_valid),
    .out_vec_flat(out_vec_flat),
    .m_axis_tdata(m_axis_tdata),
    .m_axis_tvalid(m_axis_tvalid),
    .m_axis_tready(m_axis_tready),
    .m_axis_tlast(m_axis_tlast)
  );

endmodule
