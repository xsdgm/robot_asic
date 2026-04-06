`include "policy_dims.vh"

module axis_feature_ingress #(
  parameter DATA_W = `POLICY_DATA_W,
  parameter INPUT_DIM = `POLICY_INPUT_DIM,
  parameter AXIS_W = 32
) (
  input clk,
  input rst_n,
  input [AXIS_W-1:0] s_axis_tdata,
  input s_axis_tvalid,
  output s_axis_tready,
  input s_axis_tlast,
  output reg vec_valid,
  output reg signed [DATA_W*INPUT_DIM-1:0] vec_data_flat
);

  localparam CNT_W = (INPUT_DIM <= 2) ? 1 : $clog2(INPUT_DIM);
  reg [CNT_W-1:0] wr_idx;

  assign s_axis_tready = 1'b1;

  always @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
      wr_idx <= 0;
      vec_valid <= 1'b0;
      vec_data_flat <= 0;
    end else begin
      vec_valid <= 1'b0;
      if (s_axis_tvalid && s_axis_tready) begin
        vec_data_flat[wr_idx*DATA_W +: DATA_W] <= s_axis_tdata[DATA_W-1:0];
        if ((wr_idx == INPUT_DIM - 1) || s_axis_tlast) begin
          wr_idx <= 0;
          vec_valid <= 1'b1;
        end else begin
          wr_idx <= wr_idx + 1'b1;
        end
      end
    end
  end

endmodule
