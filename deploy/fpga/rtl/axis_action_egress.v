`include "policy_dims.vh"

module axis_action_egress #(
  parameter DATA_W = `POLICY_DATA_W,
  parameter OUTPUT_DIM = `POLICY_OUTPUT_DIM,
  parameter AXIS_W = 32
) (
  input clk,
  input rst_n,
  input out_valid,
  input signed [DATA_W*OUTPUT_DIM-1:0] out_vec_flat,
  output reg [AXIS_W-1:0] m_axis_tdata,
  output reg m_axis_tvalid,
  input m_axis_tready,
  output reg m_axis_tlast
);

  localparam CNT_W = (OUTPUT_DIM <= 2) ? 1 : $clog2(OUTPUT_DIM);
  reg sending;
  reg [CNT_W-1:0] rd_idx;
  reg signed [DATA_W*OUTPUT_DIM-1:0] out_latched;

  always @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
      sending <= 1'b0;
      rd_idx <= 0;
      out_latched <= 0;
      m_axis_tdata <= 0;
      m_axis_tvalid <= 1'b0;
      m_axis_tlast <= 1'b0;
    end else begin
      if (!sending && out_valid) begin
        out_latched <= out_vec_flat;
        sending <= 1'b1;
        rd_idx <= 0;
        m_axis_tvalid <= 1'b1;
      end

      if (sending && m_axis_tvalid) begin
        m_axis_tdata <= {{(AXIS_W-DATA_W){out_latched[rd_idx*DATA_W + DATA_W - 1]}}, out_latched[rd_idx*DATA_W +: DATA_W]};
        m_axis_tlast <= (rd_idx == OUTPUT_DIM - 1);

        if (m_axis_tready) begin
          if (rd_idx == OUTPUT_DIM - 1) begin
            sending <= 1'b0;
            rd_idx <= 0;
            m_axis_tvalid <= 1'b0;
            m_axis_tlast <= 1'b0;
          end else begin
            rd_idx <= rd_idx + 1'b1;
          end
        end
      end else begin
        m_axis_tlast <= 1'b0;
      end
    end
  end

endmodule
