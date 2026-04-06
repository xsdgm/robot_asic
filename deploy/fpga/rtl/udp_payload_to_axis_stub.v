module udp_payload_to_axis_stub #(
  parameter AXIS_W = 32
) (
  input clk,
  input rst_n,
  input udp_payload_valid,
  input [7:0] udp_payload_byte,
  input udp_payload_last,
  output reg [AXIS_W-1:0] m_axis_tdata,
  output reg m_axis_tvalid,
  input m_axis_tready,
  output reg m_axis_tlast
);

  reg [1:0] byte_count;
  reg [31:0] pack_reg;

  always @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
      byte_count <= 0;
      pack_reg <= 0;
      m_axis_tdata <= 0;
      m_axis_tvalid <= 1'b0;
      m_axis_tlast <= 1'b0;
    end else begin
      if (m_axis_tvalid && m_axis_tready) begin
        m_axis_tvalid <= 1'b0;
        m_axis_tlast <= 1'b0;
      end

      if (udp_payload_valid) begin
        case (byte_count)
          2'd0: pack_reg[7:0] <= udp_payload_byte;
          2'd1: pack_reg[15:8] <= udp_payload_byte;
          2'd2: pack_reg[23:16] <= udp_payload_byte;
          default: pack_reg[31:24] <= udp_payload_byte;
        endcase

        if ((byte_count == 2'd3) || udp_payload_last) begin
          m_axis_tdata <= pack_reg;
          if (byte_count == 2'd0) m_axis_tdata[7:0] <= udp_payload_byte;
          if (byte_count == 2'd1) m_axis_tdata[15:8] <= udp_payload_byte;
          if (byte_count == 2'd2) m_axis_tdata[23:16] <= udp_payload_byte;
          if (byte_count == 2'd3) m_axis_tdata[31:24] <= udp_payload_byte;
          m_axis_tvalid <= 1'b1;
          m_axis_tlast <= udp_payload_last;
          byte_count <= 0;
        end else begin
          byte_count <= byte_count + 1'b1;
        end
      end
    end
  end

endmodule
