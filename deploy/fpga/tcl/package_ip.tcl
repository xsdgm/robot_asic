# Vivado batch TCL script for packaging policy_ip_top as a reusable IP.
# Usage example:
# vivado -mode batch -source deploy/fpga/tcl/package_ip.tcl -tclargs \
#   <proj_dir> <ip_repo_dir> <vendor> <library> <name> <version> [part]

if { $argc < 6 } {
  puts "Usage: package_ip.tcl <proj_dir> <ip_repo_dir> <vendor> <library> <name> <version> [part]"
  exit 1
}

set proj_dir  [lindex $argv 0]
set ip_repo   [lindex $argv 1]
set vendor    [lindex $argv 2]
set library   [lindex $argv 3]
set ip_name   [lindex $argv 4]
set ip_ver    [lindex $argv 5]
set fpga_part "xc5vsx95tff1136-1"
if { $argc >= 7 } {
  set fpga_part [lindex $argv 6]
}

set rtl_dir [file normalize "deploy/fpga/rtl"]
set gen_dir [file normalize "deploy/fpga/generated"]

file mkdir $proj_dir
file mkdir $ip_repo

create_project -force policy_ip_pkg $proj_dir -part $fpga_part
add_files [glob -nocomplain $rtl_dir/*.v]
add_files [glob -nocomplain $gen_dir/*.v]
add_files [glob -nocomplain $gen_dir/*.vh]
set_property include_dirs [list $gen_dir] [get_filesets sources_1]
update_compile_order -fileset sources_1

# Package current project as IP.
ipx::package_project -root_dir $ip_repo -vendor $vendor -library $library -taxonomy {/UserIP}
set core [ipx::current_core]
set_property name $ip_name $core
set_property version $ip_ver $core
set_property display_name $ip_name $core
set_property description "ONNX policy MLP inference IP with AXI4-Stream I/O" $core

# Infer and expose interfaces.
ipx::infer_bus_interface s_axis xilinx.com:interface:axis_rtl:1.0 $core
ipx::infer_bus_interface m_axis xilinx.com:interface:axis_rtl:1.0 $core
ipx::associate_bus_interfaces -clock clk -reset rst_n $core

ipx::create_xgui_files $core
ipx::update_checksums $core
ipx::save_core $core

close_project
puts "[OK] IP packaged at: $ip_repo"
