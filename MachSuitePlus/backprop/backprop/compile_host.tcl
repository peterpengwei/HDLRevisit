# Define the solution for SDAccel
create_solution -name backprop_host -dir . -force
add_device -vbnv xilinx:adm-pcie-7v3:1ddr:2.1

# Host Compiler Flags
set_property -name host_cflags -value "-g -Wall -D FPGA_DEVICE -D C_KERNEL -I/curr/pengwei/ISCA17/HDLRevisit/MachSuite/common"  -objects [current_solution]

# Host Source Files
add_files "../../common/harness.c ../../common/support.c local_support.c"
add_files "../../common/support.h backprop.h"
set_property file_type "c header files" [get_files "support.h"]
set_property file_type "c header files" [get_files "backprop.h"]

compile_host -arch x86_64

# Package the application binaries
package_system

