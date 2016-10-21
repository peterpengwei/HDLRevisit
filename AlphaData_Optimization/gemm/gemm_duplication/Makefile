all:
	sdaccel compile_bitstream.tcl
clean:
	rm -rf sdaccel* gemm/ .Xil/ output.data
run:
	./gemm/pkg/pcie/gemm.exe input.data check.data ./gemm/pkg/pcie/workload.xclbin
