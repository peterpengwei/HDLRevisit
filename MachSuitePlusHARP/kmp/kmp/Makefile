all:
	sdaccel compile_bitstream.tcl
clean:
	rm -rf sdaccel* kmp/ .Xil/ output.data
run:
	./kmp/pkg/pcie/kmp.exe input.data check.data ./kmp/pkg/pcie/workload.xclbin
