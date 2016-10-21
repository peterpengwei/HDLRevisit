all:
	sdaccel compile_bitstream.tcl
clean:
	rm -rf sdaccel* kmp/ .Xil/ output.data
run:
	./kmp/pkg/pcie/kmp.exe ../kmp_baseline/input.data ../kmp_baseline/check.data ./kmp/pkg/pcie/workload.xclbin
