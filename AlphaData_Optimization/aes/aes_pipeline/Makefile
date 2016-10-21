all:
	sdaccel compile_bitstream.tcl
clean:
	rm -rf sdaccel* aes/ .Xil/ output.data
run:
	./aes/pkg/pcie/aes.exe input.data check.data ./aes/pkg/pcie/workload.xclbin
