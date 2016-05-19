all:
	sdaccel compile_bitstream.tcl
clean:
	rm -rf sdaccel* backprop/ .Xil/ output.data
run:
	./backprop/pkg/pcie/backprop.exe input.data check.data ./backprop/pkg/pcie/workload.xclbin
