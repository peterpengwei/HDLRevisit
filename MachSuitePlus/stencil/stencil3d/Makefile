all:
	sdaccel compile_bitstream.tcl
clean:
	rm -rf sdaccel* stencil/ .Xil/ output.data
run:
	./stencil/pkg/pcie/stencil.exe input.data check.data stencil/pkg/pcie/workload.xclbin
