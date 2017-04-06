all:
	sdaccel compile_bitstream.tcl
cosim:
	sdaccel compile_bitstreamCosim.tcl
sim:
	sdaccel compile_bitstreamSim.tcl
clean:
	rm -rf sdaccel* stencil/ .Xil/ output.data
run:
	./stencil/pkg/pcie/stencil.exe /curr/pengwei/input.data /curr/pengwei/check.data ./stencil/pkg/pcie/workload.xclbin
