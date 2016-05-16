all:
	sdaccel compile_bitstream.tcl
clean:
	rm -rf sdaccel* nw/ .Xil/ output.data
run:
	./nw/pkg/pcie/nw.exe input.data check.data ./nw/pkg/pcie/workload.xclbin
