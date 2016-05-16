all:
	sdaccel compile_bitstream.tcl
clean:
	rm -rf sdaccel* md/ .Xil/ output.data
run:
	./md/pkg/pcie/md.exe input.data check.data ./md/pkg/pcie/workload.xclbin
