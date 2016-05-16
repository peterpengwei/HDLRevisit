all:
	sdaccel compile_bitstream.tcl
clean:
	rm -rf sdaccel* bfs/ .Xil/ output.data
run:
	./bfs/pkg/pcie/bfs.exe input.data check.data ./bfs/pkg/pcie/workload.xclbin
