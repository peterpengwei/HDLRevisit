all:
	sdaccel compile_bitstream.tcl
clean:
	rm -rf sdaccel* spmv/ .Xil/ output.data
run:
	./spmv/pkg/pcie/spmv.exe input.data check.data ./spmv/pkg/pcie/workload.xclbin
