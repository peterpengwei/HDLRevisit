all:
	sdaccel compile_bitstream.tcl
clean:
	rm -rf sdaccel* sort/ .Xil/ output.data
run:
	./sort/pkg/pcie/sort.exe ../merge_baseline/input.data ../merge_baseline/check.data ./sort/pkg/pcie/workload.xclbin
