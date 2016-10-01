all:
	sdaccel compile_bitstream.tcl
clean:
	rm -rf sdaccel* viterbi/ .Xil/ output.data
run:
	./viterbi/pkg/pcie/viterbi.exe input.data check.data ./viterbi/pkg/pcie/workload.xclbin
