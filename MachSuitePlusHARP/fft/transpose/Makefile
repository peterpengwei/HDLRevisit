all:
	sdaccel compile_bitstream.tcl
clean:
	rm -rf sdaccel* fft/ .Xil/ output.data
run:
	./fft/pkg/pcie/fft.exe input.data check.data fft/pkg/pcie/workload.xclbin
