#include <stdio.h>
#include <stdlib.h>

#define twoPI 6.28318530717959

#define FFT_SIZE 128

////////////////////////////////////////////////////////////////////////////////
// Test harness interface code.

struct bench_args_t {
        double real[FFT_SIZE];
        double img[FFT_SIZE];
        double real_twid[FFT_SIZE/2];
        double img_twid[FFT_SIZE/2];
};
