#include <stdio.h>
#include <stdlib.h>

#define twoPI 6.28318530717959

#define SIZE (1 << 16)

////////////////////////////////////////////////////////////////////////////////
// Test harness interface code.

struct bench_args_t {
        double real[SIZE];
        double img[SIZE];
        double real_twid[SIZE/2];
        double img_twid[SIZE/2];
};
