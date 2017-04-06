#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <inttypes.h>

#define SIZE (1 << 25)
#define TYPE int32_t

////////////////////////////////////////////////////////////////////////////////
// Test harness interface code.

struct bench_args_t {
  TYPE a[SIZE];
};
