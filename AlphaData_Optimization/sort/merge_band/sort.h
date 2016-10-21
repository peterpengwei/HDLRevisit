#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <inttypes.h>

#define SIZE (1 << 25)
#define TYPE int32_t
#define TYPE_MAX INT32_MAX
#define TILING_SIZE (1 << 17)
#define UNROLL_FACTOR (64)
#define JOBS_PER_UNROLL (TILING_SIZE / UNROLL_FACTOR)

void ms_mergesort(TYPE a[SIZE]);
void workload(TYPE *a);

////////////////////////////////////////////////////////////////////////////////
// Test harness interface code.

struct bench_args_t {
  TYPE a[SIZE];
};
