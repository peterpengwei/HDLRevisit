#include <stdio.h>
#include <stdlib.h>
#include <inttypes.h>

//Define input sizes
#define f_size 9
#define tile_size 64

//Data Bounds
#define TYPE int32_t

//Set number of iterations to execute
#define MAX_ITERATION 1

////////////////////////////////////////////////////////////////////////////////
// Test harness interface code.

struct bench_args_t {
    TYPE orig[(tile_size+2)*(tile_size+2)];
    TYPE sol[tile_size*tile_size];
    TYPE filter[f_size];
};
