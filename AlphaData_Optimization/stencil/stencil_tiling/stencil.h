#include <stdio.h>
#include <stdlib.h>
#include <inttypes.h>

//Define input sizes
#define col_size 4096 
#define row_size 4096
#define f_size 9

//Data Bounds
#define TYPE int32_t

//Set number of iterations to execute
#define MAX_ITERATION 1

////////////////////////////////////////////////////////////////////////////////
// Test harness interface code.

struct bench_args_t {
    TYPE orig[row_size*col_size];
    TYPE sol[row_size*col_size];
    TYPE filter[f_size];
};
