/*
Based on algorithm described here:
http://www.cs.berkeley.edu/~mhoemmen/matrix-seminar/slides/UCB_sparse_tutorial_1.pdf
*/

#include "spmv.h"
#include <string.h>

#define ROWS_PER_TILE 256
#define UNROLL_FACTOR 64

void ellpack(TYPE* nzval, short* cols, TYPE* vec, TYPE* out)
{
#pragma HLS INLINE off
    int i, j;
    TYPE Si, sum;

    ellpack_1 : for (i=0; i<ROWS_PER_TILE/UNROLL_FACTOR; i++) {
        sum = 0.0;
        ellpack_2 : for (j=0; j<L; j++) {
            sum = sum + nzval[j + i*L] * vec[cols[j + i*L]];
        }
        out[i] = sum;
    }
}

void workload(TYPE* nzval, short* cols, TYPE* vec, TYPE* out) {
#pragma HLS INTERFACE m_axi port=nzval offset=slave bundle=gmem1
#pragma HLS INTERFACE m_axi port=cols offset=slave bundle=gmem2
#pragma HLS INTERFACE m_axi port=vec offset=slave bundle=gmem1
#pragma HLS INTERFACE m_axi port=out offset=slave bundle=gmem1
#pragma HLS INTERFACE s_axilite port=nzval bundle=control
#pragma HLS INTERFACE s_axilite port=cols bundle=control
#pragma HLS INTERFACE s_axilite port=vec bundle=control
#pragma HLS INTERFACE s_axilite port=out bundle=control
#pragma HLS INTERFACE s_axilite port=return bundle=control

	int num_tiles = N/ROWS_PER_TILE;

	TYPE local_nzval[UNROLL_FACTOR][ROWS_PER_TILE*L/UNROLL_FACTOR];
	#pragma HLS ARRAY_PARTITION variable=local_nzval dim=1 complete
	short local_cols[UNROLL_FACTOR][ROWS_PER_TILE*L/UNROLL_FACTOR];
	#pragma HLS ARRAY_PARTITION variable=local_cols dim=1 complete

	int i,j,k;

	TYPE local_vec[UNROLL_FACTOR][N];
	#pragma HLS ARRAY_PARTITION variable=local_vec dim=1 complete
	memcpy(local_vec[0], vec, N*sizeof(TYPE));
	for(i=0; i<N; i++) {
        #pragma HLS PIPELINE
	    for(j=1; j<UNROLL_FACTOR; j++) {
	    #pragma HLS UNROLL
	        local_vec[j][i] = local_vec[0][i];
	    }
	} 

	TYPE local_out[UNROLL_FACTOR][ROWS_PER_TILE/UNROLL_FACTOR];
	#pragma HLS ARRAY_PARTITION variable=local_out dim=1 complete

	for (i=0; i<num_tiles; i++) {
	    //Step 1:
	    for (j=0; j<UNROLL_FACTOR; j++) {
	        memcpy(local_nzval[j], nzval + i*ROWS_PER_TILE*L + j*ROWS_PER_TILE/UNROLL_FACTOR*L, sizeof(TYPE)*ROWS_PER_TILE*L/UNROLL_FACTOR);
	    }
	    for (j=0; j<UNROLL_FACTOR; j++) {
	        memcpy(local_cols[j], cols + i*ROWS_PER_TILE*L + j*ROWS_PER_TILE/UNROLL_FACTOR*L, sizeof(short)*ROWS_PER_TILE*L/UNROLL_FACTOR);
	    }

	    //Step 2:
	    for (j=0; j<UNROLL_FACTOR; j++) {
	    #pragma HLS UNROLL
	        ellpack(local_nzval[j], local_cols[j], local_vec[j], local_out[j]);
	    }

	    //Step 3:
	    for (j=0; j<UNROLL_FACTOR; j++) {
	        memcpy(out + i*ROWS_PER_TILE + j*ROWS_PER_TILE/UNROLL_FACTOR, local_out[j], sizeof(TYPE)*ROWS_PER_TILE/UNROLL_FACTOR);
	    }
	}

	return;
}
