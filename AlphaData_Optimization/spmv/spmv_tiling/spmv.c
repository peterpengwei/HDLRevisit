/*
Based on algorithm described here:
http://www.cs.berkeley.edu/~mhoemmen/matrix-seminar/slides/UCB_sparse_tutorial_1.pdf
*/

#include "spmv.h"

#define ROWS_PER_TILE 256

void ellpack(TYPE* nzval, short* cols, TYPE* vec, TYPE* out)
{
#pragma HLS INLINE off
    int i, j;
    TYPE Si, sum;

    ellpack_1 : for (i=0; i<ROWS_PER_TILE; i++) {
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

	TYPE local_nzval[ROWS_PER_TILE*L];
	short local_cols[ROWS_PER_TILE*L];

	TYPE local_vec[N];
	memcpy(local_vec, vec, sizeof(TYPE)*N);

	TYPE local_out[ROWS_PER_TILE];

	int i,j,k;

	for (i=0; i<num_tiles; i++) {
	    //Step 1:
	    memcpy(local_nzval, nzval + i*ROWS_PER_TILE*L, sizeof(TYPE)*ROWS_PER_TILE*L);
	    memcpy(local_cols, cols + i*ROWS_PER_TILE*L, sizeof(short)*ROWS_PER_TILE*L);

	    //Step 2:
	    ellpack(local_nzval, local_cols, local_vec, local_out);

	    //Step 3:
	    memcpy(out + i*ROWS_PER_TILE, local_out, sizeof(TYPE)*ROWS_PER_TILE);
	}

	return;
}
