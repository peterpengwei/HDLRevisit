/*
Based on algorithm described here:
http://www.cs.berkeley.edu/~mhoemmen/matrix-seminar/slides/UCB_sparse_tutorial_1.pdf
*/

#include "spmv.h"
#include <string.h>

#define ROWS_PER_TILE 128
#define ROWS_PER_PE ((ROWS_PER_TILE+UNROLL_FACTOR-1)/UNROLL_FACTOR)

#define UNROLL_FACTOR 4

void ellpack(TYPE nzval[][L], short cols[][L], TYPE* vec, TYPE* out, int num_rows)
{
#pragma HLS inline off

    for (int k=0; k<ROWS_PER_PE; k++) {
#pragma HLS pipeline
	out[k] = 0.0;
    }
    ellpack_2 : for (int j=0; j<L; j++) {
        ellpack_1 : for (int i=0; i<ROWS_PER_PE; i++) {
#pragma HLS pipeline
	    if (i < num_rows) {
		out[i] += nzval[i][j] * vec[cols[i][j]];
	    }
        }
    }
}

void load(int flag, TYPE local_nzval[][ROWS_PER_PE][L], TYPE* nzval,
		    short local_cols[][ROWS_PER_PE][L], short* cols, int num_rows) {
#pragma HLS inline off
    if (flag) {
	memcpy(local_nzval, nzval, sizeof(TYPE)*L*num_rows);
	memcpy(local_cols, cols, sizeof(short)*L*num_rows);
    }
}

void store(int flag, TYPE* out, TYPE local_out[][ROWS_PER_PE], int num_rows) {
#pragma HLS inline off
    if (flag) {
        memcpy(out, local_out, sizeof(TYPE)*num_rows);
    }
}

void compute(int flag, TYPE nzval[][ROWS_PER_PE][L], short cols[][ROWS_PER_PE][L], TYPE vec[], TYPE out[][ROWS_PER_PE], int num_rows) {
#pragma HLS inline off
    if (flag) {
        TYPE vecs[UNROLL_FACTOR][N];
#pragma HLS array_partition variable=vecs dim=1 complete
	int i, j;
	for (i=0; i<N; i++) {
#pragma HLS pipeline
	    for (j=0; j<UNROLL_FACTOR; j++) {
#pragma HLS unroll
		vecs[j][i] = vec[i];
	    }
	}
	for (j=0; j<UNROLL_FACTOR; j++) {
#pragma HLS unroll
	    int pe_rows = num_rows - j*ROWS_PER_PE;
	    if (pe_rows > ROWS_PER_PE) pe_rows = ROWS_PER_PE;
	    if (pe_rows > 0) ellpack(nzval[j], cols[j], vecs[j], out[j], pe_rows);
	}
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

	int num_tiles = (N+ROWS_PER_TILE-1)/ROWS_PER_TILE;
	int tail_rows = N % ROWS_PER_TILE;
	if (tail_rows == 0) tail_rows = ROWS_PER_TILE;

	TYPE local_nzval_x[UNROLL_FACTOR][ROWS_PER_PE][L];
#pragma HLS array_partition variable=local_nzval_x dim=1 complete
	short local_cols_x[UNROLL_FACTOR][ROWS_PER_PE][L];
#pragma HLS array_partition variable=local_cols_x dim=1 complete
	TYPE local_nzval_y[UNROLL_FACTOR][ROWS_PER_PE][L];
#pragma HLS array_partition variable=local_nzval_y dim=1 complete
	short local_cols_y[UNROLL_FACTOR][ROWS_PER_PE][L];
#pragma HLS array_partition variable=local_cols_y dim=1 complete

	TYPE local_vec[N];
	memcpy(local_vec, vec, sizeof(TYPE)*N);

	TYPE local_out_x[UNROLL_FACTOR][ROWS_PER_PE];
#pragma HLS array_partition variable=local_out_x dim=1 complete
	TYPE local_out_y[UNROLL_FACTOR][ROWS_PER_PE];
#pragma HLS array_partition variable=local_out_y dim=1 complete

	int i,j,k;

	for (i=0; i<num_tiles+2; i++) {
	    int load_flag = i < num_tiles;
	    int compute_flag = i > 0 &&  i < num_tiles+1;
	    int store_flag = i > 1;
	    int load_rows = i == num_tiles-1? tail_rows:ROWS_PER_TILE;
	    int compute_rows = i == num_tiles? tail_rows:ROWS_PER_TILE;
	    int store_rows = i == num_tiles+1? tail_rows:ROWS_PER_TILE;

	    if (i % 2 == 0) {
	        load(load_flag, local_nzval_x, nzval+i*ROWS_PER_TILE*L,
				local_cols_x, cols+i*ROWS_PER_TILE*L, load_rows);
		compute(compute_flag, local_nzval_y, local_cols_y, local_vec, local_out_y, compute_rows);
		store(store_flag, out+(i-2)*ROWS_PER_TILE, local_out_x, store_rows);
	    }
	    else {
	        load(load_flag, local_nzval_y, nzval+i*ROWS_PER_TILE*L,
				local_cols_y, cols+i*ROWS_PER_TILE*L, load_rows);
		compute(compute_flag, local_nzval_x, local_cols_x, local_vec, local_out_x, compute_rows);
		store(store_flag, out+(i-2)*ROWS_PER_TILE, local_out_y, store_rows);
	    }
	}

	return;
}
