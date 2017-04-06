#include "sort.h"
#include <assert.h>
#include <string.h>

#define TILE_SIZE (1<<18)
#define UNROLL_FACTOR 4
#define PE_SIZE ((TILE_SIZE+UNROLL_FACTOR-1)/UNROLL_FACTOR)
#define AVG_COUNT ((PE_SIZE+15)/16)

void merge(TYPE* a, int start, int m, int stop, TYPE* temp) {
	int i, j, k;

 merge_label1:
	for (i = start; i <= m; i ++) {
#pragma HLS pipeline
#pragma HLS loop_tripcount min=8 max=8 avg=8
		temp[i] = a[i];
	}

 merge_label2:
	for (j = m + 1; j <= stop; j ++) {
#pragma HLS pipeline
#pragma HLS loop_tripcount min=8 max=8 avg=8
		temp[m + 1 + stop - j] = a[j];
	}

	i = start;
	j = stop;

 merge_label3:
	for (k = start; k <= stop; k ++) {
#pragma HLS pipeline II=2
#pragma HLS loop_tripcount min=16 max=16 avg=16
		TYPE tmp_j = temp[j];
		TYPE tmp_i = temp[i];
		if(tmp_j < tmp_i) {
			a[k] = tmp_j;
			j--;
		} else {
			a[k] = tmp_i;
			i++;
		}
	}
}

void ms_mergesort(TYPE* a, int size) {
#pragma HLS inline off
	int start, stop;
	int i, m, from, mid, to;

	start = 0;
	stop = size;
	assert (size == PE_SIZE || size == TILE_SIZE%PE_SIZE);

	TYPE temp[PE_SIZE];

 mergesort_label1:
	for (m = 1; m < stop - start; m += m) {
        mergesort_label2:
		for(i = start; i < stop; i += m + m) {
#pragma HLS loop_tripcount min=AVG_COUNT max=AVG_COUNT avg=AVG_COUNT
			from = i;
			mid = i + m - 1;
			to = i + m + m - 1;
			if(to < stop){
				merge(a, from, mid, to, temp);
			}
			else {
				merge(a, from, mid, stop, temp);
			}
		}
	}
}

void load(int flag, TYPE local_a[][PE_SIZE], TYPE* a, int size) {
#pragma HLS inline off
    if (flag) {
        memcpy(local_a, a, size*sizeof(TYPE));
    }
}

void store(int flag, TYPE* a, TYPE local_a[][PE_SIZE], int size) {
#pragma HLS inline off
    if (flag) {
        memcpy(a, local_a, size*sizeof(TYPE));
    }
}

void compute(int flag, TYPE a[][PE_SIZE], int size) {
#pragma HLS inline off
    if (flag) {
        for (int i=0; i<UNROLL_FACTOR; i++) {
#pragma HLS unroll
	    int pe_size = size - i*PE_SIZE;
	    if (pe_size > PE_SIZE) pe_size = PE_SIZE;
	    if (pe_size > 0) ms_mergesort(a[i], pe_size);
	}
    }
}

void workload(TYPE *a) {
#pragma HLS INTERFACE m_axi offset=slave port=a bundle=gmem
#pragma HLS INTERFACE s_axilite port=a bundle=control
#pragma HLS INTERFACE s_axilite port=return bundle=control

    TYPE local_a_x[UNROLL_FACTOR][PE_SIZE];
#pragma HLS array_partition variable=local_a_x dim=1 complete
    TYPE local_a_y[UNROLL_FACTOR][PE_SIZE];
#pragma HLS array_partition variable=local_a_y dim=1 complete
    TYPE local_a_z[UNROLL_FACTOR][PE_SIZE];
#pragma HLS array_partition variable=local_a_z dim=1 complete

    int num_tiles = (SIZE+TILE_SIZE-1)/TILE_SIZE;
    int tail_size = SIZE % TILE_SIZE;
    if (tail_size == 0) tail_size = TILE_SIZE;
    for (int i=0; i<num_tiles+2; i++) {
        int load_flag = i < num_tiles;
	int compute_flag = i > 0 && i < num_tiles+1;
	int store_flag = i > 1;
	int load_size = i == num_tiles-1? tail_size:TILE_SIZE;
	int compute_size = i == num_tiles? tail_size:TILE_SIZE;
	int store_size = i == num_tiles+1? tail_size:TILE_SIZE;
	if (i % 3 == 0) {
	    load(load_flag, local_a_x, a+i*TILE_SIZE, load_size);
	    compute(compute_flag, local_a_z, compute_size);
	    store(store_flag, a+(i-2)*TILE_SIZE, local_a_y, store_size);
	}
	else if (i % 3 == 1) {
	    load(load_flag, local_a_y, a+i*TILE_SIZE, load_size);
	    compute(compute_flag, local_a_x, compute_size);
	    store(store_flag, a+(i-2)*TILE_SIZE, local_a_z, store_size);
	}
	else {
	    load(load_flag, local_a_z, a+i*TILE_SIZE, load_size);
	    compute(compute_flag, local_a_y, compute_size);
	    store(store_flag, a+(i-2)*TILE_SIZE, local_a_x, store_size);
	}
    }
}
