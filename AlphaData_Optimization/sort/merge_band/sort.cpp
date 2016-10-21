#include "sort.h"
#include <string.h>

#include "ap_int.h"

typedef ap_int<128> uint128_t;
typedef ap_int<256> uint256_t;
typedef ap_int<512> uint512_t;

extern "C" {

#define WIDTH_FACTOR_INT 4

#define TYPE_INT uint128_t

void merge(TYPE a[JOBS_PER_UNROLL], int start, int m, int stop) {
	TYPE temp[JOBS_PER_UNROLL];
	int i, j, k;

 merge_label1:
	for (i = start; i <= m; i ++) {
#pragma HLS PIPELINE
		temp[i] = a[i];
	}

 merge_label2:
	for (j = m + 1; j <= stop; j ++) {
#pragma HLS PIPELINE
		temp[m + 1 + stop - j] = a[j];
	}

	i = start;
	j = stop;

 merge_label3:
	for (k = start; k <= stop; k ++) {
#pragma HLS PIPELINE
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

void merge_reduce(TYPE a[TILING_SIZE], int start, int m, int stop) {
	TYPE temp[TILING_SIZE];
	int i, j, k;

 merge_label1:
	for (i = start; i <= m; i ++) {
#pragma HLS PIPELINE
		temp[i] = a[i];
	}

 merge_label2:
	for (j = m + 1; j <= stop; j ++) {
#pragma HLS PIPELINE
		temp[m + 1 + stop - j] = a[j];
	}

	i = start;
	j = stop;

 merge_label3:
	for (k = start; k <= stop; k ++) {
#pragma HLS PIPELINE
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

void ms_mergesort(TYPE_INT global_a[JOBS_PER_UNROLL/WIDTH_FACTOR_INT]) {

	TYPE a[JOBS_PER_UNROLL];

	int i,j;

	TYPE_INT tmp_int;
	for (i=0; i<JOBS_PER_UNROLL/WIDTH_FACTOR_INT; i++) {
	#pragma HLS PIPELINE
	  tmp_int = global_a[i];
	  a[i*WIDTH_FACTOR_INT] = tmp_int.range(31, 0);
	  for (j=1; j<WIDTH_FACTOR_INT; j++) {
	    tmp_int = tmp_int >> 32;
	    a[i*WIDTH_FACTOR_INT+j] = tmp_int.range(31, 0);
	  }
	}
	

	int start, stop;
	int m, from, mid, to;

	start = 0;
	stop = JOBS_PER_UNROLL;

 mergesort_label1:
	for (m = 1; m < JOBS_PER_UNROLL; m += m) {
        mergesort_label2:
		for(i = start; i < stop; i += m + m) {
			merge(a, i, i + m - 1, i + 2 * m - 1);
		}
	}

	for (i=0; i<JOBS_PER_UNROLL/WIDTH_FACTOR_INT; i++) {
	#pragma HLS PIPELINE
	  tmp_int.range(31, 0) = a[i*WIDTH_FACTOR_INT];
	  for (j=1; j<WIDTH_FACTOR_INT; j++) {
	    tmp_int = tmp_int >> 32;
	    tmp_int.range(31, 0) = a[i*WIDTH_FACTOR_INT+j];
	  }
	  global_a[i] = tmp_int;
	}
}

void compute(int flag, TYPE_INT a[UNROLL_FACTOR][JOBS_PER_UNROLL/WIDTH_FACTOR_INT]) {
	if (flag) {
		int i, m, start = 0, stop = TILING_SIZE;
		for (i = 0; i < UNROLL_FACTOR; i ++) {
#pragma HLS unroll
			ms_mergesort(a[i]);
		}
		/* for (m = JOBS_PER_UNROLL; m < TILING_SIZE; m += m) { */
		/* mergesort_label2: */
		/* 	for(i = start; i < stop; i += m + m) { */
		/* 		merge_reduce(a[0], i, i + m - 1, i + 2 * m - 1); */
		/* 	} */
		/* } */
	}
}

void load(int flag, TYPE_INT *local_a, TYPE_INT *a) {
	if (flag) {
		memcpy(local_a, a, sizeof(TYPE) * TILING_SIZE);
	}
}

void save(int flag, TYPE_INT *local_a, TYPE_INT *a) {
	if (flag) {
		memcpy(a, local_a, sizeof(TYPE) * TILING_SIZE);
	}
}

void workload(TYPE_INT *a) {
#pragma HLS INTERFACE m_axi offset=slave port=a bundle=gmem
#pragma HLS INTERFACE s_axilite port=a bundle=control
#pragma HLS INTERFACE s_axilite port=return bundle=control

	int i, iterCount = SIZE / TILING_SIZE;
	TYPE_INT local_a_0[UNROLL_FACTOR][JOBS_PER_UNROLL/WIDTH_FACTOR_INT];
#pragma HLS ARRAY_PARTITION variable=local_a_0 complete dim=1
	TYPE_INT local_a_1[UNROLL_FACTOR][JOBS_PER_UNROLL/WIDTH_FACTOR_INT];
#pragma HLS ARRAY_PARTITION variable=local_a_1 complete dim=1
	TYPE_INT local_a_2[UNROLL_FACTOR][JOBS_PER_UNROLL/WIDTH_FACTOR_INT];
#pragma HLS ARRAY_PARTITION variable=local_a_2 complete dim=1
	
	for (i = 0; i < iterCount + 2; i ++) {
		int idx = i % 3;
		int load_flag = (i < iterCount);
		int compute_flag = (i > 0) && (i < iterCount + 1);
		int save_flag = (i > 1) && (i < iterCount + 2);
		switch (idx) {
		case 0:
			load(load_flag, local_a_0[0], a + i * TILING_SIZE/WIDTH_FACTOR_INT);
			compute(compute_flag, local_a_2);
			save(save_flag, local_a_1[0], a + (i - 2) * TILING_SIZE/WIDTH_FACTOR_INT);
			break;
		case 1:
			load(load_flag, local_a_1[0], a + i * TILING_SIZE/WIDTH_FACTOR_INT);
			compute(compute_flag, local_a_0);
			save(save_flag, local_a_2[0], a + (i - 2) * TILING_SIZE/WIDTH_FACTOR_INT);
			break;
		case 2:
			load(load_flag, local_a_2[0], a + i * TILING_SIZE/WIDTH_FACTOR_INT);
			compute(compute_flag, local_a_1);
			save(save_flag, local_a_0[0], a + (i - 2) * TILING_SIZE/WIDTH_FACTOR_INT);
			break;
		default:
			break;
		}
	}
	return;
}

}
