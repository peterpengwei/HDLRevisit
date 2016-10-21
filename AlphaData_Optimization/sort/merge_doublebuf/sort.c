#include "sort.h"
#include <string.h>

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

void ms_mergesort(TYPE a[JOBS_PER_UNROLL]) {
	int start, stop;
	int i, m, from, mid, to;

	start = 0;
	stop = JOBS_PER_UNROLL;

 mergesort_label1:
	for (m = 1; m < JOBS_PER_UNROLL; m += m) {
        mergesort_label2:
		for(i = start; i < stop; i += m + m) {
			merge(a, i, i + m - 1, i + 2 * m - 1);
		}
	}
}

void compute(int flag, TYPE a[UNROLL_FACTOR][JOBS_PER_UNROLL]) {
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

void load(int flag, TYPE *local_a, TYPE *a) {
	if (flag) {
		memcpy(local_a, a, sizeof(TYPE) * TILING_SIZE);
	}
}

void save(int flag, TYPE *local_a, TYPE *a) {
	if (flag) {
		memcpy(a, local_a, sizeof(TYPE) * TILING_SIZE);
	}
}

void workload(TYPE *a) {
#pragma HLS INTERFACE m_axi offset=slave port=a bundle=gmem
#pragma HLS INTERFACE s_axilite port=a bundle=control
#pragma HLS INTERFACE s_axilite port=return bundle=control

	int i, iterCount = SIZE / TILING_SIZE;
	TYPE local_a_0[UNROLL_FACTOR][JOBS_PER_UNROLL];
#pragma HLS ARRAY_PARTITION variable=local_a_0 complete dim=1
	TYPE local_a_1[UNROLL_FACTOR][JOBS_PER_UNROLL];
#pragma HLS ARRAY_PARTITION variable=local_a_1 complete dim=1
	TYPE local_a_2[UNROLL_FACTOR][JOBS_PER_UNROLL];
#pragma HLS ARRAY_PARTITION variable=local_a_2 complete dim=1
	
	for (i = 0; i < iterCount + 2; i ++) {
		int idx = i % 3;
		int load_flag = (i < iterCount);
		int compute_flag = (i > 0) && (i < iterCount + 1);
		int save_flag = (i > 1) && (i < iterCount + 2);
		switch (idx) {
		case 0:
			load(load_flag, local_a_0[0], a + i * TILING_SIZE);
			compute(compute_flag, local_a_2);
			save(save_flag, local_a_1[0], a + (i - 2) * TILING_SIZE);
			break;
		case 1:
			load(load_flag, local_a_1[0], a + i * TILING_SIZE);
			compute(compute_flag, local_a_0);
			save(save_flag, local_a_2[0], a + (i - 2) * TILING_SIZE);
			break;
		case 2:
			load(load_flag, local_a_2[0], a + i * TILING_SIZE);
			compute(compute_flag, local_a_1);
			save(save_flag, local_a_0[0], a + (i - 2) * TILING_SIZE);
			break;
		default:
			break;
		}
	}
	return;
}
