#include "sort.h"
#include <string.h>

void merge(TYPE a[JOBS_PER_UNROLL], int start, int m, int stop) {
#pragma HLS inline off
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
#pragma HLS inline off
	static TYPE temp[TILING_SIZE];
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
#pragma HLS inline off
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

void sort_map(TYPE a[UNROLL_FACTOR][JOBS_PER_UNROLL]) {
	int i;
	for (i = 0; i < UNROLL_FACTOR; i ++) {
#pragma HLS unroll
		ms_mergesort(a[i]);
	}
}

void sort_reduce(TYPE *a) {
#pragma HLS inline off
	int m, i, start = 0, stop = TILING_SIZE;
	for (m = JOBS_PER_UNROLL; m < TILING_SIZE; m += m) {
        mergesort_label2:
		for(i = start; i < stop; i += m + m) {
			merge_reduce(a, i, i + m - 1, i + 2 * m - 1);
		}
	}
}

void workload(TYPE *a) {
#pragma HLS INTERFACE m_axi offset=slave port=a bundle=gmem
#pragma HLS INTERFACE s_axilite port=a bundle=control
#pragma HLS INTERFACE s_axilite port=return bundle=control

	int i, iterCount = SIZE / TILING_SIZE;
	TYPE local_a[UNROLL_FACTOR][JOBS_PER_UNROLL];
#pragma HLS ARRAY_PARTITION variable=local_a complete dim=1
	for (i = 0; i < iterCount; i ++) {
		memcpy(local_a[0], a + i * TILING_SIZE, sizeof(TYPE) * TILING_SIZE);
		sort_map(local_a);
		//		sort_reduce(local_a);
		memcpy(a + i * TILING_SIZE, local_a[0], sizeof(TYPE) * TILING_SIZE);
	}
	return;
}
