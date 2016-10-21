#include "sort.h"
#include <string.h>

void merge(TYPE a[TILING_SIZE], int start, int m, int stop) {
	TYPE temp[TILING_SIZE];
	int i, j, k;

 merge_label1:
	for (i = start; i <= m; i ++) {
		temp[i] = a[i];
	}

 merge_label2:
	for (j = m + 1; j <= stop; j ++) {
		temp[m + 1 + stop - j] = a[j];
	}

	i = start;
	j = stop;

 merge_label3:
	for (k = start; k <= stop; k ++) {
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

void ms_mergesort(TYPE a[TILING_SIZE]) {
	int start, stop;
	int i, m, from, mid, to;

	start = 0;
	stop = TILING_SIZE;

 mergesort_label1:
	for (m = 1; m < stop - start; m += m) {
        mergesort_label2:
		for(i = start; i < stop; i += m + m) {
			from = i;
			mid = i + m - 1;
			to = i + m + m - 1;
			if(to < stop){
				merge(a, from, mid, to);
			}
			else {
				merge(a, from, mid, stop);
			}
		}
	}
}

void workload(TYPE *a) {
#pragma HLS INTERFACE m_axi offset=slave port=a bundle=gmem
#pragma HLS INTERFACE s_axilite port=a bundle=control
#pragma HLS INTERFACE s_axilite port=return bundle=control

	int i, iterCount = SIZE / TILING_SIZE;
	TYPE local_a[TILING_SIZE];
	for (i = 0; i < iterCount; i ++) {
		memcpy(local_a, a + i * TILING_SIZE, sizeof(local_a));
		ms_mergesort(local_a);
		memcpy(a + i * TILING_SIZE, local_a, sizeof(local_a));
	}
	return;
}
