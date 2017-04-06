#include "nw.h"
#include <assert.h>
#include <string.h>

#define MATCH_SCORE 1
#define MISMATCH_SCORE -1
#define GAP_SCORE -1

#define ALIGN '\\'
#define SKIPA '^'
#define SKIPB '<'

#define MAX(A,B) ( ((A)>(B))?(A):(B) )

#define JOBS_PER_BATCH 256

#define CYCLIC_1 1
#define CYCLIC_2 1

#define UNROLL_FACTOR 2

void needwun(char SEQA[ALEN], char SEQB[BLEN],
             char alignedA[ALEN+BLEN], char alignedB[ALEN+BLEN]){

    char M[BLEN+1][ALEN+1];
#pragma HLS array_partition variable=M dim=1 cyclic factor=1
#pragma HLS array_partition variable=M dim=2 cyclic factor=1
    char ptr[BLEN+1][ALEN+1];
#pragma HLS array_partition variable=ptr dim=1 cyclic factor=1
#pragma HLS array_partition variable=ptr dim=2 cyclic factor=1

    short score, up_left, up, left, max;
    unsigned a_idx, b_idx;
    unsigned a_str_idx, b_str_idx;

    init_row: for(a_idx=0; a_idx<(ALEN+1+CYCLIC_2-1)/CYCLIC_2; a_idx++){
#pragma HLS pipeline
	for (int i=0; i<CYCLIC_2; i++) {
#pragma HLS unroll factor=2
	    if (a_idx*CYCLIC_2+i < ALEN+1) {
                M[0][a_idx*CYCLIC_2+i] = (a_idx*CYCLIC_2+i) * GAP_SCORE;
	        ptr[0][a_idx*CYCLIC_2+i] = SKIPB; 
	    }
	}
    }
    init_col: for(b_idx=0; b_idx<(BLEN+1+CYCLIC_1-1)/CYCLIC_1; b_idx++){
#pragma HLS pipeline
	for (int i=0; i<CYCLIC_1; i++) {
#pragma HLS unroll factor=2
	    if (b_idx*CYCLIC_1+i < BLEN+1) {
                M[b_idx*CYCLIC_1+i][0] = (b_idx*CYCLIC_1+i) * GAP_SCORE;
	        ptr[b_idx*CYCLIC_1+i][0] = SKIPA;
	    }
	}
    }

    // Matrix filling loop
    fill_out: for(b_idx=1; b_idx<(BLEN+1); b_idx++){
        fill_in: for(a_idx=1; a_idx<(ALEN+1); a_idx++){
#pragma HLS pipeline
            if(SEQA[a_idx-1] == SEQB[b_idx-1]){
                score = MATCH_SCORE;
            } else {
                score = MISMATCH_SCORE;
            }

            up_left = M[b_idx-1][a_idx-1] + score;
            up      = M[b_idx-1][a_idx  ] + GAP_SCORE;
            left    = M[b_idx  ][a_idx-1] + GAP_SCORE;

            max = MAX(up_left, MAX(up, left));

            M[b_idx][a_idx] = max;
            if(max == left){
                ptr[b_idx][a_idx] = SKIPB;
            } else if(max == up){
                ptr[b_idx][a_idx] = SKIPA;
            } else{
                ptr[b_idx][a_idx] = ALIGN;
            }
        }
    }

    // TraceBack (n.b. aligned sequences are backwards to avoid string appending)
    a_idx = ALEN;
    b_idx = BLEN;
    a_str_idx = 0;
    b_str_idx = 0;

    trace: while(a_idx>0 || b_idx>0) {
#pragma HLS pipeline
#pragma HLS loop_tripcount min=128 max=256 avg=192
	switch(ptr[b_idx][a_idx]) {
	case ALIGN:
            alignedA[a_str_idx++] = SEQA[a_idx-1];
            alignedB[b_str_idx++] = SEQB[b_idx-1];
            a_idx--;
            b_idx--;
	    break;
	case SKIPB:
            alignedA[a_str_idx++] = SEQA[a_idx-1];
            alignedB[b_str_idx++] = '-';
            a_idx--;
	    break;
	default:
            alignedA[a_str_idx++] = '-';
            alignedB[b_str_idx++] = SEQB[b_idx-1];
            b_idx--;
	}
	// if (ptr[b_idx][a_idx] == ALIGN) {
        //     alignedA[a_str_idx++] = SEQA[a_idx-1];
        //     alignedB[b_str_idx++] = SEQB[b_idx-1];
        //     a_idx--;
        //     b_idx--;
	// }
	// else if (ptr[b_idx][a_idx] == SKIPB) {
        //     alignedA[a_str_idx++] = SEQA[a_idx-1];
        //     alignedB[b_str_idx++] = '-';
        //     a_idx--;
	// }
	// else {
        //     alignedA[a_str_idx++] = '-';
        //     alignedB[b_str_idx++] = SEQB[b_idx-1];
        //     b_idx--;
	// }
    }

    // Pad the result
    pad_a: for( ; a_str_idx<ALEN+BLEN; a_str_idx++ ) {
#pragma HLS pipeline
#pragma HLS loop_tripcount min=0 max=128 avg=64
      alignedA[a_str_idx] = '_';
    }
    pad_b: for( ; b_str_idx<ALEN+BLEN; b_str_idx++ ) {
#pragma HLS pipeline
#pragma HLS loop_tripcount min=0 max=128 avg=64
      alignedB[b_str_idx] = '_';
    }
}

void load(int flag, void* dst1, void* src1, size_t size1,
		    void* dst2, void* src2, size_t size2) {
#pragma HLS inline off
    if (flag) {
	memcpy(dst1, src1, size1);
	memcpy(dst2, src2, size2);
    }
}

void store(int flag, void* dst1, void* src1, size_t size1,
		     void* dst2, void* src2, size_t size2) {
#pragma HLS inline off
    if (flag) {
	memcpy(dst1, src1, size1);
	memcpy(dst2, src2, size2);
    }
}

void pe(char* seqA, char* seqB, char* alignedA, char* alignedB) {
#pragma HLS inline off
    for (unsigned i=0; i<JOBS_PER_BATCH/UNROLL_FACTOR; i++) {
        needwun(seqA+i*ALEN, seqB+i*BLEN, alignedA+i*(ALEN+BLEN), alignedB+i*(ALEN+BLEN));
    }
}

void compute(int flag, char seqA_buf[][ALEN*JOBS_PER_BATCH/UNROLL_FACTOR], char seqB_buf[][BLEN*JOBS_PER_BATCH/UNROLL_FACTOR], 
             char alignedA_buf[][(ALEN+BLEN)*JOBS_PER_BATCH/UNROLL_FACTOR], char alignedB_buf[][(ALEN+BLEN)*JOBS_PER_BATCH/UNROLL_FACTOR], int num_jobs) {
#pragma HLS inline off
    if (flag) {
        for (unsigned j=0; j<UNROLL_FACTOR; j++) {
#pragma HLS unroll
            pe(seqA_buf[j], seqB_buf[j], alignedA_buf[j], alignedB_buf[j]);
        }
    }
}

void workload(char* SEQA, char* SEQB,
              char* alignedA, char* alignedB, int num_jobs) {
#pragma HLS INTERFACE m_axi port=SEQA offset=slave bundle=gmem
#pragma HLS INTERFACE m_axi port=SEQB offset=slave bundle=gmem
#pragma HLS INTERFACE m_axi port=alignedA offset=slave bundle=gmem
#pragma HLS INTERFACE m_axi port=alignedB offset=slave bundle=gmem
#pragma HLS INTERFACE s_axilite port=SEQA bundle=control
#pragma HLS INTERFACE s_axilite port=SEQB bundle=control
#pragma HLS INTERFACE s_axilite port=alignedA bundle=control
#pragma HLS INTERFACE s_axilite port=alignedB bundle=control
#pragma HLS INTERFACE s_axilite port=num_jobs bundle=control
#pragma HLS INTERFACE s_axilite port=return bundle=control

	assert(num_jobs == (1 << 16));

	char seqA_buf_x[UNROLL_FACTOR][ALEN*JOBS_PER_BATCH/UNROLL_FACTOR];
#pragma HLS array_partition variable=seqA_buf_x dim=1 complete
	char seqB_buf_x[UNROLL_FACTOR][BLEN*JOBS_PER_BATCH/UNROLL_FACTOR];
#pragma HLS array_partition variable=seqB_buf_x dim=1 complete
	char alignedA_buf_x[UNROLL_FACTOR][(ALEN+BLEN)*JOBS_PER_BATCH/UNROLL_FACTOR];
#pragma HLS array_partition variable=alignedA_buf_x dim=1 complete
	char alignedB_buf_x[UNROLL_FACTOR][(ALEN+BLEN)*JOBS_PER_BATCH/UNROLL_FACTOR];
#pragma HLS array_partition variable=alignedB_buf_x dim=1 complete
	char seqA_buf_y[UNROLL_FACTOR][ALEN*JOBS_PER_BATCH/UNROLL_FACTOR];
#pragma HLS array_partition variable=seqA_buf_y dim=1 complete
	char seqB_buf_y[UNROLL_FACTOR][BLEN*JOBS_PER_BATCH/UNROLL_FACTOR];
#pragma HLS array_partition variable=seqB_buf_y dim=1 complete
	char alignedA_buf_y[UNROLL_FACTOR][(ALEN+BLEN)*JOBS_PER_BATCH/UNROLL_FACTOR];
#pragma HLS array_partition variable=alignedA_buf_y dim=1 complete
	char alignedB_buf_y[UNROLL_FACTOR][(ALEN+BLEN)*JOBS_PER_BATCH/UNROLL_FACTOR];
#pragma HLS array_partition variable=alignedB_buf_y dim=1 complete

	int num_batches = (num_jobs + JOBS_PER_BATCH - 1) / JOBS_PER_BATCH;
	int tail_jobs = num_jobs % JOBS_PER_BATCH;
	if (tail_jobs == 0) tail_jobs = JOBS_PER_BATCH;

	int i, j, k;
	for (i=0; i<num_batches+2; i++) {
          int load_jobs = i == num_batches-1? tail_jobs:JOBS_PER_BATCH;
          int compute_jobs = i == num_batches? tail_jobs:JOBS_PER_BATCH;
          int store_jobs = i == num_batches+1? tail_jobs:JOBS_PER_BATCH;
          int load_flag = i < num_batches;
          int compute_flag = i > 0 && i < num_batches+1;
          int store_flag = i > 1;
	  if (i % 2 == 0) {
	    load(load_flag, seqA_buf_x, SEQA+i*ALEN*JOBS_PER_BATCH, ALEN*load_jobs,
			    seqB_buf_x, SEQB+i*BLEN*JOBS_PER_BATCH, BLEN*load_jobs);
	    compute(compute_flag, seqA_buf_y, seqB_buf_y, alignedA_buf_y, alignedB_buf_y, compute_jobs);
	    store(store_flag, alignedA+(i-2)*(ALEN+BLEN)*JOBS_PER_BATCH, alignedA_buf_x, (ALEN+BLEN)*store_jobs,
			      alignedB+(i-2)*(ALEN+BLEN)*JOBS_PER_BATCH, alignedB_buf_x, (ALEN+BLEN)*store_jobs);
	  }
	  else {
	    load(load_flag, seqA_buf_y, SEQA+i*ALEN*JOBS_PER_BATCH, ALEN*load_jobs,
			    seqB_buf_y, SEQB+i*BLEN*JOBS_PER_BATCH, BLEN*load_jobs);
	    compute(compute_flag, seqA_buf_x, seqB_buf_x, alignedA_buf_x, alignedB_buf_x, compute_jobs);
	    store(store_flag, alignedA+(i-2)*(ALEN+BLEN)*JOBS_PER_BATCH, alignedA_buf_y, (ALEN+BLEN)*store_jobs,
			      alignedB+(i-2)*(ALEN+BLEN)*JOBS_PER_BATCH, alignedB_buf_y, (ALEN+BLEN)*store_jobs);
	  }
	}
	return;
}
