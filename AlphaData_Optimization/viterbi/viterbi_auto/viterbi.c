#include "viterbi.h"
#include <assert.h>
#include <string.h>

#define JOBS_PER_BATCH 256
#define UNROLL_FACTOR 2
#define JOBS_PER_PE 128
//#define JOBS_PER_PE ((JOBS_PER_BATCH+UNROLL_FACTOR-1)/UNROLL_FACTOR)

int viterbi( tok_t obs[N_OBS], prob_t init[N_STATES], prob_t transition[TRANS_SIZE], prob_t emission[EMIT_SIZE], state_t path[N_OBS] )
{
  prob_t llike[N_OBS][N_STATES];
#pragma HLS array_partition variable=llike dim=2 complete
  step_t t;
  state_t prev, curr;
  prob_t min_p, p;
  state_t min_s, s;
  // All probabilities are in -log space. (i.e.: P(x) => -log(P(x)) )
 
  // Initialize with first observation and initial probabilities
  L_init: for( s=0; s<N_STATES; s++ ) {
#pragma HLS unroll
#pragma HLS dependence variable=emission inter false
    assert (obs[0] >= 0 && obs[0] < N_TOKENS);
    llike[0][s] = init[s] + emission[s*N_TOKENS+obs[0]];
  }

  // Iteratively compute the probabilities over time
  L_timestep: for( t=1; t<N_OBS; t++ ) {
#pragma HLS pipeline
    L_curr_state: for( curr=0; curr<N_STATES; curr++ ) {
#pragma HLS unroll
#pragma HLS dependence variable=emission inter false
      assert(obs[t] >=0 && obs[t] < N_TOKENS);
      // Compute likelihood HMM is in current state and where it came from.
      prev = 0;
      min_p = llike[t-1][prev] +
              transition[prev*N_STATES+curr] +
              emission[curr*N_TOKENS+obs[t]];
      L_prev_state: for( prev=1; prev<N_STATES; prev++ ) {
#pragma HLS unroll
#pragma HLS dependence variable=emission inter false
        p = llike[t-1][prev] +
            transition[prev*N_STATES+curr] +
            emission[curr*N_TOKENS+obs[t]];
        if( p<min_p ) {
          min_p = p;
        }
      }
      llike[t][curr] = min_p;
    }
  }

  // Identify end state
  min_s = 0;
  min_p = llike[N_OBS-1][min_s];
  L_end: for( s=1; s<N_STATES; s++ ) {
#pragma HLS unroll
    p = llike[N_OBS-1][s];
    if( p<min_p ) {
      min_p = p;
      min_s = s;
    }
  }
  path[N_OBS-1] = min_s;

  // Backtrack to recover full path
  L_backtrack: for( t=N_OBS-2; t>=0; t-- ) {
#pragma HLS pipeline
    assert (min_s >=0 && min_s < N_STATES);
    assert (path[t+1] >= 0 && path[t+1] < N_STATES);
    min_s = 0;
    min_p = llike[t][min_s] + transition[min_s*N_STATES+path[t+1]];
    L_state: for( s=1; s<N_STATES; s++ ) {
#pragma HLS unroll
#pragma HLS dependence variable=transition inter false
      p = llike[t][s] + transition[s*N_STATES+path[t+1]];
      if( p<min_p ) {
        min_p = p;
        min_s = s;
      }
    }
    path[t] = min_s;
  }

  return 0;
}

void load(int flag, tok_t* dst1, tok_t* src1, size_t size1,
		    prob_t* dst2, prob_t* src2, size_t size2,
		    prob_t* dst3, prob_t* src3, size_t size3,
		    prob_t* dst4, prob_t* src4, size_t size4) {
#pragma HLS inline off
    if (flag) {
        memcpy(dst1, src1, size1);
        memcpy(dst2, src2, size2);
        memcpy(dst3, src3, size3);
        memcpy(dst4, src4, size4);
    }

}

void store(int flag, state_t* dst, state_t* src, size_t size) {
#pragma HLS inline off
    if (flag) {
	memcpy(dst, src, size);
    }
}

void pe(tok_t obs[][N_OBS], prob_t init[][N_STATES], prob_t transition[][TRANS_SIZE], prob_t emission[][EMIT_SIZE], state_t path[][N_OBS], int num_jobs) {
#pragma HLS inline off
    for (int i=0; i<num_jobs; i++) {
        viterbi(obs[i], init[i], transition[i], emission[i], path[i]);
    }
}

void compute(int flag, tok_t obs[][JOBS_PER_PE][N_OBS], prob_t init[][JOBS_PER_PE][N_STATES],
		       prob_t transition[][JOBS_PER_PE][TRANS_SIZE], prob_t emission[][JOBS_PER_PE][EMIT_SIZE],
		       state_t path[][JOBS_PER_PE][N_OBS], int num_jobs) {
#pragma HLS inline off
    if (flag) {
        for (int i=0; i<UNROLL_FACTOR; i++) {
#pragma HLS unroll
	    int pe_jobs = num_jobs-i*JOBS_PER_PE;
	    if (pe_jobs > JOBS_PER_PE) pe_jobs = JOBS_PER_PE;
	    if (pe_jobs > 0) pe(obs[i], init[i], transition[i], emission[i], path[i], pe_jobs);
	}
    
    }
}

void workload( tok_t* obs, prob_t* init, prob_t* transition, prob_t* emission, state_t* path, int num_jobs ) {
#pragma HLS INTERFACE m_axi port=obs offset=slave bundle=gmem1
#pragma HLS INTERFACE m_axi port=init offset=slave bundle=gmem2
#pragma HLS INTERFACE m_axi port=transition offset=slave bundle=gmem2
#pragma HLS INTERFACE m_axi port=emission offset=slave bundle=gmem2
#pragma HLS INTERFACE m_axi port=path offset=slave bundle=gmem1
#pragma HLS INTERFACE s_axilite port=obs bundle=control
#pragma HLS INTERFACE s_axilite port=init bundle=control
#pragma HLS INTERFACE s_axilite port=transition bundle=control
#pragma HLS INTERFACE s_axilite port=emission bundle=control
#pragma HLS INTERFACE s_axilite port=path bundle=control
#pragma HLS INTERFACE s_axilite port=num_jobs bundle=control
#pragma HLS INTERFACE s_axilite port=return bundle=control

    tok_t local_obs_x[UNROLL_FACTOR][JOBS_PER_PE][N_OBS];
#pragma HLS array_partition variable=local_obs_x dim=1 complete
    prob_t local_init_x[UNROLL_FACTOR][JOBS_PER_PE][N_STATES];
#pragma HLS array_partition variable=local_init_x dim=1 complete
#pragma HLS array_partition variable=local_init_x dim=3 complete
    prob_t local_transition_x[UNROLL_FACTOR][JOBS_PER_PE][TRANS_SIZE];
#pragma HLS array_partition variable=local_transition_x dim=1 complete
#pragma HLS array_partition variable=local_transition_x dim=3 complete
    prob_t local_emission_x[UNROLL_FACTOR][JOBS_PER_PE][EMIT_SIZE];
#pragma HLS array_partition variable=local_emission_x dim=1 complete
#pragma HLS array_partition variable=local_emission_x dim=3 complete
    state_t local_path_x[UNROLL_FACTOR][JOBS_PER_PE][N_OBS];
#pragma HLS array_partition variable=local_path_x dim=1 complete

    tok_t local_obs_y[UNROLL_FACTOR][JOBS_PER_PE][N_OBS];
#pragma HLS array_partition variable=local_obs_y dim=1 complete
    prob_t local_init_y[UNROLL_FACTOR][JOBS_PER_PE][N_STATES];
#pragma HLS array_partition variable=local_init_y dim=1 complete
#pragma HLS array_partition variable=local_init_y dim=3 complete
    prob_t local_transition_y[UNROLL_FACTOR][JOBS_PER_PE][TRANS_SIZE];
#pragma HLS array_partition variable=local_transition_y dim=1 complete
#pragma HLS array_partition variable=local_transition_y dim=3 complete
    prob_t local_emission_y[UNROLL_FACTOR][JOBS_PER_PE][EMIT_SIZE];
#pragma HLS array_partition variable=local_emission_y dim=1 complete
#pragma HLS array_partition variable=local_emission_y dim=3 complete
    state_t local_path_y[UNROLL_FACTOR][JOBS_PER_PE][N_OBS];
#pragma HLS array_partition variable=local_path_y dim=1 complete

    assert(num_jobs == (1 << 20));

    int num_batches = (num_jobs+JOBS_PER_BATCH-1) / JOBS_PER_BATCH;
    int tail_jobs = num_jobs % JOBS_PER_BATCH;
    if (tail_jobs == 0) tail_jobs = JOBS_PER_BATCH;

    int i,j;

    for (i=0; i<num_batches+2; i++) {
        int load_jobs = i == num_batches-1? tail_jobs:JOBS_PER_BATCH;
        int compute_jobs = i == num_batches? tail_jobs:JOBS_PER_BATCH;
        int store_jobs = i == num_batches+1? tail_jobs:JOBS_PER_BATCH;
        int load_flag = i < num_batches;
        int compute_flag = i > 0 && i < num_batches+1;
        int store_flag = i > 1;
	if (i % 2 == 0) {
	    load(load_flag, local_obs_x, (obs+i*N_OBS*JOBS_PER_BATCH), sizeof(tok_t)*N_OBS*load_jobs,
			    local_init_x, (init+i*N_STATES*JOBS_PER_BATCH), sizeof(prob_t)*N_STATES*load_jobs,
			    local_transition_x, (transition+i*TRANS_SIZE*JOBS_PER_BATCH), sizeof(prob_t)*TRANS_SIZE*load_jobs,
			    local_emission_x, (emission+i*EMIT_SIZE*JOBS_PER_BATCH), sizeof(prob_t)*EMIT_SIZE*load_jobs);
	    compute(compute_flag, local_obs_y, local_init_y, local_transition_y, local_emission_y, local_path_y, compute_jobs);
	    store(store_flag, path+(i-2)*N_OBS*JOBS_PER_BATCH, local_path_x, sizeof(state_t)*N_OBS*store_jobs);
	}
	else {
	    load(load_flag, local_obs_y, obs+i*N_OBS*JOBS_PER_BATCH, sizeof(tok_t)*N_OBS*load_jobs,
			    local_init_y, init+i*N_STATES*JOBS_PER_BATCH, sizeof(prob_t)*N_STATES*load_jobs,
			    local_transition_y, transition+i*TRANS_SIZE*JOBS_PER_BATCH, sizeof(prob_t)*TRANS_SIZE*load_jobs,
			    local_emission_y, emission+i*EMIT_SIZE*JOBS_PER_BATCH, sizeof(prob_t)*EMIT_SIZE*load_jobs);
	    compute(compute_flag, local_obs_x, local_init_x, local_transition_x, local_emission_x, local_path_x, compute_jobs);
	    store(store_flag, path+(i-2)*N_OBS*JOBS_PER_BATCH, local_path_y, sizeof(state_t)*N_OBS*store_jobs);
	}
    }
    return;
}
