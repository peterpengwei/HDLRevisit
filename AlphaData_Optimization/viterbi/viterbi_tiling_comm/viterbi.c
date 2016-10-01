#include "viterbi.h"

#define JOBS_PER_BATCH 256

int viterbi( tok_t obs[N_OBS], prob_t init[N_STATES], prob_t transition[N_STATES*N_STATES], prob_t emission[N_STATES*N_TOKENS], state_t path[N_OBS] )
{
  prob_t llike[N_OBS][N_STATES];
  step_t t;
  state_t prev, curr;
  prob_t min_p, p;
  state_t min_s, s;
  // All probabilities are in -log space. (i.e.: P(x) => -log(P(x)) )
 
  // Initialize with first observation and initial probabilities
  L_init: for( s=0; s<N_STATES; s++ ) {
    llike[0][s] = init[s] + emission[s*N_TOKENS+obs[0]];
  }

  // Iteratively compute the probabilities over time
  L_timestep: for( t=1; t<N_OBS; t++ ) {
    L_curr_state: for( curr=0; curr<N_STATES; curr++ ) {
      // Compute likelihood HMM is in current state and where it came from.
      prev = 0;
      min_p = llike[t-1][prev] +
              transition[prev*N_STATES+curr] +
              emission[curr*N_TOKENS+obs[t]];
      L_prev_state: for( prev=1; prev<N_STATES; prev++ ) {
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
    p = llike[N_OBS-1][s];
    if( p<min_p ) {
      min_p = p;
      min_s = s;
    }
  }
  path[N_OBS-1] = min_s;

  // Backtrack to recover full path
  L_backtrack: for( t=N_OBS-2; t>=0; t-- ) {
    min_s = 0;
    min_p = llike[t][min_s] + transition[min_s*N_STATES+path[t+1]];
    L_state: for( s=1; s<N_STATES; s++ ) {
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

void workload( tok_t* obs, prob_t* init, prob_t* transition, prob_t* emission, state_t* path, int num_jobs ) {
#pragma HLS INTERFACE m_axi port=obs offset=slave bundle=gmem1
#pragma HLS INTERFACE m_axi port=init offset=slave bundle=gmem2
#pragma HLS INTERFACE m_axi port=transition offset=slave bundle=gmem2
#pragma HLS INTERFACE m_axi port=emission offset=slave bundle=gmem2
#pragma HLS INTERFACE m_axi port=path offset=slave bundle=gmem3
#pragma HLS INTERFACE s_axilite port=obs bundle=control
#pragma HLS INTERFACE s_axilite port=init bundle=control
#pragma HLS INTERFACE s_axilite port=transition bundle=control
#pragma HLS INTERFACE s_axilite port=emission bundle=control
#pragma HLS INTERFACE s_axilite port=path bundle=control
#pragma HLS INTERFACE s_axilite port=num_jobs bundle=control
#pragma HLS INTERFACE s_axilite port=return bundle=control

    tok_t local_obs[N_OBS*JOBS_PER_BATCH];
    prob_t local_init[N_STATES*JOBS_PER_BATCH];
    prob_t local_transition[TRANS_SIZE*JOBS_PER_BATCH];
    prob_t local_emission[EMIT_SIZE*JOBS_PER_BATCH];
    state_t local_path[N_OBS*JOBS_PER_BATCH];

    int num_batches = num_jobs / JOBS_PER_BATCH;

    int i,j;
    

    for (i=0; i<num_batches; i++) {
	//step 1: copy data in
	memcpy(local_obs, obs+i*JOBS_PER_BATCH*N_OBS, sizeof(tok_t)*N_OBS*JOBS_PER_BATCH);
	memcpy(local_init, init+i*JOBS_PER_BATCH*N_STATES, sizeof(prob_t)*N_STATES*JOBS_PER_BATCH);
	memcpy(local_transition, transition+i*JOBS_PER_BATCH*TRANS_SIZE, sizeof(prob_t)*TRANS_SIZE*JOBS_PER_BATCH);
	memcpy(local_emission, emission+i*JOBS_PER_BATCH*EMIT_SIZE, sizeof(prob_t)*EMIT_SIZE*JOBS_PER_BATCH);
        // for (j=0; j<JOBS_PER_BATCH; j++) {
        //     viterbi(local_obs + j*N_OBS, local_init + j*N_STATES, local_transition + j*TRANS_SIZE, local_emission + j*EMIT_SIZE, local_path + j*N_OBS);
        // }
	memcpy(path+i*JOBS_PER_BATCH*N_OBS, local_path, sizeof(state_t)*N_OBS*JOBS_PER_BATCH);
    }
    return;
}
