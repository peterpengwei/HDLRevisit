/*
Implementation based on:
Hong, Oguntebi, Olukotun. "Efficient Parallel Graph Exploration on Multi-Core CPU and GPU." PACT, 2011.
*/

#include "bfs.h"

#define Q_PUSH(node) { queue[q_in==0?N_NODES-1:q_in-1]=node; q_in=(q_in+1)%N_NODES; }
#define Q_PEEK() (queue[q_out])
#define Q_POP() { q_out = (q_out+1)%N_NODES; }
#define Q_EMPTY() (q_in>q_out ? q_in==q_out+1 : (q_in==0)&&(q_out==N_NODES-1))

void bfs(edge_index_t *edge_begin, 
	 edge_index_t *edge_end,
	 node_index_t *dst,
         node_index_t starting_node,
	 level_t *level,
         edge_index_t *level_counts) {
  node_index_t queue[N_NODES];
  node_index_t q_in, q_out;
  node_index_t dummy;
  node_index_t n;
  edge_index_t e;

  int i;
  init_levels: for( i=0; i<N_NODES; i++ ) {
      level[i] = MAX_LEVEL;
  }
  init_horizons: for( i=0; i<N_LEVELS; i++ ) {
      level_counts[i] = 0;
  }

  q_in = 1;
  q_out = 0;
  level[starting_node] = 0;
  level_counts[0] = 1;
  Q_PUSH(starting_node);

  loop_queue: for( dummy=0; dummy<N_NODES; dummy++ ) { // Typically while(not_empty(queue)){
    if( Q_EMPTY() )
      break;
    n = Q_PEEK();
    Q_POP();
    edge_index_t tmp_begin = edge_begin[n];
    edge_index_t tmp_end = edge_end[n];
    loop_neighbors: for( e=tmp_begin; e<tmp_end; e++ ) {
      node_index_t tmp_dst = dst[e];
      level_t tmp_level = level[tmp_dst];

      if( tmp_level ==MAX_LEVEL ) { // Unmarked
        level_t tmp_level = level[n]+1;
        level[tmp_dst] = tmp_level;
        ++level_counts[tmp_level];
        Q_PUSH(tmp_dst);
      }
    }
  }

  /*
  printf("Horizons:");
  for( i=0; i<N_LEVELS; i++ )
    printf(" %d", level_counts[i]);
  printf("\n");
  */
}
void workload(edge_index_t *edge_begin, 
	      edge_index_t *edge_end,
	      node_index_t *dst,
              node_index_t *p_starting_node,
	      level_t *level,
              edge_index_t *level_counts) {
#pragma HLS INTERFACE m_axi port=edge_begin offset=slave bundle=gmem
#pragma HLS INTERFACE m_axi port=edge_end offset=slave bundle=gmem
#pragma HLS INTERFACE m_axi port=dst offset=slave bundle=gmem
#pragma HLS INTERFACE m_axi port=p_starting_node offset=slave bundle=gmem
#pragma HLS INTERFACE m_axi port=level offset=slave bundle=gmem
#pragma HLS INTERFACE m_axi port=level_counts offset=slave bundle=gmem
#pragma HLS INTERFACE s_axilite port=edge_begin bundle=control
#pragma HLS INTERFACE s_axilite port=edge_end bundle=control
#pragma HLS INTERFACE s_axilite port=dst bundle=control
#pragma HLS INTERFACE s_axilite port=p_starting_node bundle=control
#pragma HLS INTERFACE s_axilite port=level bundle=control
#pragma HLS INTERFACE s_axilite port=level_counts bundle=control
#pragma HLS INTERFACE s_axilite port=return bundle=control

	edge_index_t local_edge_begin[N_NODES];
	edge_index_t local_edge_end[N_NODES];
	node_index_t local_dst[N_EDGES];
	level_t local_level[N_NODES];
	edge_index_t local_level_counts[N_LEVELS];

	// Step 1: copy
	memcpy(local_edge_begin, edge_begin, sizeof(local_edge_begin));
	memcpy(local_edge_end, edge_end, sizeof(local_edge_end));
	memcpy(local_dst, dst, sizeof(local_dst));

	bfs(local_edge_begin, local_edge_end, local_dst, *p_starting_node, local_level, local_level_counts);

	memcpy(level, local_level, sizeof(local_level));
	memcpy(level_counts, local_level_counts, sizeof(local_level_counts));

	return;
}
