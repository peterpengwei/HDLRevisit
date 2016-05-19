/*
Implementations based on:
Harish and Narayanan. "Accelerating large graph algorithms on the GPU using CUDA." HiPC, 2007.
Hong, Oguntebi, Olukotun. "Efficient Parallel Graph Exploration on Multi-Core CPU and GPU." PACT, 2011.
*/

#include "bfs.h"

void bfs(edge_index_t *edge_begin, 
	 edge_index_t *edge_end,
	 node_index_t *dst,
         node_index_t starting_node,
	 level_t *level,
         edge_index_t *level_counts) {
  node_index_t n;
  edge_index_t e;
  level_t horizon;
  edge_index_t cnt;

  level[starting_node] = 0;
  level_counts[0] = 1;

  loop_horizons: for( horizon=0; horizon<N_LEVELS; horizon++ ) {
    cnt = 0;
    // Add unmarked neighbors of the current horizon to the next horizon
    loop_nodes: for( n=0; n<N_NODES; n++ ) {
      if( level[n]==horizon ) {
        edge_index_t tmp_begin = edge_begin[n];
        edge_index_t tmp_end = edge_end[n];
        loop_neighbors: for( e=tmp_begin; e<tmp_end; e++ ) {
          node_index_t tmp_dst = dst[e];
          level_t tmp_level = level[tmp_dst];

          if( tmp_level == MAX_LEVEL ) { // Unmarked
            level[tmp_dst] = horizon+1;
            ++cnt;
          }
        }
      }
    }
    if( (level_counts[horizon+1]=cnt)==0 )
      break;
  }
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

	bfs(edge_begin, edge_end, dst, *p_starting_node, level, level_counts);
	return;
}
