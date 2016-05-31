#include "bfs.h"
#include "support.h"
#include <string.h>
#include <unistd.h>

int INPUT_SIZE = sizeof(struct bench_args_t);

void run_benchmark( void *vargs, cl_context& context, 
  cl_command_queue& commands, cl_program& program, cl_kernel& kernel ) {
  struct bench_args_t *args = (struct bench_args_t *)vargs;
  node_index_t dst[N_EDGES];
  edge_index_t edge_begin[N_NODES];
  edge_index_t edge_end[N_NODES];
  int i;
  for (i = 0; i < N_EDGES; i++) dst[i] = args->edges[i].dst;
  for (i = 0; i < N_NODES; i++) {
  	edge_begin[i] = args->nodes[i].edge_begin;
  	edge_end[i] = args->nodes[i].edge_end;
  }
  // Create device buffers
  //
  static unsigned * edge_begin_buffer = 
    (unsigned int*)clSVMAllocAltera(context, 0, sizeof(edge_begin), 1024);
  static unsigned * edge_end_buffer = 
    (unsigned int*)clSVMAllocAltera(context, 0, sizeof(edge_end), 1024);
  static unsigned * dst_buffer = 
    (unsigned int*)clSVMAllocAltera(context, 0, sizeof(dst), 1024);
  static unsigned * level_buffer = 
    (unsigned int*)clSVMAllocAltera(context, 0, sizeof(args->level), 1024);
  static unsigned * level_counts_buffer = 
    (unsigned int*)clSVMAllocAltera(context, 0, sizeof(args->level_counts), 1024);

  // Write our data set into device buffers  
  //
  memcpy(edge_begin_buffer, edge_begin, sizeof(edge_begin));
  memcpy(edge_end_buffer, edge_end, sizeof(edge_end));
  memcpy(dst_buffer, dst, sizeof(dst));
  memcpy(level_buffer, args->level, sizeof(args->level));
  memcpy(level_counts_buffer, args->level_counts, sizeof(args->level_counts));
  node_index_t starting_node = args->starting_node;
    
  // Set the arguments to our compute kernel
  //
  int status;
  status  = clSetKernelArgSVMPointerAltera(kernel, 0, (void*)edge_begin_buffer);
  status |= clSetKernelArgSVMPointerAltera(kernel, 1, (void*)edge_end_buffer);
  status |= clSetKernelArgSVMPointerAltera(kernel, 2, (void*)dst_buffer);
  status |= clSetKernelArg(kernel, 3, sizeof(node_index_t), &starting_node);
  status |= clSetKernelArgSVMPointerAltera(kernel, 4, (void*)level_buffer);
  status |= clSetKernelArgSVMPointerAltera(kernel, 5, (void*)level_counts_buffer);
  if (status != CL_SUCCESS) {
    dump_error("Failed set args.", status);
    exit(1);
  }

  // Execute the kernel over the entire range of our 1d input data set
  // using the maximum number of work group items for this device
  //

#ifdef OPENCL_KERNEL
  status = clEnqueueTask(commands, kernel, 0, NULL, NULL);
#else
  printf("Error: C kernel is not currently supported!\n");
  exit(1);
#endif
  if (status)
  {
    printf("Error: Failed to execute kernel! %d\n", status);
    printf("Test failed\n");
    exit(1);
  }
  clFinish(commands);

  // Read back the results from the device to verify the output
  //
  memcpy(args->level, level_buffer, sizeof(args->level));
  memcpy(args->level_counts, level_counts_buffer, sizeof(args->level_counts));
}

/* Input format:
%% Section 1
uint64_t[1]: starting node
%% Section 2
uint64_t[N_NODES*2]: node structures (start and end indices of edge lists)
%% Section 3
uint64_t[N_EDGES]: edges structures (just destination node id)
*/

void input_to_data(int fd, void *vdata) {
  struct bench_args_t *data = (struct bench_args_t *)vdata;
  char *p, *s;
  uint64_t *nodes;
  int64_t i;

  // Zero-out everything.
  memset(vdata,0,sizeof(struct bench_args_t));
  // Max-ify levels
  for(i=0; i<N_NODES; i++) {
    data->level[i]=MAX_LEVEL;
  }
  // Load input string
  p = readfile(fd);
  // Section 1: starting node
  s = find_section_start(p,1);
  parse_uint64_t_array(s, (uint64_t *)(&data->starting_node), 1);

  // Section 2: node structures
  s = find_section_start(p,2);
  nodes = (uint64_t *)malloc(N_NODES*2*sizeof(uint64_t));
  parse_uint64_t_array(s, nodes, N_NODES*2);
  for(i=0; i<N_NODES; i++) {
    data->nodes[i].edge_begin = nodes[2*i];
    data->nodes[i].edge_end = nodes[2*i+1];
  }
  free(nodes);
  // Section 3: edge structures
  s = find_section_start(p,3);
  parse_uint64_t_array(s, (uint64_t *)(data->edges), N_EDGES);
}

void data_to_input(int fd, void *vdata) {
  uint64_t *nodes;
  int64_t i;

  struct bench_args_t *data = (struct bench_args_t *)vdata;
  // Section 1: starting node
  write_section_header(fd);
  write_uint64_t_array(fd, (uint64_t *)(&data->starting_node), 1);
  // Section 2: node structures
  write_section_header(fd);
  nodes = (uint64_t *)malloc(N_NODES*2*sizeof(uint64_t));
  for(i=0; i<N_NODES; i++) {
    nodes[2*i]  = data->nodes[i].edge_begin;
    nodes[2*i+1]= data->nodes[i].edge_end;
  }
  write_uint64_t_array(fd, nodes, N_NODES*2);
  free(nodes);
  // Section 3: edge structures
  write_section_header(fd);
  write_uint64_t_array(fd, (uint64_t *)(data->edges), N_EDGES);
}

/* Output format:
%% Section 1
uint64_t[N_LEVELS]: horizon counts
*/

void output_to_data(int fd, void *vdata) {
  struct bench_args_t *data = (struct bench_args_t *)vdata;
  char *p, *s;
  // Zero-out everything.
  memset(vdata,0,sizeof(struct bench_args_t));
  // Load input string
  p = readfile(fd);
  // Section 1: horizon counts
  s = find_section_start(p,1);
  parse_uint64_t_array(s, (uint64_t *)(data->level_counts), N_LEVELS);
}

void data_to_output(int fd, void *vdata) {
  struct bench_args_t *data = (struct bench_args_t *)vdata;
  // Section 1
  write_section_header(fd);
  write_uint64_t_array(fd, (uint64_t *)(data->level_counts), N_LEVELS);
}

int check_data( void *vdata, void *vref ) {
  struct bench_args_t *data = (struct bench_args_t *)vdata;
  struct bench_args_t *ref = (struct bench_args_t *)vref;
  int has_errors = 0;
  int i;

  // Check that the horizons have the same number of nodes
  for(i=0; i<N_LEVELS; i++) {
    has_errors |= (data->level_counts[i]!=ref->level_counts[i]);
  }

  // Return true if it's correct.
  return !has_errors;
}
