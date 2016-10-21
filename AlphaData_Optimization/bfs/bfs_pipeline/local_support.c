#include "bfs.h"
#include "support.h"
#include <string.h>
#include <unistd.h>
#include "my_timer.h"

int INPUT_SIZE = sizeof(struct bench_args_t);

void run_benchmark( void *vargs, cl_context& context, cl_command_queue& commands, cl_program& program, cl_kernel& kernel ) {
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

  // 0th: initialize the timer at the beginning of the program
  timespec timer = tic();

  // Create device buffers
  //
  cl_mem edge_begin_buffer = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(edge_begin), NULL, NULL);
  cl_mem edge_end_buffer = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(edge_end), NULL, NULL);
  cl_mem dst_buffer = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(dst), NULL, NULL);
  cl_mem starting_node = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(node_index_t), NULL, NULL);
  cl_mem level_buffer = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(args->level), NULL, NULL);
  cl_mem level_counts_buffer = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(args->level_counts), NULL, NULL);
  if (!edge_begin_buffer || !edge_end_buffer || !dst_buffer || !starting_node || !level_buffer || !level_counts_buffer)
  {
    printf("Error: Failed to allocate device memory!\n");
    printf("Test failed\n");
    exit(1);
  }    

  // 1st: time of buffer allocation
  toc(&timer, "buffer allocation");

  // Write our data set into device buffers  
  //
  int err;
  err = clEnqueueWriteBuffer(commands, edge_begin_buffer, CL_TRUE, 0, sizeof(edge_begin), edge_begin, 0, NULL, NULL);
  err |= clEnqueueWriteBuffer(commands, edge_end_buffer, CL_TRUE, 0, sizeof(edge_end), edge_end, 0, NULL, NULL);
  err |= clEnqueueWriteBuffer(commands, dst_buffer, CL_TRUE, 0, sizeof(dst), dst, 0, NULL, NULL);
  err |= clEnqueueWriteBuffer(commands, starting_node, CL_TRUE, 0, sizeof(node_index_t), &(args->starting_node), 0, NULL, NULL);
  if (err != CL_SUCCESS)
  {
      printf("Error: Failed to write to device memory!\n");
      printf("Test failed\n");
      exit(1);
  }

  // 2nd: time of pageable-pinned memory copy
  toc(&timer, "memory copy");
    
  // Set the arguments to our compute kernel
  //
  err   = clSetKernelArg(kernel, 0, sizeof(cl_mem), &edge_begin_buffer);
  err  |= clSetKernelArg(kernel, 1, sizeof(cl_mem), &edge_end_buffer);
  err  |= clSetKernelArg(kernel, 2, sizeof(cl_mem), &dst_buffer);
  err  |= clSetKernelArg(kernel, 3, sizeof(cl_mem), &starting_node);
  err  |= clSetKernelArg(kernel, 4, sizeof(cl_mem), &level_buffer);
  err  |= clSetKernelArg(kernel, 5, sizeof(cl_mem), &level_counts_buffer);
  if (err != CL_SUCCESS)
  {
    printf("Error: Failed to set kernel arguments! %d\n", err);
    printf("Test failed\n");
    exit(1);
  }

  // 3rd: time of setting arguments
  toc(&timer, "set arguments");

  // Execute the kernel over the entire range of our 1d input data set
  // using the maximum number of work group items for this device
  //

#ifdef C_KERNEL
  err = clEnqueueTask(commands, kernel, 0, NULL, NULL);
#else
  printf("Error: OpenCL kernel is not currently supported!\n");
  exit(1);
#endif
  if (err)
  {
    printf("Error: Failed to execute kernel! %d\n", err);
    printf("Test failed\n");
    exit(1);
  }

  // 4th: time of kernel execution
  clFinish(commands);
  toc(&timer, "kernel execution");

  // Read back the results from the device to verify the output
  //
  err = clEnqueueReadBuffer( commands, level_buffer, CL_TRUE, 0, sizeof(args->level), args->level, 0, NULL, NULL );  
  err |= clEnqueueReadBuffer( commands, level_counts_buffer, CL_TRUE, 0, sizeof(args->level_counts), args->level_counts, 0, NULL, NULL );  
  if (err != CL_SUCCESS)
  {
    printf("Error: Failed to read output array! %d\n", err);
    printf("Test failed\n");
    exit(1);
  }

  // 5th: time of data retrieving (PCIe + memcpy)
  toc(&timer, "data retrieving");
}

/* Input format:
%% Section 1
uint32_t[1]: starting node
%% Section 2
uint32_t[N_NODES*2]: node structures (start and end indices of edge lists)
%% Section 3
uint32_t[N_EDGES]: edges structures (just destination node id)
*/

void input_to_data(int fd, void *vdata) {
  struct bench_args_t *data = (struct bench_args_t *)vdata;
  char *p, *s;
  uint32_t *nodes;
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
  parse_uint32_t_array(s, &data->starting_node, 1);

  // Section 2: node structures
  s = find_section_start(p,2);
  nodes = (uint32_t *)malloc(N_NODES*2*sizeof(uint32_t));
  parse_uint32_t_array(s, nodes, N_NODES*2);
  for(i=0; i<N_NODES; i++) {
    data->nodes[i].edge_begin = nodes[2*i];
    data->nodes[i].edge_end = nodes[2*i+1];
  }
  free(nodes);
  // Section 3: edge structures
  s = find_section_start(p,3);
  parse_uint32_t_array(s, (uint32_t *)(data->edges), N_EDGES);
}

void data_to_input(int fd, void *vdata) {
  uint32_t *nodes;
  int64_t i;

  struct bench_args_t *data = (struct bench_args_t *)vdata;
  // Section 1: starting node
  write_section_header(fd);
  write_uint32_t_array(fd, &data->starting_node, 1);
  // Section 2: node structures
  write_section_header(fd);
  nodes = (uint32_t *)malloc(N_NODES*2*sizeof(uint32_t));
  for(i=0; i<N_NODES; i++) {
    nodes[2*i]  = data->nodes[i].edge_begin;
    nodes[2*i+1]= data->nodes[i].edge_end;
  }
  write_uint32_t_array(fd, nodes, N_NODES*2);
  free(nodes);
  // Section 3: edge structures
  write_section_header(fd);
  write_uint32_t_array(fd, (uint32_t *)(&data->edges), N_EDGES);
}

/* Output format:
%% Section 1
uint32_t[N_LEVELS]: horizon counts
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
  parse_uint32_t_array(s, data->level_counts, N_LEVELS);
}

void data_to_output(int fd, void *vdata) {
  struct bench_args_t *data = (struct bench_args_t *)vdata;
  // Section 1
  write_section_header(fd);
  write_uint32_t_array(fd, data->level_counts, N_LEVELS);
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
