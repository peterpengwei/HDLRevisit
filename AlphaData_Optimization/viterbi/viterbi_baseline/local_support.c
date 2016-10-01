#include "viterbi.h"
#include "support.h"
#include <string.h>
#include "my_timer.h"

int INPUT_SIZE = sizeof(struct bench_args_t);

void run_benchmark( void *vargs, cl_context& context, cl_command_queue& commands, cl_program& program, cl_kernel& kernel ) {
  struct bench_args_t *args = (struct bench_args_t *)vargs;

  int num_jobs = 1 << 20;

  tok_t* obs = (tok_t *)malloc(sizeof(args->obs) * num_jobs);
  prob_t* init = (prob_t *)malloc(sizeof(args->init) * num_jobs);
  prob_t* transition = (prob_t *)malloc(sizeof(args->transition) * num_jobs);
  prob_t* emission = (prob_t *)malloc(sizeof(args->emission) * num_jobs);
  state_t* path = (state_t *)malloc(sizeof(args->path) * num_jobs);
  int i;
  for (i=0; i<num_jobs; i++) {
    memcpy(obs+i*N_OBS, args->obs, sizeof(args->obs));
    memcpy(init+i*N_STATES, args->init, sizeof(args->init));
    memcpy(transition+i*TRANS_SIZE, args->transition, sizeof(args->transition));
    memcpy(emission+i*EMIT_SIZE, args->emission, sizeof(args->emission));
  }

  // 0th: initialize the timer at the beginning of the program
  timespec timer = tic();

  // Create device buffers
  //
  cl_mem obs_buffer = clCreateBuffer(context, CL_MEM_READ_WRITE, num_jobs * sizeof(args->obs), NULL, NULL);
  cl_mem init_buffer = clCreateBuffer(context, CL_MEM_READ_WRITE, num_jobs * sizeof(args->init), NULL, NULL);
  cl_mem transition_buffer = clCreateBuffer(context, CL_MEM_READ_WRITE, num_jobs * sizeof(args->transition), NULL, NULL);
  cl_mem emission_buffer = clCreateBuffer(context, CL_MEM_READ_WRITE, num_jobs * sizeof(args->emission), NULL, NULL);
  cl_mem path_buffer = clCreateBuffer(context, CL_MEM_READ_WRITE, num_jobs * sizeof(args->path), NULL, NULL);
  if (!obs_buffer || !init_buffer || !transition_buffer || !emission_buffer || !path_buffer)
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
  err = clEnqueueWriteBuffer(commands, obs_buffer, CL_TRUE, 0, num_jobs * sizeof(args->obs), obs, 0, NULL, NULL);
  err |= clEnqueueWriteBuffer(commands, init_buffer, CL_TRUE, 0, num_jobs * sizeof(args->init), init, 0, NULL, NULL);
  err |= clEnqueueWriteBuffer(commands, transition_buffer, CL_TRUE, 0, num_jobs * sizeof(args->transition), transition, 0, NULL, NULL);
  err |= clEnqueueWriteBuffer(commands, emission_buffer, CL_TRUE, 0, num_jobs * sizeof(args->emission), emission, 0, NULL, NULL);
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
  err  = clSetKernelArg(kernel, 0, sizeof(cl_mem), &obs_buffer);
  err  |= clSetKernelArg(kernel, 1, sizeof(cl_mem), &init_buffer);
  err  |= clSetKernelArg(kernel, 2, sizeof(cl_mem), &transition_buffer);
  err  |= clSetKernelArg(kernel, 3, sizeof(cl_mem), &emission_buffer);
  err  |= clSetKernelArg(kernel, 4, sizeof(cl_mem), &path_buffer);
  err  |= clSetKernelArg(kernel, 5, sizeof(int), &num_jobs);
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
  err = clEnqueueReadBuffer( commands, path_buffer, CL_TRUE, 0, num_jobs * sizeof(args->path), path, 0, NULL, NULL );  
  if (err != CL_SUCCESS)
  {
    printf("Error: Failed to read output array! %d\n", err);
    printf("Test failed\n");
    exit(1);
  }

  // 5th: time of data retrieving (PCIe + memcpy)
  toc(&timer, "data retrieving");
  memcpy(args->path, path, sizeof(args->path));
}

/* Input format:
%% Section 1
tok_t[N_OBS]: observation vector
%% Section 2
prob_t[N_STATES]: initial state probabilities
%% Section 3
prob_t[N_STATES*N_STATES]: transition matrix
%% Section 4
prob_t[N_STATES*N_TOKENS]: emission matrix
*/

void input_to_data(int fd, void *vdata) {
  struct bench_args_t *data = (struct bench_args_t *)vdata;
  char *p, *s;
  // Zero-out everything.
  memset(vdata,0,sizeof(struct bench_args_t));
  // Load input string
  p = readfile(fd);

  s = find_section_start(p,1);
  parse_uint8_t_array(s, data->obs, N_OBS);

  s = find_section_start(p,2);
  STAC(parse_,TYPE,_array)(s, data->init, N_STATES);

  s = find_section_start(p,3);
  STAC(parse_,TYPE,_array)(s, data->transition, N_STATES*N_STATES);

  s = find_section_start(p,4);
  STAC(parse_,TYPE,_array)(s, data->emission, N_STATES*N_TOKENS);
}

void data_to_input(int fd, void *vdata) {
  struct bench_args_t *data = (struct bench_args_t *)vdata;

  write_section_header(fd);
  write_uint8_t_array(fd, data->obs, N_OBS);

  write_section_header(fd);
  STAC(write_,TYPE,_array)(fd, data->init, N_STATES);

  write_section_header(fd);
  STAC(write_,TYPE,_array)(fd, data->transition, N_STATES*N_STATES);

  write_section_header(fd);
  STAC(write_,TYPE,_array)(fd, data->emission, N_STATES*N_TOKENS);
}

/* Output format:
%% Section 1
uint8_t[N_OBS]: most likely state chain
*/

void output_to_data(int fd, void *vdata) {
  struct bench_args_t *data = (struct bench_args_t *)vdata;
  char *p, *s;
  // Zero-out everything.
  memset(vdata,0,sizeof(struct bench_args_t));
  // Load input string
  p = readfile(fd);

  s = find_section_start(p,1);
  parse_uint8_t_array(s, data->path, N_OBS);
}

void data_to_output(int fd, void *vdata) {
  struct bench_args_t *data = (struct bench_args_t *)vdata;

  write_section_header(fd);
  write_uint8_t_array(fd, data->path, N_OBS);
}

int check_data( void *vdata, void *vref ) {
  struct bench_args_t *data = (struct bench_args_t *)vdata;
  struct bench_args_t *ref = (struct bench_args_t *)vref;
  int has_errors = 0;
  int i;

  for(i=0; i<N_OBS; i++) {
    has_errors |= (data->path[i]!=ref->path[i]);
  }

  // Return true if it's correct.
  return !has_errors;
}
