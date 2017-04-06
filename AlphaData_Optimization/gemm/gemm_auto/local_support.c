#include "gemm.h"
#include "support.h"
#include <string.h>
#include "my_timer.h"

int INPUT_SIZE = sizeof(struct bench_args_t);

#define EPSILON ((TYPE)1.0e-6)

void run_benchmark( void *vargs, cl_context& context, cl_command_queue& commands, cl_program& program, cl_kernel& kernel ) {
  struct bench_args_t *args = (struct bench_args_t *)vargs;

  // 0th: initialize the timer at the beginning of the program
  timespec timer = tic();

  // Create device buffers
  //
  cl_mem m1_buffer = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(args->m1), NULL, NULL);
  cl_mem m2_buffer = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(args->m2), NULL, NULL);
  cl_mem prod_buffer = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(args->prod), NULL, NULL);
  if (!m1_buffer || !m2_buffer)
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
  err = clEnqueueWriteBuffer(commands, m1_buffer, CL_TRUE, 0, sizeof(args->m1), args->m1, 0, NULL, NULL);
  err |= clEnqueueWriteBuffer(commands, m2_buffer, CL_TRUE, 0, sizeof(args->m2), args->m2, 0, NULL, NULL);
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
  err  = clSetKernelArg(kernel, 0, sizeof(cl_mem), &m1_buffer);
  err  |= clSetKernelArg(kernel, 1, sizeof(cl_mem), &m2_buffer);
  err  |= clSetKernelArg(kernel, 2, sizeof(cl_mem), &prod_buffer);
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
  err = clEnqueueReadBuffer( commands, prod_buffer, CL_TRUE, 0, sizeof(args->prod), args->prod, 0, NULL, NULL );  
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
TYPE[N]: matrix 1
%% Section 2
TYPE[N]: matrix 2
*/

void input_to_data(int fd, void *vdata) {
  struct bench_args_t *data = (struct bench_args_t *)vdata;
  char *p, *s;
  // Zero-out everything.
  memset(vdata,0,sizeof(struct bench_args_t));
  // Load input string
  p = readfile(fd);

  s = find_section_start(p,1);
  STAC(parse_,TYPE,_array)(s, data->m1, N);

  s = find_section_start(p,2);
  STAC(parse_,TYPE,_array)(s, data->m2, N);

}

void data_to_input(int fd, void *vdata) {
  struct bench_args_t *data = (struct bench_args_t *)vdata;

  write_section_header(fd);
  STAC(write_,TYPE,_array)(fd, data->m1, N);

  write_section_header(fd);
  STAC(write_,TYPE,_array)(fd, data->m2, N);
}

/* Output format:
%% Section 1
TYPE[N]: output matrix
*/

void output_to_data(int fd, void *vdata) {
  struct bench_args_t *data = (struct bench_args_t *)vdata;
  char *p, *s;
  // Load input string
  p = readfile(fd);

  s = find_section_start(p,1);
  STAC(parse_,TYPE,_array)(s, data->prod, N);
}

void data_to_output(int fd, void *vdata) {
  struct bench_args_t *data = (struct bench_args_t *)vdata;

  write_section_header(fd);
  STAC(write_,TYPE,_array)(fd, data->prod, N);
}

int check_data( void *vdata, void *vref ) {
  struct bench_args_t *data = (struct bench_args_t *)vdata;
  struct bench_args_t *ref = (struct bench_args_t *)vref;
  int has_errors = 0;
  int r,c;
  TYPE diff;

  for( r=0; r<row_size; r++ ) {
    for( c=0; c<col_size; c++ ) {
      diff = data->prod[r*row_size + c] - ref->prod[r*row_size+c];
      has_errors |= (diff<-EPSILON) || (EPSILON<diff);
    }
  }

  // Return true if it's correct.
  return !has_errors;
}
