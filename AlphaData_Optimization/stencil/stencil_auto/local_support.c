#include "stencil.h"
#include "support.h"
#include <string.h>
#include "my_timer.h"

int INPUT_SIZE = sizeof(struct bench_args_t);

#define EPSILON (1.0e-6)

void run_benchmark( void *vargs, cl_context& context, cl_command_queue& commands, cl_program& program, cl_kernel& kernel ) {
  struct bench_args_t *args = (struct bench_args_t *)vargs;
  int num_imgs = (1 << 12);
  TYPE* orig = (TYPE*)malloc(sizeof(args->orig)*num_imgs);
  TYPE* sol = (TYPE*)malloc(sizeof(args->sol)*num_imgs);

  for (int i=0; i<num_imgs; i++) {
      memcpy(orig+i*(tile_size+2)*(tile_size+2), args->orig, sizeof(args->orig));
  }

  // 0th: initialize the timer at the beginning of the program
  timespec timer = tic();

  // Create device buffers
  //
  cl_mem orig_buffer = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(args->orig)*num_imgs, NULL, NULL);
  cl_mem sol_buffer = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(args->sol)*num_imgs, NULL, NULL);
  cl_mem filter_buffer = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(args->filter), NULL, NULL);
  if (!orig_buffer || !sol_buffer || !filter_buffer)
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
  err = clEnqueueWriteBuffer(commands, orig_buffer, CL_TRUE, 0, sizeof(args->orig)*num_imgs, orig, 0, NULL, NULL);
  err |= clEnqueueWriteBuffer(commands, filter_buffer, CL_TRUE, 0, sizeof(args->filter), args->filter, 0, NULL, NULL);
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
  err  = clSetKernelArg(kernel, 0, sizeof(cl_mem), &orig_buffer);
  err  |= clSetKernelArg(kernel, 1, sizeof(cl_mem), &sol_buffer);
  err  |= clSetKernelArg(kernel, 2, sizeof(cl_mem), &filter_buffer);
  err  |= clSetKernelArg(kernel, 3, sizeof(int), &num_imgs);
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
  err = clEnqueueReadBuffer( commands, sol_buffer, CL_TRUE, 0, sizeof(args->sol)*num_imgs, sol, 0, NULL, NULL );  
  if (err != CL_SUCCESS)
  {
    printf("Error: Failed to read output array! %d\n", err);
    printf("Test failed\n");
    exit(1);
  }

  // 5th: time of data retrieving (PCIe + memcpy)
  toc(&timer, "data retrieving");

  memcpy(args->sol, sol, sizeof(args->sol));
  free(orig); free(sol);

}

/* Input format:
%% Section 1
TYPE[row_size*col_size]: input matrix
%% Section 2
TYPE[f_size]: filter coefficients
*/

void input_to_data(int fd, void *vdata) {
  struct bench_args_t *data = (struct bench_args_t *)vdata;
  char *p, *s;
  // Zero-out everything.
  memset(vdata,0,sizeof(struct bench_args_t));
  // Load input string
  p = readfile(fd);

  s = find_section_start(p,1);
  STAC(parse_,TYPE,_array)(s, data->orig, (tile_size+2)*(tile_size+2));

  s = find_section_start(p,2);
  STAC(parse_,TYPE,_array)(s, data->filter, f_size);
}

void data_to_input(int fd, void *vdata) {
  struct bench_args_t *data = (struct bench_args_t *)vdata;

  write_section_header(fd);
  STAC(write_,TYPE,_array)(fd, data->orig, (tile_size+2)*(tile_size+2));

  write_section_header(fd);
  STAC(write_,TYPE,_array)(fd, data->filter, f_size);
}

/* Output format:
%% Section 1
TYPE[row_size*col_size]: solution matrix
*/

void output_to_data(int fd, void *vdata) {
  struct bench_args_t *data = (struct bench_args_t *)vdata;
  char *p, *s;
  // Zero-out everything.
  memset(vdata,0,sizeof(struct bench_args_t));
  // Load input string
  p = readfile(fd);

  s = find_section_start(p,1);
  STAC(parse_,TYPE,_array)(s, data->sol, tile_size*tile_size);
}

void data_to_output(int fd, void *vdata) {
  struct bench_args_t *data = (struct bench_args_t *)vdata;

  write_section_header(fd);
  STAC(write_,TYPE,_array)(fd, data->sol, tile_size*tile_size);
}

int check_data( void *vdata, void *vref ) {
  struct bench_args_t *data = (struct bench_args_t *)vdata;
  struct bench_args_t *ref = (struct bench_args_t *)vref;
  int has_errors = 0;
  int row, col;
  TYPE diff;

  for(row=0; row<tile_size; row++) {
    for(col=0; col<tile_size; col++) {
      diff = data->sol[row*tile_size + col] - ref->sol[row*tile_size + col];
      has_errors |= (diff<-EPSILON) || (EPSILON<diff);
    }
  }

  // Return true if it's correct.
  return !has_errors;
}
