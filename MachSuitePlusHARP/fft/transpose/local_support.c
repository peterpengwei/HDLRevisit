#include "fft.h"
#include "support.h"
#include <string.h>

int INPUT_SIZE = sizeof(struct bench_args_t);

#define EPSILON ((TYPE)1.0e-6)

void run_benchmark( void *vargs, cl_context& context, cl_command_queue& commands, cl_program& program, cl_kernel& kernel ) {
  struct bench_args_t *args = (struct bench_args_t *)vargs;
  // Create device buffers
  //
  cl_mem work_x_buffer = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(args->work_x), NULL, NULL);
  cl_mem work_y_buffer = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(args->work_y), NULL, NULL);
  if (!work_x_buffer || !work_y_buffer)
  {
    printf("Error: Failed to allocate device memory!\n");
    printf("Test failed\n");
    exit(1);
  }    

  // Write our data set into device buffers  
  //
  int err;
  err = clEnqueueWriteBuffer(commands, work_x_buffer, CL_TRUE, 0, sizeof(args->work_x), args->work_x, 0, NULL, NULL);
  err |= clEnqueueWriteBuffer(commands, work_y_buffer, CL_TRUE, 0, sizeof(args->work_y), args->work_y, 0, NULL, NULL);
  if (err != CL_SUCCESS)
  {
      printf("Error: Failed to write to device memory!\n");
      printf("Test failed\n");
      exit(1);
  }
    
  // Set the arguments to our compute kernel
  //
  err  = clSetKernelArg(kernel, 0, sizeof(cl_mem), &work_x_buffer);
  err  |= clSetKernelArg(kernel, 1, sizeof(cl_mem), &work_y_buffer);
  if (err != CL_SUCCESS)
  {
    printf("Error: Failed to set kernel arguments! %d\n", err);
    printf("Test failed\n");
    exit(1);
  }

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

  // Read back the results from the device to verify the output
  //
  err = clEnqueueReadBuffer( commands, work_x_buffer, CL_TRUE, 0, sizeof(args->work_x), args->work_x, 0, NULL, NULL );  
  err |= clEnqueueReadBuffer( commands, work_y_buffer, CL_TRUE, 0, sizeof(args->work_y), args->work_y, 0, NULL, NULL );  
  if (err != CL_SUCCESS)
  {
    printf("Error: Failed to read output array! %d\n", err);
    printf("Test failed\n");
    exit(1);
  }
}

/* Input format:
%% Section 1
TYPE[512]: signal (real part)
%% Section 2
TYPE[512]: signal (complex part)
*/

void input_to_data(int fd, void *vdata) {
  struct bench_args_t *data = (struct bench_args_t *)vdata;
  char *p, *s;
  // Load input string
  p = readfile(fd);

  s = find_section_start(p,1);
  STAC(parse_,TYPE,_array)(s, data->work_x, 512);

  s = find_section_start(p,2);
  STAC(parse_,TYPE,_array)(s, data->work_y, 512);
}

void data_to_input(int fd, void *vdata) {
  struct bench_args_t *data = (struct bench_args_t *)vdata;

  write_section_header(fd);
  STAC(write_,TYPE,_array)(fd, data->work_x, 512);

  write_section_header(fd);
  STAC(write_,TYPE,_array)(fd, data->work_y, 512);
}

/* Output format:
%% Section 1
TYPE[512]: freq (real part)
%% Section 2
TYPE[512]: freq (complex part)
*/

void output_to_data(int fd, void *vdata) {
  struct bench_args_t *data = (struct bench_args_t *)vdata;
  char *p, *s;
  // Load input string
  p = readfile(fd);
 
  s = find_section_start(p,1);
  STAC(parse_,TYPE,_array)(s, data->work_x, 512);

  s = find_section_start(p,2);
  STAC(parse_,TYPE,_array)(s, data->work_y, 512);
}

void data_to_output(int fd, void *vdata) {
  struct bench_args_t *data = (struct bench_args_t *)vdata;

  write_section_header(fd);
  STAC(write_,TYPE,_array)(fd, data->work_x, 512);

  write_section_header(fd);
  STAC(write_,TYPE,_array)(fd, data->work_y, 512);
}

int check_data( void *vdata, void *vref ) {
  struct bench_args_t *data = (struct bench_args_t *)vdata;
  struct bench_args_t *ref = (struct bench_args_t *)vref;
  int has_errors = 0;
  int i;
  double real_diff, img_diff;

  for(i=0; i<512; i++) {
    real_diff = data->work_x[i] - ref->work_x[i];
    img_diff = data->work_y[i] - ref->work_y[i];
    has_errors |= (real_diff<-EPSILON) || (EPSILON<real_diff);
    //if( has_errors )
      //printf("%d (real): %f (%f)\n", i, real_diff, EPSILON);
    has_errors |= (img_diff<-EPSILON) || (EPSILON<img_diff);
    //if( has_errors )
      //printf("%d (img): %f (%f)\n", i, img_diff, EPSILON);
  }

  // Return true if it's correct.
  return !has_errors;
}
