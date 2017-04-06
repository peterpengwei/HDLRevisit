#include "fft.h"
#include "support.h"
#include <string.h>
#include "my_timer.h"

int INPUT_SIZE = sizeof(struct bench_args_t);

#define EPSILON ((double)1.0e-6)

void run_benchmark( void *vargs, cl_context& context, cl_command_queue& commands, cl_program& program, cl_kernel& kernel ) {
  struct bench_args_t *args = (struct bench_args_t *)vargs;
  // Create device buffers
  //
  int num_strides = (1 << 18);
  double* real = (double *)malloc(sizeof(args->real)*num_strides);
  double* img = (double *)malloc(sizeof(args->img)*num_strides);
  double* real_twid = (double *)malloc(sizeof(args->real_twid)*num_strides);
  double* img_twid = (double *)malloc(sizeof(args->img_twid)*num_strides);
  for (int i=0; i<num_strides; i++) {
      memcpy(real+i*FFT_SIZE, args->real, sizeof(args->real));
      memcpy(img+i*FFT_SIZE, args->img, sizeof(args->img));
      memcpy(real_twid+i*FFT_SIZE/2, args->real_twid, sizeof(args->real_twid));
      memcpy(img_twid+i*FFT_SIZE/2, args->img_twid, sizeof(args->img_twid));
  }

  // 0th: initialize the timer at the beginning of the program
  timespec timer = tic();

  cl_mem real_buffer = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(args->real)*num_strides, NULL, NULL);
  cl_mem img_buffer = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(args->img)*num_strides, NULL, NULL);
  cl_mem real_twid_buffer = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(args->real_twid)*num_strides, NULL, NULL);
  cl_mem img_twid_buffer = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(args->img_twid)*num_strides, NULL, NULL);
  if (!real_buffer || !img_buffer || !real_twid_buffer || !img_twid_buffer)
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
  err = clEnqueueWriteBuffer(commands, real_buffer, CL_TRUE, 0, num_strides*sizeof(args->real), real, 0, NULL, NULL);
  err |= clEnqueueWriteBuffer(commands, img_buffer, CL_TRUE, 0, num_strides*sizeof(args->img), img, 0, NULL, NULL);
  err |= clEnqueueWriteBuffer(commands, real_twid_buffer, CL_TRUE, 0, num_strides*sizeof(args->real_twid), real_twid, 0, NULL, NULL);
  err |= clEnqueueWriteBuffer(commands, img_twid_buffer, CL_TRUE, 0, num_strides*sizeof(args->img_twid), img_twid, 0, NULL, NULL);
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
  err  = clSetKernelArg(kernel, 0, sizeof(cl_mem), &real_buffer);
  err  |= clSetKernelArg(kernel, 1, sizeof(cl_mem), &img_buffer);
  err  |= clSetKernelArg(kernel, 2, sizeof(cl_mem), &real_twid_buffer);
  err  |= clSetKernelArg(kernel, 3, sizeof(cl_mem), &img_twid_buffer);
  err  |= clSetKernelArg(kernel, 4, sizeof(int), &num_strides);
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
  err = clEnqueueReadBuffer( commands, real_buffer, CL_TRUE, 0, num_strides*sizeof(args->real), real, 0, NULL, NULL );  
  err |= clEnqueueReadBuffer( commands, img_buffer, CL_TRUE, 0, num_strides*sizeof(args->img), img, 0, NULL, NULL );  
  if (err != CL_SUCCESS)
  {
    printf("Error: Failed to read output array! %d\n", err);
    printf("Test failed\n");
    exit(1);
  }

  // 5th: time of data retrieving (PCIe + memcpy)
  toc(&timer, "data retrieving");

  memcpy(args->real, real, sizeof(args->real));
  memcpy(args->img, img, sizeof(args->img));

  free(real); free(img); free(real_twid); free(img_twid);
}

/* Input format:
%% Section 1
double: signal (real part)
%% Section 2
double: signal (complex part)
%% Section 3
double: twiddle factor (real part)
%% Section 4
double: twiddle factor (complex part)
*/

void input_to_data(int fd, void *vdata) {
  struct bench_args_t *data = (struct bench_args_t *)vdata;
  char *p, *s;
  // Load input string
  p = readfile(fd);

  s = find_section_start(p,1);
  parse_double_array(s, data->real, FFT_SIZE);

  s = find_section_start(p,2);
  parse_double_array(s, data->img, FFT_SIZE);

  s = find_section_start(p,3);
  parse_double_array(s, data->real_twid, FFT_SIZE/2);

  s = find_section_start(p,4);
  parse_double_array(s, data->img_twid, FFT_SIZE/2);
}

void data_to_input(int fd, void *vdata) {
  struct bench_args_t *data = (struct bench_args_t *)vdata;

  write_section_header(fd);
  write_double_array(fd, data->real, FFT_SIZE);

  write_section_header(fd);
  write_double_array(fd, data->img, FFT_SIZE);

  write_section_header(fd);
  write_double_array(fd, data->real_twid, FFT_SIZE/2);

  write_section_header(fd);
  write_double_array(fd, data->img_twid, FFT_SIZE/2);
}

/* Output format:
%% Section 1
double: freq (real part)
%% Section 2
double: freq (complex part)
*/

void output_to_data(int fd, void *vdata) {
  struct bench_args_t *data = (struct bench_args_t *)vdata;
  char *p, *s;
  // Zero-out everything.
  memset(vdata,0,sizeof(struct bench_args_t));
  // Load input string
  p = readfile(fd);

  s = find_section_start(p,1);
  parse_double_array(s, data->real, FFT_SIZE);

  s = find_section_start(p,2);
  parse_double_array(s, data->img, FFT_SIZE);
}

void data_to_output(int fd, void *vdata) {
  struct bench_args_t *data = (struct bench_args_t *)vdata;

  write_section_header(fd);
  write_double_array(fd, data->real, FFT_SIZE);

  write_section_header(fd);
  write_double_array(fd, data->img, FFT_SIZE);
}

int check_data( void *vdata, void *vref ) {
  struct bench_args_t *data = (struct bench_args_t *)vdata;
  struct bench_args_t *ref = (struct bench_args_t *)vref;
  int has_errors = 0;
  int i;
  double real_diff, img_diff;

  for(i=0; i<FFT_SIZE; i++) {
    real_diff = data->real[i] - ref->real[i];
    img_diff = data->img[i] - ref->img[i];
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
