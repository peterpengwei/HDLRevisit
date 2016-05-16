#include "md.h"
#include "support.h"
#include <string.h>

int INPUT_SIZE = sizeof(struct bench_args_t);

#define EPSILON ((TYPE)1.0e-6)

void run_benchmark( void *vargs, cl_context& context, cl_command_queue& commands, cl_program& program, cl_kernel& kernel ) {
  struct bench_args_t *args = (struct bench_args_t *)vargs;
  // Create device buffers
  //
  cl_mem force_x_buffer = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(args->force_x), NULL, NULL);
  cl_mem force_y_buffer = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(args->force_y), NULL, NULL);
  cl_mem force_z_buffer = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(args->force_z), NULL, NULL);
  cl_mem position_x_buffer = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(args->position_x), NULL, NULL);
  cl_mem position_y_buffer = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(args->position_y), NULL, NULL);
  cl_mem position_z_buffer = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(args->position_z), NULL, NULL);
  cl_mem NL_buffer = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(args->NL), NULL, NULL);
  if (!force_x_buffer || !force_y_buffer || !force_z_buffer || !position_x_buffer || !position_y_buffer || !position_z_buffer || !NL_buffer)
  {
    printf("Error: Failed to allocate device memory!\n");
    printf("Test failed\n");
    exit(1);
  }    

  // Write our data set into device buffers  
  //
  int err;
  err = clEnqueueWriteBuffer(commands, force_x_buffer, CL_TRUE, 0, sizeof(args->force_x), args->force_x, 0, NULL, NULL);
  err |= clEnqueueWriteBuffer(commands, force_y_buffer, CL_TRUE, 0, sizeof(args->force_y), args->force_y, 0, NULL, NULL);
  err |= clEnqueueWriteBuffer(commands, force_z_buffer, CL_TRUE, 0, sizeof(args->force_z), args->force_z, 0, NULL, NULL);
  err |= clEnqueueWriteBuffer(commands, position_x_buffer, CL_TRUE, 0, sizeof(args->position_x), args->position_x, 0, NULL, NULL);
  err |= clEnqueueWriteBuffer(commands, position_y_buffer, CL_TRUE, 0, sizeof(args->position_y), args->position_y, 0, NULL, NULL);
  err |= clEnqueueWriteBuffer(commands, position_z_buffer, CL_TRUE, 0, sizeof(args->position_z), args->position_z, 0, NULL, NULL);
  err |= clEnqueueWriteBuffer(commands, NL_buffer, CL_TRUE, 0, sizeof(args->NL), args->NL, 0, NULL, NULL);
  if (err != CL_SUCCESS)
  {
      printf("Error: Failed to write to device memory!\n");
      printf("Test failed\n");
      exit(1);
  }
    
  // Set the arguments to our compute kernel
  //
  err  = clSetKernelArg(kernel, 0, sizeof(cl_mem), &force_x_buffer);
  err  |= clSetKernelArg(kernel, 1, sizeof(cl_mem), &force_y_buffer);
  err  |= clSetKernelArg(kernel, 2, sizeof(cl_mem), &force_z_buffer);
  err  |= clSetKernelArg(kernel, 3, sizeof(cl_mem), &position_x_buffer);
  err  |= clSetKernelArg(kernel, 4, sizeof(cl_mem), &position_y_buffer);
  err  |= clSetKernelArg(kernel, 5, sizeof(cl_mem), &position_z_buffer);
  err  |= clSetKernelArg(kernel, 6, sizeof(cl_mem), &NL_buffer);
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
  err = clEnqueueReadBuffer( commands, force_x_buffer, CL_TRUE, 0, sizeof(args->force_x), args->force_x, 0, NULL, NULL );  
  err |= clEnqueueReadBuffer( commands, force_y_buffer, CL_TRUE, 0, sizeof(args->force_y), args->force_y, 0, NULL, NULL );  
  err |= clEnqueueReadBuffer( commands, force_z_buffer, CL_TRUE, 0, sizeof(args->force_z), args->force_z, 0, NULL, NULL );  
  if (err != CL_SUCCESS)
  {
    printf("Error: Failed to read output array! %d\n", err);
    printf("Test failed\n");
    exit(1);
  }
}

/* Input format:
%% Section 1
TYPE[nAtoms]: x positions
%% Section 2
TYPE[nAtoms]: y positions
%% Section 3
TYPE[nAtoms]: z positions
%% Section 4
int32_t[nAtoms*maxNeighbors]: neighbor list
*/

void input_to_data(int fd, void *vdata) {
  struct bench_args_t *data = (struct bench_args_t *)vdata;
  char *p, *s;
  // Zero-out everything.
  memset(vdata,0,sizeof(struct bench_args_t));
  // Load input string
  p = readfile(fd);

  s = find_section_start(p,1);
  STAC(parse_,TYPE,_array)(s, data->position_x, nAtoms);

  s = find_section_start(p,2);
  STAC(parse_,TYPE,_array)(s, data->position_y, nAtoms);

  s = find_section_start(p,3);
  STAC(parse_,TYPE,_array)(s, data->position_z, nAtoms);

  s = find_section_start(p,4);
  parse_int32_t_array(s, data->NL, nAtoms*maxNeighbors);

}

void data_to_input(int fd, void *vdata) {
  struct bench_args_t *data = (struct bench_args_t *)vdata;

  write_section_header(fd);
  STAC(write_,TYPE,_array)(fd, data->position_x, nAtoms);

  write_section_header(fd);
  STAC(write_,TYPE,_array)(fd, data->position_y, nAtoms);

  write_section_header(fd);
  STAC(write_,TYPE,_array)(fd, data->position_z, nAtoms);

  write_section_header(fd);
  write_int32_t_array(fd, data->NL, nAtoms*maxNeighbors);

}

/* Output format:
%% Section 1
TYPE[nAtoms]: new x force
%% Section 2
TYPE[nAtoms]: new y force
%% Section 3
TYPE[nAtoms]: new z force
*/

void output_to_data(int fd, void *vdata) {
  struct bench_args_t *data = (struct bench_args_t *)vdata;
  char *p, *s;
  // Zero-out everything.
  memset(vdata,0,sizeof(struct bench_args_t));
  // Load input string
  p = readfile(fd);

  s = find_section_start(p,1);
  STAC(parse_,TYPE,_array)(s, data->force_x, nAtoms);

  s = find_section_start(p,2);
  STAC(parse_,TYPE,_array)(s, data->force_y, nAtoms);

  s = find_section_start(p,3);
  STAC(parse_,TYPE,_array)(s, data->force_z, nAtoms);

}

void data_to_output(int fd, void *vdata) {
  struct bench_args_t *data = (struct bench_args_t *)vdata;

  write_section_header(fd);
  STAC(write_,TYPE,_array)(fd, data->force_x, nAtoms);

  write_section_header(fd);
  STAC(write_,TYPE,_array)(fd, data->force_y, nAtoms);

  write_section_header(fd);
  STAC(write_,TYPE,_array)(fd, data->force_z, nAtoms);
}

int check_data( void *vdata, void *vref ) {
  struct bench_args_t *data = (struct bench_args_t *)vdata;
  struct bench_args_t *ref = (struct bench_args_t *)vref;
  int has_errors = 0;
  int i;
  TYPE diff_x, diff_y, diff_z;

  for( i=0; i<nAtoms; i++ ) {
    diff_x = data->force_x[i] - ref->force_x[i];
    diff_y = data->force_y[i] - ref->force_y[i];
    diff_z = data->force_z[i] - ref->force_z[i];
    has_errors |= (diff_x<-EPSILON) || (EPSILON<diff_x);
    has_errors |= (diff_y<-EPSILON) || (EPSILON<diff_y);
    has_errors |= (diff_z<-EPSILON) || (EPSILON<diff_z);
  }

  // Return true if it's correct.
  return !has_errors;
}
