#include "backprop.h"
#include "support.h"
#include <string.h>

int INPUT_SIZE = sizeof(struct bench_args_t);

#define EPSILON (1.0e-6)

void run_benchmark( void *vargs, cl_context& context, cl_command_queue& commands, cl_program& program, cl_kernel& kernel ) {
  struct bench_args_t *args = (struct bench_args_t *)vargs;
  // Create device buffers
  //
  cl_mem weights1_buffer = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(args->weights1), NULL, NULL);
  cl_mem weights2_buffer = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(args->weights2), NULL, NULL);
  cl_mem weights3_buffer = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(args->weights3), NULL, NULL);
  cl_mem biases1_buffer = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(args->biases1), NULL, NULL);
  cl_mem biases2_buffer = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(args->biases2), NULL, NULL);
  cl_mem biases3_buffer = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(args->biases3), NULL, NULL);
  cl_mem training_data_buffer = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(args->training_data), NULL, NULL);
  cl_mem training_targets_buffer = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(args->training_targets), NULL, NULL);
  if (!weights1_buffer || !weights2_buffer || !weights3_buffer || !biases1_buffer ||
		  !biases2_buffer || !biases3_buffer || !training_data_buffer ||
		  !training_targets_buffer)
  {
    printf("Error: Failed to allocate device memory!\n");
    printf("Test failed\n");
    exit(1);
  }    

  // Write our data set into device buffers  
  //
  int err;
  err = clEnqueueWriteBuffer(commands, weights1_buffer, CL_TRUE, 0, sizeof(args->weights1), args->weights1, 0, NULL, NULL);
  err |= clEnqueueWriteBuffer(commands, weights2_buffer, CL_TRUE, 0, sizeof(args->weights2), args->weights2, 0, NULL, NULL);
  err |= clEnqueueWriteBuffer(commands, weights3_buffer, CL_TRUE, 0, sizeof(args->weights3), args->weights3, 0, NULL, NULL);
  err |= clEnqueueWriteBuffer(commands, biases1_buffer, CL_TRUE, 0, sizeof(args->biases1), args->biases1, 0, NULL, NULL);
  err |= clEnqueueWriteBuffer(commands, biases2_buffer, CL_TRUE, 0, sizeof(args->biases2), args->biases2, 0, NULL, NULL);
  err |= clEnqueueWriteBuffer(commands, biases3_buffer, CL_TRUE, 0, sizeof(args->biases3), args->biases3, 0, NULL, NULL);
  err |= clEnqueueWriteBuffer(commands, training_data_buffer, CL_TRUE, 0, sizeof(args->training_data), args->training_data, 0, NULL, NULL);
  err |= clEnqueueWriteBuffer(commands, training_targets_buffer, CL_TRUE, 0, sizeof(args->training_targets), args->training_targets, 0, NULL, NULL);
  if (err != CL_SUCCESS)
  {
      printf("Error: Failed to write to device memory!\n");
      printf("Test failed\n");
      exit(1);
  }
    
  // Set the arguments to our compute kernel
  //
  err  = clSetKernelArg(kernel, 0, sizeof(cl_mem), &weights1_buffer);
  err  |= clSetKernelArg(kernel, 1, sizeof(cl_mem), &weights2_buffer);
  err  |= clSetKernelArg(kernel, 2, sizeof(cl_mem), &weights3_buffer);
  err  |= clSetKernelArg(kernel, 3, sizeof(cl_mem), &biases1_buffer);
  err  |= clSetKernelArg(kernel, 4, sizeof(cl_mem), &biases2_buffer);
  err  |= clSetKernelArg(kernel, 5, sizeof(cl_mem), &biases3_buffer);
  err  |= clSetKernelArg(kernel, 6, sizeof(cl_mem), &training_data_buffer);
  err  |= clSetKernelArg(kernel, 7, sizeof(cl_mem), &training_targets_buffer);
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
  err = clEnqueueReadBuffer( commands, weights1_buffer, CL_TRUE, 0, sizeof(args->weights1), args->weights1, 0, NULL, NULL );  
  err |= clEnqueueReadBuffer( commands, weights2_buffer, CL_TRUE, 0, sizeof(args->weights2), args->weights2, 0, NULL, NULL );  
  err |= clEnqueueReadBuffer( commands, weights3_buffer, CL_TRUE, 0, sizeof(args->weights3), args->weights3, 0, NULL, NULL );  
  err |= clEnqueueReadBuffer( commands, biases1_buffer, CL_TRUE, 0, sizeof(args->biases1), args->biases1, 0, NULL, NULL );  
  err |= clEnqueueReadBuffer( commands, biases2_buffer, CL_TRUE, 0, sizeof(args->biases2), args->biases2, 0, NULL, NULL );  
  err |= clEnqueueReadBuffer( commands, biases3_buffer, CL_TRUE, 0, sizeof(args->biases3), args->biases3, 0, NULL, NULL );  
  if (err != CL_SUCCESS)
  {
    printf("Error: Failed to read output array! %d\n", err);
    printf("Test failed\n");
    exit(1);
  }
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
  STAC(parse_,TYPE,_array)(s, data->weights1, input_dimension*nodes_per_layer);

  s = find_section_start(p,2);
  STAC(parse_,TYPE,_array)(s, data->weights2, nodes_per_layer*nodes_per_layer);

  s = find_section_start(p,3);
  STAC(parse_,TYPE,_array)(s, data->weights3, nodes_per_layer*possible_outputs);

  s = find_section_start(p,4);
  STAC(parse_,TYPE,_array)(s, data->biases1, nodes_per_layer);

  s = find_section_start(p,5);
  STAC(parse_,TYPE,_array)(s, data->biases2, nodes_per_layer);

  s = find_section_start(p,6);
  STAC(parse_,TYPE,_array)(s, data->biases3, possible_outputs);

  s = find_section_start(p,7);
  STAC(parse_,TYPE,_array)(s, data->training_data, training_sets*input_dimension);

  s = find_section_start(p,8);
  STAC(parse_,TYPE,_array)(s, data->training_targets, training_sets*possible_outputs);
}

void data_to_input(int fd, void *vdata) {
  struct bench_args_t *data = (struct bench_args_t *)vdata;

  write_section_header(fd);
  STAC(write_,TYPE,_array)(fd, data->weights1, input_dimension*nodes_per_layer);

  write_section_header(fd);
  STAC(write_,TYPE,_array)(fd, data->weights2, nodes_per_layer*nodes_per_layer);

  write_section_header(fd);
  STAC(write_,TYPE,_array)(fd, data->weights3, nodes_per_layer*possible_outputs);

  write_section_header(fd);
  STAC(write_,TYPE,_array)(fd, data->biases1, nodes_per_layer);

  write_section_header(fd);
  STAC(write_,TYPE,_array)(fd, data->biases2, nodes_per_layer);

  write_section_header(fd);
  STAC(write_,TYPE,_array)(fd, data->biases3, possible_outputs);

  write_section_header(fd);
  STAC(write_,TYPE,_array)(fd, data->training_data, training_sets*input_dimension);

  write_section_header(fd);
  STAC(write_,TYPE,_array)(fd, data->training_targets, training_sets*possible_outputs);
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
  STAC(parse_,TYPE,_array)(s, data->weights1, input_dimension*nodes_per_layer);

  s = find_section_start(p,2);
  STAC(parse_,TYPE,_array)(s, data->weights2, nodes_per_layer*nodes_per_layer);

  s = find_section_start(p,3);
  STAC(parse_,TYPE,_array)(s, data->weights3, nodes_per_layer*possible_outputs);

  s = find_section_start(p,4);
  STAC(parse_,TYPE,_array)(s, data->biases1, nodes_per_layer);

  s = find_section_start(p,5);
  STAC(parse_,TYPE,_array)(s, data->biases2, nodes_per_layer);

  s = find_section_start(p,6);
  STAC(parse_,TYPE,_array)(s, data->biases3, possible_outputs);

}

void data_to_output(int fd, void *vdata) {
  struct bench_args_t *data = (struct bench_args_t *)vdata;

  write_section_header(fd);
  STAC(write_,TYPE,_array)(fd, data->weights1, input_dimension*nodes_per_layer);

  write_section_header(fd);
  STAC(write_,TYPE,_array)(fd, data->weights2, nodes_per_layer*nodes_per_layer);

  write_section_header(fd);
  STAC(write_,TYPE,_array)(fd, data->weights3, nodes_per_layer*possible_outputs);

  write_section_header(fd);
  STAC(write_,TYPE,_array)(fd, data->biases1, nodes_per_layer);

  write_section_header(fd);
  STAC(write_,TYPE,_array)(fd, data->biases2, nodes_per_layer);

  write_section_header(fd);
  STAC(write_,TYPE,_array)(fd, data->biases3, possible_outputs);

}

int check_data( void *vdata, void *vref ) {
  struct bench_args_t *data = (struct bench_args_t *)vdata;
  struct bench_args_t *ref = (struct bench_args_t *)vref;
  int has_errors = 0;
  int i, j;
  TYPE diff;

  for(i=0; i<input_dimension; i++) {
    for(j=0; j<nodes_per_layer; j++) {
      diff = data->weights1[i*nodes_per_layer + j] - ref->weights1[i*nodes_per_layer + j];
      has_errors |= (diff<-EPSILON) || (EPSILON<diff);
    }
  }
  for(i=0; i<nodes_per_layer; i++) {
    for(j=0; j<nodes_per_layer; j++) {
      diff = data->weights2[i*nodes_per_layer + j] - ref->weights2[i*nodes_per_layer + j];
      has_errors |= (diff<-EPSILON) || (EPSILON<diff);
    }
  }
  for(i=0; i<nodes_per_layer; i++) {
    for(j=0; j<possible_outputs; j++) {
      diff = data->weights3[i*possible_outputs + j] - ref->weights3[i*possible_outputs + j];
      has_errors |= (diff<-EPSILON) || (EPSILON<diff);
    }
  }
  for(i=0; i<nodes_per_layer; i++) {
    diff = data->biases1[i] - ref->biases1[i];
    has_errors |= (diff<-EPSILON) || (EPSILON<diff);
  }
  for(i=0; i<nodes_per_layer; i++) {
    diff = data->biases2[i] - ref->biases2[i];
    has_errors |= (diff<-EPSILON) || (EPSILON<diff);
  }
  for(i=0; i<possible_outputs; i++) {
    diff = data->biases3[i] - ref->biases3[i];
    has_errors |= (diff<-EPSILON) || (EPSILON<diff);
  }
  // Return true if it's correct.
  return !has_errors;
}
