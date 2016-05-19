#include "md.h"
#include "support.h"
#include <string.h>

int INPUT_SIZE = sizeof(struct bench_args_t);

#define EPSILON ((TYPE)1.0e-6)

void run_benchmark( void *vargs, cl_context& context, cl_command_queue& commands, cl_program& program, cl_kernel& kernel ) {
  struct bench_args_t *args = (struct bench_args_t *)vargs;
  TYPE force_x[blockSide][blockSide][blockSide][densityFactor];
  TYPE force_y[blockSide][blockSide][blockSide][densityFactor];
  TYPE force_z[blockSide][blockSide][blockSide][densityFactor];
  TYPE position_x[blockSide][blockSide][blockSide][densityFactor];
  TYPE position_y[blockSide][blockSide][blockSide][densityFactor];
  TYPE position_z[blockSide][blockSide][blockSide][densityFactor];
  int i,j,k,l;
  for (i=0; i<blockSide; i++) 
    for (j=0; j<blockSide; j++)
      for (k=0; k<blockSide; k++)
	for (l=0; l<densityFactor; l++)
        {
          force_x[i][j][k][l] = (args->force)[i][j][k][l].x;
          force_y[i][j][k][l] = (args->force)[i][j][k][l].y;
          force_z[i][j][k][l] = (args->force)[i][j][k][l].z;
          position_x[i][j][k][l] = (args->position)[i][j][k][l].x;
          position_y[i][j][k][l] = (args->position)[i][j][k][l].y;
          position_z[i][j][k][l] = (args->position)[i][j][k][l].z;
        }
  // Create device buffers
  //
  cl_mem force_x_buffer = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(force_x), NULL, NULL);
  cl_mem force_y_buffer = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(force_y), NULL, NULL);
  cl_mem force_z_buffer = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(force_z), NULL, NULL);
  cl_mem position_x_buffer = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(position_x), NULL, NULL);
  cl_mem position_y_buffer = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(position_y), NULL, NULL);
  cl_mem position_z_buffer = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(position_z), NULL, NULL);
  cl_mem n_points_buffer = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(args->n_points), NULL, NULL);
  if (!force_x_buffer || !force_y_buffer || !force_z_buffer || !position_x_buffer || !position_y_buffer || !position_z_buffer || !n_points_buffer)
  {
    printf("Error: Failed to allocate device memory!\n");
    printf("Test failed\n");
    exit(1);
  }    

  // Write our data set into device buffers  
  //
  int err;
  err = clEnqueueWriteBuffer(commands, force_x_buffer, CL_TRUE, 0, sizeof(force_x), force_x, 0, NULL, NULL);
  err |= clEnqueueWriteBuffer(commands, force_y_buffer, CL_TRUE, 0, sizeof(force_y), force_y, 0, NULL, NULL);
  err |= clEnqueueWriteBuffer(commands, force_z_buffer, CL_TRUE, 0, sizeof(force_z), force_z, 0, NULL, NULL);
  err |= clEnqueueWriteBuffer(commands, position_x_buffer, CL_TRUE, 0, sizeof(position_x), position_x, 0, NULL, NULL);
  err |= clEnqueueWriteBuffer(commands, position_y_buffer, CL_TRUE, 0, sizeof(position_y), position_y, 0, NULL, NULL);
  err |= clEnqueueWriteBuffer(commands, position_z_buffer, CL_TRUE, 0, sizeof(position_z), position_z, 0, NULL, NULL);
  err |= clEnqueueWriteBuffer(commands, n_points_buffer, CL_TRUE, 0, sizeof(args->n_points), args->n_points, 0, NULL, NULL);
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
  err  |= clSetKernelArg(kernel, 6, sizeof(cl_mem), &n_points_buffer);
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
  err = clEnqueueReadBuffer( commands, force_x_buffer, CL_TRUE, 0, sizeof(force_x), force_x, 0, NULL, NULL );  
  err |= clEnqueueReadBuffer( commands, force_y_buffer, CL_TRUE, 0, sizeof(force_y), force_y, 0, NULL, NULL );  
  err |= clEnqueueReadBuffer( commands, force_z_buffer, CL_TRUE, 0, sizeof(force_z), force_z, 0, NULL, NULL );  
  if (err != CL_SUCCESS)
  {
    printf("Error: Failed to read output array! %d\n", err);
    printf("Test failed\n");
    exit(1);
  }
  for (i=0; i<blockSide; i++) 
    for (j=0; j<blockSide; j++)
      for (k=0; k<blockSide; k++)
	for (l=0; l<densityFactor; l++)
        {
          (args->force)[i][j][k][l].x = force_x[i][j][k][l];
          (args->force)[i][j][k][l].y = force_y[i][j][k][l];
          (args->force)[i][j][k][l].z = force_z[i][j][k][l];
        }
}

/* Input format:
%% Section 1
int32_t[blockSide^3]: grid populations
%% Section 2
TYPE[blockSide^3*densityFactor]: positions
*/

void input_to_data(int fd, void *vdata) {
  struct bench_args_t *data = (struct bench_args_t *)vdata;
  char *p, *s;
  // Zero-out everything.
  memset(vdata,0,sizeof(struct bench_args_t));
  // Load input string
  p = readfile(fd);

  s = find_section_start(p,1);
  parse_int32_t_array(s, (int32_t *)(data->n_points), blockSide*blockSide*blockSide);

  s = find_section_start(p,2);
  STAC(parse_,TYPE,_array)(s, (double *)(data->position), 3*blockSide*blockSide*blockSide*densityFactor);
}

void data_to_input(int fd, void *vdata) {
  struct bench_args_t *data = (struct bench_args_t *)vdata;

  write_section_header(fd);
  write_int32_t_array(fd, (int32_t *)(data->n_points), blockSide*blockSide*blockSide);

  write_section_header(fd);
  STAC(write_,TYPE,_array)(fd, (double *)(data->position), 3*blockSide*blockSide*blockSide*densityFactor);

}

/* Output format:
%% Section 1
TYPE[blockSide^3*densityFactor]: force
*/

void output_to_data(int fd, void *vdata) {
  struct bench_args_t *data = (struct bench_args_t *)vdata;
  char *p, *s;
  // Zero-out everything.
  memset(vdata,0,sizeof(struct bench_args_t));
  // Load input string
  p = readfile(fd);

  s = find_section_start(p,1);
  STAC(parse_,TYPE,_array)(s, (double *)data->force, 3*blockSide*blockSide*blockSide*densityFactor);
}

void data_to_output(int fd, void *vdata) {
  struct bench_args_t *data = (struct bench_args_t *)vdata;

  write_section_header(fd);
  STAC(write_,TYPE,_array)(fd, (double *)data->force, 3*blockSide*blockSide*blockSide*densityFactor);
}

int check_data( void *vdata, void *vref ) {
  struct bench_args_t *data = (struct bench_args_t *)vdata;
  struct bench_args_t *ref = (struct bench_args_t *)vref;
  int has_errors = 0;
  int i, j, k, d;
  TYPE diff_x, diff_y, diff_z;

  for(i=0; i<blockSide; i++) {
    for(j=0; j<blockSide; j++) {
      for(k=0; k<blockSide; k++) {
        for(d=0; d<densityFactor; d++) {
          diff_x = data->force[i][j][k][d].x - ref->force[i][j][k][d].x;
          diff_y = data->force[i][j][k][d].y - ref->force[i][j][k][d].y;
          diff_z = data->force[i][j][k][d].z - ref->force[i][j][k][d].z;
          has_errors |= (diff_x<-EPSILON) || (EPSILON<diff_x);
          has_errors |= (diff_y<-EPSILON) || (EPSILON<diff_y);
          has_errors |= (diff_z<-EPSILON) || (EPSILON<diff_z);
        }
      }
    }
  }

  // Return true if it's correct.
  return !has_errors;
}
