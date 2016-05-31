#include "md.h"
#include "support.h"
#include <string.h>

int INPUT_SIZE = sizeof(struct bench_args_t);

#define EPSILON ((TYPE)1.0e-6)

void run_benchmark( void *vargs, cl_context& context, cl_command_queue& commands, cl_program& program, cl_kernel& kernel ) {
  struct bench_args_t *args = (struct bench_args_t *)vargs;
  // Create device buffers
  //
  static unsigned *force_x_buffer = (unsigned int*)clSVMAllocAltera(context, 0, sizeof(args->force_x), 1024); 
  static unsigned *force_y_buffer = (unsigned int*)clSVMAllocAltera(context, 0, sizeof(args->force_y), 1024); 
  static unsigned *force_z_buffer = (unsigned int*)clSVMAllocAltera(context, 0, sizeof(args->force_z), 1024); 
  static unsigned *position_x_buffer = (unsigned int*)clSVMAllocAltera(context, 0, sizeof(args->position_x), 1024); 
  static unsigned *position_y_buffer = (unsigned int*)clSVMAllocAltera(context, 0, sizeof(args->position_y), 1024); 
  static unsigned *position_z_buffer = (unsigned int*)clSVMAllocAltera(context, 0, sizeof(args->position_z), 1024); 
  static unsigned *NL_buffer = (unsigned int*)clSVMAllocAltera(context, 0, sizeof(args->NL), 1024); 

  // Write our data set into device buffers  
  //
  memcpy(force_x_buffer, args->force_x, sizeof(args->force_x));
  memcpy(force_y_buffer, args->force_y, sizeof(args->force_y));
  memcpy(force_z_buffer, args->force_z, sizeof(args->force_z));
  memcpy(position_x_buffer, args->position_x, sizeof(args->position_x));
  memcpy(position_y_buffer, args->position_y, sizeof(args->position_y));
  memcpy(position_z_buffer, args->position_z, sizeof(args->position_z));
  memcpy(NL_buffer, args->NL, sizeof(args->NL));
    
  // Set the arguments to our compute kernel
  //
  int status;
  status = clSetKernelArgSVMPointerAltera(kernel, 0, (void*)force_x_buffer);
  status |= clSetKernelArgSVMPointerAltera(kernel, 1, (void*)force_y_buffer);
  status |= clSetKernelArgSVMPointerAltera(kernel, 2, (void*)force_z_buffer);
  status |= clSetKernelArgSVMPointerAltera(kernel, 3, (void*)position_x_buffer);
  status |= clSetKernelArgSVMPointerAltera(kernel, 4, (void*)position_y_buffer);
  status |= clSetKernelArgSVMPointerAltera(kernel, 5, (void*)position_z_buffer);
  status |= clSetKernelArgSVMPointerAltera(kernel, 6, (void*)NL_buffer);
  if(status != CL_SUCCESS) {
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
  memcpy(args->force_x, force_x_buffer, sizeof(args->force_x));
  memcpy(args->force_y, force_y_buffer, sizeof(args->force_y));
  memcpy(args->force_z, force_z_buffer, sizeof(args->force_z));
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
