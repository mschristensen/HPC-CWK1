/* Functions pertinent to the outer simulation steps */

#include <math.h>
#include <stdio.h>
#include "lbm.h"

/*void swap(speed_t** one, speed_t** two){
  speed_t* temp = *one;
  *one = *two;
  *two = temp;
}*/

float timestep(const param_t params, const accel_area_t accel_area,
    lbm_context_t* lbm_context,
    speed_t** cells_ptr, speed_t** tmp_cells_ptr, char* obstacles)
{
    cl_int GRID_SIZE = params.nx * params.ny;
    speed_t* cells = *cells_ptr;
    speed_t* tmp_cells = *tmp_cells_ptr;

    // OPENCL PROPAGATE ---------------------------------------------------------------------
    // set work-item problem dimensions
    #define NUM_DIMENSIONS 2
    size_t global_work_size[NUM_DIMENSIONS] = {params.nx, params.ny};   //total problem size
    size_t local_work_size[NUM_DIMENSIONS]  = {32,32};                  //work-group size

    cl_int err;
    setArgs(lbm_context, cells, tmp_cells, GRID_SIZE);
    err = clEnqueueNDRangeKernel(lbm_context->queue, lbm_context->kernels[0].kernel,
                                 NUM_DIMENSIONS, NULL,
                                 global_work_size, local_work_size, 0, NULL, NULL);
    if (CL_SUCCESS != err) DIE("OpenCL enqueue kernel 1 error %d!\n", err);

    err  = clEnqueueReadBuffer(lbm_context->queue, lbm_context->kernels[0].args[0],
                              CL_TRUE, 0,
                              sizeof(speed_t) * GRID_SIZE, cells,
                              0, NULL, NULL);
    if (CL_SUCCESS != err) DIE("OpenCL error reading back cell arrays %d!\n", err);
    err = clEnqueueReadBuffer(lbm_context->queue, lbm_context->kernels[0].args[1],
                              CL_TRUE, 0,
                              sizeof(speed_t) * GRID_SIZE, tmp_cells,
                              0, NULL, NULL);
    if (CL_SUCCESS != err) DIE("OpenCL error reading back tmp_cell arrays %d!\n", err);
    cl_float* tot_u = malloc(sizeof(cl_float) * GRID_SIZE);
    err = clEnqueueReadBuffer(lbm_context->queue, lbm_context->kernels[0].args[3],
                              CL_TRUE, 0,
                              sizeof(cl_float) * GRID_SIZE, tot_u,
                              0, NULL, NULL);
    if (CL_SUCCESS != err) DIE("OpenCL error reading back tot_u array %d!\n", err);

    // CALCULATE AV_VELS OUTPUT FROM tot_u ARRAY
    int ii, jj, index;
    float tot_u_out = 0.0;
    int tot_cells = 0;
    for(ii = 0; ii < params.ny; ii++)
    {
      for(jj = 0; jj < params.nx; jj++)
      {
        index = ii*params.nx + jj;
        if(tot_u[index] >= 0)
        {
          tot_u_out += tot_u[index];
          tot_cells++;
        }
      }
    }
    return tot_u_out / (float)tot_cells;
}

void setArgs(lbm_context_t* lbm_context,
    speed_t* cells, speed_t* tmp_cells, int GRID_SIZE)
{
  cl_int err;
  err  = clEnqueueWriteBuffer(lbm_context->queue, lbm_context->kernels[0].args[0],
                             CL_TRUE, 0,
                             sizeof(speed_t) * GRID_SIZE, cells,
                             0, NULL, NULL);
  err |= clEnqueueWriteBuffer(lbm_context->queue, lbm_context->kernels[0].args[1],
                             CL_TRUE, 0,
                             sizeof(speed_t) * GRID_SIZE, tmp_cells,
                             0, NULL, NULL);
  if (CL_SUCCESS != err) DIE("OpenCL error %d writing to buffers", err);

  // set the kernel args
  err  = clSetKernelArg(lbm_context->kernels[0].kernel, 1, sizeof(cl_mem), &(lbm_context->kernels[0].args[0]));
  err |= clSetKernelArg(lbm_context->kernels[0].kernel, 2, sizeof(cl_mem), &(lbm_context->kernels[0].args[1]));
  if (CL_SUCCESS != err) DIE("OpenCL error %d setting kernel args", err);
}
