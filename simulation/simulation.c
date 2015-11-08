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
    speed_t** cells_ptr, speed_t** tmp_cells_ptr, char* obstacles, int iter_num, unsigned int obstacle_count)
{
    cl_int GRID_SIZE = params.nx * params.ny;
    cl_int NUM_WORK_GROUPS = GRID_SIZE / 1024;
    //printf("NUM_WORK_GROUPS = %d / %d = %d\n", GRID_SIZE, WORK_GROUP_SIZE, NUM_WORK_GROUPS);
    speed_t* cells = *cells_ptr;
    speed_t* tmp_cells = *tmp_cells_ptr;

    // set work-item problem dimensions
    #define NUM_DIMENSIONS 2
    size_t global_work_size[NUM_DIMENSIONS] = {params.nx, params.ny};                //total problem size
    size_t local_work_size[NUM_DIMENSIONS]  = {WORK_GROUP_SIZE_X,WORK_GROUP_SIZE_Y}; //work-group size

    // Set kernel args according to parity of iter_num
    setArgs(lbm_context, cells, tmp_cells, GRID_SIZE, iter_num);

    // Run the kernel
    cl_int err;
    err = clEnqueueNDRangeKernel(lbm_context->queue, lbm_context->kernels[0].kernel,
                                 NUM_DIMENSIONS, NULL,
                                 global_work_size, local_work_size, 0, NULL, NULL);
    if (CL_SUCCESS != err) DIE("OpenCL enqueue kernel 1 error %d!\n", err);

    // Only read back the final cell arrays on the last iteration
    if(iter_num == params.max_iters - 1)
    {
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
    }

    // Read back the av_vels data
    cl_float* tot_u = malloc(sizeof(cl_float) * GRID_SIZE);
    err = clEnqueueReadBuffer(lbm_context->queue, lbm_context->kernels[0].args[3],
                              CL_TRUE, 0,
                              sizeof(cl_float) * NUM_WORK_GROUPS, tot_u,
                              0, NULL, NULL);
    if (CL_SUCCESS != err) DIE("OpenCL error reading back tot_u array %d!\n", err);



    // CALCULATE AV_VELS OUTPUT FROM tot_u ARRAY
    int ii, jj, index;
    float tot_u_out = 0.0;
    int tot_cells = 0;
    /*
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
    }*/
    for(ii = 0; ii < NUM_WORK_GROUPS; ii++)
    {
      tot_u_out += tot_u[ii];
      //printf("%f ", tot_u[ii]);
    }
    printf("result = %f/%d = %f\n", tot_u_out, obstacle_count, tot_u_out / (float)obstacle_count);
    return tot_u_out / (float)obstacle_count;
}

void setArgs(lbm_context_t* lbm_context,
    speed_t* cells, speed_t* tmp_cells, int GRID_SIZE, int iter_num)
{
  cl_int err;

  // Must write to the buffers on the first iteration, otherwise av_vels[0] = 0,
  // and values only start correctly at av_vels[1]
  if(iter_num == 0)
  {
    err  = clEnqueueWriteBuffer(lbm_context->queue, lbm_context->kernels[0].args[0],
                               CL_TRUE, 0,
                               sizeof(speed_t) * GRID_SIZE, cells,
                               0, NULL, NULL);
    err |= clEnqueueWriteBuffer(lbm_context->queue, lbm_context->kernels[0].args[1],
                               CL_TRUE, 0,
                               sizeof(speed_t) * GRID_SIZE, tmp_cells,
                               0, NULL, NULL);
    if (CL_SUCCESS != err) DIE("OpenCL error %d writing to buffers", err);
  }

  // set the kernel args, swapping cells and tmp_cells every iteration
  char odd = iter_num % 2;
  err  = clSetKernelArg(lbm_context->kernels[0].kernel, odd ? 4 : 3, sizeof(cl_mem), &(lbm_context->kernels[0].args[0]));
  err |= clSetKernelArg(lbm_context->kernels[0].kernel, odd ? 3 : 4, sizeof(cl_mem), &(lbm_context->kernels[0].args[1]));
  if (CL_SUCCESS != err) DIE("OpenCL error %d setting kernel args", err);
}
