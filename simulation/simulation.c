/* Functions pertinent to the outer simulation steps */

#include <math.h>
#include <stdio.h>
#include "lbm.h"

void swap(speed_t** one, speed_t** two){
  speed_t* temp = *one;
  *one = *two;
  *two = temp;
}

float timestep(const param_t params, const accel_area_t accel_area,
    lbm_context_t* lbm_context,
    speed_t** cells_ptr, speed_t** tmp_cells_ptr, speed_t** tmp_tmp_cells_ptr, char* obstacles)
{

    cl_int GRID_SIZE = params.nx * params.ny;
    speed_t* cells = *cells_ptr;
    speed_t* tmp_cells = *tmp_cells_ptr;
    speed_t* tmp_tmp_cells = *tmp_tmp_cells_ptr;

    accelerate_flow(params,accel_area,cells,obstacles);
/*
    // set work-item problem dimensions
    #define NUM_DIMENSIONS 2
    size_t global_work_size[NUM_DIMENSIONS] = {params.nx, params.ny};   //total problem size
    size_t local_work_size[NUM_DIMENSIONS]  = {32,32};                  //work-group size

    cl_int err;
    setArgs2(lbm_context, cells, tmp_cells, tmp_tmp_cells, GRID_SIZE);
    err = clEnqueueNDRangeKernel(lbm_context->queue, lbm_context->kernels[0].kernel,
                                 NUM_DIMENSIONS, NULL,
                                 global_work_size, local_work_size, 0, NULL, NULL);
    if (CL_SUCCESS != err) DIE("OpenCL enqueue kernel error %d!\n", err);

    err = clEnqueueReadBuffer(lbm_context->queue, lbm_context->kernels[0].args[0],
                              CL_TRUE, 0,
                              sizeof(speed_t) * GRID_SIZE, cells,
                              0, NULL, NULL);
    if (CL_SUCCESS != err) DIE("OpenCL error reading back cell arrays %d!\n", err);
    err = clEnqueueReadBuffer(lbm_context->queue, lbm_context->kernels[0].args[1],
                              CL_TRUE, 0,
                              sizeof(speed_t) * GRID_SIZE, tmp_cells,
                              0, NULL, NULL);
    if (CL_SUCCESS != err) DIE("OpenCL error reading back tmp_cell arrays %d!\n", err);
    err = clEnqueueReadBuffer(lbm_context->queue, lbm_context->kernels[0].args[2],
                              CL_TRUE, 0,
                              sizeof(speed_t) * GRID_SIZE, tmp_tmp_cells,
                              0, NULL, NULL);
    if (CL_SUCCESS != err) DIE("OpenCL error reading back tmp_tmp_cell arrays %d!\n", err);
    cl_float* tot_u = malloc(sizeof(cl_float) * GRID_SIZE);
    err = clEnqueueReadBuffer(lbm_context->queue, lbm_context->kernels[0].args[4],
                              CL_TRUE, 0,
                              sizeof(cl_float) * GRID_SIZE, tot_u,
                              0, NULL, NULL);
    if (CL_SUCCESS != err) DIE("OpenCL error reading back tot_u array %d!\n", err);

    //SWAP
    speed_t* temp = *cells_ptr;
    *cells_ptr = *tmp_tmp_cells_ptr;
    *tmp_tmp_cells_ptr = temp;

    //Update
    cells = *cells_ptr;
    tmp_cells = *tmp_cells_ptr;
    tmp_tmp_cells = *tmp_tmp_cells_ptr;
*/


    // OPENCL PROPAGATE ---------------------------------------------------------------------
    // set work-item problem dimensions
    #define NUM_DIMENSIONS 2
    size_t global_work_size[NUM_DIMENSIONS] = {params.nx, params.ny};   //total problem size
    size_t local_work_size[NUM_DIMENSIONS]  = {32,32};                  //work-group size

    cl_int err;
    setArgs(lbm_context, cells, tmp_cells, tmp_tmp_cells, GRID_SIZE, 0);
    err = clEnqueueNDRangeKernel(lbm_context->queue, lbm_context->kernels[0].kernel,
                                 NUM_DIMENSIONS, NULL,
                                 global_work_size, local_work_size, 0, NULL, NULL);
    if (CL_SUCCESS != err) DIE("OpenCL enqueue kernel 0 error %d!\n", err);

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

    // OPENCL REBOUND ---------------------------------------------------------------------
    setArgs(lbm_context, cells, tmp_cells, tmp_tmp_cells, GRID_SIZE, 1);
    err = clEnqueueNDRangeKernel(lbm_context->queue, lbm_context->kernels[1].kernel,
                                 NUM_DIMENSIONS, NULL,
                                 global_work_size, local_work_size, 0, NULL, NULL);
    if (CL_SUCCESS != err) DIE("OpenCL enqueue kernel 1 error %d!\n", err);

    err  = clEnqueueReadBuffer(lbm_context->queue, lbm_context->kernels[1].args[0],
                              CL_TRUE, 0,
                              sizeof(speed_t) * GRID_SIZE, cells,
                              0, NULL, NULL);
    if (CL_SUCCESS != err) DIE("OpenCL error reading back cell arrays %d!\n", err);
    err = clEnqueueReadBuffer(lbm_context->queue, lbm_context->kernels[1].args[1],
                              CL_TRUE, 0,
                              sizeof(speed_t) * GRID_SIZE, tmp_cells,
                              0, NULL, NULL);
    if (CL_SUCCESS != err) DIE("OpenCL error reading back tmp_cell arrays %d!\n", err);
/*
    // OPENCL COLLISION ---------------------------------------------------------------------
    setArgs(lbm_context, cells, tmp_cells, tmp_tmp_cells, GRID_SIZE, 2);
    err = clEnqueueNDRangeKernel(lbm_context->queue, lbm_context->kernels[2].kernel,
                                 NUM_DIMENSIONS, NULL,
                                 global_work_size, local_work_size, 0, NULL, NULL);
    if (CL_SUCCESS != err) DIE("OpenCL enqueue kernel 2 error %d!\n", err);

    err  = clEnqueueReadBuffer(lbm_context->queue, lbm_context->kernels[2].args[0],
                              CL_TRUE, 0,
                              sizeof(speed_t) * GRID_SIZE, cells,
                              0, NULL, NULL);
    if (CL_SUCCESS != err) DIE("OpenCL error reading back cell arrays %d!\n", err);
    err = clEnqueueReadBuffer(lbm_context->queue, lbm_context->kernels[2].args[1],
                              CL_TRUE, 0,
                              sizeof(speed_t) * GRID_SIZE, tmp_cells,
                              0, NULL, NULL);
    if (CL_SUCCESS != err) DIE("OpenCL error reading back tmp_cell arrays %d!\n", err);
*/
    // OPENCL AV_VELOCITY ---------------------------------------------------------------------
    setArgs(lbm_context, cells, tmp_cells, tmp_tmp_cells, GRID_SIZE, 3);
    err = clEnqueueNDRangeKernel(lbm_context->queue, lbm_context->kernels[3].kernel,
                                 NUM_DIMENSIONS, NULL,
                                 global_work_size, local_work_size, 0, NULL, NULL);
    if (CL_SUCCESS != err) DIE("OpenCL enqueue kernel 3 error %d!\n", err);

    err  = clEnqueueReadBuffer(lbm_context->queue, lbm_context->kernels[3].args[0],
                              CL_TRUE, 0,
                              sizeof(speed_t) * GRID_SIZE, cells,
                              0, NULL, NULL);
    if (CL_SUCCESS != err) DIE("OpenCL error reading back cell arrays %d!\n", err);

    cl_float* tot_u = malloc(sizeof(cl_float) * GRID_SIZE);
    err = clEnqueueReadBuffer(lbm_context->queue, lbm_context->kernels[3].args[2],
                              CL_TRUE, 0,
                              sizeof(cl_float) * GRID_SIZE, tot_u,
                              0, NULL, NULL);
    if (CL_SUCCESS != err) DIE("OpenCL error reading back tot_u array %d!\n", err);


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

void setArgs2(lbm_context_t* lbm_context,
    speed_t* cells, speed_t* tmp_cells, speed_t* tmp_tmp_cells, int GRID_SIZE)
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
  err |= clEnqueueWriteBuffer(lbm_context->queue, lbm_context->kernels[0].args[2],
                             CL_TRUE, 0,
                             sizeof(speed_t) * GRID_SIZE, tmp_tmp_cells,
                             0, NULL, NULL);
  if (CL_SUCCESS != err) DIE("OpenCL error writing to buffers %d!\n", err);

  // set the kernel args
  err  = clSetKernelArg(lbm_context->kernels[0].kernel, 1, sizeof(cl_mem), &(lbm_context->kernels[0].args[0]));
  err |= clSetKernelArg(lbm_context->kernels[0].kernel, 2, sizeof(cl_mem), &(lbm_context->kernels[0].args[1]));
  err |= clSetKernelArg(lbm_context->kernels[0].kernel, 3, sizeof(cl_mem), &(lbm_context->kernels[0].args[2]));
  if (CL_SUCCESS != err) DIE("OpenCL error %d setting kernel args", err);
}

void setArgs(lbm_context_t* lbm_context,
    speed_t* cells, speed_t* tmp_cells, speed_t* tmp_tmp_cells, int GRID_SIZE, int KERNEL)
{
  cl_int err;
  err  = clEnqueueWriteBuffer(lbm_context->queue, lbm_context->kernels[KERNEL].args[0],
                             CL_TRUE, 0,
                             sizeof(speed_t) * GRID_SIZE, cells,
                             0, NULL, NULL);
  if(KERNEL != 3)
  {
    err |= clEnqueueWriteBuffer(lbm_context->queue, lbm_context->kernels[KERNEL].args[1],
                               CL_TRUE, 0,
                               sizeof(speed_t) * GRID_SIZE, tmp_cells,
                               0, NULL, NULL);
  }
  // set the kernel args
  err  = clSetKernelArg(lbm_context->kernels[0].kernel, 1, sizeof(cl_mem), &(lbm_context->kernels[0].args[0]));
  if(KERNEL != 3)
  {
    err |= clSetKernelArg(lbm_context->kernels[0].kernel, 2, sizeof(cl_mem), &(lbm_context->kernels[0].args[1]));
  }
  if (CL_SUCCESS != err) DIE("OpenCL error %d setting kernel args", err);
}

void accelerate_flow(const param_t params, const accel_area_t accel_area,
    speed_t* cells, char* obstacles)
{
    int ii,jj;     /* generic counters */
    double w1,w2;  /* weighting factors */

    /* compute weighting factors */
    w1 = params.density * params.accel / 9.0;
    w2 = params.density * params.accel / 36.0;

    if (accel_area.col_or_row == ACCEL_COLUMN)
    {
        jj = accel_area.idx;

        for (ii = 0; ii < params.ny; ii++)
        {
            /* if the cell is not occupied and
            ** we don't send a density negative */
            if (!obstacles[ii*params.nx + jj] &&
            (cells[ii*params.nx + jj].speeds[4] - w1) > 0.0 &&
            (cells[ii*params.nx + jj].speeds[7] - w2) > 0.0 &&
            (cells[ii*params.nx + jj].speeds[8] - w2) > 0.0 )
            {
                /* increase 'north-side' densities */
                cells[ii*params.nx + jj].speeds[2] += w1;
                cells[ii*params.nx + jj].speeds[5] += w2;
                cells[ii*params.nx + jj].speeds[6] += w2;
                /* decrease 'south-side' densities */
                cells[ii*params.nx + jj].speeds[4] -= w1;
                cells[ii*params.nx + jj].speeds[7] -= w2;
                cells[ii*params.nx + jj].speeds[8] -= w2;
            }
        }
    }
    else
    {
        ii = accel_area.idx;

        for (jj = 0; jj < params.nx; jj++)
        {
            /* if the cell is not occupied and
            ** we don't send a density negative */
            if (!obstacles[ii*params.nx + jj] &&
            (cells[ii*params.nx + jj].speeds[3] - w1) > 0.0 &&
            (cells[ii*params.nx + jj].speeds[6] - w2) > 0.0 &&
            (cells[ii*params.nx + jj].speeds[7] - w2) > 0.0 )
            {
                /* increase 'east-side' densities */
                cells[ii*params.nx + jj].speeds[1] += w1;
                cells[ii*params.nx + jj].speeds[5] += w2;
                cells[ii*params.nx + jj].speeds[8] += w2;
                /* decrease 'west-side' densities */
                cells[ii*params.nx + jj].speeds[3] -= w1;
                cells[ii*params.nx + jj].speeds[6] -= w2;
                cells[ii*params.nx + jj].speeds[7] -= w2;
            }
        }
    }
}
