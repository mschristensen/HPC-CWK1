/* Functions pertinent to the outer simulation steps */

#include <math.h>
#include <stdio.h>
#include "lbm.h"

float timestep(const param_t params, const accel_area_t accel_area,
    lbm_context_t lbm_context,
    speed_t* cells, speed_t* tmp_cells, speed_t* tmp_tmp_cells, char* obstacles)
{
    accelerate_flow(params,accel_area,cells,obstacles);

    // set work-item problem dimensions
    #define NUM_DIMENSIONS 2
    cl_int GRID_SIZE = params.nx * params.ny;
    size_t global_work_size[NUM_DIMENSIONS] = {params.nx, params.ny};   //total problem size
    size_t local_work_size[NUM_DIMENSIONS]  = {32,32};                    //work-group size
    // {1, 1} worked with box! Try it again with fixed kernel code
    // First make sure different values are output on each iteration of timestep:
    //   -Use a {1, 1} kernel to temporaily avoid parallel bugs (e.g. barriers needed etc)
    //   -Only do timestep once and see if tot_cell and tot_u contain different values in each entry
    //   -If this is the case then maybe the args arent being properly updated across timesteps
    // Then look at slide 119 at how to use barriers - I think some are needed in the kernel!

    cl_int err;
    err = clEnqueueNDRangeKernel(lbm_context.queue, lbm_context.kernels[0].kernel,
                                 NUM_DIMENSIONS, NULL,
                                 global_work_size, local_work_size, 0, NULL, NULL);
    if (CL_SUCCESS != err) DIE("OpenCL enqueue error %d!\n", err);

    cl_float tot_u[GRID_SIZE];
    cl_int tot_cells[GRID_SIZE];
    err  = clEnqueueReadBuffer(lbm_context.queue, lbm_context.kernels[0].args[1],
                              CL_TRUE, 0,
                              sizeof(speed_t) * GRID_SIZE, cells,
                              0, NULL, NULL);
    err |= clEnqueueReadBuffer(lbm_context.queue, lbm_context.kernels[0].args[2],
                              CL_TRUE, 0,
                              sizeof(speed_t) * GRID_SIZE, tmp_cells,
                              0, NULL, NULL);
    err |= clEnqueueReadBuffer(lbm_context.queue, lbm_context.kernels[0].args[3],
                              CL_TRUE, 0,
                              sizeof(speed_t) * GRID_SIZE, tmp_tmp_cells,
                              0, NULL, NULL);
    if (CL_SUCCESS != err) DIE("OpenCL error reading back cell arrays %d!\n", err);
    err = clEnqueueReadBuffer(lbm_context.queue, lbm_context.kernels[0].args[5],
                              CL_TRUE, 0,
                              sizeof(cl_float) * GRID_SIZE, tot_u,
                              0, NULL, NULL);
    if (CL_SUCCESS != err) DIE("OpenCL error reading back tot_u %d!\n", err);
    err = clEnqueueReadBuffer(lbm_context.queue, lbm_context.kernels[0].args[6],
                              CL_TRUE, 0,
                              sizeof(cl_int) *   GRID_SIZE, tot_cells,
                              0, NULL, NULL);
    if (CL_SUCCESS != err) DIE("OpenCL error reading back tot_cells %d!\n", err);

    setArgs(lbm_context, cells, tmp_cells, tmp_tmp_cells, GRID_SIZE);

    float tot_u_out = 0.0;
    int tot_cells_out = 0;
    int ii, jj, index;
    for (ii = 0; ii < params.ny; ii++)
    {
      for (jj = 0; jj < params.nx; jj++)
      {
        index = ii*params.nx + jj;
        /* accumulate the norm of x- and y- velocity components */
        if(tot_u[index] > 0.0)
        {
          tot_u_out += tot_u[index];
        }
        //printf("tot_u %f\n", tot_u[index]);
        /* increase counter of inspected cells */
        if(tot_cells[index] != 0)
        {
          tot_cells_out++;
        }
      }
    }

    float output = tot_u_out / (float)tot_cells_out;
    // print results
    printf("OUTPUT = %f/%d = %f\n", tot_u_out, tot_cells_out, output);
    if (CL_SUCCESS != err) DIE("OpenCL error %d!\n", err);
    return output;

  //return 0.0;
  /*
    *   TODO
    *   Run OpenCL kernels on the device
    */
}

void setArgs(lbm_context_t lbm_context,
    speed_t* cells, speed_t* tmp_cells, speed_t* tmp_tmp_cells, int GRID_SIZE)
{
  //cl_mem d_cells          = clCreateBuffer(lbm_context.context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, sizeof(speed_t) * GRID_SIZE, cells,        NULL);
  //cl_mem d_tmp_cells      = clCreateBuffer(lbm_context.context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, sizeof(speed_t) * GRID_SIZE, tmp_cells,    NULL);
  //cl_mem d_tmp_tmp_cells  = clCreateBuffer(lbm_context.context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, sizeof(speed_t) * GRID_SIZE, tmp_tmp_cells,NULL);

  // allocate memory for the kernel args
  //lbm_context.kernels[0].args[1] = d_cells;
  //lbm_context.kernels[0].args[2] = d_tmp_cells;
  //lbm_context.kernels[0].args[3] = d_tmp_tmp_cells;
  cl_int err;
  err  = clEnqueueWriteBuffer(lbm_context.queue, lbm_context.kernels[0].args[1],
                             CL_TRUE, 0,
                             sizeof(speed_t) * GRID_SIZE, cells,
                             0, NULL, NULL);
  err |= clEnqueueWriteBuffer(lbm_context.queue, lbm_context.kernels[0].args[2],
                             CL_TRUE, 0,
                             sizeof(speed_t) * GRID_SIZE, tmp_cells,
                             0, NULL, NULL);
  err |= clEnqueueWriteBuffer(lbm_context.queue, lbm_context.kernels[0].args[3],
                             CL_TRUE, 0,
                             sizeof(speed_t) * GRID_SIZE, tmp_tmp_cells,
                             0, NULL, NULL);
  if (CL_SUCCESS != err) DIE("OpenCL error writing to buffers %d!\n", err);

  // set the kernel args
  err  = clSetKernelArg(lbm_context.kernels[0].kernel, 1, sizeof(cl_mem), &lbm_context.kernels[0].args[1]);
  err |= clSetKernelArg(lbm_context.kernels[0].kernel, 2, sizeof(cl_mem), &lbm_context.kernels[0].args[2]);
  err |= clSetKernelArg(lbm_context.kernels[0].kernel, 3, sizeof(cl_mem), &lbm_context.kernels[0].args[3]);
  if (CL_SUCCESS != err) DIE("OpenCL error %d setting kernel args", err);
}

void accelerate_flow(const param_t params, const accel_area_t accel_area,
    speed_t* cells, char* obstacles)
{
    int ii,jj;    /* generic counters */
    /* weighting factors */
    float w1 = params.density * params.accel / 9.0;
    float w2 = params.density * params.accel / 36.0;
    int index;    /* array index for obstacles and speeds */

    if (accel_area.col_or_row == ACCEL_COLUMN)
    {
        jj = accel_area.idx;
        for (ii = 0; ii < params.ny; ii++)
        {
            index = ii*params.nx + jj;
            /* if the cell is not occupied and
            ** we don't send a density negative */
            if (!obstacles[index] &&
            (cells[index].speeds[4] - w1) > 0.0 &&
            (cells[index].speeds[7] - w2) > 0.0 &&
            (cells[index].speeds[8] - w2) > 0.0 )
            {
                /* increase 'north-side' densities */
                cells[index].speeds[2] += w1;
                cells[index].speeds[5] += w2;
                cells[index].speeds[6] += w2;
                /* decrease 'south-side' densities */
                cells[index].speeds[4] -= w1;
                cells[index].speeds[7] -= w2;
                cells[index].speeds[8] -= w2;
            }
        }
    }
    else
    {
        ii = accel_area.idx;
        for (jj = 0; jj < params.nx; jj++)
        {
            index = ii*params.nx + jj;
            /* if the cell is not occupied and
            ** we don't send a density negative */
            if (!obstacles[index] &&
            (cells[index].speeds[3] - w1) > 0.0 &&
            (cells[index].speeds[6] - w2) > 0.0 &&
            (cells[index].speeds[7] - w2) > 0.0 )
            {
                /* increase 'east-side' densities */
                cells[index].speeds[1] += w1;
                cells[index].speeds[5] += w2;
                cells[index].speeds[8] += w2;
                /* decrease 'west-side' densities */
                cells[index].speeds[3] -= w1;
                cells[index].speeds[6] -= w2;
                cells[index].speeds[7] -= w2;
            }
        }
    }
}
