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
    speed_t* cells = *cells_ptr;
    speed_t* tmp_cells = *tmp_cells_ptr;
    speed_t* tmp_tmp_cells = *tmp_tmp_cells_ptr;

    accelerate_flow(params,accel_area,cells,obstacles);
    /*  -Have established that all cell arrays are correctly updated by the kernels.
    *   -This is true at all steps. This is true between timesteps.
    *   -The problem is with the values that the kernels are assigning to the arrays.
    *   -Somehow, due to the way work is split up or something, the values are always updated to exactly what they were before.
    *   -i.e. a call like tmp_cells[index].speeds[1] = cells[ii *params->nx + x_w].speeds[1] means that both sides of the assignment are THE SAME VALUE.
    *   -So strange... see the OpenMP coursework to view how values *should* change between iterations... they just aren't, somehow.
    *
    *   -ASK IN LAB:
    *       -- Values don't change! :@
    *       -- Need > 1 work groups for even smallest prob, but cant sync...
    *       -- Don't need to sync as all self-contained in kernel?
    *       -- Login to newblue4?
    *       -- Dont
    */

    // set work-item problem dimensions
    #define NUM_DIMENSIONS 2
    cl_int GRID_SIZE = params.nx * params.ny;
    size_t global_work_size[NUM_DIMENSIONS] = {params.nx, params.ny};   //total problem size
    size_t local_work_size[NUM_DIMENSIONS]  = {32,32};                  //work-group size
    int print_cell_index = 50;
    /*
    printf("NEXT TIMESTEP\n");
    printf("CELLS  @ (%d) = {%f, %f, %f, %f, %f, %f, %f, %f, %f}\n", print_cell_index,
          cells[print_cell_index*params.nx + print_cell_index].speeds[0],
          cells[print_cell_index*params.nx + print_cell_index].speeds[1],
          cells[print_cell_index*params.nx + print_cell_index].speeds[2],
          cells[print_cell_index*params.nx + print_cell_index].speeds[3],
          cells[print_cell_index*params.nx + print_cell_index].speeds[4],
          cells[print_cell_index*params.nx + print_cell_index].speeds[5],
          cells[print_cell_index*params.nx + print_cell_index].speeds[6],
          cells[print_cell_index*params.nx + print_cell_index].speeds[7],
          cells[print_cell_index*params.nx + print_cell_index].speeds[8]);
    printf("TTCELLS@ (%d) = {%f, %f, %f, %f, %f, %f, %f, %f, %f}\n", print_cell_index,
          tmp_tmp_cells[print_cell_index*params.nx + print_cell_index].speeds[0],
          tmp_tmp_cells[print_cell_index*params.nx + print_cell_index].speeds[1],
          tmp_tmp_cells[print_cell_index*params.nx + print_cell_index].speeds[2],
          tmp_tmp_cells[print_cell_index*params.nx + print_cell_index].speeds[3],
          tmp_tmp_cells[print_cell_index*params.nx + print_cell_index].speeds[4],
          tmp_tmp_cells[print_cell_index*params.nx + print_cell_index].speeds[5],
          tmp_tmp_cells[print_cell_index*params.nx + print_cell_index].speeds[6],
          tmp_tmp_cells[print_cell_index*params.nx + print_cell_index].speeds[7],
          tmp_tmp_cells[print_cell_index*params.nx + print_cell_index].speeds[8]);*/


    cl_int err;
    err = clEnqueueNDRangeKernel(lbm_context->queue, lbm_context->kernels[0].kernel,
                                 NUM_DIMENSIONS, NULL,
                                 global_work_size, local_work_size, 0, NULL, NULL);
    if (CL_SUCCESS != err) DIE("OpenCL enqueue error %d!\n", err);

    cl_float tot_u[GRID_SIZE];
    cl_int tot_cells[GRID_SIZE];
    err  = clEnqueueReadBuffer(lbm_context->queue, lbm_context->kernels[0].args[1],
                              CL_TRUE, 0,
                              sizeof(speed_t) * GRID_SIZE, cells,
                              0, NULL, NULL);
    err |= clEnqueueReadBuffer(lbm_context->queue, lbm_context->kernels[0].args[2],
                              CL_TRUE, 0,
                              sizeof(speed_t) * GRID_SIZE, tmp_cells,
                              0, NULL, NULL);
    err |= clEnqueueReadBuffer(lbm_context->queue, lbm_context->kernels[0].args[3],
                              CL_TRUE, 0,
                              sizeof(speed_t) * GRID_SIZE, tmp_tmp_cells,
                              0, NULL, NULL);
    if (CL_SUCCESS != err) DIE("OpenCL error reading back cell arrays %d!\n", err);
    err = clEnqueueReadBuffer(lbm_context->queue, lbm_context->kernels[0].args[5],
                              CL_TRUE, 0,
                              sizeof(cl_float) * GRID_SIZE, tot_u,
                              0, NULL, NULL);
    if (CL_SUCCESS != err) DIE("OpenCL error reading back tot_u %d!\n", err);
    /*err = clEnqueueReadBuffer(lbm_context->queue, lbm_context->kernels[0].args[6],
                              CL_TRUE, 0,
                              sizeof(cl_int) *   GRID_SIZE, tot_cells,
                              0, NULL, NULL);
    if (CL_SUCCESS != err) DIE("OpenCL error reading back tot_cells %d!\n", err);*/

    clFinish(lbm_context->queue);
    //SWAP
    speed_t* temp = *cells_ptr;
    *cells_ptr = *tmp_tmp_cells_ptr;
    *tmp_tmp_cells_ptr = temp;

    //Update
    cells = *cells_ptr;
    tmp_cells = *tmp_cells_ptr;
    tmp_tmp_cells = *tmp_tmp_cells_ptr;

/*
    printf("KERNEL EXECUTED + SWAPPED\n");
    printf("CELLS  @ (%d) = {%f, %f, %f, %f, %f, %f, %f, %f, %f}\n", print_cell_index,
          cells[print_cell_index*params.nx + print_cell_index].speeds[0],
          cells[print_cell_index*params.nx + print_cell_index].speeds[1],
          cells[print_cell_index*params.nx + print_cell_index].speeds[2],
          cells[print_cell_index*params.nx + print_cell_index].speeds[3],
          cells[print_cell_index*params.nx + print_cell_index].speeds[4],
          cells[print_cell_index*params.nx + print_cell_index].speeds[5],
          cells[print_cell_index*params.nx + print_cell_index].speeds[6],
          cells[print_cell_index*params.nx + print_cell_index].speeds[7],
          cells[print_cell_index*params.nx + print_cell_index].speeds[8]);
    printf("TTCELLS@ (%d) = {%f, %f, %f, %f, %f, %f, %f, %f, %f}\n", print_cell_index,
          tmp_tmp_cells[print_cell_index*params.nx + print_cell_index].speeds[0],
          tmp_tmp_cells[print_cell_index*params.nx + print_cell_index].speeds[1],
          tmp_tmp_cells[print_cell_index*params.nx + print_cell_index].speeds[2],
          tmp_tmp_cells[print_cell_index*params.nx + print_cell_index].speeds[3],
          tmp_tmp_cells[print_cell_index*params.nx + print_cell_index].speeds[4],
          tmp_tmp_cells[print_cell_index*params.nx + print_cell_index].speeds[5],
          tmp_tmp_cells[print_cell_index*params.nx + print_cell_index].speeds[6],
          tmp_tmp_cells[print_cell_index*params.nx + print_cell_index].speeds[7],
          tmp_tmp_cells[print_cell_index*params.nx + print_cell_index].speeds[8]);*/

    //Write new values to the kernel buffers and re-set args
    setArgs(lbm_context, cells, tmp_cells, tmp_tmp_cells, GRID_SIZE);

    float tot_u_out = 0.0;
    int tot_cells_out = 0;
    int ii, jj, index;
    for (ii = 0; ii < params.ny; ii++)
    {
      for (jj = 0; jj < params.nx; jj++)
      {
        index = ii*params.nx + jj;
        // accumulate the norm of x- and y- velocity components
        if(tot_u[index] > 0.0)
        {
          tot_u_out += tot_u[index];
        }
        //printf("tot_u %f\n", tot_u[index]);
        // increase counter of inspected cells
        if(tot_cells[index] != 0)
        {
          tot_cells_out++;
        }
      }
    }

    float output = tot_u_out / (float)tot_cells_out;
    //printf("OUTPUT = %f/%d = %f\n", tot_u_out, tot_cells_out, output);
    if (CL_SUCCESS != err) DIE("OpenCL error %d!\n", err);
    return output;
}

void setArgs(lbm_context_t* lbm_context,
    speed_t* cells, speed_t* tmp_cells, speed_t* tmp_tmp_cells, int GRID_SIZE)
{
  //cl_mem d_cells          = clCreateBuffer(lbm_context->context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, sizeof(speed_t) * GRID_SIZE, cells,        NULL);
  //cl_mem d_tmp_cells      = clCreateBuffer(lbm_context->context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, sizeof(speed_t) * GRID_SIZE, tmp_cells,    NULL);
  //cl_mem d_tmp_tmp_cells  = clCreateBuffer(lbm_context->context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, sizeof(speed_t) * GRID_SIZE, tmp_tmp_cells,NULL);

  // allocate memory for the kernel args
  //lbm_context->kernels[0].args[1] = d_cells;
  //lbm_context->kernels[0].args[2] = d_tmp_cells;
  //lbm_context->kernels[0].args[3] = d_tmp_tmp_cells;
  cl_int err;
  err  = clEnqueueWriteBuffer(lbm_context->queue, lbm_context->kernels[0].args[1],
                             CL_TRUE, 0,
                             sizeof(speed_t) * GRID_SIZE, cells,
                             0, NULL, NULL);
  err |= clEnqueueWriteBuffer(lbm_context->queue, lbm_context->kernels[0].args[2],
                             CL_TRUE, 0,
                             sizeof(speed_t) * GRID_SIZE, tmp_cells,
                             0, NULL, NULL);
  err |= clEnqueueWriteBuffer(lbm_context->queue, lbm_context->kernels[0].args[3],
                             CL_TRUE, 0,
                             sizeof(speed_t) * GRID_SIZE, tmp_tmp_cells,
                             0, NULL, NULL);
  if (CL_SUCCESS != err) DIE("OpenCL error writing to buffers %d!\n", err);
  clFinish(lbm_context->queue);

  // set the kernel args
  err  = clSetKernelArg(lbm_context->kernels[0].kernel, 1, sizeof(cl_mem), &(lbm_context->kernels[0].args[1]));
  err |= clSetKernelArg(lbm_context->kernels[0].kernel, 2, sizeof(cl_mem), &(lbm_context->kernels[0].args[2]));
  err |= clSetKernelArg(lbm_context->kernels[0].kernel, 3, sizeof(cl_mem), &(lbm_context->kernels[0].args[3]));
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
