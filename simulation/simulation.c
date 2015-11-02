/* Functions pertinent to the outer simulation steps */

#include <math.h>
#include <stdio.h>
#include "lbm.h"

#define GUIDED_THRESHOLD 65536  //Input > 256 * 256 uses guided scheduling

float timestep(const param_t params, const accel_area_t accel_area,
    lbm_context_t lbm_context,
    speed_t* cells, speed_t* tmp_cells, speed_t* tmp_tmp_cells, char* obstacles)
{

    accelerate_flow(params,accel_area,cells,obstacles);
    //return d2q9bgk(params,cells,tmp_cells,tmp_tmp_cells,obstacles);

    // set work-item problem dimensions
    #define NUM_DIMENSIONS 1
    size_t global_work_size[NUM_DIMENSIONS] = {params.nx * params.ny}; //total problem size
    size_t local_work_size[NUM_DIMENSIONS]  = {1}; //per work-item size

    cl_int err;
    err = clEnqueueNDRangeKernel(lbm_context.queue, lbm_context.kernels[0].kernel, NUM_DIMENSIONS, NULL,
                        global_work_size, local_work_size,0,NULL,NULL);
    if (CL_SUCCESS != err) printf("OpenCL enqueue error %d!\n", err);

    float output[1];
    // read output array (blocking so data is ready after this call)
    err = clEnqueueReadBuffer(lbm_context.queue, lbm_context.kernels[0].args[5],
                              CL_TRUE, 0,
                              sizeof(cl_float), output,
                              0, NULL, NULL);

    // print results
    printf("output = %f\n", output[0]);
    if (CL_SUCCESS != err) printf("OpenCL error %d!\n", err);
    return output[0];

  //return 0.0;
  /*
    *   TODO
    *   Run OpenCL kernels on the device
    */
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
