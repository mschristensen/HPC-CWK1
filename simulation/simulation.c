/* Functions pertinent to the outer simulation steps */

#include <math.h>
#include <stdio.h>
#include <omp.h>
#include "lbm.h"

#define LARGE_PIPE_SIZE 1000
//TODO::
//  -Compare parallel speed without collapse(2)
//  -Try scheduling to optimise cache?
//  -Try sections to minimise thread spawning overhead?
//  -Try prefetching?
//  -add if-condition (an omp directive) to parallelise only for large inputs? e.g. for accelerate_flow
//  -spawn threads only once and add barriers? may need a single monolithic function, or pass the vars in as args (in a struct?)
//  -Parallelise utils.c
//  -Reduce cache-thrashing where possible
//  -Experiment with scheduling
//  -Experiment with task-level parallelism for triple nested loops

float timestep(const param_t params, const accel_area_t accel_area,
    speed_t* cells, speed_t* tmp_cells, speed_t* tmp_tmp_cells, char* obstacles)
{
    accelerate_flow(params,accel_area,cells,obstacles);
    return d2q9bgk(params,cells,tmp_cells,tmp_tmp_cells,obstacles);
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
        #pragma omp parallel for if(params.ny >= LARGE_PIPE_SIZE) default(none) shared(cells,obstacles) private(index,ii) firstprivate(jj,w1,w2)
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
        #pragma omp parallel for if(params.nx >= LARGE_PIPE_SIZE) default(none) shared(cells,obstacles) private(index,jj) firstprivate(ii,w1,w2)
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

float d2q9bgk(const param_t params, speed_t* cells, speed_t* tmp_cells, speed_t* tmp_tmp_cells, char* obstacles)
{
    int ii,jj,kk;                /* generic counters */
    int x_e,x_w,y_n,y_s;         /* indices of neighbouring cells */
    int index;                   /* current cell array index */

    const float c_sq = 1.0/3.0;  /* square of speed of sound */
    const float w0 = 4.0/9.0;    /* weighting factor */
    const float w1 = 1.0/9.0;    /* weighting factor */
    const float w2 = 1.0/36.0;   /* weighting factor */

    float u_x,u_y;               /* av. velocities in x and y directions */
    float u_sq;                  /* squared velocity */
    float local_density;         /* sum of densities in a particular cell */
    float u[NSPEEDS];            /* directional velocities */
    float d_equ[NSPEEDS];        /* equilibrium densities */

    int tot_cells = 0;           /* no. of cells used in calculation */
    float tot_u = 0.0;           /* accumulated magnitudes of velocity for each cell */

    #pragma omp parallel default(none) shared(cells,tmp_cells,obstacles,tmp_tmp_cells,tot_u,tot_cells) private(ii,jj,kk,index,u_x,u_y,u_sq,local_density,u,d_equ,y_n,x_e,y_s,x_w) firstprivate(c_sq,w0,w1,w2)
    {
        #pragma omp for collapse(2) reduction(+:tot_u,tot_cells) schedule(guided) nowait
        /* loop over _all_ cells */
        for (ii = 0; ii < params.ny; ii++)
        {
            for (jj = 0; jj < params.nx; jj++)
            {
                y_n = (ii + 1) % params.ny;
                x_e = (jj + 1) % params.nx;
                y_s = (ii == 0) ? (ii + params.ny - 1) : (ii - 1);
                x_w = (jj == 0) ? (jj + params.nx - 1) : (jj - 1);

                index = ii*params.nx + jj;

                /* Propagate densities to neighbouring cells, following
                ** appropriate directions of travel.
                ** Now uses A-A pattern (rather than A-B) as per
                ** https://www.cs.arizona.edu/~pbailey/Accelerating_GPU_LBM.pdf
                ** to facilitate merge with rebound step */
                tmp_cells[index].speeds[0] = cells[index].speeds[0];                //central cell
                tmp_cells[index].speeds[1] = cells[ii *params.nx + x_w].speeds[1];  //east speed from west-side cell
                tmp_cells[index].speeds[2] = cells[y_s*params.nx + jj].speeds[2];   //north speed from south-side cell
                tmp_cells[index].speeds[3] = cells[ii *params.nx + x_e].speeds[3];  //west speed from east-side cell
                tmp_cells[index].speeds[4] = cells[y_n*params.nx + jj].speeds[4];   //south speed from north-side cell
                tmp_cells[index].speeds[5] = cells[y_s*params.nx + x_w].speeds[5];  //north-east speed from south-west-side cell
                tmp_cells[index].speeds[6] = cells[y_s*params.nx + x_e].speeds[6];  //north-west speed from south-east-side cell
                tmp_cells[index].speeds[7] = cells[y_n*params.nx + x_e].speeds[7];  //south-west speed from north-east-side cell
                tmp_cells[index].speeds[8] = cells[y_n*params.nx + x_w].speeds[8];  //south-east speed from north-west-side cell

                //tmp_tmp_cells[index].speeds now correct
                //cells[index] cannot be written to as original values are needed in later iterations

                /* if the cell contains an obstacle */
                if (obstacles[index])
                {
                    /* REBOUND STEP */
                    tmp_tmp_cells[index].speeds[1] = tmp_cells[index].speeds[3];
                    tmp_tmp_cells[index].speeds[2] = tmp_cells[index].speeds[4];
                    tmp_tmp_cells[index].speeds[3] = tmp_cells[index].speeds[1];
                    tmp_tmp_cells[index].speeds[4] = tmp_cells[index].speeds[2];
                    tmp_tmp_cells[index].speeds[5] = tmp_cells[index].speeds[7];
                    tmp_tmp_cells[index].speeds[6] = tmp_cells[index].speeds[8];
                    tmp_tmp_cells[index].speeds[7] = tmp_cells[index].speeds[5];
                    tmp_tmp_cells[index].speeds[8] = tmp_cells[index].speeds[6];
                } else {
                    /* COLLISION STEP */
                    /* compute local density total */
                    local_density = 0.0;

                    for (kk = 0; kk < NSPEEDS; kk++)
                    {
                        local_density += tmp_cells[index].speeds[kk];
                    }

                    /* compute x velocity component */
                    u_x = (tmp_cells[index].speeds[1] +
                            tmp_cells[index].speeds[5] +
                            tmp_cells[index].speeds[8]
                        - (tmp_cells[index].speeds[3] +
                            tmp_cells[index].speeds[6] +
                            tmp_cells[index].speeds[7]))
                        / local_density;

                    /* compute y velocity component */
                    u_y = (tmp_cells[index].speeds[2] +
                            tmp_cells[index].speeds[5] +
                            tmp_cells[index].speeds[6]
                        - (tmp_cells[index].speeds[4] +
                            tmp_cells[index].speeds[7] +
                            tmp_cells[index].speeds[8]))
                        / local_density;

                    /* velocity squared */
                    u_sq = u_x * u_x + u_y * u_y;

                    /* directional velocity components */
                    u[1] =   u_x;        /* east */
                    u[2] =         u_y;  /* north */
                    u[3] = - u_x;        /* west */
                    u[4] =       - u_y;  /* south */
                    u[5] =   u_x + u_y;  /* north-east */
                    u[6] = - u_x + u_y;  /* north-west */
                    u[7] = - u_x - u_y;  /* south-west */
                    u[8] =   u_x - u_y;  /* south-east */

                    /* equilibrium densities */
                    /* zero velocity density: weight w0 */
                    d_equ[0] = w0 * local_density * (1.0 - u_sq / (2.0 * c_sq));
                    /* axis speeds: weight w1 */
                    d_equ[1] = w1 * local_density * (1.0 + u[1] / c_sq
                        + (u[1] * u[1]) / (2.0 * c_sq * c_sq)
                        - u_sq / (2.0 * c_sq));
                    d_equ[2] = w1 * local_density * (1.0 + u[2] / c_sq
                        + (u[2] * u[2]) / (2.0 * c_sq * c_sq)
                        - u_sq / (2.0 * c_sq));
                    d_equ[3] = w1 * local_density * (1.0 + u[3] / c_sq
                        + (u[3] * u[3]) / (2.0 * c_sq * c_sq)
                        - u_sq / (2.0 * c_sq));
                    d_equ[4] = w1 * local_density * (1.0 + u[4] / c_sq
                        + (u[4] * u[4]) / (2.0 * c_sq * c_sq)
                        - u_sq / (2.0 * c_sq));
                    /* diagonal speeds: weight w2 */
                    d_equ[5] = w2 * local_density * (1.0 + u[5] / c_sq
                        + (u[5] * u[5]) / (2.0 * c_sq * c_sq)
                        - u_sq / (2.0 * c_sq));
                    d_equ[6] = w2 * local_density * (1.0 + u[6] / c_sq
                        + (u[6] * u[6]) / (2.0 * c_sq * c_sq)
                        - u_sq / (2.0 * c_sq));
                    d_equ[7] = w2 * local_density * (1.0 + u[7] / c_sq
                        + (u[7] * u[7]) / (2.0 * c_sq * c_sq)
                        - u_sq / (2.0 * c_sq));
                    d_equ[8] = w2 * local_density * (1.0 + u[8] / c_sq
                        + (u[8] * u[8]) / (2.0 * c_sq * c_sq)
                        - u_sq / (2.0 * c_sq));


                    /* local density total */
                    local_density = 0.0;
                    for (kk = 0; kk < NSPEEDS; kk++)
                    {
                        /* relaxation step (part of COLLISION STEP) */
                        tmp_tmp_cells[index].speeds[kk] =
                            (tmp_cells[index].speeds[kk] + params.omega *
                            (d_equ[kk] - tmp_cells[index].speeds[kk]));

                        /* AV_VELS STEP */
                        local_density += tmp_tmp_cells[index].speeds[kk];
                    }

                    /* x-component of velocity */
                    u_x = (tmp_tmp_cells[index].speeds[1] +
                            tmp_tmp_cells[index].speeds[5] +
                            tmp_tmp_cells[index].speeds[8]
                        - (tmp_tmp_cells[index].speeds[3] +
                            tmp_tmp_cells[index].speeds[6] +
                            tmp_tmp_cells[index].speeds[7])) /
                        local_density;

                    /* compute y velocity component */
                    u_y = (tmp_tmp_cells[index].speeds[2] +
                            tmp_tmp_cells[index].speeds[5] +
                            tmp_tmp_cells[index].speeds[6]
                        - (tmp_tmp_cells[index].speeds[4] +
                            tmp_tmp_cells[index].speeds[7] +
                            tmp_tmp_cells[index].speeds[8])) /
                        local_density;

                    /* accumulate the norm of x- and y- velocity components */
                    tot_u += sqrt(u_x*u_x + u_y*u_y);
                    /* increase counter of inspected cells */
                    ++tot_cells;
                }
            }
        }
    }

    return tot_u / (float)tot_cells;
}
