//To use double data types, uncomment the following line:
//#pragma OPENCL EXTENSION cl_khr_fp64 : enable

#define NSPEEDS 9

/* struct to hold the 'speed' values */
typedef struct {
    float speeds[NSPEEDS];
} speed_t;

/* struct to hold the parameter values */
typedef struct {
    int nx;            /* no. of cells in x-direction */
    int ny;            /* no. of cells in y-direction */
    int max_iters;      /* no. of iterations */
    int reynolds_dim;  /* dimension for Reynolds number */
    float density;       /* density per link */
    float accel;         /* density redistribution */
    float omega;         /* relaxation parameter */
} param_t;

typedef enum { ACCEL_ROW=0, ACCEL_COLUMN=1 } accel_e;
typedef struct {
    int col_or_row;
    int idx;
} accel_area_t;

/*
*   TODO
*   Write OpenCL kernels
*/
__kernel void vadd(__global const float *a,
				     __global const float *b,
				     __global       float *c)
 {
     int gid = get_global_id(0);
     c[gid]  = a[gid] + b[gid];
 }

//GREAT RESOURCE FOR UNDERSTANDING GLOBAL/LOCAL IDS
//https://jorudolph.wordpress.com/2012/02/03/opencl-work-item-ids-globalgrouplocal/

//TEST KERNEL
// shared_read_only_small   = [1, 2, 3]
// shared_read_write_1      = [1, 1, 1, 1, 1, ...]
// shared_read_write_2      = [2, 2, 2, 2, 2, ...]
// single_output            = [x]
__kernel void test(__global const int* shared_read_only_small, __global int* shared_read_write_1, __global int* shared_read_write_2, __global int* single_output, __global const param_t* params)
{
  //write 2 to all positions in shared_read_write_1
  //add together shared_read_write_1 and shared_read_write_2 and write to shared_read_write_2
  //add shared_read_only_small elements together into output
  int gid = get_global_id(0);
  shared_read_write_1[gid] = 2;
  shared_read_write_2[gid] = shared_read_write_1[gid] + shared_read_write_2[gid];
  single_output[0] = params->nx;
}


 /* This function is the result of a merge between propagate, rebound, collision and av_velocity.
 ** Please refer to the report.
 ** Code is deliberately non-DRY because not encapsulating the loop body in a function yields a performance increase. */
 __kernel void d2q9bgk(__global const param_t* params, __global speed_t* cells, __global speed_t* tmp_cells, __global speed_t* tmp_tmp_cells, __global char* obstacles, __global float* output)
 {
     int ii,jj,kk;                /* generic counters */
     ii = get_global_id(0);
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

     /* loop over _all_ cells */
     //for (ii = 0; ii < params.ny; ii++)
     //{
         for (jj = 0; jj < params->nx; jj++)
         {
             y_n = (ii + 1) % params->ny;
             x_e = (jj + 1) % params->nx;
             y_s = (ii == 0) ? (ii + params->ny - 1) : (ii - 1);
             x_w = (jj == 0) ? (jj + params->nx - 1) : (jj - 1);

             index = ii*params->nx + jj;

             /* Propagate densities to neighbouring cells, following
             ** appropriate directions of travel.
             ** Now updates *all* of the speed values in the *current* cell
             ** by *reading* from the neighbouring cells, (to facilitate
             ** the merge with rebound-collision-av_velocity step) */
             tmp_cells[index].speeds[0] = cells[index].speeds[0];                //central cell
             tmp_cells[index].speeds[1] = cells[ii *params->nx + x_w].speeds[1];  //east speed from west-side cell
             tmp_cells[index].speeds[2] = cells[y_s*params->nx + jj].speeds[2];   //north speed from south-side cell
             tmp_cells[index].speeds[3] = cells[ii *params->nx + x_e].speeds[3];  //west speed from east-side cell
             tmp_cells[index].speeds[4] = cells[y_n*params->nx + jj].speeds[4];   //south speed from north-side cell
             tmp_cells[index].speeds[5] = cells[y_s*params->nx + x_w].speeds[5];  //north-east speed from south-west-side cell
             tmp_cells[index].speeds[6] = cells[y_s*params->nx + x_e].speeds[6];  //north-west speed from south-east-side cell
             tmp_cells[index].speeds[7] = cells[y_n*params->nx + x_e].speeds[7];  //south-west speed from north-east-side cell
             tmp_cells[index].speeds[8] = cells[y_n*params->nx + x_w].speeds[8];  //south-east speed from north-west-side cell

             /* tmp_tmp_cells[index].speeds now correct, but
             ** cells[index] cannot be written to as original values are
             ** needed in later iterations; therefore a new scratch-space
             ** grid, tmp_tmp_cells, is used. */

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
                         (tmp_cells[index].speeds[kk] + params->omega *
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
      //}

     output[0] = tot_u / (float)tot_cells;
 }
