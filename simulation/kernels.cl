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
__kernel void propagate(const param_t params, __global speed_t* cells, __global speed_t* tmp_cells)
{
    int ii,jj;            /* generic counters */
    ii = get_global_id(1);
    jj = get_global_id(0);

    int x_e,x_w,y_n,y_s;  /* indices of neighbouring cells */
    /* determine indices of axis-direction neighbours
    ** respecting periodic boundary conditions (wrap around) */
    y_n = (ii + 1) % params.ny;
    x_e = (jj + 1) % params.nx;
    y_s = (ii == 0) ? (ii + params.ny - 1) : (ii - 1);
    x_w = (jj == 0) ? (jj + params.nx - 1) : (jj - 1);
    /* propagate densities to neighbouring cells, following
    ** appropriate directions of travel and writing into
    ** scratch space grid */
    tmp_cells[ii *params.nx + jj].speeds[0]  = cells[ii*params.nx + jj].speeds[0]; /* central cell, */
                                             /* no movement   */
    tmp_cells[ii *params.nx + x_e].speeds[1] = cells[ii*params.nx + jj].speeds[1]; /* east */
    tmp_cells[y_n*params.nx + jj].speeds[2]  = cells[ii*params.nx + jj].speeds[2]; /* north */
    tmp_cells[ii *params.nx + x_w].speeds[3] = cells[ii*params.nx + jj].speeds[3]; /* west */
    tmp_cells[y_s*params.nx + jj].speeds[4]  = cells[ii*params.nx + jj].speeds[4]; /* south */
    tmp_cells[y_n*params.nx + x_e].speeds[5] = cells[ii*params.nx + jj].speeds[5]; /* north-east */
    tmp_cells[y_n*params.nx + x_w].speeds[6] = cells[ii*params.nx + jj].speeds[6]; /* north-west */
    tmp_cells[y_s*params.nx + x_w].speeds[7] = cells[ii*params.nx + jj].speeds[7]; /* south-west */
    tmp_cells[y_s*params.nx + x_e].speeds[8] = cells[ii*params.nx + jj].speeds[8]; /* south-east */
}


__kernel void rebound_collision_av_vels(const param_t params, __global speed_t* cells, __global speed_t* tmp_cells, __global char* obstacles, __global float* tot_u)
{
  int ii,jj,kk;  /* generic counters */
  ii = get_global_id(1);
  jj = get_global_id(0);

  const float c_sq = 1.0/3.0;  /* square of speed of sound */
  const float w0 = 4.0/9.0;    /* weighting factor */
  const float w1 = 1.0/9.0;    /* weighting factor */
  const float w2 = 1.0/36.0;   /* weighting factor */

  float u_x,u_y;               /* av. velocities in x and y directions */
  float u_sq;                  /* squared velocity */
  float local_density;         /* sum of densities in a particular cell */
  float u[NSPEEDS];            /* directional velocities */
  float d_equ[NSPEEDS];        /* equilibrium densities */

  /* if the cell contains an obstacle */
  if (obstacles[ii*params.nx + jj])
  {
      /* called after propagate, so taking values from scratch space
      ** mirroring, and writing into main grid */
      cells[ii*params.nx + jj].speeds[1] = tmp_cells[ii*params.nx + jj].speeds[3];
      cells[ii*params.nx + jj].speeds[2] = tmp_cells[ii*params.nx + jj].speeds[4];
      cells[ii*params.nx + jj].speeds[3] = tmp_cells[ii*params.nx + jj].speeds[1];
      cells[ii*params.nx + jj].speeds[4] = tmp_cells[ii*params.nx + jj].speeds[2];
      cells[ii*params.nx + jj].speeds[5] = tmp_cells[ii*params.nx + jj].speeds[7];
      cells[ii*params.nx + jj].speeds[6] = tmp_cells[ii*params.nx + jj].speeds[8];
      cells[ii*params.nx + jj].speeds[7] = tmp_cells[ii*params.nx + jj].speeds[5];
      cells[ii*params.nx + jj].speeds[8] = tmp_cells[ii*params.nx + jj].speeds[6];

      tot_u[ii*params.nx + jj] = -1.0;
  } else {
    /* compute local density total */
    local_density = 0.0;

    for (kk = 0; kk < NSPEEDS; kk++)
    {
        local_density += tmp_cells[ii*params.nx + jj].speeds[kk];
    }

    /* compute x velocity component */
    u_x = (tmp_cells[ii*params.nx + jj].speeds[1] +
            tmp_cells[ii*params.nx + jj].speeds[5] +
            tmp_cells[ii*params.nx + jj].speeds[8]
        - (tmp_cells[ii*params.nx + jj].speeds[3] +
            tmp_cells[ii*params.nx + jj].speeds[6] +
            tmp_cells[ii*params.nx + jj].speeds[7]))
        / local_density;

    /* compute y velocity component */
    u_y = (tmp_cells[ii*params.nx + jj].speeds[2] +
            tmp_cells[ii*params.nx + jj].speeds[5] +
            tmp_cells[ii*params.nx + jj].speeds[6]
        - (tmp_cells[ii*params.nx + jj].speeds[4] +
            tmp_cells[ii*params.nx + jj].speeds[7] +
            tmp_cells[ii*params.nx + jj].speeds[8]))
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

    /* relaxation step */
    for (kk = 0; kk < NSPEEDS; kk++)
    {
        cells[ii*params.nx + jj].speeds[kk] =
            (tmp_cells[ii*params.nx + jj].speeds[kk] + params.omega *
            (d_equ[kk] - tmp_cells[ii*params.nx + jj].speeds[kk]));
    }

    //AV_VELS STEP

    /* local density total */
    local_density = 0.0;

    for (kk = 0; kk < NSPEEDS; kk++)
    {
        local_density += cells[ii*params.nx + jj].speeds[kk];
    }

    /* x-component of velocity */
    u_x = (cells[ii*params.nx + jj].speeds[1] +
            cells[ii*params.nx + jj].speeds[5] +
            cells[ii*params.nx + jj].speeds[8]
        - (cells[ii*params.nx + jj].speeds[3] +
            cells[ii*params.nx + jj].speeds[6] +
            cells[ii*params.nx + jj].speeds[7])) /
        local_density;

    /* compute y velocity component */
    u_y = (cells[ii*params.nx + jj].speeds[2] +
            cells[ii*params.nx + jj].speeds[5] +
            cells[ii*params.nx + jj].speeds[6]
        - (cells[ii*params.nx + jj].speeds[4] +
            cells[ii*params.nx + jj].speeds[7] +
            cells[ii*params.nx + jj].speeds[8])) /
        local_density;

    /* accumulate the norm of x- and y- velocity components */
    tot_u[ii*params.nx + jj] = sqrt(u_x*u_x + u_y*u_y);
  }
}

__kernel void d2q9bgk(const param_t params, __global speed_t* cells, __global speed_t* tmp_cells, __global speed_t* tmp_tmp_cells, __global char* obstacles, __global float* tot_u)
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

  y_n = (ii + 1) % params.ny;
  x_e = (jj + 1) % params.nx;
  y_s = (ii == 0) ? (ii + params.ny - 1) : (ii - 1);
  x_w = (jj == 0) ? (jj + params.nx - 1) : (jj - 1);

  index = ii*params.nx + jj;

  /* Propagate densities to neighbouring cells, following
  ** appropriate directions of travel.
  ** Now updates *all* of the speed values in the *current* cell
  ** by *reading* from the neighbouring cells, (to facilitate
  ** the merge with rebound-collision-av_velocity step) */
  tmp_cells[index].speeds[0] = cells[index].speeds[0];                //central cell
  tmp_cells[index].speeds[1] = cells[ii *params.nx + x_w].speeds[1];  //east speed from west-side cell
  tmp_cells[index].speeds[2] = cells[y_s*params.nx + jj].speeds[2];   //north speed from south-side cell
  tmp_cells[index].speeds[3] = cells[ii *params.nx + x_e].speeds[3];  //west speed from east-side cell
  tmp_cells[index].speeds[4] = cells[y_n*params.nx + jj].speeds[4];   //south speed from north-side cell
  tmp_cells[index].speeds[5] = cells[y_s*params.nx + x_w].speeds[5];  //north-east speed from south-west-side cell
  tmp_cells[index].speeds[6] = cells[y_s*params.nx + x_e].speeds[6];  //north-west speed from south-east-side cell
  tmp_cells[index].speeds[7] = cells[y_n*params.nx + x_e].speeds[7];  //south-west speed from north-east-side cell
  tmp_cells[index].speeds[8] = cells[y_n*params.nx + x_w].speeds[8];  //south-east speed from north-west-side cell

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

      tot_u[index] = -1.0;
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
      tot_u[index] = sqrt(u_x*u_x + u_y*u_y);
  }
}
