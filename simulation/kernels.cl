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

__kernel void d2q9bgk(const param_t params, const accel_area_t accel_area, __local float* sums, __global speed_t* cells, __global speed_t* tmp_cells, __global char* obstacles, __global float* tot_u)
{
  int ii,jj,kk;  /* generic counters */
  ii = get_global_id(1);
  jj = get_global_id(0);

  int lid_x = get_local_id(0);
  int lid_y = get_local_id(1);
  int lsz_x = get_local_size(0);
  int lsz_y = get_local_size(1);

  const float c_sq = 1.0/3.0;  /* square of speed of sound */
  const float w0 = 4.0/9.0;    /* weighting factor */
  float w1 = 1.0/9.0;    /* weighting factor */
  float w2 = 1.0/36.0;   /* weighting factor */

  float u_x,u_y;               /* av. velocities in x and y directions */
  float u_sq;                  /* squared velocity */
  float local_density;         /* sum of densities in a particular cell */
  float u[NSPEEDS];            /* directional velocities */
  float d_equ[NSPEEDS];        /* equilibrium densities */

  float tmp[NSPEEDS];

  int x_e,x_w,y_n,y_s;  /* indices of neighbouring cells */

  // Check whether this work item is within the valid range...
  // ...it may not be due to padding the problem size to be a multiple of
  // the work group size! If it is not, the behaviour is the same as an obstacle,
  // i.e. set the sum for the work item to 0. Note that the barriers still need to be
  // executed after this block.


  if(ii >= params.ny || jj >= params.nx)
  {
    sums[lid_y * lsz_x + lid_x] = 0.0;
  } else {
    /* determine indices of axis-direction neighbours
    ** respecting periodic boundary conditions (wrap around) */
    y_n = (ii + 1) % params.ny;
    x_e = (jj + 1) % params.nx;
    y_s = (ii == 0) ? (ii + params.ny - 1) : (ii - 1);
    x_w = (jj == 0) ? (jj + params.nx - 1) : (jj - 1);

    // PROPAGATE STEP
    tmp[0] = cells[ii *params.nx + jj].speeds[0];                //central cell
    tmp[1] = cells[ii *params.nx + x_w].speeds[1];  //east speed from west-side cell
    tmp[2] = cells[y_s*params.nx + jj].speeds[2];   //north speed from south-side cell
    tmp[3] = cells[ii *params.nx + x_e].speeds[3];  //west speed from east-side cell
    tmp[4] = cells[y_n*params.nx + jj].speeds[4];   //south speed from north-side cell
    tmp[5] = cells[y_s*params.nx + x_w].speeds[5];  //north-east speed from south-west-side cell
    tmp[6] = cells[y_s*params.nx + x_e].speeds[6];  //north-west speed from south-east-side cell
    tmp[7] = cells[y_n*params.nx + x_e].speeds[7];  //south-west speed from north-east-side cell
    tmp[8] = cells[y_n*params.nx + x_w].speeds[8];  //south-east speed from north-west-side cell

    /* if the cell contains an obstacle */
    if (obstacles[ii*params.nx + jj])
    {
        /* called after propagate, so taking values from scratch space
        ** mirroring, and writing into main grid */
        tmp_cells[ii*params.nx + jj].speeds[1] = tmp[3];
        tmp_cells[ii*params.nx + jj].speeds[2] = tmp[4];
        tmp_cells[ii*params.nx + jj].speeds[3] = tmp[1];
        tmp_cells[ii*params.nx + jj].speeds[4] = tmp[2];
        tmp_cells[ii*params.nx + jj].speeds[5] = tmp[7];
        tmp_cells[ii*params.nx + jj].speeds[6] = tmp[8];
        tmp_cells[ii*params.nx + jj].speeds[7] = tmp[5];
        tmp_cells[ii*params.nx + jj].speeds[8] = tmp[6];

        // Obstacle here so only add 0 to the av_vels sum
        sums[lid_y * lsz_x + lid_x] = 0.0;
    } else {
      /* compute local density total */
      local_density = 0.0;

      local_density += tmp[0];
      local_density += tmp[1];
      local_density += tmp[2];
      local_density += tmp[3];
      local_density += tmp[4];
      local_density += tmp[5];
      local_density += tmp[6];
      local_density += tmp[7];
      local_density += tmp[8];

      /* compute x velocity component */
      u_x = (tmp[1] +
              tmp[5] +
              tmp[8]
          - (tmp[3] +
              tmp[6] +
              tmp[7]))
          / local_density;

      /* compute y velocity component */
      u_y = (tmp[2] +
              tmp[5] +
              tmp[6]
          - (tmp[4] +
              tmp[7] +
              tmp[8]))
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
      tmp_cells[ii*params.nx + jj].speeds[0] = (tmp[0] + params.omega * (d_equ[0] - tmp[0]));
      tmp_cells[ii*params.nx + jj].speeds[1] = (tmp[1] + params.omega * (d_equ[1] - tmp[1]));
      tmp_cells[ii*params.nx + jj].speeds[2] = (tmp[2] + params.omega * (d_equ[2] - tmp[2]));
      tmp_cells[ii*params.nx + jj].speeds[3] = (tmp[3] + params.omega * (d_equ[3] - tmp[3]));
      tmp_cells[ii*params.nx + jj].speeds[4] = (tmp[4] + params.omega * (d_equ[4] - tmp[4]));
      tmp_cells[ii*params.nx + jj].speeds[5] = (tmp[5] + params.omega * (d_equ[5] - tmp[5]));
      tmp_cells[ii*params.nx + jj].speeds[6] = (tmp[6] + params.omega * (d_equ[6] - tmp[6]));
      tmp_cells[ii*params.nx + jj].speeds[7] = (tmp[7] + params.omega * (d_equ[7] - tmp[7]));
      tmp_cells[ii*params.nx + jj].speeds[8] = (tmp[8] + params.omega * (d_equ[8] - tmp[8]));

      //AV_VELS STEP

      /* local density total */
      local_density = 0.0;
      local_density += tmp_cells[ii*params.nx + jj].speeds[0];
      local_density += tmp_cells[ii*params.nx + jj].speeds[1];
      local_density += tmp_cells[ii*params.nx + jj].speeds[2];
      local_density += tmp_cells[ii*params.nx + jj].speeds[3];
      local_density += tmp_cells[ii*params.nx + jj].speeds[4];
      local_density += tmp_cells[ii*params.nx + jj].speeds[5];
      local_density += tmp_cells[ii*params.nx + jj].speeds[6];
      local_density += tmp_cells[ii*params.nx + jj].speeds[7];
      local_density += tmp_cells[ii*params.nx + jj].speeds[8];

      /* x-component of velocity */
      u_x = (tmp_cells[ii*params.nx + jj].speeds[1] +
              tmp_cells[ii*params.nx + jj].speeds[5] +
              tmp_cells[ii*params.nx + jj].speeds[8]
          - (tmp_cells[ii*params.nx + jj].speeds[3] +
              tmp_cells[ii*params.nx + jj].speeds[6] +
              tmp_cells[ii*params.nx + jj].speeds[7])) /
          local_density;

      /* compute y velocity component */
      u_y = (tmp_cells[ii*params.nx + jj].speeds[2] +
              tmp_cells[ii*params.nx + jj].speeds[5] +
              tmp_cells[ii*params.nx + jj].speeds[6]
          - (tmp_cells[ii*params.nx + jj].speeds[4] +
              tmp_cells[ii*params.nx + jj].speeds[7] +
              tmp_cells[ii*params.nx + jj].speeds[8])) /
          local_density;

      /* accumulate the norm of x- and y- velocity components */
      sums[lid_y * lsz_x + lid_x] = sqrt(u_x*u_x + u_y*u_y);
    }


    // ACCELERATE_FLOW STEP

    // compute weighting factors
    w1 = params.density * params.accel / 9.0;
    w2 = params.density * params.accel / 36.0;

    if (accel_area.col_or_row == ACCEL_COLUMN)
    {
      if(jj == accel_area.idx)
      {
        // if the cell is not occupied and
        // we don't send a density negative
        if (!obstacles[ii*params.nx + jj] &&
        (tmp_cells[ii*params.nx + jj].speeds[4] - w1) > 0.0 &&
        (tmp_cells[ii*params.nx + jj].speeds[7] - w2) > 0.0 &&
        (tmp_cells[ii*params.nx + jj].speeds[8] - w2) > 0.0 )
        {
            // increase 'north-side' densities
            tmp_cells[ii*params.nx + jj].speeds[2] += w1;
            tmp_cells[ii*params.nx + jj].speeds[5] += w2;
            tmp_cells[ii*params.nx + jj].speeds[6] += w2;
            // decrease 'south-side' densities
            tmp_cells[ii*params.nx + jj].speeds[4] -= w1;
            tmp_cells[ii*params.nx + jj].speeds[7] -= w2;
            tmp_cells[ii*params.nx + jj].speeds[8] -= w2;
        }
      }
    }
    else
    {
      if(ii == accel_area.idx)
      {
        // if the cell is not occupied and
        // we don't send a density negative
        if (!obstacles[ii*params.nx + jj] &&
        (tmp_cells[ii*params.nx + jj].speeds[3] - w1) > 0.0 &&
        (tmp_cells[ii*params.nx + jj].speeds[6] - w2) > 0.0 &&
        (tmp_cells[ii*params.nx + jj].speeds[7] - w2) > 0.0 )
        {
            // increase 'east-side' densities
            tmp_cells[ii*params.nx + jj].speeds[1] += w1;
            tmp_cells[ii*params.nx + jj].speeds[5] += w2;
            tmp_cells[ii*params.nx + jj].speeds[8] += w2;
            // decrease 'west-side' densities
            tmp_cells[ii*params.nx + jj].speeds[3] -= w1;
            tmp_cells[ii*params.nx + jj].speeds[6] -= w2;
            tmp_cells[ii*params.nx + jj].speeds[7] -= w2;
        }
      }
    }
  }

  // AV_VELS REDUCTION
  // See: http://web.engr.oregonstate.edu/~mjb/cs575/Handouts/opencl.reduction.2pp.pdf
  int offset;
  int lsz = lsz_x * lsz_y;
  int tnum = lid_y * lsz_x + lid_x;
  int wgnum = get_group_id(1) * get_num_groups(0) + get_group_id(0);

  for(offset = 1; offset < lsz; offset *= 2)
  {
    int mask = 2*offset - 1;
    barrier(CLK_LOCAL_MEM_FENCE);
    if((tnum & mask) == 0)
    {
      sums[tnum] += sums[tnum + offset];
    }
  }

  barrier(CLK_LOCAL_MEM_FENCE);
  if(tnum == 0)
  {
    tot_u[wgnum] = sums[0];
  }

}
