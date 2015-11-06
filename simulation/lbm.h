#ifndef LBM_HDR_FILE
#define LBM_HDR_FILE

#if __APPLE__
#include <OpenCL/cl.h>
#else
#include <CL/cl.h>
#endif

#define NSPEEDS         9

/* Size of box in imaginary 'units */
#define BOX_X_SIZE (100.0)
#define BOX_Y_SIZE (100.0)

/* struct to hold the parameter values */
typedef struct {
    cl_int nx;            /* no. of cells in x-direction */
    cl_int ny;            /* no. of cells in y-direction */
    cl_int max_iters;      /* no. of iterations */
    cl_int reynolds_dim;  /* dimension for Reynolds number */
    cl_float density;       /* density per link */
    cl_float accel;         /* density redistribution */
    cl_float omega;         /* relaxation parameter */
} param_t;

/* obstacle positions */
typedef struct {
    float obs_x_min;
    float obs_x_max;
    float obs_y_min;
    float obs_y_max;
} obstacle_t;

// Custom struct to allow kernels to be described in an lbm_context_t
typedef struct {
    cl_kernel kernel;         //the actual kernel
    cl_mem* args;             //array of kernel args
} lbm_kernel_t;

typedef struct {
    cl_context context;
    cl_device_id device;
    cl_command_queue queue;
    lbm_kernel_t* kernels;    //array of kernels
    cl_program program;
} lbm_context_t;

/* struct to hold the 'speed' values */
typedef struct {
    cl_float speeds[NSPEEDS];
} speed_t;

typedef enum { ACCEL_ROW=0, ACCEL_COLUMN=1 } accel_e;
typedef struct {
    cl_int col_or_row;
    cl_int idx;
} accel_area_t;

/* Parse command line arguments to get filenames */
void parse_args (int argc, char* argv[],
    char** final_state_file, char** av_vels_file, char** param_file, int * device_id);

void initialise(const char* paramfile, accel_area_t * accel_area,
    param_t* params, speed_t** cells_ptr, speed_t** tmp_cells_ptr,
    char** obstacles_ptr, float** av_vels_ptr);

void opencl_initialise(int device_id, param_t params, accel_area_t accel_area,
    lbm_context_t * lbm_context, speed_t * cells, speed_t * tmp_cells, char * obstacles);
void opencl_finalise(lbm_context_t lbm_context);

void list_opencl_platforms(void);

void write_values(const char * final_state_file, const char * av_vels_file,
    const param_t params, speed_t* cells, char* obstacles, float* av_vels);

void finalise(speed_t** cells_ptr, speed_t** tmp_cells_ptr,
    char** obstacles_ptr, float** av_vels_ptr);

float timestep(const param_t params, const accel_area_t accel_area,
    lbm_context_t* lbm_context,
    speed_t** cells_ptr, speed_t** tmp_cells_ptr, char* obstacles);

void accelerate_flow(const param_t params, const accel_area_t accel_area,
    speed_t* cells, char* obstacles);
void propagate(const param_t params, speed_t* cells, speed_t* tmp_cells);
void rebound(const param_t params, speed_t* cells, speed_t* tmp_cells, char* obstacles);
void collision(const param_t params, speed_t* cells, speed_t* tmp_cells, char* obstacles);
double av_velocity(const param_t params, speed_t* cells, char* obstacles);

void setArgs(lbm_context_t* lbm_context,
    speed_t* cells, speed_t* tmp_cells, int GRID_SIZE);
/* Sum all the densities in the grid.
** The total should remain constant from one timestep to the next. */
float total_density(const param_t params, speed_t* cells);

/* calculate Reynolds number */
float calc_reynolds(const param_t params, const float last_av_vel);

/* Exit, printing out formatted string */
#define DIE(...) exit_with_error(__LINE__, __FILE__, __VA_ARGS__)
void exit_with_error(int line, const char* filename, const char* format, ...)
__attribute__ ((format (printf, 3, 4)));

#endif
