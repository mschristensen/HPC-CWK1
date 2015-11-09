/* utilities for opencl to get device, etc */

#include <stdio.h>
#include <string.h>

#include "lbm.h"

void get_opencl_platforms(cl_platform_id ** platforms, cl_uint * num_platforms)
{
    cl_int err;

    err = clGetPlatformIDs(0, NULL, num_platforms);
    if (CL_SUCCESS != err) DIE("OpenCL error %d getting number of platforms", err);

    *platforms = (cl_platform_id *) calloc(*num_platforms, sizeof(cl_platform_id));
    err = clGetPlatformIDs(*num_platforms, *platforms, NULL);
    if (CL_SUCCESS != err) DIE("OpenCL error %d getting platforms", err);
}

char * get_platform_info(cl_platform_info param_name, cl_platform_id platform)
{
    cl_int err;
    size_t return_size;

    err = clGetPlatformInfo(platform, param_name, 0, NULL, &return_size);
    if (CL_SUCCESS != err) DIE("OpenCL error %d getting return size of platform parameter", err);

    char * return_string = (char *) calloc(return_size, sizeof(char));
    err = clGetPlatformInfo(platform, param_name, return_size, return_string, NULL);
    if (CL_SUCCESS != err) DIE("OpenCL error %d getting platform info", err);

    return return_string;
}

void get_platform_devices(cl_platform_id platform, cl_device_id ** devices, cl_uint * num_devices)
{
    cl_int err;

    err = clGetDeviceIDs(platform, CL_DEVICE_TYPE_ALL, 0, NULL, num_devices);
    if (CL_SUCCESS != err) DIE("OpenCL error %d getting number of devices", err);

    *devices = (cl_device_id *) calloc(*num_devices, sizeof(cl_device_id));
    err = clGetDeviceIDs(platform, CL_DEVICE_TYPE_ALL, *num_devices, *devices, NULL);
    if (CL_SUCCESS != err) DIE("OpenCL error %d getting device IDs", err);
}

char * get_device_info(cl_device_info param_name, cl_device_id device)
{
    cl_int err;

    char * return_string = NULL;
    size_t return_size;

    err = clGetDeviceInfo(device, param_name, 0, NULL, &return_size);
    if (CL_SUCCESS != err) DIE("OpenCL error %d getting return size of device parameter", err);

    switch (param_name)
    {
    case CL_DEVICE_TYPE:
        return_string = (char *) calloc(50, sizeof(char));
        cl_device_type device_type;
        err = clGetDeviceInfo(device, param_name, return_size, &device_type, NULL);

        switch (device_type)
        {
        case CL_DEVICE_TYPE_GPU:
            strcat(return_string, "GPU");
            break;
        case CL_DEVICE_TYPE_CPU:
            strcat(return_string, "CPU");
            break;
        case CL_DEVICE_TYPE_ACCELERATOR:
            strcat(return_string, "ACCELERATOR");
            break;
        default:
            strcat(return_string, "DEFAULT");
            break;
        }
        break;
    case CL_DEVICE_NAME:
        return_string = (char *) calloc(return_size, sizeof(char));
        err = clGetDeviceInfo(device, param_name, return_size, return_string, NULL);
        break;
    default:
        DIE("Other device_info types not implemented\n");
    }

    if (CL_SUCCESS != err) DIE("OpenCL error %d getting device parameter", err);

    return return_string;
}

void print_device_info(cl_device_id device, int device_id)
{
    char * device_name = get_device_info(CL_DEVICE_NAME, device);
    char * device_type = get_device_info(CL_DEVICE_TYPE, device);

    fprintf(stdout, " Device %u: %s (%s)\n", device_id, device_name, device_type);

    free(device_name);
    free(device_type);
}

void list_opencl_platforms(void)
{
    cl_platform_id * platforms = NULL;
    cl_uint num_platforms;
    get_opencl_platforms(&platforms, &num_platforms);

    cl_uint i, d;

    for (i = 0; i < num_platforms; i++)
    {
        cl_uint num_devices = 0;
        cl_device_id * devices = NULL;

        get_platform_devices(platforms[i], &devices, &num_devices);

        char * profile = get_platform_info(CL_PLATFORM_PROFILE, platforms[i]);
        char * version = get_platform_info(CL_PLATFORM_VERSION, platforms[i]);
        char * name = get_platform_info(CL_PLATFORM_NAME, platforms[i]);
        char * vendor = get_platform_info(CL_PLATFORM_VENDOR, platforms[i]);

        fprintf(stdout, "Platform %u: %s - %s (OpenCL profile = %s, version = %s)\n",
            i, vendor, name, profile, version);

        for (d = 0; d < num_devices; d++)
        {
            print_device_info(devices[d], d);
        }

        free(profile);
        free(version);
        free(name);
        free(vendor);

        free(devices);
    }

    free(platforms);

    exit(EXIT_SUCCESS);
}

void opencl_initialise(int device_id, param_t params, accel_area_t accel_area,
    lbm_context_t * lbm_context, speed_t * cells, speed_t * tmp_cells, char * obstacles, int work_group_size_x, int work_group_size_y)
{
    /* get device etc. */
    cl_platform_id * platforms = NULL;
    cl_uint num_platforms;
    get_opencl_platforms(&platforms, &num_platforms);

    cl_device_id * devices = NULL;
    cl_uint total_devices;
    cl_uint i;

    total_devices = 0;

    for (i = 0; i < num_platforms; i++)
    {
        cl_uint num_platform_devices = 0;
        cl_device_id * platform_devices = NULL;

        get_platform_devices(platforms[i], &platform_devices, &num_platform_devices);

        devices = (cl_device_id *) realloc(devices, sizeof(cl_device_id)*(total_devices + num_platform_devices));
        memcpy(&devices[total_devices], platform_devices, num_platform_devices*sizeof(cl_device_id));

        total_devices += num_platform_devices;

        free(platform_devices);
    }

    if (device_id >= (int) total_devices)
    {
        DIE("Asked for device %d but there were only %u available!\n", device_id, total_devices);
    }

    lbm_context->device = devices[device_id];

    free(devices);
    free(platforms);

    fprintf(stdout, "Got OpenCL device:\n");
    print_device_info(lbm_context->device, device_id);

    cl_int err;

    /* create the context and command queue */
    lbm_context->context = clCreateContext(NULL, 1, &lbm_context->device, NULL, NULL, &err);
    if (CL_SUCCESS != err) DIE("OpenCL error %d creating context", err);

    lbm_context->queue = clCreateCommandQueue(lbm_context->context, lbm_context->device, 0, &err);
    if (CL_SUCCESS != err) DIE("OpenCL error %d creating command queue", err);

    fprintf(stdout, "Created OpenCL context\n");

    /* Get kernels from file etc */
    FILE * source_fp;

    #define KERNEL_FILE "kernels.cl"

    source_fp = fopen(KERNEL_FILE, "r");

    if (NULL == source_fp)
    {
        DIE("Unable to open kernel file %s", KERNEL_FILE);
    }

    size_t source_size;
    fseek(source_fp, 0, SEEK_END);
    source_size = ftell(source_fp);

    char * source = (char *) calloc(source_size + 1, sizeof(char));
    fseek(source_fp, 0, SEEK_SET);
    size_t bytes_read = fread(source, 1, source_size, source_fp);

    if (bytes_read != source_size)
    {
        DIE("Expected to read %lu bytes from kernel file, actually read %lu bytes", source_size, bytes_read);
    }

    source[source_size] = '\0';

    lbm_context->program = clCreateProgramWithSource(lbm_context->context, 1, (const char**)&source, NULL, &err);

    free(source);
    fclose(source_fp);

    if (CL_SUCCESS != err) DIE("OpenCL error %d creating program", err);

    fprintf(stdout, "Building program\n");

    err = clBuildProgram(lbm_context->program, 1, &lbm_context->device, NULL, NULL, NULL);

    if (err == CL_BUILD_PROGRAM_FAILURE)
    {
        cl_int build_err;
        size_t log_size;

        build_err = clGetProgramBuildInfo(lbm_context->program, lbm_context->device, CL_PROGRAM_BUILD_LOG, 0, NULL, &log_size);
        if (CL_SUCCESS != build_err) DIE("OpenCL error %d getting size of build log", build_err);

        char * build_log = (char *) calloc(log_size + 1, sizeof(char));
        build_err = clGetProgramBuildInfo(lbm_context->program, lbm_context->device, CL_PROGRAM_BUILD_LOG, log_size, build_log, NULL);
        if (CL_SUCCESS != build_err) DIE("OpenCL error %d getting build log", build_err);

        printf("OpenCL program build log:\n%s\n", build_log);
        free(build_log);
    }

    if (CL_SUCCESS != err) DIE("OpenCL error %d building program", err);

    /*
    *   Allocate memory and create kernels
    */

    #define KERNEL_NUM 1
    lbm_context->kernels = malloc(sizeof(lbm_kernel_t) * KERNEL_NUM);
    lbm_context->kernels[0].kernel = clCreateKernel(lbm_context->program, "d2q9bgk", &err);
    if (CL_SUCCESS != err) DIE("OpenCL error %d creating kernel 0", err);

    // Set OpenCL kernel dimension arguments

    // Default wg size is 32x32, however these can also be specified on the command line
    lbm_context->kernels[0].dimensions.WORK_GROUP_SIZE_X = (work_group_size_x == 0) ? 32 : work_group_size_x;
    lbm_context->kernels[0].dimensions.WORK_GROUP_SIZE_Y = (work_group_size_y == 0) ? 32 : work_group_size_y;

    // Problem size is params.nx by params.ny iff. each is divisible by the corresponding wg size.
    // Otherwise, it is padded to the nearest multiple of the corresponding wg size.
    lbm_context->kernels[0].dimensions.PROBLEM_SIZE_X = ((params.nx % lbm_context->kernels[0].dimensions.WORK_GROUP_SIZE_X) == 0) ? params.nx : (params.nx + lbm_context->kernels[0].dimensions.WORK_GROUP_SIZE_X - (params.nx % lbm_context->kernels[0].dimensions.WORK_GROUP_SIZE_X));
    lbm_context->kernels[0].dimensions.PROBLEM_SIZE_Y = ((params.ny % lbm_context->kernels[0].dimensions.WORK_GROUP_SIZE_Y) == 0) ? params.ny : (params.ny + lbm_context->kernels[0].dimensions.WORK_GROUP_SIZE_Y - (params.ny % lbm_context->kernels[0].dimensions.WORK_GROUP_SIZE_Y));

    // Grid size is PROBLEM_SIZE_X * PROBLEM_SIZE_Y, which may be larger than params.nx * params.ny in the case of padding
    lbm_context->kernels[0].dimensions.GRID_SIZE = lbm_context->kernels[0].dimensions.PROBLEM_SIZE_X * lbm_context->kernels[0].dimensions.PROBLEM_SIZE_Y;

    // Number of work groups is calculated by the problem_size/wg_size
    lbm_context->kernels[0].dimensions.NUM_WORK_GROUPS = (lbm_context->kernels[0].dimensions.GRID_SIZE) /
                                                         ((lbm_context->kernels[0].dimensions.WORK_GROUP_SIZE_X) *
                                                         (lbm_context->kernels[0].dimensions.WORK_GROUP_SIZE_Y));

    // Print out dimension data
    printf("PARAMS.NX: %d\nPARAMS.NY: %d\nPROBLEM_SIZE_X: %d\nPROBLEM_SIZE_Y: %d\nWORK_GROUP_SIZE_X: %d\nWORK_GROUP_SIZE_Y: %d\nNUM_WORK_GROUPS: %d\n",
          params.nx, params.ny,
          lbm_context->kernels[0].dimensions.PROBLEM_SIZE_X, lbm_context->kernels[0].dimensions.PROBLEM_SIZE_Y,
          lbm_context->kernels[0].dimensions.WORK_GROUP_SIZE_X, lbm_context->kernels[0].dimensions.WORK_GROUP_SIZE_Y,
          lbm_context->kernels[0].dimensions.NUM_WORK_GROUPS);

    // Create the buffers. Note that the size of the cells/tmp_cells/obstacles arrays
    // is still the same as originally, i.e. params.nx * params.ny, and therefore does not include the padding.
    // Instead, the kernel itself checks that the current work item is within a valid range, and if not it does not
    // perform any calculation.
    cl_mem d_cells          = clCreateBuffer(lbm_context->context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, sizeof(speed_t) * params.nx * params.ny, cells,     NULL);
    cl_mem d_tmp_cells      = clCreateBuffer(lbm_context->context, CL_MEM_READ_WRITE,                        sizeof(speed_t) * params.nx * params.ny, NULL,      NULL);
    cl_mem d_obstacles      = clCreateBuffer(lbm_context->context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,  sizeof(char) * params.nx * params.ny, obstacles, NULL);
    cl_mem d_tot_u          = clCreateBuffer(lbm_context->context, CL_MEM_WRITE_ONLY,                        sizeof(cl_float)* lbm_context->kernels[0].dimensions.NUM_WORK_GROUPS, NULL,NULL);

    // allocate memory for the kernel 0 args
    lbm_context->kernels[0].args = malloc(sizeof(cl_mem) * 4);
    lbm_context->kernels[0].args[0] = d_cells;
    lbm_context->kernels[0].args[1] = d_tmp_cells;
    lbm_context->kernels[0].args[2] = d_obstacles;
    lbm_context->kernels[0].args[3] = d_tot_u;

    // set kernel 0 args
    err   = clSetKernelArg(lbm_context->kernels[0].kernel, 0, sizeof(param_t), &params);
    err  |= clSetKernelArg(lbm_context->kernels[0].kernel, 1, sizeof(accel_area_t), &accel_area);
    err  |= clSetKernelArg(lbm_context->kernels[0].kernel, 2, sizeof(cl_float) * (lbm_context->kernels[0].dimensions.WORK_GROUP_SIZE_X) * (lbm_context->kernels[0].dimensions.WORK_GROUP_SIZE_Y), NULL);
    err  |= clSetKernelArg(lbm_context->kernels[0].kernel, 3, sizeof(cl_mem), &lbm_context->kernels[0].args[0]);
    err  |= clSetKernelArg(lbm_context->kernels[0].kernel, 4, sizeof(cl_mem), &lbm_context->kernels[0].args[1]);
    err  |= clSetKernelArg(lbm_context->kernels[0].kernel, 5, sizeof(cl_mem), &lbm_context->kernels[0].args[2]);
    err  |= clSetKernelArg(lbm_context->kernels[0].kernel, 6, sizeof(cl_mem), &lbm_context->kernels[0].args[3]);
    if (CL_SUCCESS != err) DIE("OpenCL error %d setting kernel 0 args", err);

    fprintf(stdout, "Finished initialising OpenCL\n");
}

void opencl_finalise(lbm_context_t lbm_context)
{
    clReleaseCommandQueue(lbm_context.queue);
    clReleaseContext(lbm_context.context);
}
