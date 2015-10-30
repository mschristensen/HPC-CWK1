# MY NOTES

This is branch is for HPC CWK2, using OpenCL.
Program restructuring from CWK1 is carried forward; OpenMP is not.

This private repository is labelled `mygithub`. Only push to this repo, using `git push mygithub master`... do NOT use `origin`.

# HPC Coursework (original markdown):

Base coursework for HPC 2015 class

* Source code for simulation is in the simulation/ folder
* Inputs are in the inputs/ folder
* Results checking scripts are in the check/ folder

## Compiling and running

To compile, go into the simulation folder and type `make`. Editing the values for `CC` and `CFLAGS` in the Makefile can be used to enable different compiler options or use a different compiler. These can also be passed on the command line:

    $ make CFLAGS="-O3 -fopenmp"

Inputs, final state output files, and parameter files are all specified on the command line of the `lbm` executable. eg:

    $ ./lbm -a av_vels.dat -f final_state.dat -p ../inputs/box.params

The options for the program can be found by passing the `-h` flag to the executable.

It should be possible to compile and run the program in one by typing `make run `. By default there are some standard settings in the makefile which control which input files to use and where to output to. To use different input files, this can also be specified on the `make` command line like such:

    $ make run FINAL_STATE_FILE=./other_output_file.dat PARAM_FILE=./my_input_file.params

By default, all of these options will use the 128x128 box input parameter file and the associated reference results.

## Checking results

An automated result checking function is provided that requires you to load Python 2.7 (e.g. module load languages/python-2.7.6). Running `make check` will check the output file (average veclocities and final state) against some reference results. By default, it should look something like this:

    $ make check
    python ../check/check.py --ref-av-vels-file=../check/box.av_vels.dat --ref-final-state-file=../check/box.final_state.dat --av-vels-file=./av_vels.dat --final-state-file=./final_state.dat
    Total difference in av_vels : 5.270812566515E-11
    Biggest difference (at step 1219) : 1.000241556248E-14
      1.595203170657E-02 vs. 1.595203170658E-02 = 6.3e-11%

    Total difference in final_state : 5.962977334129E-11
    Biggest difference (at coord (6,2)) : 1.000588500943E-14
      3.329122639178E-02 vs. 3.329122639179E-02 = 3e-11%

    Both tests passed!

This script takes both the reference results and the results to check (both average velocities and final state). This is also specified in the makefile and can be changed like the other options:

    $ make run PARAM_FILE=./my_input_file.params
    ./lbm -a ./av_vels.dat -f ./final_state.dat -p ./my_input_file.params
    ...
    $ make check REF_FINAL_STATE_FILE=./my_final_state_ref.dat REF_AV_VELS_FILE=./my_av_vels_ref.dat
    python ../check/check.py --ref-av-vels-file=./my_av_vels_ref.dat --ref-final-state-file=./my_final_state_ref.dat --av-vels-file=./av_vels.dat --final-state-file=./final_state.dat
    ...

All the options for this script can be examined by passing the --help flag to it.

    $ python ../check/check.py --help
    usage: check.py [-h] [--tolerance TOLERANCE] --ref-av-vels-file
                    REF_AV_VELS_FILE --ref-final-state-file REF_FINAL_STATE_FILE
    ...

A plot of the final state can also be done by doing `make plot`, which will create a .png file in the directory. _NOTE: This only works with a final state file called final\_state.plt_. If you want to run this on your own computer, you need gnuplot installed.

## Running on BlueCrystal Phase 3

The `make run` command will run the executable locally. When you wish
to submit a job to the queuing system on BlueCrystal, you should use
the job submission script provided.

    $ qsub lbm_job_submit

This will dispatch a job to the queue, which you can monitor using the
`qstat` command:

    $ qstat | grep $USER

When finished, the output from your job will be in a file called
`LBM.out`:

    $ less LBM.out

If you wish to run a different set of input parameters, you should
modify `lbm_job_submit` to update the value assigned to `PARAM_FILE`.

## Checking SAFE submission

Running `make submission` will copy all files needed for submission to
a `submission` sub-directory. It will then check that they build
correctly with a clean environment, which is how the auto-marker will
see your submission.

If you have created additional files that are needed to build the
program, you should add them to the `SUBMIT_FILES` variable inside the
Makefile.

If you are using any modules that are not defaults, or need to set any
additional environment variables for your program to work correctly,
then you should create an `env.sh` file containing a set of `module
load` and `export` commands that will be run before compiling and
running your code.
