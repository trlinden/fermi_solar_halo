This is a code that calculates the gamma-ray flux from the inverse-Compton Scattering halo that surrounds the Sun, using Fermi-LAT telescope data. It is based on 2505.04625, which you should cite (potentially along with 2012.04654 and 2104.02068, which used earlier versions of the code), if you are using this code.

To run, simply execute the run.sh file on a server with at least 16 cores and at least 64 GB of RAM. The code also requires the fermitools to be installed (e.g., gtselect), along with fermipy, sunpy, healpy. The machine must support mpiexec for multiprocessing. 

The default run executs the input.yaml file -- which runs one week of data on a relatively coarse healpix grid, which means it takes about an hour on the above machine. Based on the very small dataset, it will give you mostly junk as data. You will need to run more data at higher precision to get scientifically useful answers.

If you want to run the full scientific analysis, you can run the file input_fullanalysis.yaml. However, on a 16 core machine, this will take a year or two. The first step, for example, takes about 1 week on 512 cores. However, this should (in theory) execute correctly - though you should change the run.sh files to make mpiexec use more cores.
