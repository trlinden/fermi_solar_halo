#!/bin/bash -l

# This slurm job runs HPL benchmark.

#SBATCH -p cops          # Partition
#SBATCH -N 1           # Number of nodes
#SBATCH -J solar_background_model           # Job name
#SBATCH -t 140:00:00       # Wall time (H:M:S)
#SBATCH -o slurm-%j.out   # Job output file

#module unload openmpi3
#module load openmpi4 ucx mkl

ulimit -s unlimited

conda activate fermipy

##This runs through and makes the hourly counts and exposure maps, that will be used for the analysis
##The python script creates, and then runs a list of gtselect, gtltcube, gtexpcube2 functions for each time step
##This extensively uses a ramdisk, because the gtltcubes are large and need to be made in each every timestep, but is then destroyed, because only the exposure cube and psf file are necessary
##The exposures are stored in a new folder, exposures, and the psfs are stored in a folder named psfs
mpiexec -n 16 python step1_make_maps_async.py

##This takes the data and pre-sorts it into a numpy file - it's just a helper function which makes it so that you don't overload the ram by loading less efficient fits files when you are running on big datasets
##It produces a simple numpy file that has all the info that you need from the fits files, that file is in a new folder called solar_data
python step2_prerun_make_numpy_file.py

##This calculates the position of each photon in helioprojective coordinates, and saves it in into a text file, along with other key information about each photon (like its distance from the moon, energy, time, theta and phi coordinates, etc.)
mpiexec -n 16 python step2_make_helioprojective_models_sourcemask.py

##This is a set of three codes that converts the exposure files that were produced in step1 into helioprojective coordinates, producing both a set of "on" exposures in solar coordinates, and a set of "off exposures" in RA/DEC, but with the Sun removed
##First, we need to make the temporary files "exposures_sourcecuts" which are the cutting of the galactic plane, moon, and nearby bright sources, whcih are common in each model
##Then the on exposures are stored in solar_exposures/ the off exposures are stored in "exposure_nosun_nomoon" which eliminates emission from the Sun and Moon, to produce the astrophysical background

##The first file here creates the sourcecut map, by going through every coordinate in r.a. and dec, and determining whether it is cut (by a nearby source, the galactic plane, or the moon) or partially cut (in which case it calcualtes the fractional exposure)
mpiexec -n 16 python step3_make_exposure_map_cuts_based_on_sourcecuts_async.py input.yaml

##This code applies those cuts to every exposure file generated in step 1, creating exposures_sourcecuts, which will be used for both the on and off part of the analysis
python step3_apply_source_cuts_to_all_exposures.py input.yaml

##This code takes the exposure_sourcecuts files as input, calculates the helioprojective (and moon distance) of each coordinate at each timeslot, moving the "on" exposure into solar_exposure in helioprojective coordinates, and the "off" exposure" into exposures_nosun_nomoon
mpiexec -n 16 python step3_convert_exposures_to_helioprojective_coordinates_and_make_background_model.py

##This is a small helper function that sorts the solar_data into strictly monotonically increasing time. This is useful because the mpi execution in step 2 can make things get out of time -- and it is much faster to do the time binning in the next step if things are strictly increasing time.
sort -g -k 1 solar_data/helioprojective_data.txt > solar_data/helioprojective_data_sorted.txt

##This function bins all of the solar data into different timesteps - saving the result into text files, so they can be made into a map.
python step4_sort_counts_into_timesteps.py 

##This function moves all of the solar data into helioprojective coordinate maps (in healpix format), now that the files are produced.
mpiexec -n 16 python step4_sort_counts_into_helioprojective_coordinates_maps_in_each_timestep_async.py

##This applies the timecuts for the moon position, and the solar disk position and source position into the remaining off and on exposure and data maps (solar_data and solar_exposure are for the "on") and data and exposure are for the "off"
mpiexec -n 16 python step5_apply_timecuts_to_data_nosun_nomoon_async_60degmooncut.py input.yaml extracuts_60degmoon_starttimes.npy extracuts_60degmoon_endtimes.npy 
mpiexec -n 16 python step5_apply_timecuts_to_exposure_nosun_nomoon_async_60degmooncut.py input.yaml extracuts_60degmoon_starttimes.npy extracuts_60degmoon_endtimes.npy 
mpiexec -n 16 python step5_apply_timecuts_to_solar_data_async_60degmooncut.py input.yaml extracuts_60degmoon_starttimes.npy extracuts_60degmoon_endtimes.npy 
mpiexec -n 16 python step5_apply_timecuts_to_solar_exposure_async_60degmooncut.py input.yaml extracuts_60degmoon_starttimes.npy extracuts_60degmoon_endtimes.npy 

wait

##This finally builds the astrophysical diffuse model in helioprojective coordinates- by integrating over every timestep included in our analysis, saving the result in background_models/background_model.starttime.stoptime.cutmodelname.npz
##The result is in a healpix file, where the entries are photon counts (the number of expected counts given the flux in the background model and the helioprojective exposure -- and the size of the data is (energybins, healpix_angular_bins)
mpiexec -n 16 python step6_calculate_diffuse_and_moon_models_in_helioprojective_coordinates_async_60degmooncut.py input.yaml extracuts_60degmoon_starttimes.npy extracuts_60degmoon_endtimes.npy

##This builds the ICS and Disk Models for our analysis - smearing them out over the point spread function of the instrument, and plotting the results in helioprojective coordinates
##The data is again given in counts, with the same binning (energy bins, healpix_angular_bins) as before
##This data is normed to 1 count in each energy bin -- because we are going to be fitting this to the data later
mpiexec -n 16 python step7_calculate_solar_disk_model_async_60degmooncut.py input.yaml extracuts_60degmoon_starttimes.npy extracuts_60degmoon_endtimes.npy &
mpiexec -n 16 python step7_calculate_solar_ics_models_async_60degmooncut.py input.yaml extracuts_60degmoon_starttimes.npy extracuts_60degmoon_endtimes.npy ics_models/list_IC_morphology_files.txt &

wait

##Now we run the specific fitting algorithms to get results for our chosen model
mkdir residuals

#This is the last step that finds the correct normalizations for the physical ICS - based on a standard theoretical model. This multiplies the ICS model above - and gives an expected number of real counts in each energy and angular bin, for models where we fit the data to an exact ICS model
python step8_find_correct_normalizations_for_physical_ICS.py ics_models/theoretical_IC_morphology_electrons_Phi_0.txt input.yaml
python step8_find_correct_normalizations_for_physical_ICS.py ics_models/theoretical_IC_morphology_electrons_Phi_1000.txt input.yaml
python step8_find_correct_normalizations_for_physical_ICS.py ics_models/theoretical_IC_morphology_positrons_Phi_0.txt input.yaml
python step8_find_correct_normalizations_for_physical_ICS.py ics_models/theoretical_IC_morphology_positrons_Phi_1000.txt input.yaml

##This is the likelihood fitting function, it writes the folder residuals/ and shows the total residual of the fit, the residual + true ICS flux (e.g., the ICS map), and calculates the likelihood of the fit. 
##In this case, we import 4 different models for modulation 0MV modulation for electrons, 1000 MV modulation for electrons, and the same two choices for positrons. The normalization of each is allowed to float independently
##This is equivalent to the choices that make our main results for the true ICS flux
python step8_fit_fixeddiffuse_floatingics_fourcomponents.py 0MV_electrons 1000MV_electrons 0MV_positrons 1000MV_positrons input.yaml

