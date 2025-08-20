from random import randint
from time import sleep
from astropy.io import fits
import sunpy.coordinates
import astropy
import math

##Astropy packages to get solar system coordinates
from astropy.coordinates import SkyCoord
from astropy.coordinates import solar_system_ephemeris, ICRS, get_body
from astropy.time import Time
from astropy import units as u

import fermisun ##This has the variables that control how we analyze the Sun
import numpy as np
import os
import healpy as hp

'''
1.) We need to take all the helioprojective data listed in the text file, and place it into healpix maps centered on the sun
2.) The second task is to go through the same file -- find all the photons that are not near the Sun, and add them into a numpy array defined like the exposure files
2.) At present, this job is just done on one core, because dividing up the job into different cores that would likely be working on different files is challenging
3.) Fortunately, we've already computed all of the correct pixel space, so it is really just going through lines and binning the data
'''


fs = fermisun.fermisun('input.yaml', False) ##Initialize and read the yaml file. We do not need to copy over the ft2 file here.

###order is metval, raval, decval, thetaval, phival, zenithval, earthazimuth, eventid, energyval, lval, bval, bitval, frontback
infile = open(fs.yaml_wkdir + '/solar_data/' + 'helioprojective_data_sorted.txt', 'r') ##This is a text file that includes all of the photon data

##Get the timesteps we want to open the files for:
##Time steps
timesteps = np.arange(fs.yaml_starttime, fs.yaml_endtime, fs.yaml_expstep)
if(timesteps[-1] != fs.yaml_endtime): ##need to add the explicit endtime into the calculation if it is not included
     timesteps = np.append(timesteps, fs.yaml_endtime)
print(timesteps)

total_lines = len(timesteps)

##Note, here we take advantage of the fact that the photons are already sorted in MET, so it is going directly up. If this were not true, this code would mess up
timestep_counter=0 ##This is the timestep we are in -- we need to check whether we need to increment this for every photon
photon_counter=0 ##number of photons we have gone through, so we can initialize correctly
total_photon_counter=0 ##number of photons we move through, to see what number we didn't add
outfile = open('solar_data/counts.' + str(timesteps[0]) + '.' + str(timesteps[1]) + '.txt', 'w')
for line in infile.readlines():
    params = line.split()
    time = float(params[0])

    while(time > timesteps[timestep_counter+1]): ##we have moved into the next timebin, in theory we might move multiple timebins, so run this iteratively
        print("Moving to New Timestep. Total photons added is: ", photon_counter)
        outfile.close()
        timestep_counter += 1 ##go to the next time bins
        outfile = open('solar_data/counts.' + str(timesteps[timestep_counter]) + '.' + str(timesteps[timestep_counter+1]) + '.txt', 'w')
    if(time > timesteps[timestep_counter] and time <= timesteps[timestep_counter+1]):
        print(line, file=outfile, end='')
        photon_counter += 1
    total_photon_counter += 1

print("Added ", photon_counter, "photons out of ", total_photon_counter, "available photons.")
