###NOTE: The angle distribution of the input ics file is text coded in, so that the first line is a 0 angle, and then the lines after that start at 0.2 degrees and increment by 0.05 degrees
###      If this is wrong, you need to change the file 

from astropy.io import fits
import sunpy.coordinates
from sunpy.coordinates import frames
import astropy
import math
import healpy as hp
import random
import sys

from scipy import interpolate

##Astropy packages to get solar system coordinates
from astropy.coordinates import SkyCoord
from astropy.coordinates import solar_system_ephemeris, ICRS, get_body
from astropy.time import Time
from astropy import units as u

import fermisun ##This has the variables that control how we analyze the Sun
import numpy as np
import os

fs = fermisun.fermisun(sys.argv[2], False)

exposure_zipped = np.load("solar_exposure/solar_exposure." + str(fs.yaml_starttime) + "." + str(fs.yaml_endtime) + "." + str(fs.yaml_flarecut_name) + ".npz")
exposure = exposure_zipped['arr_0']
print(exposure.shape)
num_str = 4.0*math.pi / exposure.shape[1] ##This is the number of steradians in each pixel

infile = open(sys.argv[1], 'r')
angle = []
fluxes = []
rad2deg = 180.0/math.pi

for line in infile.readlines():
    params = line.split()
    if(line[0] != '#'):
        if(float(params[0]) * rad2deg > 0.15): #Then we care about this bin
            angle += [float(params[0]) * rad2deg] ##Get this in degrees
            fluxes += [[float(params[1]), float(params[2]), float(params[3]), float(params[4]), float(params[5]), float(params[6]), float(params[7]), float(params[8]), float(params[9]), float(params[10]), float(params[11]), float(params[12]), float(params[13]), float(params[14]), float(params[15]), float(params[16]), float(params[17]), float(params[18]), float(params[19]), float(params[20]), float(params[21]), float(params[22]), float(params[23]), float(params[24]), float(params[25]), float(params[26]), float(params[27]), float(params[28]), float(params[29]), float(params[30]), float(params[31]), float(params[32]), float(params[33]), float(params[34]), float(params[35]), float(params[36])]]
infile.close()

#The angles work as 0.0, 0.2, 0.25, 0.3......

##Now calculate the number of expected events 
## Go through each pixel in the exposure map, figure out its degrees, get the correct angle from the fluxes, and then calculate the expected number of events

counts_at_energies = np.zeros(fs.yaml_ebins) ##36 energy bins here

for i in range(0, exposure.shape[1]):
    (Txdeg, Tydeg) = fs.convert_pix_to_ang(i, 512)
    distance = fs.distance_degrees(0.0, 0.0, Tydeg, Txdeg)
    if(distance > 0.2 and distance <  fs.yaml_suncut): ##this is the range we care about
        for en in range(0, 36):
            emin = 0.01 * math.pow(10.0, 0.5) * math.pow(10.0, en/8.0)
            emax = 0.01 * math.pow(10.0, 0.5) * math.pow(10.0, en/8.0) * math.pow(10.0, 1.0/8.0)
            angle_index = round((distance - 0.2)/0.05)
            counts_at_energies[en] += fluxes[angle_index][en] * exposure[en][i] * num_str * (emax-emin)

for x in range(0, fs.yaml_ebins):
    emin = 0.01 * math.pow(10.0, 0.5) * math.pow(10.0, x/8.0)
    emax = 0.01 * math.pow(10.0, 0.5) * math.pow(10.0, x/8.0) * math.pow(10.0, 1.0/8.0)
    print(x, emin, emax, counts_at_energies[x])

