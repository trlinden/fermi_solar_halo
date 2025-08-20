from astropy.io import fits
import os
import math
import healpy as hp

import fermisun ##This has the variables that control how we analyze the Sun
import numpy as np
import os

fs = fermisun.fermisun('input.yaml', False) ##Initialize and read the yaml file. We do not need to copy over the ft2 file here.

os.system('mkdir exposures_sourcecuts')

##Get the timesteps we want to open the files for:
##Time steps
timesteps = np.arange(fs.yaml_starttime, fs.yaml_endtime, fs.yaml_expstep)
if(timesteps[-1] != fs.yaml_endtime): ##need to add the explicit endtime into the calculation if it is not included
     timesteps = np.append(timesteps, fs.yaml_endtime)
print(timesteps)

total_lines = len(timesteps)

##Load the masking map, which is a numpy array
masking_map_zipped = np.load(fs.yaml_source_mask_name)
masking_map = masking_map_zipped['arr_0']

for a in range(0, len(timesteps)-1):
    runline=0
    try:
        exposure_input = fits.open('exposures/nexp2.' + str(timesteps[a]) + '.' + str(timesteps[a+1]) + '.fits')
        runline=1
    except:
        print("File not found " + str(timesteps[a]) + ' ' + str(timesteps[a+1]))
        pass
    if(runline==1):
        exposure_data = np.asarray(exposure_input[0].data)
        exposure_output = exposure_data * masking_map
        fits.writeto('exposures_sourcecuts/nexp2.' + str(timesteps[a]) + '.' + str(timesteps[a+1])+ '.fits', exposure_output)
        print(timesteps[a])
