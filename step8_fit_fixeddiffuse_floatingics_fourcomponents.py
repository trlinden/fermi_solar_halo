import math
import sys
import numpy as np
import os
import healpy as hp
from iminuit import Minuit
import json
from scipy import special
import fermisun 
import scipy.stats

rad2deg = 180.0/math.pi
fs = fermisun.fermisun(sys.argv[5], False) ##Initialize and read the yaml file, second command asks whether to copy ft2 file over to the ramdir or not

#Importing the Maps
counts_data_zipped = np.load("solar_data_npz/solar_data." + str(fs.yaml_starttime) + "." + str(fs.yaml_endtime) + '.' + fs.yaml_flarecut_name + '.npz')
counts_data_all = counts_data_zipped['arr_0']

disk_data_zipped = np.load("background_model/solar_disk." + str(fs.yaml_starttime) + "." + str(fs.yaml_endtime) + '.' + fs.yaml_flarecut_name + '.npz')
disk_data_all = disk_data_zipped['arr_0']

ics_data_zipped = np.load("background_model/ics_model_" + sys.argv[1] + '.' + str(fs.yaml_starttime) + "." + str(fs.yaml_endtime) + '.' + fs.yaml_flarecut_name + '.npz')
ics_data_all = ics_data_zipped['arr_0']

ics_data_zipped2 = np.load("background_model/ics_model_" + sys.argv[2] + '.' + str(fs.yaml_starttime) + "." + str(fs.yaml_endtime) + '.' + fs.yaml_flarecut_name + '.npz') ##positrons
ics_data_all2 = ics_data_zipped2['arr_0']

ics_data_zipped3 = np.load("background_model/ics_model_" + sys.argv[3] + '.' + str(fs.yaml_starttime) + "." + str(fs.yaml_endtime) + '.' + fs.yaml_flarecut_name + '.npz')
ics_data_all3 = ics_data_zipped3['arr_0']

ics_data_zipped4 = np.load("background_model/ics_model_" + sys.argv[4] + '.' + str(fs.yaml_starttime) + "." + str(fs.yaml_endtime) + '.' + fs.yaml_flarecut_name + '.npz') ##positrons
ics_data_all4 = ics_data_zipped4['arr_0']

diffuse_data_zipped = np.load("background_model/background_model." + str(fs.yaml_starttime) + "." + str(fs.yaml_endtime) + '.' + fs.yaml_flarecut_name + '.npz')
diffuse_data_all = diffuse_data_zipped['arr_0']


##There appear to be some edge effect problems (ugh) , set the fit to go up to 44.9 degrees instead of 60 degrees, which should fix things
##Note, the edge effect problems happen becasue the counts map cared whether the photon itself was within 60 degrees, the diffuse map cared whether the center of the pixel was within 45 degrees
##Can stay 0.25 degrees away from the edge (at nside 512) and be safe
for ipix in range(fs.yaml_healpix_numpixels):
	(Txdeg, Tydeg) = fs.convert_pix_to_ang(ipix, fs.yaml_healpix_nside)
	distance = fs.distance_degrees(0.0, 0.0, Tydeg, Txdeg)
	if(distance > fs.yaml_suncut - 0.25): ##stay a quarter of a degree away, which is safer than we had previously programmed TL: 20-June 2023
		for en in range(0, fs.yaml_ebins):
			counts_data_all[en][ipix] = 0
			disk_data_all[en][ipix] = 0
			ics_data_all[en][ipix] = 0
			ics_data_all2[en][ipix] = 0
			ics_data_all3[en][ipix] = 0
			ics_data_all4[en][ipix] = 0
			diffuse_data_all[en][ipix]=1.0e-50 #set this to be slightly non-zero, so that the model prediction is always infantesimally greater than 0, and there are no divide by 0 errors, do this outside of loops


#print("After Asphericity Correction: ", np.sum(ics_data_all), np.sum(ics_data_all2), np.sum(ics_data_all) + np.sum(ics_data_all2))

#print("Difference is: ", np.sum(np.fabs(np.fabs(save_ics_data_all) - np.fabs(ics_data_all) - np.fabs(ics_data_all2))))

##Fitting Function
def fit(cube, countsmap, diskmap, icsmap, icsmap2, icsmap3, icsmap4, diffusemap):
    ##Cube[0] = disknorm
    ##Cube[2] = diffusenorm
	normeddiskmap = cube[0] * diskmap
	normedicsmap  = cube[1] * icsmap
	normedicsmap2  = cube[2] * icsmap2
	normedicsmap3  = cube[3] * icsmap3
	normedicsmap4  = cube[4] * icsmap4
	normeddiffusemap = diffusemap + 1.0e-50 ##prevent divide by 0 errors

	modelmap = normeddiskmap + normedicsmap + normedicsmap2 + normedicsmap3 + normedicsmap4 + normeddiffusemap
	lg_total_probability = scipy.stats.poisson.logpmf(countsmap, modelmap) 

	totalfit = np.sum(lg_total_probability)
	return -1.0*totalfit ##-1.0 because pymultinest maximizes instead of minimizes

def loglikelihood(disknorm, icsnorm, icsnorm2, icsnorm3, icsnorm4):
        cube = [disknorm, icsnorm, icsnorm2, icsnorm3, icsnorm4]
        return fit(cube, counts_data, disk_data, ics_data, ics_data2, ics_data3, ics_data4, diffuse_data)

residual_map = np.zeros([fs.yaml_ebins,fs.yaml_healpix_numpixels])
residual_map_noics = np.zeros([fs.yaml_ebins,fs.yaml_healpix_numpixels])
ics_only_map = np.zeros([fs.yaml_ebins,fs.yaml_healpix_numpixels])


total_fval = 0.0
for energyval in range(0, int(fs.yaml_ebins)): ##100 MeV to 31.6 GeV in 20 bins.. for solar modulation test figures
	counts_data = counts_data_all[energyval]
	disk_data = disk_data_all[energyval]
	ics_data = ics_data_all[energyval]
	ics_data2 = ics_data_all2[energyval]
	ics_data3 = ics_data_all3[energyval]
	ics_data4 = ics_data_all4[energyval]
	diffuse_data = diffuse_data_all[energyval]

	factorialcountsmap = special.factorial(counts_data) ##This is always the same number, save time by only computing it once

	m = Minuit(loglikelihood, disknorm=0, icsnorm=0, icsnorm2=0, icsnorm3=0, icsnorm4=0)

	m.errordef = 0.5
	m.errors["disknorm"] = 1.0
	m.limits["disknorm"] = (0, 2e5)
	m.errors["icsnorm"] = 1.0
	m.limits["icsnorm"] = (0, 2e5)
	m.errors["icsnorm2"] = 1.0
	m.limits["icsnorm2"] = (0, 2e5)
	m.errors["icsnorm3"] = 1.0
	m.limits["icsnorm3"] = (0, 2e5)
	m.errors["icsnorm4"] = 1.0
	m.limits["icsnorm4"] = (0, 2e5)

	m.migrad()

	m.hesse()

	energystep = (np.log10(fs.yaml_emax) - np.log10(fs.yaml_emin)) / fs.yaml_ebins
	energy_in_bin = fs.yaml_emin * np.power(10.0, (energyval+0.5) * energystep) ##0.5 puts us in the middle of the energy bin
	print(energy_in_bin, m.values['disknorm'], m.values['icsnorm'], m.values['icsnorm2'], m.values['icsnorm3'], m.values['icsnorm4'], m.fval) ##updated this energy printing TL 20-June 2023
	total_fval += m.fval
	disknormvalue = float(m.values['disknorm'])
	icsnormvalue = float(m.values['icsnorm'])
	icsnormvalue2 = float(m.values['icsnorm2'])
	icsnormvalue3 = float(m.values['icsnorm3'])
	icsnormvalue4 = float(m.values['icsnorm4'])
        
	residual_map[energyval] = counts_data - disknormvalue * disk_data - icsnormvalue*ics_data - icsnormvalue2*ics_data2 - icsnormvalue3*ics_data3 - icsnormvalue4*ics_data4 - diffuse_data 
	residual_map_noics[energyval] = counts_data - disknormvalue * disk_data - diffuse_data 

print(total_fval, " is the final likelihood value")
np.savez_compressed('residuals/residual_' + fs.yaml_flarecut_name + "_nomoon_fixeddiffuse_floatingics_fourcomponents.npz", residual_map)
np.savez_compressed('residuals/residual_noics_' + fs.yaml_flarecut_name + "_nomoon_fixeddiffuse_floatingics_fourcomponents.npz", residual_map_noics)

outfile = open('residuals/allfits_' + fs.yaml_flarecut_name + "_nomoon_fixeddiffuse_floatingics_fourcomponents.txt", 'a')
print(sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4], total_fval, file=outfile)
outfile.close()
