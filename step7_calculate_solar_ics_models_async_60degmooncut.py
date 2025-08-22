from mpi4py import MPI
from random import randint
from time import sleep
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

'''
This code calculates the expected astrophysical background in Helioprojective coordinates
	This is calculated as: (astrophysical Background counts) / (astrophysical background exposure) * helioprojective exposure = helioprojective background counts
	The astrophysical background counts and astrophysical background exposure are calculated from models where both the Sun and Moon are removed
	The tricky part is that the correlation between Helioprojective coordinates and astrophysical background coordinates is time-dependent
	The astrophysical background flux is assumed to be constant
'''

COMM = MPI.COMM_WORLD # the communicator
RANK = COMM.Get_rank() # identifies each node
SIZE = COMM.Get_size() # the number of nodes

def distance_degrees(Ty1, Tx1, Ty2, Tx2): ##both input and output in degrees, every node needs access to this
	Ty1 = Ty1*math.pi/180.0
	Tx1 = Tx1*math.pi/180.0
	Ty2 = Ty2*math.pi/180.0
	Tx2 = Tx2*math.pi/180.0

	delta_y = math.fabs(Ty2-Ty1)
	delta_x = math.fabs(Tx2-Tx1)

	aval = math.pow(math.sin(delta_y/2.0), 2.0) + math.cos(Ty1)*math.cos(Ty2) * math.pow(math.sin(delta_x/2.0), 2.0)
	return 180.0/math.pi*2.0 * math.atan2(math.sqrt(aval), math.sqrt(1.0-aval))


def getJD(met): ##Every node needs access to this
	if met > 252460801: met-=1 # 2008 leap second
	if met > 362793601: met-=1 # 2012 leap second
	if met > 457401602: met-=1 # 2015 leap second
	if met > 504921603: met-=1 # 2016 leap second
	return 2454683.15527778 + (met-239557417) / 86400.0 ##convert to julian day


def worker(i, timesteps_start_removed, timesteps_end_removed, angular_size_of_sun_degrees):
	while True:
		job = COMM.recv(source=0, tag=11)
		if len(job) < 2: ##These will be the -1 lines
			if(job[0] == 'kill'):
				return 0
			else:
				sleep(2) ##This prevents the worker from continually asking for jobs and overloading the manager, at the cost of making the program take a few seconds to exit
				COMM.send([0], dest=0, tag=12)
		else:
			startline = int(job[0])
			endline = int(job[1])	
			successful_bins_run = 0
			psf_arrays = np.zeros([fs.yaml_ebins, 4500]) ##4500 is the number of bin files in the gtpsf data, should be added as an option at some point
			weighted_psf_array = np.zeros([fs.yaml_ebins, 4500]) ##4500 number of bin files in the gtpsf data, should be added as an option at some point
			total_exposure = np.zeros(fs.yaml_ebins)
			angular_array = np.zeros(fs.yaml_psf_thetabins)
			for counter in range(startline, endline):
				runtimestep=1
				try:
					psffile = fits.open('psfs/psf.' + str(timesteps_start_removed[counter]) + '.' + str(timesteps_end_removed[counter]) + '.fits')	
				except:
					runtimestep=0
					print("Did not Run Time series: ", str(timesteps_start_removed[counter]), str(timesteps_end_removed[counter])) 
					pass

				if(runtimestep):
					scidata = np.asarray(psffile[1].data)
					if(successful_bins_run == 0): ##Run this only at the first timestep, and make it work even if the first step doesnt exist
						angular_array = np.asarray(psffile[2].data) ##Get the actual angular binning to print out
					for en in range(0, fs.yaml_ebins):
						psf_arrays[en] += scidata[en][1] * scidata[en][2]
						total_exposure[en] += scidata[en][1]
					successful_bins_run += 1

			COMM.send([psf_arrays, total_exposure, angular_array, endline-startline], dest=0, tag=12)
			savearray = 0 ##Clear this
					
def manager(p, n, timesteps_start_removed, timesteps_end_removed, angular_size_of_sun_degrees, num_timesteps_per_worker):
	result = ""
	jobnbr = 0 # current job number, we start with job number 0
	cntrcv = 0 # number of received results
	linenumber=0
	numlines = len(timesteps_start_removed)
	psf_arrays = 0.0 ##This is going to be a numpy array that stores all of the information and gets printed out
	total_exposure= 0.0 ##This is the moon model data, which we also want to save, but separately
	numlines_received = 0
	angular_array = np.zeros(fs.yaml_psf_thetabins)
	for i in range(1, p):
		startline = linenumber
		endline = linenumber + num_timesteps_per_worker
		if(endline > numlines):
			endline = numlines
		array_to_send = [startline, endline]
		print('sending job', jobnbr, 'of ', n, 'to', i, 'which includes lines', startline, 'to', endline)
		COMM.send(array_to_send, dest=i, tag=11)
		jobnbr = jobnbr + 1
		linenumber = endline
		if jobnbr > n-1: break ##stop at n-1, since we are losing one core

	while (1): ##Continuing doing this until things break
		state = MPI.Status()
		okay = COMM.Iprobe(source=MPI.ANY_SOURCE, \
		tag=MPI.ANY_TAG, status=state)

		if not okay:
			sleep(0.1)

		else:
			node = state.Get_source()
			c = COMM.recv(source=node, tag=12)
			if(len(c) > 1): ##This wasnt just the null sendback
				print("Received: ", cntrcv, " of ", n)
				cntrcv = cntrcv + 1
				psf_arrays = np.add(psf_arrays, c[0])
				total_exposure = np.add(total_exposure, c[1])
				#print("This is angular_array", angular_array)
				#if(angular_array == 0): ##just do this once
				angular_array = np.asarray(c[2]) ##can reset this every time, it is always the same
				numlines_received += c[3]
				if(cntrcv == n):
					##Here there is suddenly quite a bit to do, that can only be done on the master node				
					weighted_psf_array = np.zeros([fs.yaml_ebins, 4500])					
					
					##Angular array actually holds an array of arrays, for some dumb reason, you have to fix this
					fix_angular_array = np.zeros(fs.yaml_psf_thetabins)
					for fixcounter in range(0, len(angular_array)):
						fix_angular_array[fixcounter] = angular_array[fixcounter][0]
					angular_array = fix_angular_array * math.pi/180.0
					
					for en in range(0, fs.yaml_ebins):
						print(en, total_exposure[en], psf_arrays[en][0], psf_arrays[en][10], psf_arrays[en][20], psf_arrays[en][4000])

					
					##Get the weighted psf array by dividing
					for en in range(0, fs.yaml_ebins):
						weighted_psf_array[en] = psf_arrays[en] / total_exposure[en] ##should be a 36x4500 result
				
					##Now make a model of the ICS without smearing.
					##Note: From discussions with the Fermi collaboration, the correct way to deal with this is to weight it
					##      by the exposure, and then to smear with the PSF. So we need to grab the ICS model and the exposure
					## 	over the full sky. Multiply those, then smear it with the PSF, and then normalize it

					##We first need to open all of the different ics models that we want, this might be a list with a few
					ics_healpix_maps = []
					ics_infilenames = [] 
					ics_infilelabels = []
					ics_infile_list = open(sys.argv[4], 'r') ##I have temporarily hacked this so it doesn't read the yaml file anymore, and gets input from the command line, so i can run multiple ones at the same time: TL 20250226
					for line in ics_infile_list.readlines():
						params = line.split()
						ics_infilenames += [params[0]]
						ics_infilelabels += [params[1]] ##This is the unique tag we add when we write outputs with the ICS files
					ics_infile_list.close()

					##Now open the fits files and get the angular data
					##We will put these into interpolated functions for fast processing
					##NEW: From Jung-Tsung's recent models, these are now text files, and not fits files, need to change the input
					ics_arrays = []
					for infile in ics_infilenames:
						ics_interpolated_at_energies = [] 
						infile_ics = open(infile, 'r')
						
						angle_vals = [0.0] ##start with a flux of 0 at the origin, which is very close to true
						flux_vals = [np.zeros(36)] ##a 36 element array, for each energy value, at a radius of 0 

						for ics_line in infile_ics.readlines(): ##each line is a specific radius and has 33 components, a defined radius and the flux at 36 energy values
							if(ics_line[0] != '#'): ##Not a commented out line
								flux_vals_at_angle = [] #Reset this every time, so we get 36 things into a single numpy array and then add them together
								ics_params = ics_line.split() ##Split into the 36 choices
								angle_vals += [float(ics_params[0])*180.0/math.pi] ##This is in degrees
								for en in range(0, fs.yaml_ebins): ##Go through our ebins -- do this instead of stopping at the end of the line, since we want it to break if the number of bins isn't right
									flux_vals_at_angle += [float(ics_params[en+1])]
								flux_vals_at_angle = np.asarray(flux_vals_at_angle)
								flux_vals += [flux_vals_at_angle] ##Turn this into a multidimensional array
						flux_vals = np.asarray(flux_vals)
						angle_vals = np.asarray(angle_vals)	
						for en in range(0, fs.yaml_ebins): ##Go through every energy value and make an interpolated function
							interpolated_function = interpolate.interp1d(angle_vals, flux_vals[:,en], kind='linear', bounds_error=True)
							ics_interpolated_at_energies += [interpolated_function] ##This will be 36 interpolator functions, for a single ICS model
						ics_arrays += [ics_interpolated_at_energies] ##This will be the number of ICS models * the number of energy values, and will have an interpolated function for each
					
					##Now we need to figure out every healpix pixel we need, and its distance away from the center
					##This way, we can open all of the maps at once, and then just assign the right weights to the right pixels
					##instead of computing the correlation between healpix pixel and angle multiple times
					##This result is true for both the disk and the ICS
					##HOWEVER: Since we are going through the correct pixel space at this point, it is worthwhile to calculate the ICS maps in healpix coordinates with no smearing
					
					healpix_pixels_to_update = [] ##This is a list of the healpix pixels to update and their distance
					ics_healpix_models = np.zeros([len(ics_arrays), fs.yaml_ebins, fs.yaml_healpix_numpixels]) ##This is a big set of 0s that is a set of 36 healpix maps for every ICS model we are considering, we need to populate it
					for ipix in range(0, fs.yaml_healpix_numpixels):
						if(ipix % 10000 == 0):
							print("Ipix is: ", ipix)
						(Ty, Tx) = hp.pixelfunc.pix2ang(fs.yaml_healpix_nside, ipix, nest=False)
						Tydeg = -180.0/math.pi * Ty + 90.0
						Txdeg = -180.0/math.pi * Tx
						if(Tx > math.pi):
							Txdeg = -180.0/math.pi * Tx + 360.0
						distance = distance_degrees(0.0, 0.0, Tydeg, Txdeg)
						if(distance < fs.yaml_suncut): ##the size of our simulation
							healpix_pixels_to_update += [(ipix, distance)] ##The angular distance in degrees
							for ics_model_num in range(0, len(ics_arrays)):
								for en in range(0, fs.yaml_ebins):
									ics_healpix_models[ics_model_num][en][ipix] = ics_arrays[ics_model_num][en](distance) ##we have changed the interpolator to degrees to match this call
						
					##Now we have all of our ICS maps, but we need to open the Fermi-LAT exposure model and bias them -- the normalization of this doesn't matter because we renormalize to 1 count at the end anyway
					exposure_model_zipped = np.load('solar_exposure/solar_exposure.' + str(fs.yaml_starttime) + '.' + str(fs.yaml_endtime) + '.' + str(fs.yaml_flarecut_name) + '.npz')
					exposure_model = exposure_model_zipped['arr_0'] ##this unzips the exposure model
					
					###This is the model without cuts. We first smear by this, and then use the psf, and then take the ratio
					exposure_model_nomask_zipped = np.load('solar_exposure_nomask/solar_exposure.' + str(fs.yaml_starttime) + '.' + str(fs.yaml_endtime) + '.' + str(fs.yaml_flarecut_name) + '.npz')
					exposure_model_nomask = exposure_model_nomask_zipped['arr_0'] ##this unzips the exposure model

					for ics_model_num in range(0, len(ics_arrays)): ##exposure model is a 36xhealpix_nbins matrix, so we just need to multiply these by each other
					##We don't need the unnormalized version anymore - so we can just save the output back into the original array
						ics_healpix_models[ics_model_num] = np.asarray(ics_healpix_models[ics_model_num]) * exposure_model_nomask ##should both be 36xhealpix_nbins
					
					##Now we are done! We just need to smear this with our smoothing function, and save all the answers						

					##Finally smear with the psf we calculated, which is actually pretty fast
					##Calculate the spherical harmonic transform of the psf
					##note that the angular array needs to be in radians and not degrees
					smoothed_ics_maps = np.zeros([len(ics_arrays), fs.yaml_ebins, fs.yaml_healpix_numpixels])

					for en in range(0, fs.yaml_ebins):
						print ("Energy: ", en , "completed.")		
						psf_spherical_harmonic = hp.sphtfunc.beam2bl(weighted_psf_array[en], angular_array, 100000) ##Compute b(l) out to 10000 steps (should be 360.0/100000 degrees, which is small?)
						for ics_map_num in range(0, len(ics_arrays)):
							smoothed_ics_maps[ics_map_num][en] = hp.sphtfunc.smoothing(ics_healpix_models[ics_map_num][en], beam_window=psf_spherical_harmonic)
					

					##Now we need to take the ratio to get to the correct exposure
					for ics_map_num in range(0, len(ics_arrays)):
						smoothed_ics_maps[ics_map_num] = np.multiply(smoothed_ics_maps[ics_map_num], np.divide(exposure_model, (exposure_model_nomask+1e-30))) ##get the ratio of teh exposure maps to generate the exposure with cuts

					##Finally, we need to go back and still normalize this to one photon
					for en in range(0, fs.yaml_ebins):
						for ics_map_num in range(0, len(ics_arrays)):
							ics_normalization_constant = np.sum(smoothed_ics_maps[ics_map_num][en])
							smoothed_ics_maps[ics_map_num][en] = np.divide(smoothed_ics_maps[ics_map_num][en], ics_normalization_constant) ##Renormalize to one photon

					##Save all ebins in one file
					for ics_map_num in range(0, len(ics_arrays)):
						np.savez_compressed('background_model/ics_model_' + str(ics_infilelabels[ics_map_num]) + '.' + str(fs.yaml_starttime) + '.' + str(fs.yaml_endtime) + '.' + fs.yaml_flarecut_name + '.npz', smoothed_ics_maps[ics_map_num])

					##Clean up by ending the workers
					for i in range(1, p):
						COMM.send(['kill'], dest=i, tag=11)
					return 0
			else:
				print("Received null from ", node)

			if linenumber >= numlines:
				print('sending -1 to', node)
				COMM.send([-1], dest=node, tag=11)
			else:
				startline = linenumber
				endline = linenumber + num_timesteps_per_worker
				if(endline > numlines):
					endline = numlines
				array_to_send = [startline, endline]
				print('sending job', jobnbr, 'of ', n, 'to', node, 'which includes lines', startline, 'to', endline)
				COMM.send(array_to_send, dest=node, tag=11)
				jobnbr = jobnbr + 1
				linenumber = endline

def main(timesteps_start_removed, timesteps_end_removed, angular_size_of_sun_degrees):
	if(RANK == 0):
		num_timesteps_per_worker = 200
		##Add one to make sure the extra x jobs at the end get completed
		nbr = int(len(timesteps_start_removed)/num_timesteps_per_worker + 1) ##When the worker returns, it has to send back a ton of data, so we want it to do like 1000 steps at a time
		manager(SIZE, nbr, timesteps_start_removed, timesteps_end_removed, angular_size_of_sun_degrees, num_timesteps_per_worker)
	else:
		worker(RANK, timesteps_start_removed, timesteps_end_removed, angular_size_of_sun_degrees)


##We can actually do the prep on every node - it wastes like 20 seconds of processing, but makes sure that everything is on the same page
if RANK >= 0: ##Only need to do this on the main node

	rad2deg = 180.0/math.pi ##1 radian to degrees
	fs = fermisun.fermisun(sys.argv[1], False) ##Initialize and read the yaml file. We do not need to copy over the ft2 file here.
	solar_system_ephemeris.set('jpl')
	angular_size_of_sun_degrees = 0.266450226 ##Angular size of the sun in degrees
	
	if(RANK == 0):
		os.system('mkdir background_model')

	ebins = fs.yaml_ebins
	suncut = fs.yaml_suncut
	healpix_nside = fs.yaml_healpix_nside
	healpix_numpixels = fs.yaml_healpix_numpixels
	
	##Time Periods for the Analysis
	flarecut_name = fs.yaml_flarecut_name
	flarecut_filename = fs.yaml_flarecut_filename
	flarecut_buffer = float(fs.yaml_flarecut_buffer)

	timesteps_start_removed = np.load(sys.argv[2])
	timesteps_end_removed = np.load(sys.argv[3])
	
	total_lines = len(timesteps_start_removed)

	if(RANK == 0):
		sleep(10) ##make sure the primary process ends last

##We need to run this same code for each of the four filesets, so we just run main 4 times
main(timesteps_start_removed, timesteps_end_removed, angular_size_of_sun_degrees)
