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
			psf_arrays = np.zeros([fs.yaml_ebins, fs.yaml_psf_thetabins]) ##4500 is the number of bin files in the gtpsf data, should be added as an option at some point
			weighted_psf_array = np.zeros([fs.yaml_ebins, fs.yaml_psf_thetabins]) ##4500 number of bin files in the gtpsf data, should be added as an option at some point
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
			sleep(0.01)

		else:
			node = state.Get_source()
			c = COMM.recv(source=node, tag=12)
			if(len(c) > 1): ##This wasnt just the null sendback
				print("Received: ", cntrcv, " of ", n)
				cntrcv = cntrcv + 1
				psf_arrays = np.add(psf_arrays, c[0])
				total_exposure = np.add(total_exposure, c[1])
				angular_array = np.asarray(c[2]) ##can reset this every time, it is always the same
				numlines_received += c[3]
				if(cntrcv == n):
					##Here there is suddenly quite a bit to do, that can only be done on the master node				
					weighted_psf_array = np.zeros([fs.yaml_ebins, fs.yaml_psf_thetabins])					
					##Angular array actually holds an array of arrays, for some dumb reason, you have to fix this
					fix_angular_array = np.zeros(fs.yaml_psf_thetabins)
					for fixcounter in range(0, len(angular_array)):
						fix_angular_array[fixcounter] = angular_array[fixcounter]
					angular_array = fix_angular_array * math.pi/180.0
					
					for en in range(0, fs.yaml_ebins):
						print(en, total_exposure[en], psf_arrays[en][0], psf_arrays[en][10], psf_arrays[en][20], psf_arrays[en][4000])

					
					##Get the weighted psf array by dividing
					for en in range(0, fs.yaml_ebins):
						weighted_psf_array[en] = psf_arrays[en] / total_exposure[en] ##should be a 32x4500 result

					##Now make the model of the disk without smearing --  Monte Carlo to get the pixel edges correct in healpix format
					num_sun_points=10000
					x=0
					sun_map_no_smearing = np.zeros(fs.yaml_healpix_numpixels)
					while(x < num_sun_points):
						Tx = -angular_size_of_sun_degrees + 2.0 * random.random() * angular_size_of_sun_degrees ##WRONG! : This should have been 2.0, but was set to 1.5, which seems non-sensical? (TL 20-June 2023)
						Ty = -angular_size_of_sun_degrees + 2.0 * random.random() * angular_size_of_sun_degrees
						if(fs.distance_degrees(0.0, 0.0, Ty, Tx) < angular_size_of_sun_degrees): ##on the surface of the Sun
							ipix = fs.convert_ang_to_pix(Tx, Ty, fs.yaml_healpix_nside)
							sun_map_no_smearing[ipix] += 1.0/num_sun_points ##start with this normalized
							x+=1
					
					##Finally smear with the psf we calculated, which is actually pretty fast
					##Calculate the spherical harmonic transform of the psf
					##note that the angular array needs to be in radians and not degrees
					smoothed_map = np.zeros([fs.yaml_ebins, fs.yaml_healpix_numpixels])
					for en in range(0, fs.yaml_ebins):
						print ("Energy: ", en , "completed.")		
						psf_spherical_harmonic = hp.sphtfunc.beam2bl(weighted_psf_array[en], angular_array, 100000) ##Compute b(l) out to 10000 steps (should be 360.0/100000 degrees, which is small?)
						smoothed_map[en] = hp.sphtfunc.smoothing(sun_map_no_smearing, beam_window=psf_spherical_harmonic)

					#The psf is only defined to 45 degrees, there can be numerical issues right at the boundaries
					##For safety, set everything to 0 outside of 35 degrees, it should be 0 there anyway.
					for ipix in range(0, fs.yaml_healpix_numpixels):
						(Txdeg, Tydeg) = fs.convert_pix_to_ang(ipix, fs.yaml_healpix_nside) 
						distance = fs.distance_degrees(0.0, 0.0, Tydeg, Txdeg)
						if(distance > 35.0):
							for en in range(0, fs.yaml_ebins):
								smoothed_map[en][ipix] = 0.0

					for en in range(0, fs.yaml_ebins):
						normalization_constant = np.sum(smoothed_map[en])
						smoothed_map[en] = np.divide(smoothed_map[en], normalization_constant)
					
					##Save all ebins in one file
					np.savez_compressed('background_model/solar_disk.' + str(fs.yaml_starttime) + '.' + str(fs.yaml_endtime) + '.' + fs.yaml_flarecut_name + '.npz', smoothed_map)

					##Clean up by ending the workers
					for i in range(1, p):
						COMM.send(['kill'], dest=i, tag=11)
					return 0
					exit()
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
		num_timesteps_per_worker = 10
		##Add one to make sure the extra x jobs at the end get completed
		nbr = int(len(timesteps_start_removed)/num_timesteps_per_worker + 1) ##When the worker returns, it has to send back a ton of data, so we want it to do like 1000 steps at a time
		manager(SIZE, nbr, timesteps_start_removed, timesteps_end_removed, angular_size_of_sun_degrees, num_timesteps_per_worker)
	else:
		worker(RANK, timesteps_start_removed, timesteps_end_removed, angular_size_of_sun_degrees)


##We can actually do the prep on every node - it wastes like 20 seconds of processing, but makes sure that everything is on the same page
if RANK >= 0: ##Only need to do this on the main node
	fs = fermisun.fermisun(sys.argv[1], False) ##Initialize and read the yaml file. We do not need to copy over the ft2 file here.
	solar_system_ephemeris.set('jpl')
	 ##Angular size of the sun in degrees
	
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
main(timesteps_start_removed, timesteps_end_removed, fs.yaml_angular_size_of_sun_degrees)
