from mpi4py import MPI
from random import randint
from time import sleep
from astropy.io import fits
import sunpy.coordinates
from sunpy.coordinates import frames
import astropy
import math
import healpy as hp
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


def worker(i, timesteps_start_removed, timesteps_end_removed, in_pixels, in_Tyvals, in_Txvals, background_flux, source_mask_dataset):
	while True:
		job = COMM.recv(source=0, tag=11)
		if len(job) < 2: ##These will be the -1 lines
			exit()
		else:
			## Now we have a bunch of setup that runs equivilently on every core
			startline = int(job[0])
			endline = int(job[1])
			model_counts = np.zeros([fs.yaml_ebins, fs.yaml_healpix_numpixels]) ##This is an array that stores the output number of counts predicted over the timeframe we care about
			moon_model = np.zeros([fs.yaml_ebins, fs.yaml_healpix_numpixels]) ##This is an array that stores the output number of counts predicted over the timeframe we care about
			runnum=0
			healpix_pixelsize = 4*math.pi / fs.yaml_healpix_numpixels ##4PI sr on the sky, each pixel is the same
			for x in range(startline, endline):
				fileexists=1
				try:
					exposurefile = np.load('solar_exposure/solar_exposure.' + str(timesteps_start_removed[x]) + '.' + str(timesteps_end_removed[x]) + '.npz')
				except:
					fileexists=0
					print("Missing File: ", 'solar_exposure/solar_exposure.' + str(timesteps_start_removed[x]) + '.' + str(timesteps_end_removed[x]) + '.npz')
					pass
				if(fileexists):
					exposurearray = exposurefile['a']
				
					avg_time = (timesteps_start_removed[x] + timesteps_end_removed[x]) / 2.0 ##We need to get the solar position at the center of this timestep
					jdval = fs.getJD(avg_time)
					jd_formatted = Time(jdval, format='jd')
					sunEarthdistance = sunpy.coordinates.sun.earth_distance(jd_formatted)	
					
					##We need to do the following things - get the Helioprojective coordinate system at this time
					##Then go through all the pixels in helioprojective coordinates, and calculate their r.a./dec coordinates
					##Then take the flux in that r.a./dec multiply it by the solar exposure in the helioprojective frame, and add it to the expected counts
					
					helio_positions = SkyCoord(in_Txvals*u.degree, in_Tyvals*u.degree, frame=frames.Helioprojective, distance=sunEarthdistance, observer="Earth", obstime=jd_formatted)
					ra_positions = helio_positions.transform_to('precessedgeocentric')

					ravals = ra_positions.ra.degree
					decvals = ra_positions.dec.degree


					##Now we loop through all the ra/dec values that are in the helioprojective model, which we can use for both diffuse/moon
					##This first part deals with the diffuse file, which is defined everywhere
					for x in range(0, len(ravals)): ##Now we do need to make a loop, but the long part of calculating the transforms is over
						in_pixel = in_pixels[x]
						in_Txval = in_Txvals[x]
						in_Tyval = in_Tyvals[x]
						raval = ravals[x]
						decval = decvals[x]

						##Now get the right pixels for the flux map -- need to check that the exposure map and the counts map are defined the same way
						decpix = int( (decval+90.0) / fs.yaml_binsz)
						if(raval <= 180.0):
							rapix = int( (180.0-raval) / fs.yaml_binsz)
						else:
							rapix = int( (540.0-raval) / fs.yaml_binsz)

        					##Now go through every energy level and create the counts map and exposure map we want
        					#for energy in range(0, 32):, can do this instead with an array slice
						model_counts[:, in_pixel] += healpix_pixelsize * exposurearray[:, in_pixel] * background_flux[:, decpix, rapix]
			

			COMM.send([model_counts,moon_model, endline-startline], dest=0, tag=12)
			savearray = 0 ##Clear this
					
def manager(p, n, timesteps_start_removed, timesteps_end_removed, in_pixels, in_Tyvals, in_Txvals, background_flux, num_timesteps_per_worker, source_mask_dataset):
	result = ""
	jobnbr = 0 # current job number, we start with job number 0
	cntrcv = 0 # number of received results
	linenumber=0
	numlines = len(timesteps_start_removed)
	total_result = 0.0 ##This is going to be a numpy array that stores all of the information and gets printed out
	total_moon = 0.0 ##This is the moon model data, which we also want to save, but separately
	numlines_received = 0
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
				total_result = np.add(total_result, c[0])
				total_moon = np.add(total_moon, c[1])
				numlines_received += c[2]
				if(cntrcv == n):
					print("Got here!")
					np.savez_compressed('background_model/background_model' + '.' + str(fs.yaml_starttime) + '.' + str(fs.yaml_endtime) + '.' + flarecut_name + '.npz', total_result)
					##Before saving the moon, we want to normalize it to 1 count
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

def main(timesteps_start_removed, timesteps_end_removed, in_pixels, in_Tyvals, in_Txvals, background_flux, source_mask_dataset):
	if(RANK == 0):
		num_timesteps_per_worker = 100
		##Add one to make sure the extra x jobs at the end get completed
		nbr = int(len(timesteps_start_removed)/num_timesteps_per_worker + 1) ##When the worker returns, it has to send back a ton of data, so we want it to do like 1000 steps at a time
		manager(SIZE, nbr, timesteps_start_removed, timesteps_end_removed, in_pixels, in_Tyvals, in_Txvals, background_flux, num_timesteps_per_worker, source_mask_dataset)
	else:
		worker(RANK, timesteps_start_removed, timesteps_end_removed, in_pixels, in_Tyvals, in_Txvals, background_flux, source_mask_dataset)


##We can actually do the prep on every node - it wastes like 20 seconds of processing, but makes sure that everything is on the same page
if RANK >= 0: ##Only need to do this on the main node

	rad2deg = 180.0/math.pi ##1 radian to degrees
	fs = fermisun.fermisun(sys.argv[1], False) ##Initialize and read the yaml file. We do not need to copy over the ft2 file here.
	solar_system_ephemeris.set('jpl')
	
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

	timesteps_start_removed_temp = np.load(sys.argv[2])
	timesteps_end_removed_temp = np.load(sys.argv[3])
	
	timesteps_start_removed = []
	timesteps_end_removed = []
	for x in range(0, len(timesteps_start_removed_temp)):
		if(timesteps_start_removed_temp[x] >= fs.yaml_starttime and timesteps_end_removed_temp[x] <= fs.yaml_endtime):
			timesteps_start_removed.append(timesteps_start_removed_temp[x])
			timesteps_end_removed.append(timesteps_end_removed_temp[x])			
	
	timesteps_start_removed = np.asarray(timesteps_start_removed)
	timesteps_end_removed = np.asarray(timesteps_end_removed)

	total_lines = len(timesteps_start_removed)
	
	pixel_array = []
	in_pixels = []
	in_Txvals = []
	in_Tyvals = []

	for ipix in range(0, healpix_numpixels):
		(Ty, Tx) = hp.pixelfunc.pix2ang(healpix_nside, ipix, nest=False)
		Tydeg = -180.0/math.pi * Ty + 90.0
		Txdeg = -180.0/math.pi * Tx
		if(Tx > math.pi):
			Txdeg = -180.0/math.pi * Tx + 360.0
		if(fs.distance_degrees(Tydeg, Txdeg, 0.0, 0.0) < suncut): ##Switch to 60 degrees for larger analysis
			pixel_array += [[ipix, Tydeg, Txdeg]]
			in_pixels += [ipix]
			in_Tyvals += [Tydeg]
			in_Txvals += [Txdeg]

	##Finally, we need to make the diffuse flux map, which requires opening the exposure and counts files, and dividing them
	background_exposure_map = np.load('exposures_nosun_nomoon/nexp2.' + str(fs.yaml_starttime) + '.' + str(fs.yaml_endtime) + '.' + fs.yaml_flarecut_name +'.npz')
	background_exposure = background_exposure_map['arr_0']
	background_exposure_bin_center = np.sqrt(background_exposure[:-1,:,:]*background_exposure[1:,:,:]) ##Change to logarithmic average of exposure in bin
	background_counts_map = np.load('data_nosun_nomoon/data.' + str(fs.yaml_starttime) + '.' + str(fs.yaml_endtime) + '.' + fs.yaml_flarecut_name +'.npz')
	background_counts = background_counts_map['arr_0']

	angular_size = np.zeros(background_counts.shape) ##We define this, and give every value a term cos(angle), which we divide by to get the relative counts per helioprojective pixel
	print(angular_size.shape)
	#This array should be something like (Ebins, decbins, rabins)
	##This will hold the angular size of each bin in sr, and then we will get the relative size of the bin compared to the helioprojective map to get the answer
	for a in range(0, angular_size.shape[1]):
		angle = -90.0 + (1.0*a+0.5)*fs.yaml_binsz ##want to get the center of the bin
		angular_size[:,a,:] = np.cos(angle/rad2deg) * fs.yaml_binsz ** 2.0 / rad2deg / rad2deg ##This is the size of the pixel in sr, we divide out to get an intrinsic flux
	background_flux = background_counts / (background_exposure_bin_center+1.0e-30) / angular_size ##This is the flux we need for every node


	##We need to add the information regarding the Moon mask -- The moon hasn't been masked yet, and we need to take the flux of the moon and multiply it by the mask fraction to renormalize it
	source_mask_dataset_array = np.load(fs.yaml_source_mask_name)
	source_mask_dataset = source_mask_dataset_array['arr_0']



	if(RANK == 0):
		sleep(10) ##make sure the primary process ends last

##We need to run this same code for each of the four filesets, so we just run main 4 times
main(timesteps_start_removed, timesteps_end_removed, in_pixels, in_Tyvals, in_Txvals, background_flux, source_mask_dataset)
