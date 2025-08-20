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
This code adds up all the different time slices that are not cut by solar flares, and produces models for the:
	1.) background counts (no sun, no moon)
	2.) background exposure (no sun, no moon)
	3.) solar counts 
	4.) solar exposure
It sets up the last chapter, which will be to calculate the background flux, and multiply it by the solar exposure in background coordinates
which will give us the predicted diffuse contribution
'''

COMM = MPI.COMM_WORLD # the communicator
RANK = COMM.Get_rank() # identifies each node
SIZE = COMM.Get_size() # the number of nodes


def worker(i, timesteps_start_removed, timesteps_end_removed, file_to_add, filetype, npz_array_location):
	while True:
		job = COMM.recv(source=0, tag=11)
		if len(job) < 2: ##These will be the -1 lines
			exit()
		else:
			## Now we ddhave a bunch of setup that runs equivilently on every core
			startline = int(job[0])
			endline = int(job[1])
			runnum=0
			if(filetype == '.fits'):
				savearray = np.zeros((fs.yaml_ebins+1, int(180.0/fs.yaml_binsz), int(360.0/fs.yaml_binsz))) ##make sure this exists to send back
				for x in range(startline, endline):
					fileexists=1
					try:
						openfile = fits.open(file_to_add + '.' + str(timesteps_start_removed[x]) + '.' + str(timesteps_end_removed[x]) + '.fits')
					except:
						fileexists=0
						pass
					if(fileexists):
						array = np.asarray(openfile[0].data)
						if(x == startline):
							savearray = array
							runnum += 1
						else:
							savearray = np.add(savearray, array)
				COMM.send([savearray,endline-startline], dest=0, tag=12)
				savearray = 0 ##Clear this
					
def manager(p, n, timesteps_start_removed, timesteps_end_removed, file_to_add, filetype, npz_array_location, flarecut_name, num_timesteps_per_worker):
	jobnbr = 0 # current job number, we start with job number 0
	cntrcv = 0 # number of received results
	linenumber=0
	numlines_received = 0
	total_result = 0.0 ##This is going to be a numpy array that stores all of the information and gets printed out
	numlines = len(timesteps_start_removed)
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
				total_result = np.add(total_result, c[0])
				numlines_received += c[1]
				if(cntrcv == n):
					print("Got here!")
					np.savez_compressed(file_to_add + '.' + str(fs.yaml_starttime) + '.' + str(fs.yaml_endtime) + '.' + flarecut_name + '.npz', total_result)
					##Clean up by ending the workers
					for i in range(1, p):
						COMM.send(["kill"], dest=i, tag=11)
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

##This recieves the singular (not array values) of files_to_add, filetypes, and npz_array_locations
def main(timesteps_start_removed, timesteps_end_removed, total_lines, file_to_add, filetype, npz_array_location, flarecut_name):
	if(RANK == 0):
		num_timesteps_per_worker = 20
		##Add one to make sure the extra x jobs at the end get completed
		nbr = int(len(timesteps_start_removed)/num_timesteps_per_worker + 1) ##When the worker returns, it has to send back a ton of data, so we want it to do like 1000 steps at a time
		manager(SIZE, nbr, timesteps_start_removed, timesteps_end_removed, file_to_add, filetype, npz_array_location, flarecut_name, num_timesteps_per_worker)
	else:
		worker(RANK, timesteps_start_removed, timesteps_end_removed, file_to_add, filetype, npz_array_location)


##We can actually do the prep on every node - it wastes like 20 seconds of processing, but makes sure that everything is on the same page
if RANK >= 0: ##Only need to do this on the main node
	fs = fermisun.fermisun(sys.argv[1], False) ##Initialize and read the yaml file. We do not need to copy over the ft2 file here.

	
	files_to_add = 'exposures_nosun_nomoon/nexp2' ##There are four things to add, so we want to run 5 cores
	filetypes = '.fits'
	npz_array_locations = '0'
	
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
	outputfile = open('write_output_file_for_exposure_nosun_nomoon_of_yr2_timesteps.txt', 'w')
	for counter in range(0, len(timesteps_start_removed)):
		print(timesteps_start_removed[counter], timesteps_end_removed[counter], file=outputfile)
	outputfile.close()

	if(RANK == 0):
		sleep(10) ##make sure the primary process ends last

##We need to run this same code for each of the four filesets, so we just run main 4 times
main(timesteps_start_removed, timesteps_end_removed, total_lines, files_to_add, filetypes, npz_array_locations, flarecut_name)
