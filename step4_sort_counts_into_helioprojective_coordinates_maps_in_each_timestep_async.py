from mpi4py import MPI
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
1.) We Need to take every photon from the dataset, and calculate its position in helioprojective coordinates
2.) We need to build an exposure map of the Sun in helioprojective coordinates
2.) We need to build an exposure map that has the Sun and Moon Removed
3.) We need to build a counts map of the full sky that has the Sun and Moon Removed
4.) We need to use #2 and #3 to build a flux map of the full sky during periods when the Sun and Moon aren't there.
5.) We need to build disk, ICS, and Moon models in helioprojective coordinates, and then exposure correct them
'''

COMM = MPI.COMM_WORLD # the communicator
RANK = COMM.Get_rank() # identifies each node
SIZE = COMM.Get_size() # the number of nodes
num_lines_per_task = 1000 ##Each processor runs 10000 photons before coming back to ask for more. This is some balance between dividing tasks well, and not making file IO a nightmare


def worker(RANK, timesteps, healpix_nside, emin, emax, ebins, binsz, suncut, mooncut, healpix_numpixels, dec_bins, ra_bin):
	while True:
		job = COMM.recv(source=0, tag=11)
		if len(job) < 2: ##This will be the -1 lines
			sleep(2) ##This prevents the worker from continually asking for jobs and overloading the manager, at the cost of making the program take a few seconds to exit
			break
		else:
			heliodata = np.zeros([ebins,healpix_numpixels])
			backgrounddata = np.zeros([ebins, dec_bins, ra_bins])
			timestep = job[0]
			starttime = str(timesteps[timestep])
			try:
				endtime = str(timesteps[timestep+1])
			except:
				endtime = str(fs.yaml_endtime)
				pass
			if(int(endtime) == int(starttime)):
				sleep(2)
				COMM.send(0, dest=0, tag=12)
				break
			else:
				infile = open('solar_data/counts.' + starttime + '.' + endtime + '.txt', 'r')
				for line in infile.readlines():
					params = line.split()
					time = float(params[0])
					raval = float(params[2])
					decval = float(params[3])
					Txval = float(params[6])
					Tyval =	float(params[7])
					helioprojective_bin = int(params[8])
					energybin = int(params[9])
					lunar_distance = float(params[-1])
				
					if(raval <= 180.0):
						rabin = int( (180.0-raval)/binsz)
					if(raval > 180.0):
						rabin = int((540.0 - raval) / binsz)
					decbin = int(decval/binsz + 90.0/binsz) ##want to go from -90 to 90 in declination
					solar_distance = fs.distance_degrees(Tyval, Txval, 0.0, 0.0)

					if(solar_distance > suncut and lunar_distance > mooncut): ##Need to add the mooncut here, which means we miss some photons
						backgrounddata[energybin, decbin, rabin] += 1
					if(solar_distance  < suncut):
						heliodata[energybin, helioprojective_bin] += 1
				infile.close()	
				np.savez_compressed('solar_data_npz/solar_data.' + str(timesteps[timestep]) + '.' + str(timesteps[timestep+1]) + '.npz', heliodata)
				np.savez_compressed('data_nosun_nomoon/' + 'data.' + str(timesteps[timestep]) + '.' + str(timesteps[timestep+1]) + '.npz', backgrounddata)
				COMM.send(0, dest=0, tag=12)


def manager(p, n, timesteps, healpix_nside, emin, emax, ebins, binsz, suncut, mooncut, healpix_numpixels, dec_bins, ra_bin):
	result = ""
	n=n-1 ##There are one less jobs than timesteps, because we send the final time as well.
	jobnbr = 0 # current job number, we start with job number 0
	cntrcv = 0 # number of received results
	for i in range(1, p):
		print('sending job', jobnbr, 'to', i)
		array_to_send = [jobnbr, 1] ##just need to make this array longer than length 1
		COMM.send(array_to_send, dest=i, tag=11)
		jobnbr = jobnbr + 1
		if jobnbr > n-1: break ##stop at n-1, since we are losing one core

	while cntrcv < n+1:
		state = MPI.Status()
		okay = COMM.Iprobe(source=MPI.ANY_SOURCE, \
		tag=MPI.ANY_TAG, status=state)

		if not okay:
			sleep(0.1)

		else:
			node = state.Get_source()
			c = COMM.recv(source=node, tag=12)
			print("Received: ", cntrcv, " of ", n)
			cntrcv = cntrcv + 1
			print(cntrcv, n+1)
			if jobnbr > n: ##because the timesteps go to the final timestep
				print('sending -1 to', node)
				COMM.send([-1], dest=node, tag=11)
			else:
				print('sending job', jobnbr, 'to', node)
				array_to_send = [jobnbr, 1]
				COMM.send(array_to_send, dest=node, tag=11)
				jobnbr = jobnbr + 1

def main(timesteps, healpix_nside, emin, emax, ebins, binsz, suncut, mooncut, healpix_numpixels, dec_bins, ra_bin):
	if(RANK == 0):
		nbr = len(timesteps) ##We have a number of jobs which is the number of total lines divided by the number of lines we send in a single job
		manager(SIZE, nbr, timesteps, healpix_nside, emin, emax, ebins, binsz, suncut, mooncut, healpix_numpixels, dec_bins, ra_bin)
	else:
		worker(RANK, timesteps, healpix_nside, emin, emax, ebins, binsz, suncut, mooncut, healpix_numpixels, dec_bins, ra_bin)


##Now we do a bunch of prep on the main node
if RANK == 0: ##Only need to do this on the main node
	fs = fermisun.fermisun('input.yaml', False) ##Initialize and read the yaml file. We do not need to copy over the ft2 file here.


	###order is metval, raval, decval, thetaval, phival, zenithval, earthazimuth, eventid, energyval, lval, bval, bitval, frontback
	healpix_nside = int(fs.yaml_healpix_nside)
	emin = float(fs.yaml_emin)
	emax = float(fs.yaml_emax)
	ebins = int(fs.yaml_ebins)
	lgemin = np.log10(emin)
	lgemax = np.log10(emax)


	os.system('mkdir ' + fs.yaml_wkdir + '/data_nosun_nomoon/')
	os.system('mkdir ' + fs.yaml_wkdir + '/solar_data_npz/')

	healpix_nside = int(fs.yaml_healpix_nside)
	emin = float(fs.yaml_emin)
	emax = float(fs.yaml_emax)
	ebins = int(fs.yaml_ebins)
	binsz = float(fs.yaml_binsz) ##This is in RA/DEC, and is the binsize to make a map
	suncut = float(fs.yaml_suncut)
	mooncut = float(fs.yaml_mooncut) ##Need to add in a mooncut to get the diffuse map

	healpix_numpixels = 12 * healpix_nside * healpix_nside ##by definition
	dec_bins = int(180.0/binsz)
	ra_bins = int(360.0/binsz)

	##Get the timesteps we want to open the files for:
	##Time steps
	timesteps = np.arange(fs.yaml_starttime, fs.yaml_endtime, fs.yaml_expstep)
	if(timesteps[-1] != fs.yaml_endtime): ##need to add the explicit endtime into the calculation if it is not included
		 timesteps = np.append(timesteps, fs.yaml_endtime)
	print(timesteps)

	total_lines = len(timesteps)

	sleep(5) ##This makes sure that main exists after the other processes

if RANK > 0:
	scidata = [] ##These are blank, just so that there isn't a segfault issue when i send it
	total_lines = 0
	outfilename = 'null'

	##Compute things locally here as well
	fs = fermisun.fermisun('input.yaml', False) ##Initialize and read the yaml file. We do not need to copy over the ft2 file here.
	healpix_nside = int(fs.yaml_healpix_nside) ##easiest way to get all nodes on the same page with respect to this number
	emin = float(fs.yaml_emin)
	emax = float(fs.yaml_emax)
	ebins = int(fs.yaml_ebins)
	binsz = float(fs.yaml_binsz) ##This is in RA/DEC, and is the binsize to make a map
	suncut = float(fs.yaml_suncut)
	mooncut = float(fs.yaml_mooncut) ##Need to add in a mooncut to get the diffuse map

	healpix_numpixels = 12 * healpix_nside * healpix_nside ##by definition
	dec_bins = int(180.0/binsz)
	ra_bins = int(360.0/binsz)

	timesteps = np.arange(fs.yaml_starttime, fs.yaml_endtime, fs.yaml_expstep)
	if(timesteps[-1] != fs.yaml_endtime): ##need to add the explicit endtime into the calculation if it is not included
		 timesteps = np.append(timesteps, fs.yaml_endtime)
	print(timesteps)


main(timesteps, healpix_nside, emin, emax, ebins, binsz, suncut, mooncut, healpix_numpixels, dec_bins, ra_bins) ##This is a huge file if it is main, and a blank array for the nodes
