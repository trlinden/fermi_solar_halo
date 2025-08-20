from mpi4py import MPI
from random import randint
from time import sleep
from astropy.io import fits
import sunpy.coordinates
from sunpy.coordinates import frames
import astropy
import math
import healpy as hp

##Astropy packages to get solar system coordinates
from astropy.coordinates import SkyCoord
from astropy.coordinates import solar_system_ephemeris, ICRS, get_body
from astropy.time import Time
from astropy import units as u

import fermisun ##This has the variables that control how we analyze the Sun
import numpy as np
import os

'''
1.) We need to build an exposure map that has the Sun and Moon Removed
2.) We need to build an exposure map of the solar region in helioprojective coordinates and in healpix
'''

COMM = MPI.COMM_WORLD # the communicator
RANK = COMM.Get_rank() # identifies each node
SIZE = COMM.Get_size() # the number of nodes


def worker(i, in_pixels, in_Tyvals, in_Txvals):
    while True:

        job = COMM.recv(source=0, tag=11)
        if( len(job) < 2):
            break
        elif os.path.isfile('solar_exposure/solar_exposure.' + str(job[0]) + '.' + str(job[1]) + '.npz') or os.path.isfile('exposures_sourcecuts/nexp2.' + str(job[0]) + '.' + str(job[1]) + '.fits') == 0: ##This will be the -1 lines
            if(os.path.isfile('solar_exposure/solar_exposure.' + str(job[0]) + '.' + str(job[1]) + '.npz')):
                print('File: solar_exposure/solar_exposure.' + str(job[0]) + '.' + str(job[1]) + '.npz was found') 
            if(os.path.isfile('exposures_sourcecuts/nexp2.' + str(job[0]) + '.' + str(job[1]) + '.fits') == 0):
                print('File: exposures_sourcecuts/nexp2.' + str(job[0]) + '.' + str(job[1]) + '.fits was not found')
            sleep(0.2) ##This prevents the worker from continually asking for jobs and overloading the manager, at the cost of making the program take a few seconds to exit
            COMM.send(0, dest=0, tag=12)

        else:
            ## Now we have a bunch of setup that runs equivilently on every core

            ##First figure out what the central time is, then figure out where the Sun and Moon are.
            ##Then remove an ROI around them, and save that file as an exposure file
            ##Then also make a new exposure file in heliocentric coordinates. Then save both output files


            starttime = int(job[0])
            endtime = int(job[1])
            suncut = float(job[2])
            mooncut = float(job[3])
            energybins = int(job[4])
            binsz = float(job[5])
            healpix_numpixels = int(job[6])

            avgtime = (job[0] + job[1]) / 2.0

            ##Load the right complete exposure file

            hdulist = fits.open('exposures_sourcecuts/nexp2.' + str(starttime) + '.' + str(endtime) + '.fits')

            scidata = hdulist[0].data
            scidata_full = np.copy(scidata) ##Use this for the helioprojective calculation in a minute

            jdval = fs.getJD(avgtime)
            jd_formatted = Time(jdval, format='jd')
            sunEarthdistance = sunpy.coordinates.sun.earth_distance(jd_formatted)

            moon_pos = astropy.coordinates.get_moon(jd_formatted)
            moon_radeg = moon_pos.ra.degree
            moon_decdeg = moon_pos.dec.degree

            ##Use this info to get the position of the Sun in RA and DEC, for some reason SkyCoord messes up if a coordinate is exactly 0, so use 1e-6
            sun_helio_position = SkyCoord(1.0e-6*u.degree, 1.0e-6*u.degree, frame=frames.Helioprojective, distance=sunEarthdistance, observer="Earth", obstime=jd_formatted)
            sun_radec_position = sun_helio_position.transform_to('precessedgeocentric')
            sun_raval = sun_radec_position.ra.degree
            sun_decval = sun_radec_position.dec.degree

            ##Now go through every position in the file, and calculate its distance form sun_raval, and sun_decval, if it is in border region, subdivide
            for y in range(0, scidata.shape[1]): ##dec coordinate
                for x in range(0, scidata.shape[2]): ##ra coordinate
                    decval = -90.0 + (y+0.5) * binsz
                    raval = 180.0 - (x+0.5) * binsz
                    if(raval < 0.0):
                        raval += 360.0 ##goes from 180 -> 0 , then from 360-180
                    sundist = fs.distance_degrees(sun_decval, sun_raval, decval, raval)
                    if(sundist < suncut - binsz):
                        for en in range(0, scidata.shape[0]): ##Go through every energy bin
                            scidata[en][y][x] = 0.0
                    elif(sundist < suncut + binsz): ##We are in the boundary region
                        sub_divide_dec = 10
                        sub_divide_ra = 10
                        total_points = sub_divide_dec * sub_divide_ra
                        total_points_in = 0.0
                        for suby in range(0, sub_divide_dec):
                            for subx in range(0, sub_divide_ra):
                                sub_decval = -90.0 + y * binsz + (suby + 0.5) / sub_divide_dec * binsz
                                sub_raval = 180.0 - x * binsz + (subx + 0.5) / sub_divide_ra * binsz
                                if(fs.distance_degrees(sun_decval, sun_raval, sub_decval, sub_raval) > suncut):
                                    total_points_in += 1.0
                                ratio_in = 1.0 * total_points_in / total_points
                        for en in range(0, scidata.shape[0]):
                            scidata[en][y][x] = ratio_in * scidata[en][y][x]

                    ##Now do the same thing for the moon
                    moondist = fs.distance_degrees(moon_decdeg, moon_radeg, decval, raval)
                    if(moondist < mooncut - binsz):
                        for en in range(0, scidata.shape[0]):
                            scidata[en][y][x] = 0.0
                    elif(moondist < mooncut + binsz):
                        sub_divide_dec = 10
                        sub_divide_ra = 10
                        total_points_moon = sub_divide_dec * sub_divide_ra
                        total_points_in_moon = 0.0
                        for suby in range(0, sub_divide_dec):
                            for subx in range(0, sub_divide_ra):
                                sub_decval = -90.0 + y * binsz + (suby + 0.5) / sub_divide_dec * binsz
                                sub_raval = 180.0 - x * binsz + (subx + 0.5) / sub_divide_ra * binsz
                                if(fs.distance_degrees(moon_decdeg, moon_radeg, sub_decval, sub_raval) > mooncut):
                                    total_points_in_moon += 1.0
                        ratio_in_moon = 1.0 * total_points_in_moon / total_points_moon
                        for en in range(0, scidata.shape[0]):
                            scidata[en][y][x] = ratio_in_moon * scidata[en][y][x]
            fits.writeto('exposures_nosun_nomoon/nexp2.' + str(starttime) + '.' + str(endtime) + '.fits', scidata)

            ##Now need to go through each point in helioprojective coordinates, get the right file, and then save the exposure data from this
            sub_exposure_array = np.zeros([energybins,healpix_numpixels])

            sub_helio_positions = SkyCoord(in_Txvals*u.degree, in_Tyvals*u.degree, frame=frames.Helioprojective, distance=sunEarthdistance, observer="Earth", obstime=jd_formatted)
            sub_ra_positions = sub_helio_positions.transform_to('precessedgeocentric')

            in_sub_ravals = sub_ra_positions.ra.degree
            in_sub_decvals = sub_ra_positions.dec.degree

            for x in range(0, len(in_sub_ravals)): ##Now we do need to make a loop, but the long part of calculating the transforms is over
                in_sub_pixel = in_pixels[x]
                in_sub_Txval = in_Txvals[x]
                in_sub_Tyval = in_Tyvals[x]
                in_sub_raval = in_sub_ravals[x]
                in_sub_decval = in_sub_decvals[x]

                ##Now get the right pixels for the exposure map
                in_sub_decpix_exp = int( (in_sub_decval+90.0) / binsz)
                if(in_sub_raval <= 180.0):
                    in_sub_rapix_exp = int( (180.0-in_sub_raval) / binsz)
                else:
                    in_sub_rapix_exp = int( (540.0-in_sub_raval) / binsz)

                ##Now go through every energy level and create the counts map and exposure map we want
                for in_sub_energy in range(0, energybins):
                    in_sub_log_average_energy_exposure = math.sqrt(scidata_full[in_sub_energy][in_sub_decpix_exp][in_sub_rapix_exp] * scidata_full[in_sub_energy+1][in_sub_decpix_exp][in_sub_rapix_exp])
                    sub_exposure_array[in_sub_energy][in_sub_pixel] = in_sub_log_average_energy_exposure
            np.savez_compressed('solar_exposure/solar_exposure.' + str(starttime) + '.' + str(endtime) + '.npz', a=sub_exposure_array)
            hdulist.close()
            COMM.send(0, dest=0, tag=12)

def manager(p, n, commands, total_lines, suncut, mooncut, energybins, binsz, healpix_numpixels, in_pixels, in_Tyvals, in_Txvals):
    result = ""
    jobnbr = 0 # current job number, we start with job number 0
    cntrcv = 0 # number of received results
    for i in range(1, p):
        print('sending job', jobnbr, 'of ', n, 'to', i)
        starttime = commands[jobnbr]
        endtime = commands[jobnbr+1]
        print("Sending: ", starttime, endtime)
        array_to_send = [starttime, endtime, suncut, mooncut, energybins, binsz, healpix_numpixels]
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

            if jobnbr > n:
                print('sending -1 to', node)
                COMM.send([-1], dest=node, tag=11)
            else:
                print('sending job', jobnbr, 'of total', n, 'to', node)
                starttime = commands[jobnbr]
                if(jobnbr + 1 > n):
                    endtime = fs.yaml_endtime
                else:
                    endtime = commands[jobnbr+1]
                print("Sending: ", starttime, endtime)
                array_to_send = [starttime, endtime, suncut, mooncut, energybins, binsz, healpix_numpixels]
                COMM.send(array_to_send, dest=node, tag=11)
                jobnbr = jobnbr + 1

def main(timesteps, total_lines, suncut, mooncut, energybins, binsz, healpix_numpixels, in_pixels, in_Tyvals, in_Txvals):
    if(RANK == 0):
        nbr = int(len(timesteps)-1) ##In this case, this will take awhile, so we only need to send one job at a time to a worker, and so the nbr is just the number of timesteps
        manager(SIZE, nbr, timesteps, total_lines, suncut, mooncut, energybins, binsz, healpix_numpixels, in_pixels, in_Tyvals, in_Txvals)
    else:
        worker(RANK, in_pixels, in_Tyvals, in_Txvals)


##Now we do a bunch of prep on the main node
if RANK == 0: ##Only need to do this on the main node
    fs = fermisun.fermisun('input.yaml', False) ##Initialize and read the yaml file. We do not need to copy over the ft2 file here.

    os.system('mkdir ' + str(fs.yaml_wkdir) + '/solar_exposure')
    os.system('mkdir ' + str(fs.yaml_wkdir) + '/exposures_nosun_nomoon')

    ## Now we have a bunch of setup that runs equivilently on every core
    solar_system_ephemeris.set('jpl')

    ##Regions for analysis
    suncut = float(fs.yaml_suncut)
    mooncut = float(fs.yaml_mooncut)
    energybins = int(fs.yaml_ebins)
    binsz = float(fs.yaml_binsz)
    healpix_nside = int(fs.yaml_healpix_nside)
    healpix_numpixels = int(fs.yaml_healpix_numpixels)



    ##Get the timesteps we want to open the files for:
    ##Time steps
    timesteps = np.arange(fs.yaml_starttime, fs.yaml_endtime, fs.yaml_expstep)
    if(timesteps[-1] != fs.yaml_endtime): ##need to add the explicit endtime into the calculation if it is not included
         timesteps = np.append(timesteps, fs.yaml_endtime)
    print(timesteps)

    total_lines = len(timesteps)


    in_pixels = []
    in_Txvals = []
    in_Tyvals = []

    ##We want to convert the pixels near the Sun into healpix and helioprojective coordinates, so we need to know what pixels those are:
    for ipix in range(0, healpix_numpixels):
        (Txdeg, Tydeg) = fs.convert_pix_to_ang(ipix, healpix_nside)
        if(fs.distance_degrees(Tydeg, Txdeg, 0.0, 0.0) < suncut):
            in_pixels += [ipix]
            in_Tyvals += [Tydeg]
            in_Txvals += [Txdeg]
    print("Suncut includes: ", len(in_pixels), "of", healpix_numpixels, "pixels")
    sleep(20) ##This ensures that the workers all manage to complete their calculations before main starts sending them jobs
    ##Once the worker processes get the timesteps, they should know what to do with the rest

if RANK > 0:

    ##We can do the array calculation here too, so we don't need to send it anymore
    fs = fermisun.fermisun('input.yaml', False)

    
    ##Regions for analysis
    timesteps = []
    total_lines = []
    suncut = float(fs.yaml_suncut)
    mooncut = float(fs.yaml_mooncut)
    energybins = int(fs.yaml_ebins)
    binsz = float(fs.yaml_binsz)
    healpix_nside = int(fs.yaml_healpix_nside)
    healpix_numpixels = int(fs.yaml_healpix_numpixels)

    scidata = [] ##These are blank, just so that there isn't a segfault issue when i send it

    in_pixels = []
    in_Txvals = []
    in_Tyvals = []
    ##We want to convert the pixels near the Sun into healpix and helioprojective coordinates, so we need to know what pixels those are:
    for ipix in range(0, healpix_numpixels):
        (Txdeg, Tydeg) = fs.convert_pix_to_ang(ipix, healpix_nside)
        if(fs.distance_degrees(Tydeg, Txdeg, 0.0, 0.0) < suncut):
                in_pixels += [ipix]
                in_Tyvals += [Tydeg]
                in_Txvals += [Txdeg]
    print("Suncut includes: ", len(in_pixels), "of", healpix_numpixels, "pixels")
    outfilename = 'null'

main(timesteps, total_lines, suncut, mooncut, energybins, binsz, healpix_numpixels, in_pixels, in_Tyvals, in_Txvals) ##This is a huge file if it is main, and a blank array for the nodes
