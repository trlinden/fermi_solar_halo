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


def worker(i):
    while True:
        job = COMM.recv(source=0, tag=11)
        if len(job) < 2: ##This will be the -1 lines
            sleep(1) ##This prevents the worker from continually asking for jobs and overloading the manager, at the cost of making the program take a few seconds to exit
            break
        else:
            result = []
            ## Now we have a bunch of setup that runs equivilently on every core
            for x in range(len(job)): ##Go through every line in the job, which is hopefully a few thousand lines of data
                ##order is metval, raval, decval, thetaval, phival, zenithval, earthazimuth, eventid, energyval, lval, bval, frontback
                if(len(job[x]) < 5):
                    print("This got here: ", job[x])
                metval = job[x][0]
                raval  = job[x][1]
                decval = job[x][2]
                thetaval = job[x][3]
                phival = job[x][4]
                zenithval = job[x][5]
                earthazimuth = job[x][6]
                eventid = job[x][7]
                energyval = job[x][8]
                lval = job[x][9]
                bval = job[x][10]
                frontback = job[x][11]


		##See if we keep the photon or throw it out
                keepphoton = 1
                if(bval < fs.yaml_latitudecut and bval > -fs.yaml_latitudecut):
                    keepphoton = 0
                if(keepphoton):
                    for counter in range(0, len(source_cut_l)):
                        if(fs.distance_degrees(source_cut_b[counter], source_cut_l[counter], bval, lval) < fs.yaml_source_angular_cut):
                            keepphoton = 0
                            break ##We have gotten rid of the photon, don't need to keep trying
                if(keepphoton):
                	#Converted values
                    jdval = fs.getJD(metval)
                    jd_formatted = Time(jdval, format='jd')
                    #sunEarthdistance = sunpy.coordinates.ephemeris.get_sunearth_distance(jd_formatted)
                    sunEarthdistance = sunpy.coordinates.sun.earth_distance(jd_formatted)
                    radec_formatted = SkyCoord(raval*u.degree, decval*u.degree, sunEarthdistance, frame='precessedgeocentric', obstime=jd_formatted, equinox='J2000')
                    helioframe_formatted = sunpy.coordinates.Helioprojective(observer="Earth", obstime=jd_formatted)
                    helioframe_TxTy=radec_formatted.transform_to(helioframe_formatted)

                    ##We also want to calculate what healpix pixel this is in, so that we can produce the right files afterwards
                    healpix_index = fs.convert_ang_to_pix(helioframe_TxTy.Tx.degree, helioframe_TxTy.Ty.deg, healpix_nside)
                    if(np.log10(energyval) > lgemin and np.log10(energyval) < lgemax):
                        energy_bin = int( (np.log10(energyval)-lgemin)/(lgemax-lgemin) * ebins) ##the energy bin we find ourselves in

                        moon_pos = astropy.coordinates.get_moon(jd_formatted)
                        moon_radeg = moon_pos.ra.degree
                        moon_decdeg = moon_pos.dec.degree
                        moondistance = fs.distance_degrees(decval, raval, moon_decdeg, moon_radeg)

                        outstring = str(metval) + " " + str(energyval) + " " + str(raval) + " " + str(decval) + " " + str(lval) + " " + str(bval) + " " + str(helioframe_TxTy.Tx.degree) + " " + str(helioframe_TxTy.Ty.degree) + " " + str(healpix_index) + " " + str(energy_bin) + " " + str(thetaval) + " " + str(phival) + " " + str(zenithval) + " " + str(earthazimuth) + " " + str(eventid) + " " + str(frontback) + " " + str(moon_radeg) + " " + str(moon_decdeg) + " " + str(moondistance)
                        result += [outstring]

        COMM.send(result, dest=0, tag=12)


def manager(p, n, commands, total_lines, outfilename, healpix_nside):
    result = ""
    jobnbr = 0 # current job number, we start with job number 0
    cntrcv = 0 # number of received results
    output_file = open(outfilename, 'w')
    print("Here")
    for i in range(1, p):
        print('sending job', jobnbr, 'to', i)
        minline = int(jobnbr*num_lines_per_task)
        maxline = int((jobnbr+1)*num_lines_per_task)
        if(maxline > total_lines):
            maxline = total_lines ##Don't go past the number of actual lines in scidata
        print("Sending: ", minline, maxline)
        array_to_send = commands[minline:maxline]
        COMM.send(array_to_send, dest=i, tag=11)
        jobnbr = jobnbr + 1
        if jobnbr > n-1: break ##stop at n-1, since we are losing one core

    while cntrcv < n+1:
        state = MPI.Status()
        okay = COMM.Iprobe(source=MPI.ANY_SOURCE, \
        tag=MPI.ANY_TAG, status=state)

        if not okay:
            sleep(0.2)

        else:
            node = state.Get_source()
            c = COMM.recv(source=node, tag=12)
            print("Received: ", cntrcv, " of ", n)
            for x in range(0, len(c)):
                output_file.write(c[x] + "\n")
            cntrcv = cntrcv + 1

            if jobnbr > n:
                print('sending -1 to', node)
                COMM.send([-1], dest=node, tag=11)
            else:
                print('sending job', jobnbr, 'to', node)
                minline = int(jobnbr*num_lines_per_task)
                maxline = int((jobnbr+1)*num_lines_per_task)
                if(maxline > total_lines):
                    maxline = total_lines ##Don't go past the number of actual lines in scidata
                array_to_send = commands[minline:maxline]
                COMM.send(array_to_send, dest=node, tag=11)
                jobnbr = jobnbr + 1

def main(commands, total_lines, outfilename, healpix_nside):
    if(RANK == 0):
        nbr = int(len(commands)/num_lines_per_task) ##We have a number of jobs which is the number of total lines divided by the number of lines we send in a single job
        manager(SIZE, nbr, commands, total_lines, outfilename, healpix_nside)
    else:
        worker(RANK)


##Now we do a bunch of prep on the main node
if RANK == 0: ##Only need to do this on the main node
    fs = fermisun.fermisun('input.yaml', False) ##Initialize and read the yaml file. We do not need to copy over the ft2 file here.

    ## Now we have a bunch of setup that runs equivilently on every core
    solar_system_ephemeris.set('jpl')

    ###order is metval, raval, decval, thetaval, phival, zenithval, earthazimuth, eventid, energyval, lval, bval, bitval, frontback
    scidata = np.load('solar_data/' + fs.yaml_ft1_numpy_name) ##This is a numpy array that includes the data that we need -- which can be sent to the relevant nodes much quicker
    total_lines = len(scidata)
    outfilename = str(fs.yaml_wkdir + '/solar_data/' + 'helioprojective_data.txt')
    healpix_nside = int(fs.yaml_healpix_nside)
    emin = float(fs.yaml_emin)
    emax = float(fs.yaml_emax)
    ebins = float(fs.yaml_ebins)
    lgemin = np.log10(emin)
    lgemax = np.log10(emax)


    ##Get the list of sources that we are going to cut at some angular spacing
    infile = open(fs.yaml_source_cut_filename, 'r')
    source_cut_l = []
    source_cut_b = []

    for line in infile.readlines():
    	if(line[0] != '#'):
    		params = line.split()
    		source_cut_l += [float(params[3])]
    		source_cut_b += [float(params[4])]	
    
    sleep(5) ##This makes sure that main exists after the other processes

if RANK > 0:
    scidata = [] ##These are blank, just so that there isn't a segfault issue when i send it
    total_lines = 0
    outfilename = 'null'

    fs = fermisun.fermisun('input.yaml', False) ##Initialize and read the yaml file. We do not need to copy over the ft2 file here.
    healpix_nside = int(fs.yaml_healpix_nside) ##easiest way to get all nodes on the same page with respect to this number
    emin = float(fs.yaml_emin)
    emax = float(fs.yaml_emax)
    ebins = float(fs.yaml_ebins)
    lgemin = np.log10(emin)
    lgemax = np.log10(emax)

    infile = open(fs.yaml_source_cut_filename, 'r')
    source_cut_l = []
    source_cut_b = []

    for line in infile.readlines():
    	if(line[0] != '#'):
    		params = line.split()
    		source_cut_l += [float(params[3])]
    		source_cut_b += [float(params[4])]	
    infile.close()
 
main(scidata, total_lines, outfilename, healpix_nside) ##This is a huge file if it is main, and a blank array for the nodes
