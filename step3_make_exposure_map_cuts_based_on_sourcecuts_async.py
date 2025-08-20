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


COMM = MPI.COMM_WORLD # the communicator
RANK = COMM.Get_rank() # identifies each node
SIZE = COMM.Get_size() # the number of nodes

def worker(i):
    while True:
        job = COMM.recv(source=0, tag=11)
        if(len(job) == 1): ##Then this is a 0 line
            sleep(3) 
            if(job[0] == "kill"):
                  exit()
            COMM.send([-1], dest=0, tag=12)
        else:
            y = job[0]
            x = job[1]
            relative_exposure = 1.0
            decval = -90.0 + (y+0.5) * fs.yaml_binsz
            raval = 180.0 - (x+0.5) * fs.yaml_binsz
            if(raval < 0.0):
                raval += 360.0 ##goes from 180 -> 0 , then from 360-180

            ##First impose the galactic cut
            exp_coord = SkyCoord(raval * u.deg, decval * u.deg, frame='gcrs')
            kill_sources = 0 ##These are the sources that are less than fs.yaml_source_angular_cut - fs.yaml_binsz, this sets the pixel to 0 and kills the loop
            critical_sources = [] ##These are the sources that are between fs.yaml_source_angular_cut - fs.yaml_binsz and fs.yaml_source_angular_cut + fs.yaml_binsz
            if(math.fabs(exp_coord.galactic.b.deg) < fs.yaml_latitudecut - fs.yaml_binsz): ##Note, latitudecut needs to be a multiple of the binsz, or else there will be edge effects
                kill_sources = 1
                relative_exposure = 0.0       
            ##We also need to deal with pixels in the edge region -- but this has to be done simultaneously with sources, to deal with correlations, so we do it later
            if(kill_sources == 0): ##We didn't get rid of the flux for the galactic plane cut
                for counter in range(0, len(source_cut_ra)):
                    sourcedist = fs.distance_degrees(source_cut_dec[counter], source_cut_ra[counter], decval, raval)
                    if(sourcedist < fs.yaml_source_angular_cut - fs.yaml_binsz):
                        kill_sources=1
                        relative_exposure = 0.0       
                        break
                    elif(sourcedist < fs.yaml_source_angular_cut + fs.yaml_binsz):
                        critical_sources += [counter] ##This is a list of all of the sources that are critical for this pixel
                if(kill_sources == 0 and len(critical_sources) > 0 or math.fabs(exp_coord.galactic.b.deg) < fs.yaml_latitudecut + fs.yaml_binsz): ##We haven't killed the pixel, but there are critical ones
                    sub_divide_dec = 10 ##Want to do a 50x50 map here - this has been changed from 100x100 for the test version
                    sub_divide_ra = 10
                    total_points = 0.0 ##We need to get the number of total points with the weight from the declination of the point correct
                    total_points_in = 0.0
                    for suby in range(0, sub_divide_dec+1): ##Need to get all of the corners, to avoid edge effects
                        for subx in range(0, sub_divide_ra+1):
                            sub_decval = decval - fs.yaml_binsz/2.0 + suby / sub_divide_dec * fs.yaml_binsz
                            sub_raval = raval - fs.yaml_binsz/2.0 + subx / sub_divide_ra * fs.yaml_binsz
                            test_latitude_portion=0
                            if(math.fabs(exp_coord.galactic.b.deg) < fs.yaml_latitudecut + fs.yaml_binsz):
                                sub_exp_coord = SkyCoord(sub_raval * u.deg, sub_decval * u.deg, frame='gcrs')
                                test_latitude_portion = 1
                            solid_angle_of_point = math.cos(sub_decval/fs.yaml_rad2deg) ##Get the solid angle of this point
                            total_points += solid_angle_of_point
                            kill_point_in_loop = 0 
                            for counter in range(0, len(critical_sources)): ##Only run on the critical sources
                                if(fs.distance_degrees(source_cut_dec[critical_sources[counter]], source_cut_ra[critical_sources[counter]], sub_decval, sub_raval) < fs.yaml_source_angular_cut):
                                    kill_point_in_loop = 1
                            if(test_latitude_portion == 1 and np.fabs(sub_exp_coord.galactic.b.deg) < fs.yaml_latitudecut): ##We are also outside the galactic plane mask
                                kill_point_in_loop = 1
                            if(kill_point_in_loop == 0):
                                total_points_in += solid_angle_of_point
                    relative_exposure = 1.0 * total_points_in / total_points
            COMM.send([y,x,relative_exposure], dest=0, tag=12)

def manager(p, n, scidata):
    jobnbr = 0 # current job number, we start with job number 0
    cntrcv = 0 # number of received results
    linenumber=0
    numlines_received = 0
    total_result = 0.0 ##This is going to be a numpy array that stores all of the information and gets printed out
    num_jobs = scidata.shape[1] * scidata.shape[2] ##The number of jobs we have
    xtracker = 0
    ytracker = 0
    jobnbr=0
    for i in range(1, p):
        array_to_send = [ytracker, xtracker]
        xtracker += 1
        if(xtracker == scidata.shape[2]): #We have overshot xtracker
            xtracker = 0
            ytracker += 1 ##Keep going
        print('sending job', ytracker, 'and ', xtracker, 'to', i)
        COMM.send([ytracker, xtracker], dest=i, tag=11)
        jobnbr = jobnbr + 1
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
            print(c)
            if(len(c) > 2): ##This wasnt just the null sendback
                print("Received: ", cntrcv, " of ", n)
                cntrcv = cntrcv + 1
                for en in range(0, fs.yaml_ebins+1):
                    scidata[en][c[0]][c[1]] = float(c[2])
                if(cntrcv == n-1):
                    np.savez_compressed(fs.yaml_source_mask_name, scidata)
                    ##Clean up by ending the workers
                    for i in range(1, p):
                        COMM.send(["kill"], dest=i, tag=11)
                    exit()
                    return 0
            else:
                print("Received null from ", node)

            if jobnbr >= num_jobs:
                print('sending -1 to', node)
                COMM.send([-1], dest=node, tag=11)
            else:
                array_to_send = [ytracker, xtracker]
                xtracker += 1
                if(xtracker == scidata.shape[2]): #We have overshot xtracker
                    xtracker = 0
                    ytracker += 1 ##Keep going                
                print('sending job', ytracker, 'and ', xtracker, 'to', i)
                COMM.send(array_to_send, dest=node, tag=11)
                jobnbr = jobnbr + 1

##This recieves the singular (not array values) of files_to_add, filetypes, and npz_array_locations
def main(scidata):
	if(RANK == 0):
		nbr = scidata.shape[1] * scidata.shape[2] ##The number of jobs we have
		manager(SIZE, nbr, scidata)
	else:
		worker(RANK)
            


##We can actually do the prep on every node - it wastes like 20 seconds of processing, but makes sure that everything is on the same page
if RANK >= 0: ##Only need to do this on the main node
    fs = fermisun.fermisun(sys.argv[1], False) ##Initialize and read the yaml file. We do not need to copy over the ft2 file here.

    ## Need to get these in terms of r.a. and dec because that is what the exposure file is in
    infile = open(fs.yaml_source_cut_filename, 'r')

    source_cut_l = [] 
    source_cut_b = []
    for line in infile.readlines():
        if(line[0] != '#'):
                params = line.split()
                source_cut_l += [float(params[3])]
                source_cut_b += [float(params[4])]
    infile.close()

    source_cut_ra = []
    source_cut_dec = []
    for x in range(0, len(source_cut_l)):
        c = SkyCoord(source_cut_l[x]*u.deg, source_cut_b[x]*u.deg, frame='galactic')
        source_cut_ra += [c.gcrs.ra.deg]
        source_cut_dec += [c.gcrs.dec.deg]

    print(source_cut_ra[0], source_cut_dec[0], source_cut_l[0], source_cut_b[0])
    scidata = np.ones((fs.yaml_ebins+1, int(180.0/fs.yaml_binsz), int(360.0/fs.yaml_binsz))) ##This is our data array, which is a numpy array we will save and multiply maps by
    print(scidata.shape)

    if(RANK == 0):
    ##We use ones here, because this is a modification via a source mask, where untouchd pixels retain their original value
        sleep(5) ## Make sure this node finishes last

main(scidata)

