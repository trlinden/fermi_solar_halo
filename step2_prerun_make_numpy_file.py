import fermisun ##This has the variables that control how we analyze the Sun
import numpy as np
import os
from astropy.io import fits
import sys

'''
This file is sort of an annoyance:
What we want to do is open the ft1 fits file, take lines from the file and dispatch them to the nodes, which can then convert each photon (1 per line) into
helioprojective coordinates. The problem is that sending the lines from the master process to the node seems to slow things down in terms of I/O. An option would
be have all the nodes know about all the lines, and then just dispatch the line numbers you want them to run, but the issue is there is not enough ram for each
core to have access to a 20 GB file.

So here what we do is open the file on a single processor, save the important information into a numpy array (which makes it smaller), and then we use that numpy
array for the subsequent computations. This is just the intermediate step that runs the 1 processor part

A smarter person would think of a more effective way to skip this step... i am not that person.
'''

fs = fermisun.fermisun('input.yaml', False) ##Initialize and read the yaml file. We do not need to copy over the ft2 file here.
os.system('mkdir ' + str(fs.yaml_wkdir) + '/solar_data')

hdulist = fits.open(fs.yaml_ft1_directory + '/' + fs.yaml_ft1_name) ##This opens the fits file
scidata_all = hdulist[1].data ##This is now a list, with every single photon in it... it is going to be a large array

total_lines = len(scidata_all)

if(len(sys.argv) > 1): ##If we give it a startline and endline
	startline = int(sys.argv[1])
	endline = int(sys.argv[2])
	if(endline > total_lines):
		endline = total_lines
		print("Changed final line to : ", endline , "to match file.")
	scidata = np.asarray(scidata_all[startline:endline]) ##cut some of the file for memory reasons
else:
	scidata = np.asarray(scidata_all)

##Clean up some memory
del scidata_all ##get rid of this and close the fits file
hdulist.close()

dataset = []
##We need to clean this of weird integers that mess things up, which will take time on one core, unfortunately
for a in range(0, len(scidata)):
	intime = float(scidata[a][9])
	if(intime > fs.yaml_starttime and intime < fs.yaml_endtime):
		##order is metval, raval, decval, thetaval, phival, zenithval, earthazimuth, eventid, energyval, lval, bval, bitval, frontback
		dataline = [scidata[a][9], scidata[a][1], scidata[a][2], scidata[a][5], scidata[a][6], scidata[a][7], scidata[a][8], scidata[a][10], scidata[a][0], scidata[a][3], scidata[a][4], scidata[a][16]]
		dataset += [dataline]
	if(a % 10000 == 0):
		print("First ", str(a), "lines analyzed")
dataset = np.asarray(dataset)
np.save(str(fs.yaml_wkdir) + '/solar_data/reduced_data_' + str(fs.yaml_starttime) + '_' + str(fs.yaml_endtime) + '.npy', dataset)
