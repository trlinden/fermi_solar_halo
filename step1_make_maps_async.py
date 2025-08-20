from mpi4py import MPI
from random import randint
from time import sleep

COMM = MPI.COMM_WORLD # the communicator
RANK = COMM.Get_rank() # identifies each node
SIZE = COMM.Get_size() # the number of nodes

def worker(i):
    while True:
        job = COMM.recv(source=0, tag=11)
        if job == -1:
            sleep(1) ##This prevents the worker from continually asking for jobs and overloading the manager, at the cost of making the program take a few seconds to exit
            break
        else:
            os.system(job)
        result = 'Complete'
        print('worker', i, 'sends', result)
        COMM.send(result, dest=0, tag=12)


def manager(p, n, commands):
    result = ""
    jobnbr = 0 # current job number, we start with job number 0
    cntrcv = 0 # number of received results
    for i in range(1, p):
        print('sending job', jobnbr, 'to', i)
        COMM.send(commands[jobnbr], dest=i, tag=11)
        jobnbr = jobnbr + 1
        if jobnbr > n-1: break ##stop at n-1, since we are losing one core

    while cntrcv < n:
        state = MPI.Status()
        okay = COMM.Iprobe(source=MPI.ANY_SOURCE, \
        tag=MPI.ANY_TAG, status=state)

        if not okay:
            sleep(0.2)
        else:
            node = state.Get_source()
            c = COMM.recv(source=node, tag=12)
            print('received', c, 'from', node)
            cntrcv = cntrcv + 1
            result = result + c

            if jobnbr >= n:
                print('sending -1 to', node)
                COMM.send(-1, dest=node, tag=11)
            else:
                print('sending job', jobnbr, 'to', node)
                COMM.send(commands[jobnbr], dest=node, tag=11)
                jobnbr = jobnbr + 1

def main(commands):
    if(RANK == 0):
        nbr = len(commands)
        manager(SIZE, nbr, commands)
    else:
        worker(RANK)

##Before We Run Main - which has the manager send out jobs to the recipients, we need to first write the list of system commands we are running.
##We also need to recruit one process per node in order to copy the information over onto the ramdisk.

import fermisun ##This has the variables that control how we analyze the Sun
import numpy as np
import os

##Astropy packages to get solar system coordinates
from astropy.coordinates import SkyCoord
from astropy.coordinates import solar_system_ephemeris, ICRS, get_body
from astropy.time import Time
from astropy import units as u

##This determines a host processor on every node, that is responsible for copying the base files over to the ramdir at the beginning
### and then deleting them at the end. It does this by finding the first processor with a different node number
### then it runs a slightly different version of the initialization code, which either copies things over or doesn't

processor_info = (RANK, MPI.Get_processor_name())
allprocessors = COMM.allgather(processor_info)

##This loads fermisun once per node, not actually necessary, but also takes no time
host_processors = np.ones(SIZE)
for a in range(0, len(allprocessors)):
    for b in range(0, a):
        if(allprocessors[b][1] == allprocessors[a][1]):
                host_processors[a] = 0

if(host_processors[RANK] == 1) :
	fs = fermisun.fermisun('input.yaml', True) ##Initialize and read the yaml file, second command asks whether to copy ft2 file over to the ramdir or not


##Now we do a bunch of prep on the main node
if RANK == 0: ##Only need to do this on the main node
    os.system('mkdir ' + str(fs.yaml_wkdir) + '/exposures')
    os.system('mkdir ' + str(fs.yaml_wkdir) + '/psfs')

    ## Now we have a bunch of setup that runs equivilently on every core
    solar_system_ephemeris.set('jpl')

    ##Time steps
    timesteps = np.arange(fs.yaml_starttime, fs.yaml_endtime, fs.yaml_expstep)
    if(timesteps[-1] != fs.yaml_endtime): ##need to add the explicit endtime into the calculation if it is not included
    	 timesteps = np.append(timesteps, fs.yaml_endtime)
    print(timesteps)

    ##Central energy steps (the gtpsf file needs the central point of each energy bin)
    emin_avg = np.power(10.0, (np.log10(fs.yaml_emin) + (np.log10(fs.yaml_emax) - np.log10(fs.yaml_emin)) / (2.0*fs.yaml_ebins)))
    emax_avg = np.power(10.0, (np.log10(fs.yaml_emin) + (2.0*fs.yaml_ebins-1.0) * (np.log10(fs.yaml_emax) - np.log10(fs.yaml_emin)) / (2.0*fs.yaml_ebins)))

    ##We compute all of the timesteps on every node the same way, but the indexes they actually run will be different, this wastes time, but like 30 seconds
    commands = []
    for timestep in range(0, len(timesteps)-1): ##The last timestep is only an endstep, so go to -1 in range

    	##Get the position of the Sun and Moon at this timestep for our analysis
        avg_time = (timesteps[timestep] + timesteps[timestep+1]) / 2.0 ##We need to get the solar position at the center of this timestep, and the moon position to do the Moon model
        jdval = fs.getJD(avg_time)
        jd_formatted = Time(jdval, format='jd')
        moon_position = get_body('moon', jd_formatted)
        sun_position = get_body('sun', jd_formatted)
        sunra = sun_position.ra.degree
        sundec = sun_position.dec.degree
        moonra = moon_position.ra.degree
        moondec = moon_position.dec.degree

        t_starttime = str(timesteps[timestep])
        t_endtime = str(timesteps[timestep+1])

        ##Input Files
        ram_ft1 = str(fs.yaml_ramdir) + "/" + fs.yaml_ft1_template_name
        ram_ft2 = str(fs.yaml_ramdir) + "/" + fs.yaml_ft2_name

        #Permanent Files
        t_gtexpcube = str(fs.yaml_wkdir) + "/exposures/nexp2." + t_starttime + "." + t_endtime + ".fits"
        t_psf = str(fs.yaml_wkdir) + "/psfs/psf." + t_starttime + "." + t_endtime + ".fits"
        t_gtexpcube = str(fs.yaml_wkdir) + "/exposures/nexp2." + t_starttime + "." + t_endtime + ".fits"

        #Temporary Ramdisk Files
        t_gtltcube = str(fs.yaml_ramdir) + "/gtltcube." + t_starttime + "." + t_endtime + ".fits"
        t_srcmap = str(fs.yaml_ramdir) + "/gtsrcmaps." + t_starttime + "." + t_endtime + ".fits"
        t_bft1 = str(fs.yaml_ramdir) + "/bft1." + t_starttime + "." + t_endtime + ".fits"
        t_ft1 = str(fs.yaml_ramdir) + "/ft1." + t_starttime + "." + t_endtime + ".fits"

        ##Number of exposure bins:
        exp_longbins = int(360.0 / fs.yaml_binsz)
        exp_latbins = int(180.0 / fs.yaml_binsz)

    	#Check if the moon model exists -- since that is the last step we do, that means the files have already been generated, and we do not need to make them:
        if(1): ##The file does not exist

            run_command = "gtselect infile=" + ram_ft1 + " outfile=" + t_ft1 + " ra=0.0 dec=0.0 rad=180.0 tmin=" + str(t_starttime) + " tmax=" + str(t_endtime) + " emin=499000.0 emax=500000.0 zmin=0.0 zmax=" + str(fs.yaml_zenith) + " evclass=" + str(fs.yaml_evclass) + " evtype=" + str(fs.yaml_evtype) + " convtype=-1 phasemin=0.0 phasemax=1.0 evtable=\"EVENTS\" chatter=3 clobber=yes debug=no gui=no mode=\"ql\"" ##this makes a cut for 499 GeV - 500 GeV, which allows us to get the right header information, without having a large photon file
            run_command += " && "
            run_command += "gtltcube evfile=" + t_ft1 + " evtable=\"EVENTS\" scfile=" + ram_ft2 + " sctable=\"SC_DATA\" outfile=" + t_gtltcube + " dcostheta=" + str(fs.yaml_dcostheta) + " binsz=" + str(fs.yaml_binsz) + " phibins=" + str(fs.yaml_phibins) + " tmin=0.0 tmax=0.0 file_version=\"1\" zmin=0.0 zmax=" + str(fs.yaml_zenith) + " chatter=2 clobber=yes debug=no gui=no mode=\"ql\" "
            run_command += " && "
            run_command += "gtexpcube2 infile=" + t_gtltcube + " cmap=none outfile=" + t_gtexpcube + " irfs=" + fs.yaml_irfs + " evtype=" + str(fs.yaml_evtype) + " edisp_bins=-1 nxpix=" + str(exp_longbins) + " nypix=" + str(exp_latbins) + " binsz=" + str(fs.yaml_binsz) + " coordsys=CEL xref=0.0 yref=0.0 axisrot=0.0 proj=CAR ebinalg=LOG emin=" + str(fs.yaml_emin) + " emax=" + str(fs.yaml_emax) + " enumbins=" + str(fs.yaml_ebins) + " ebinfile=NONE "
            run_command += "hpx_ordering_scheme=RING hpx_order=6 bincalc=EDGE ignorephi=no thmax=180.0 thmin=0.0 table=EXPOSURE chatter=3 clobber=yes debug=no mode=ql" ##This is too long for one python line
            run_command += " && "
            run_command += "gtpsf expcube=" + t_gtltcube + " outfile=" + t_psf + " irfs=" + str(fs.yaml_irfs) + " evtype=" + str(fs.yaml_evtype) + " ra=" + str(sunra) + " dec=" + str(sundec) + " emin=" + str(emin_avg) + " emax=" + str(emax_avg) + " nenergies=" + str(fs.yaml_ebins) + " thetamax=45 ntheta=4500"
            run_command += " ; " ## want rm to run regardless of how the other commands function
            run_command += "rm " + t_gtltcube + " " + t_srcmap + " " + t_bft1 + " " + t_ft1
            commands += [run_command] ##Make a list of these, and then we will distribute them across nodes
        print ("number of commands is: ", len(commands), t_starttime, t_endtime)

if RANK > 0:
    commands= [] ##These are blank, just so that there isn't a segfault issue when i send it

main(commands)
