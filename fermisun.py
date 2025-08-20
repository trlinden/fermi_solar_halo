import sys
import yaml
import os
import healpy as hp
import math

class fermisun:
    def __init__(self, yaml_filename, copyft2):

        ##Read the yaml file:
        with open(yaml_filename) as file:
            yaml_parameters = yaml.load(file, Loader=yaml.FullLoader)

        self.yaml_wkdir = yaml_parameters['main']['wkdir']
        self.yaml_ramdir = yaml_parameters['main']['ramdir']
        self.yaml_mpicores = yaml_parameters['main']['mpicores']

        self.yaml_ft1_directory = yaml_parameters['data']['ft1_directory']
        self.yaml_ft2_directory = yaml_parameters['data']['ft2_directory']
        self.yaml_ft1_name = yaml_parameters['data']['ft1_name']
        self.yaml_ft1_numpy_name = yaml_parameters['data']['ft1_numpy_name']
        self.yaml_ft1_template_name = yaml_parameters['data']['ft1_template_name'] ##This is a mostly blank ft1 file, when we need something with the right selections, but no photons
        self.yaml_ft2_name = yaml_parameters['data']['ft2_name']
        self.yaml_starttime = yaml_parameters['data']['starttime']
        self.yaml_endtime = yaml_parameters['data']['endtime']
        self.yaml_expstep = yaml_parameters['data']['exposurestep']
        self.yaml_evclass = yaml_parameters['data']['eventclass']
        self.yaml_evtype = yaml_parameters['data']['eventtype']
        self.yaml_zenith = yaml_parameters['data']['zenithangle']
        self.yaml_phibins = yaml_parameters['data']['phibins']
        self.yaml_dcostheta = yaml_parameters['data']['dcostheta']
        self.yaml_binsz = yaml_parameters['data']['binsz']
        self.yaml_emin = yaml_parameters['data']['energymin']
        self.yaml_emax = yaml_parameters['data']['energymax']
        self.yaml_ebins = yaml_parameters['data']['energybins']
        self.yaml_filter = yaml_parameters['data']['filter']
        self.yaml_irfs = yaml_parameters['data']['instrumentresponsefunction']
        self.yaml_psf_thetamax = yaml_parameters['data']['psf_thetamax']
        self.yaml_psf_thetabins = yaml_parameters['data']['psf_thetabins']
        self.yaml_ics_infilelist = yaml_parameters['data']['ics_infilelist']

        self.yaml_suncut = yaml_parameters['cuts']['suncut']
        self.yaml_mooncut = yaml_parameters['cuts']['mooncut']
        self.yaml_latitudecut = yaml_parameters['cuts']['latitudecut']
        self.yaml_flarecut_name = yaml_parameters['cuts']['flarecut_name']
        self.yaml_flarecut_fullbackground_name = yaml_parameters['cuts']['flarecut_fullbackground_name']
        self.yaml_flarecut_fullbackground_starttime = yaml_parameters['cuts']['flarecut_fullbackground_starttime']
        self.yaml_flarecut_fullbackground_endtime = yaml_parameters['cuts']['flarecut_fullbackground_endtime']
        self.yaml_flarecut_filename = yaml_parameters['cuts']['flarecut_filename']
        self.yaml_flarecut_buffer = yaml_parameters['cuts']['flarecut_buffer']
        self.yaml_source_angular_cut = yaml_parameters['cuts']['source_angular_cut']
        self.yaml_source_cut_filename = yaml_parameters['cuts']['source_cut_filename']
        self.yaml_source_mask_name = yaml_parameters['cuts']['source_mask_name']

        self.yaml_healpix_nside = yaml_parameters['healpix']['nside']
        self.yaml_healpix_numpixels = 12 * int(self.yaml_healpix_nside) * int(self.yaml_healpix_nside)

        self.yaml_angular_size_of_sun_degrees = yaml_parameters['constants']['angular_size_of_sun_degrees']
        self.yaml_rad2deg = yaml_parameters['constants']['rad2deg']

        ##This copies the ft2 file over to the ramdisk, or whatever other working directory has been specified
        if(copyft2 == True):
            os.system('cp ' + self.yaml_ft2_directory + '/' + self.yaml_ft2_name + ' ' + self.yaml_ramdir)
            os.system('cp ' + self.yaml_ft1_directory + '/' + self.yaml_ft1_template_name+ ' ' + self.yaml_ramdir)

    def getJD(self, met): ##need to add new leap seconds here if they arrive
        if met > 252460801: met-=1 # 2008 leap second
        if met > 362793601: met-=1 # 2012 leap second
        if met > 457401602: met-=1 # 2015 leap second
        if met > 504921603: met-=1 # 2016 leap second
        return 2454683.15527778 + (met-239557417) / 86400.0 ##convert to julian day

    def distance_degrees(self, Ty1, Tx1, Ty2, Tx2): ##both input and output in degrees
        Ty1 = Ty1*math.pi/180.0
        Tx1 = Tx1*math.pi/180.0
        Ty2 = Ty2*math.pi/180.0
        Tx2 = Tx2*math.pi/180.0

        delta_y = math.fabs(Ty2-Ty1)
        delta_x = math.fabs(Tx2-Tx1)

        aval = math.pow(math.sin(delta_y/2.0), 2.0) + math.cos(Ty1)*math.cos(Ty2) * math.pow(math.sin(delta_x/2.0), 2.0)
        return 180.0/math.pi*2.0 * math.atan2(math.sqrt(aval), math.sqrt(1.0-aval))
    
    def convert_ang_to_pix(self, Tx_deg, Ty_deg, healpix_nside): ##takes in the x pixel, ypixel and nside for the map. 
        Ty = -math.pi/180.0 * (Ty_deg-90.0)
        Tx = -math.pi/180.0 * Tx_deg
        if(Tx > math.pi):
            Tx = -180.0 / math.pi * (Tx_deg-360.0)
        return hp.ang2pix(healpix_nside, Ty, Tx, nest=False)
    
    def convert_pix_to_ang(self, ipix, healpix_nside):
        (Ty, Tx) = hp.pixelfunc.pix2ang(healpix_nside, ipix, nest=False)
        Tydeg = -180.0/math.pi * Ty + 90.0
        Txdeg = -180.0/math.pi * Tx
        if(Tx > math.pi):
            Txdeg = -180.0/math.pi * Tx + 360.0
        return(Txdeg, Tydeg)
