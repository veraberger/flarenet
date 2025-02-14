import numpy as np
#import os
import lightkurve as lk
import pandas as pd
from .flare_model import * 
#from .cosmic_ray_extension import *
from .utils import *
import matplotlib.pyplot as plt

from astropy.coordinates import SkyCoord

#import tempfile
from typing import Union
import warnings
from . import PACKAGEDIR
import lksearch

all = ['TessStar']

class TessStar(object):
    """
    Contains all the functions to search for & download a lightkurve TargetPixelFile, and
    Extract time series and metadata for input into our model.
    The object has umbrella attributes including target ID, sector, and exposure time, and then 
    its .tpf attribute is a lightkurve TargetPixelFile object
    and the .lc attribute is a lightkurve LightCurve object
    """

    def __init__(self, 
                 ticid : Union[str, int, SkyCoord],
                 sector : int, 
                 exptime : Union[int, float] = 20, 
                 download_dir : str = None, 
                 add_cosmic_rays : bool = True, 
                 cloud : bool = False,
                 ):
        """
        Instantiates the object by downloading the TPF through lightkurve 

        Parameters
        ----------
        ticid : str, int, or astropy.coordinates.SkyCoord object
            Input target ID. See lightkurve search_lightcurve API for details.
        sector : int
            TESS sector within which to search for data.
            If None, then what? Download all and stitch, or raise an error?
        exptime : int, float
            Cadence of data product in seconds. 
        download_dir : str
            Directory into which to save downloaded TPF
        add_cosmic_rays : bool
            If True, add back in the cosmic rays otherwise removed by NASA's TESS processing pipeline. 
            Defaults to True.
        flare_fraction : float
            If add_flares = True, specifies the fraction of the lightcurve that should have flares
        cloud : bool
            If true, download TPF files from the cloud storage location
    
        """
        self.ticid = ticid
        self.sector = sector
        self.exptime = exptime 
        self.add_cosmic_rays = add_cosmic_rays
        self.tpf, self.lc, self.cr_flags = self._get_TESS_data(download_dir=download_dir, cloud=cloud)




    def _get_TESS_data(self,
                     download_dir: str = 'TPFs',
                     cloud : bool = False
                      ):
        """
        Downloads a TPF using lightkurve, optionally injects cosmic rays, 
        computes a light curve, and returns each.
        
        Returns
        -------
        tpf : lightkurve TESSTargetPixelFile object
        lc : lightkurve LightCurve object
        cr_flags : array of zeros and ones corresponding to cosmic ray locations in the lc
        """
        if cloud == True:
            cloud_uri = lksearch.TESSSearch(self.ticid, sector=self.sector, exptime=self.exptime, pipeline='SPOC').cubedata.cloud_uris
            tpf = lk.io.read(cloud_uri[0])
        else:
            local_path = lksearch.TESSSearch(self.ticid, sector=self.sector, exptime=self.exptime, pipeline='SPOC').cubedata.download(download_dir=download_dir)['Local Path'][0]
            tpf = lk.io.read(local_path)
        
        # this is slightly messy but I need the original tpf's light curve to compare to the CR light curve and figure out where the CRs are
        lc = tpf.to_lightcurve(aperture_mask=tpf.pipeline_mask)
        #Get lc standard deviation before adding flares. Only use part of lc in case it is variable
        # 20s x 90 = 30 minutes
        self.lc_std = np.nanmedian([lc['flux'].value[i:i+90].std() for i in range(len(lc)-90)])#np.nanstd(lc.flux[:1000]) 
        #print(f"STD comp: original = { np.nanstd(lc.flux[:1000])} new = {self.lc_std}")
        if self.add_cosmic_rays == True:
            if not self.exptime == 20:
                raise TypeError("Cosmic ray injection is only possible for 20-second datasets.")
            # inject cosmic rays
            cosmic_ray_cube = get_cosmicrays(tpf)
            tpf_cr = tpf+cosmic_ray_cube
            lc_cr = tpf_cr.to_lightcurve(aperture_mask=tpf.pipeline_mask)
            lc_cr_arr = list(map(int, (np.nansum(cosmic_ray_cube[:, tpf.pipeline_mask], axis=1)>0)))

            return tpf_cr, lc_cr, lc_cr_arr 
        
        else: 
            return tpf, lc, np.zeros(lc.flux.value.shape)


    

    def get_metadata(self): 
        """ 
        Get relevant metadata from the headers of a TESS TargetPixelFile object
            If outdir is specified, save the array into an .npy file


        Returns
        -------
        metaArr : array
            Right ascension, declination, TESS magnitude, effective temperature, stellar radius, sector, CDPP, crowdsap, camera, CCD, and log10 surface gravity
        """
        

        return np.array([self.tpf.ra, self.tpf.dec,  
                         self.tpf.get_header()['TESSMAG'], 
                         self.tpf.get_header()['TEFF'],  
                         self.tpf.get_header()['RADIUS'], 
                         self.tpf.sector, 
                         self.tpf.hdu[1].header['CDPP1_0'],  
                         self.tpf.hdu[1].header['CROWDSAP'], 
                         self.tpf.camera, self.tpf.ccd,  
                         self.tpf.get_header()['LOGG']])



        
    
    
    def make_orbit_files(self, 
                         modified_flux = None, 
                         labels = None,
                         output_dir = 'training_data/',
                         extra_fname_descriptor = '',
                         plot = False):
        """
        Take a light curve and supplemental information for some target and sector, split the data by orbits, 
        Save the output into an npy file. 

        Parameters
        ----------
        modified_flux : array
            (Optional) the flux with added flares and/or variability
        labels : array
            (Optional) if injected flares were added, provide an array with labels
            1 for flares, 0 for no flares
        output_dir : str
            Where to save the data
        extra_fname_descriptor : str
            text to append to the filename
        plot : bool
            Whether or not to save a plot of the two TESS orbits

        datadir : directory to save orbit data into
        
        Returns
        -------
        Nothing (results saved to datadir)
        """
        # Alternative: Use the MIT observing times table. Need to update this regularly though.
        # https://tess.mit.edu/public/files/TESS_FFI_observation_times.csv
        # Instead of treating each orbit separately, do we just want to split anytime there is a gap greater than X time?

        if isinstance(modified_flux, np.ndarray):
            df = pd.DataFrame(data=[self.lc.time.value, self.lc.flux.value, self.lc.flux_err.value, self.lc.quality.value, self.cr_flags, modified_flux, labels],
                    index=['time','og_flux','flux_err', 'quality','cr_flags','flux','flare_flags']).T

            #df.rename(columns=['time','flux','flux_err', 'quality','pos_corr', 'c_dist','cr_flags','flare_flags','lc_injected_flares'], inplace=True)
        else:
            df = pd.DataFrame(data=[self.lc.time.value, self.lc.flux.value, self.lc.flux_err.value, self.lc.quality.value, self.cr_flags, normalize_flux(self.lc.flux.value, type='standard'), np.zeros_like(self.lc.flux.value)],
                              index=['time','og_flux','flux_err', 'quality','cr_flags', 'flux', 'flare_flags']).T
            #df.rename(columns=['time','flux','flux_err', 'quality','pos_corr','c_dist','cr_flags','flare_flags','lc_flux'], inplace=True)

        dt = df['time'].diff()
        gap_index = df.index[dt == dt.max()].item() 
        orbit1 = df.iloc[:gap_index]
        orbit2 = df.iloc[gap_index:]

        orbit1.to_csv(f"{PACKAGEDIR}/{output_dir}/{self.ticid}_{self.sector}_1_data{extra_fname_descriptor}.csv", index=False)
        orbit2.to_csv(f"{PACKAGEDIR}/{output_dir}/{self.ticid}_{self.sector}_2_data{extra_fname_descriptor}.csv", index=False)
        #np.save(datadir+str(self.ticid)+'_'+str(self.sector)+'_1_data.npy', np.asarray(orbit1))
        #np.save(datadir+str(self.ticid)+'_'+str(self.sector)+'_2_data.npy', np.asarray(orbit2))
        
        if plot:
            fig, ax = plt.subplots(2, figsize=(14,5))
            ax[0].scatter(orbit1['time'], orbit1['flux'], c=orbit1['flare_flags'].values, s=2)
            ax[1].scatter(orbit2['time'], orbit2['flux'], c=orbit2['flare_flags'].values, s=2)
            ax[0].set_title("orbit 1")
            ax[1].set_title("orbit 2") 

            plt.savefig(f"{PACKAGEDIR}/{output_dir}/{self.ticid}_{self.sector}_orbits{extra_fname_descriptor}.png")
            plt.close()


            

    
