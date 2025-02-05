import numpy as np
#import os
import lightkurve as lk
import pandas as pd
from .flare_model import * 
#from .cosmic_ray_extension import *
from .utils import *
import matplotlib.pyplot as plt
from numpy.random import choice, randint
from astropy.coordinates import SkyCoord
from astropy.io import ascii
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




    #@staticmethod
    #TODO: turn back into a static method?
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

        Parameters
        ----------
        outdir : str
            Directory to save metadata into
        
        Returns
        -------
        metaArr : array
            Right ascension, declination, TESS magnitude, effective temperature, stellar radius, sector, CDPP, crowdsap, camera, CCD, and log10 surface gravity
        """
        

        '''if outdir is not None:
            if not os.path.exists(outdir):
                os.mkdir(outdir)
            np.save(f"{outdir}{self.ticid}_{self.sector}_meta.npy", metaArr)'''
        return np.array([self.tpf.ra, self.tpf.dec,  
                         self.tpf.get_header()['TESSMAG'], 
                         self.tpf.get_header()['TEFF'],  
                         self.tpf.get_header()['RADIUS'], 
                         self.tpf.sector, 
                         self.tpf.hdu[1].header['CDPP1_0'],  
                         self.tpf.hdu[1].header['CROWDSAP'], 
                         self.tpf.camera, self.tpf.ccd,  
                         self.tpf.get_header()['LOGG']])






    def generate_flares(self, 
                        #flare_fraction : float = None,
                        num_flares : int = 100,
                        ): 
        """
        Take time and flux time series 
        Generate and inject artificial flares into randomized points in the light curve, 
        such that a desired percentage of the light curve timespan is occupied by a flare
        keep injecting until t1/2 sum reaches a given fraction of the time

        Parameters
        ----------
        all_ampls : ndarray
            Array of relative amplitudes to draw from
        all_fwhms : ndarray
            Array of FWHMs to draw from
        fraction_flare : float
            Fraction of the light curve to cover with flares, between 0 and 1. 
            This is an approximation, as time covered by overlapping flares will be counted by both
        
        Returns
        -------
        flare_flux : ndarray
            Array of flare fluxes with the same length as the input object light curve
        paramsArr : ndarray
            Array of parameters for each generated flare
            [Time of peak, relative amplitude, and FWHM]
        """
        # # **** read in gunther flares file for flare parameters ****
        flaredf = ascii.read(f"{PACKAGEDIR}/supplemental_files/gunther_flares_all.txt")
        all_ampls = flaredf['Amp']
        all_fwhms = flaredf['FWHM']

        #if ((flare_fraction <0) | (flare_fraction > 1)):
        #    raise ValueError("fraction_flare must be a value between 0 and 1.")
        
        paramsArr = np.array([])
        flare_time = 0
        flare_flux = np.zeros(len(self.lc.time))


        #while flare_time < (flare_fraction * (len(self.lc.time)*(self.exptime / 24 / 60 / 60))): 
        nf = 0
        while nf < num_flares:
            # generate artificial flare parameters
            tpeak = choice(self.lc.time.value)
            
            rand_ind = randint(0, len(all_ampls)) # get random index for flare parameters
            ampl = all_ampls[rand_ind]
            fwhm = all_fwhms[rand_ind]

            flare_time += fwhm * 3 #The FWHM doesn't capture the whole tail, so extend it when determining how "full" the lc is
            
            # add flare parameters to array
            paramsArr = np.append(paramsArr, np.asarray([tpeak, ampl, fwhm])) 
            #print(self.lc.time.value, tpeak, fwhm, ampl)
            # generate the flare, add to flux array
            flare = flare_model(self.lc.time.value, tpeak, fwhm, ampl)
            flare = np.nan_to_num(flare)
            #flare_flux = np.add(flare_flux, flare)
            flare_flux += flare
            nf += 1
        


        return flare_flux, paramsArr
    
    def get_flare_flags(self,
                    flare_flux,
                    threshold=0.01):
        """
        Creates an array of 0s and 1s where the input flux is above a given threshold.

        Parameters
        ----------
        flare_flux : ndarray
            Array of flare fluxes
        threshold : float
            Value above which flux is considered to be part of a flare
        Returns
        -------
            : array
            Array of 1s where flux is high enough to be considered part of a flare; 0s otherwise
        
        """
        #print(flare_flux)
        flare_flag = np.where(flare_flux > threshold, 1, 0)
        self.flare_flag = flare_flag
        #return flare_flag
    

    def inject_flares(self, 
                      flare_flux : np.ndarray, 
                      #plot : bool = False
                      ): 
        """
        Parameters
        ----------
        flare_flux : ndarray
            Array of flare fluxes to inject
        
        Returns
        -------
            : ndarray
            Array of flare fluxes added to light curve
        """

        flux_with_flares = self.lc.flux.value + (flare_flux * self.lc.flux.value)
        self.flux_with_flares = self.normalize(flux_with_flares, norm_type='standard')


    def inject_asteroid_crossing(self):
        """
        Simulate an asteroid passing in front of a star in TESS.
        Parameters:
        t (array): Time array
        """
        # Ranges loosely based on https://iopscience.iop.org/article/10.3847/1538-4357/ace9df
        amp = np.random.uniform(0.01, 0.2)
        sig = np.random.uniform(0.01, .2)
        t_mid = np.random.choice(self.lc.time.value)
        
        #signal = amp * np.exp(-((t - t_mid) ** 2) / (2 * sig ** 2)) # Regular gaussian
        # Make a 'flat-top' gaussian instead
        signal = amp * np.exp(-((self.lc.time.value - t_mid) / (2 * sig))**4)
        self.flux_with_flares = self.flux_with_flares + signal
    
    def inject_stellar_pulsations(self):
        periods = np.random.uniform(0.1, 5, size=3)
        amplitudes = np.random.uniform(0.01, 0.2, size=3)
        phases = np.random.uniform(0, 2*np.pi, size=3)

        signal = np.zeros_like(self.lc.time.value)
        for period, amplitude, phase in zip(periods, amplitudes, phases):
            signal += amplitude * np.sin(2 * np.pi * (self.lc.time.value / period + phase))
        signal /= (1 + np.median(signal))
        self.flux_with_flares = self.flux_with_flares + signal

    def inject_rr_lyrae(self):
        """
        Simulate an RR Lyrae variable star light curve.
        Parameters:
        t (array): Time array
        period (float): Pulsation period in days
        amplitude (float): Amplitude of the variation
        rise_fraction (float): Fraction of the period spent in the rising phase
        phase_offset (float): Phase offset for the pulsation
        """
        period =  np.random.uniform(0.5, 35),
        amplitude = np.random.uniform(0.1, 0.5)
        rise_fraction = np.random.uniform(0.1, 0.3)
        phase_offset = np.random.uniform(0, 1)

        
        phase = ((self.lc.time.value / period) + phase_offset) % 1
        signal = np.zeros_like(self.lc.time.value)
        # Rising phase (usually steeper)
        rising = phase < rise_fraction
        signal[rising] = amplitude * (phase[rising] / rise_fraction)
        # Falling phase
        falling = ~rising
        signal[falling] = amplitude * (1 - ((phase[falling] - rise_fraction) / (1 - rise_fraction)))
        # Add some asymmetry and non-linearity to make it more realistic
        signal = signal - 0.1 * amplitude * np.sin(4 * np.pi * phase)
        # Normalize around 1
        signal /= np.median(signal)
        self.flux_with_flares = self.flux_with_flares + signal - 1

        
    
    
    def make_orbit_files(self, 
                         #injected_flux=None, 
                         output_dir='training_data/',
                         extra_fname_descriptor='',
                         plot=False):
        """
        Take a light curve and supplemental information for some target and sector, split the data by orbits, 
        Save the output into an npy file. 

        Parameters
        ----------
        injected_flux : array

        datadir : directory to save orbit data into
        
        Returns
        -------
        Nothing (results saved to datadir)
        """
        # Alternative: Use the MIT observing times table. Need to update this regularly though.
        # https://tess.mit.edu/public/files/TESS_FFI_observation_times.csv
        # Instead of treating each orbit separately, do we just want to split anytime there is a gap greater than X time?

        if not hasattr(self, 'flare_flags'):
            df = pd.DataFrame(data=[self.lc.time.value, self.lc.flux.value, self.lc.flux_err.value, self.lc.quality.value, self.cr_flags, self.normalize(self.lc.flux.value, norm_type='standard')],
                              index=['time','flux','flux_err', 'quality','cr_flags', 'normalized_flux']).T
            #df.rename(columns=['time','flux','flux_err', 'quality','pos_corr','c_dist','cr_flags','flare_flags','lc_flux'], inplace=True)
        else:
            df = pd.DataFrame(data=[self.lc.time.value, self.lc.flux.value, self.lc.flux_err.value, self.lc.quality.value, self.cr_flags, self.flux_with_flares, self.flare_flags],
                              index=['time','flux','flux_err', 'quality','cr_flags','flux_with_flares','flare_flags']).T

            #df.rename(columns=['time','flux','flux_err', 'quality','pos_corr', 'c_dist','cr_flags','flare_flags','lc_injected_flares'], inplace=True)
        dt = df['time'].diff()
        gap_index = df.index[dt == dt.max()].item() # also must be a less handwavy to do all this
        orbit1 = df.iloc[:gap_index]
        orbit2 = df.iloc[gap_index:]
        #if datadir is not None:
        #    if not os.path.exists(datadir):
        #        os.mkdir(datadir)
            
            # could save pandas dataframe, but they take almost 2x more memory
        orbit1.to_csv(f"{PACKAGEDIR}/{output_dir}/{self.ticid}_{self.sector}_1_data{extra_fname_descriptor}.csv", index=False)
        orbit2.to_csv(f"{PACKAGEDIR}/{output_dir}/{self.ticid}_{self.sector}_2_data{extra_fname_descriptor}.csv", index=False)
            #np.save(datadir+str(self.ticid)+'_'+str(self.sector)+'_1_data.npy', np.asarray(orbit1))
            #np.save(datadir+str(self.ticid)+'_'+str(self.sector)+'_2_data.npy', np.asarray(orbit2))
        
        if plot and hasattr(self, 'flare_flags'):
            fig, ax = plt.subplots(2, figsize=(14,5))
            ax[0].scatter(orbit1['time'], orbit1['flux_with_flares'], c=orbit1['flare_flags'].values, s=2)
            ax[1].scatter(orbit2['time'], orbit2['flux_with_flares'], c=orbit2['flare_flags'].values, s=2)
            ax[0].set_title("orbit 1")
            ax[1].set_title("orbit 2") 

            plt.savefig(f"{PACKAGEDIR}/{output_dir}/{self.ticid}_{self.sector}_orbits{extra_fname_descriptor}.png")
            plt.close()


            

    def normalize(self,
                  lc_flux : np.ndarray,
                   norm_type : str ='median'):
        """
        Normalization for flux
        
        Parameters
        ----------
        lc_flux : 1D array of flux values
        type : Supported types are 'median' or 'standard'

        Returns
        -------
        nparray containing normalized flux values
        """
        # TODO: go back to median normalization I thinK?
        if norm_type == 'median':
            normalized = lc_flux / np.nanmedian(lc_flux)
        elif norm_type == 'standard':
            normalized = (lc_flux - np.nanmean(lc_flux)) / np.nanstd(lc_flux)
        return normalized
    



    def get_cosmicrays(self, tpf):
        """Function to access and return the cosmic ray extension

        Unlike other TESS data products, the TESS 20-second target pixel files
        do not have on-board cosmic ray correction. Instead, cosmic rays are identified
        by the pipeline and removed from the data. The removed flux is saved
        in an extension in the FITS file. 
        See https://heasarc.gsfc.nasa.gov/docs/tess/TESS-CosmicRayPrimer.html
        for details
        
        Parameters:
        -----------
        tpf: lk.TargetPixelFile
            Input Target Pixel File
        
        Returns:
        -------
        cr: np.ndarray
            Array containing cosmic ray fluxes of shape tpf.shape
        """
        cadenceno = np.asarray(tpf.cadenceno)

        # Get cosmic ray information
        l = np.where([hdu.name == 'TARGET COSMIC RAY' for hdu in tpf.hdu])[0]
        if len(l) != 1:
            #raise ValueError("TPF product has no cosmic ray extension.")
            warnings.warn(f"No Cosmic Ray Extension found for {tpf.ticid}. Returning cube of 0s. ")
            return np.zeros(tpf.shape)
        hdu = tpf.hdu[l[0]]
        c, x, y, f = [hdu.data[attr].copy() for attr in ['CADENCENO', 'RAWX', 'RAWY', 'COSMIC_RAY']]
        if len(c) == 0:
            warnings.warn(f"No Cosmic Rays identified for {tpf.ticid}. Returning cube of 0s. ")
            np.zeros(tpf.shape)
            #raise ValueError("No cosmic rays identified. Is this a 20-second dataset?")
        x -= tpf.column
        y -= tpf.row

        # Mask down to only cosmic rays in cadences within the TPF
        k = np.in1d(c, cadenceno)
        c, x, y, f = c[k], x[k], y[k], f[k]

        # Map cosmic ray cadences to array indices
        r = {cadenceno[idx]:idx for idx in range(tpf.shape[0])}
        inv = np.asarray([r[c1] for c1 in c])

        # Build a cube
        cr = np.zeros(tpf.shape)
        cr[inv, y, x] = f
        return cr
    
