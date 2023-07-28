import numpy as np
import lightkurve as lk
import pandas as pd
from sklearn.preprocessing import RobustScaler
from flare_model import * 
from cosmic_ray_extension import *
import matplotlib.pyplot as plt
from numpy.random import uniform, normal, choice, randint


all = ['InputTPF']

class InputTPF(object):
    """
    Contains all the functions to search for & download a lightkurve TargetPixelFile, 
    extract time series and meta data for input into our model, ...
    """

    def __init__(self, ticid, sector, exptime=20, download_dir='tpfs/', cosmic_rays=True, inject_flares=True):
        """
        Download a TPF 
        """
        self.targetid = ticid
        self.sector = sector
        self.exptime = exptime # what if someone puts "fast" instead of 20? should i just require that exptime is a number
        self.tpf, self.lc, self.crArr = InputTPF.download_tpf('TIC '+str(ticid), sector, exptime=exptime, download_dir=download_dir, cr=cosmic_rays)
        self.centroid_col, self.centroid_row = self.tpf.estimate_centroids(aperture_mask='default', method='moments')
        self.pos_corr = self._get_pos_corr()
        self.c_dist = self._get_centroid_shift()


    @staticmethod
    def download_tpf(ticid, sector, exptime=20, download_dir='tpfs', cr=True):
        """

        """
       
        tpf = lk.search_targetpixelfile(ticid, mission='TESS', author='SPOC', exptime=exptime, sector=sector).download(download_dir=download_dir)

        # raise error if there's no file to download
        if tpf is None:
            raise ValueError(f"Unable to find data for target ID {ticid} and sector {sector} with {exptime} exposure time.")
        
        # this is so messy but I need the original tpf's light curve to compare to the CR light curve and figure out where the CRs are
        lc = tpf.to_lightcurve(aperture_mask=tpf.pipeline_mask)

        if cr == True:
            if not exptime == 20:
                raise TypeError("Cosmic ray injection is only possible for 20-second datasets.")
            # inject cosmic rays
            cosmic_ray_cube = load_cosmicray_extension(tpf) 
            tpf_cr = tpf+cosmic_ray_cube
            lc_cr = tpf_cr.to_lightcurve(aperture_mask=tpf.pipeline_mask)

            return tpf_cr, lc_cr, np.where(lc_cr.flux - lc.flux != 0, 1, 0)
        
        elif cr == False:
            return tpf, lc, np.zeros(lc.shape)
        else: 
            raise ValueError("Not a valid input for cr. Please enter True or False.")
    

    def get_metadata(self, outdir='/Users/veraberger/nasa/meta_training/'): 
        """ Get relevant metadata from the headers of a TESS TargetPixelFile object
            If outdir is specified, save the array into an .npy file

        Parameters
        ----------
        outdir : str
            Directory to save metadata into
        
        Returns
        -------
        nothing - do i say that or just leave this blank?
        """
        # metaArr = np.array([self.ra, self.dec,  self.tessmag, self.teff,  self.radius, self.sector, self.cdpp1_0,  self.crowdsap, self.camera, self.ccd,  self.logg]) 
        metaArr = np.array([self.tpf.ra, self.tpf.dec,  self.tpf.get_header()['TESSMAG'], self.tpf.get_header()['TEFF'],  self.tpf.get_header()['RADIUS'], self.tpf.sector, self.tpf.hdu[1].header['CDPP1_0'],  self.tpf.hdu[1].header['CROWDSAP'], self.tpf.camera, self.tpf.ccd,  self.tpf.get_header()['LOGG']])
        if outdir is not None:
            np.save(outdir+str(self.targetid)+'_'+str(self.sector)+'_meta.npy', metaArr)
        else:
            return metaArr

    def _get_centroid_shift(self):
        """
        Retrieves the centroid time series from a TargetPixelFile, 
        returns the magnitude of the 2D shift of the centroid from the median

        Parameters
        ----------
        outdir : str
            Directory to save metadata into
        
        Returns
        -------
        Square root of sum of squares of the centroid row and column time series
        """
        c_cols = self.centroid_col-np.nanmedian(self.centroid_col)
        c_rows = self.centroid_row-np.nanmedian(self.centroid_row)
        return np.sqrt(c_cols**2+c_rows**2).value

    def _get_pos_corr(self):
        # magnitude of vector given by poscorr 1 and 2
        return np.sqrt(self.tpf.pos_corr1**2+self.tpf.pos_corr2**2) 

        
    def get_crArr(lc, lc_cr):
        """
        For a light curve of length n, 
        create a 1xn array with 1s where a cosmic ray was identified by the TESS pipeline, and 0s otherwise.
        
        Parameters
        ----------
        lc : lk.LightCurve object 
            Light curve without CRs injected
        lc_cr : lk.LightCurve object 
            Light curve with CRs injected
        
        Returns
        -------
        crArr : array of 0s and 1s marking where in the light curve cosmic rays have been identified
        """
        crArr = np.where(lc_cr.flux - lc.flux != 0, 1, 0) # is there a better way to do this?
        return crArr

    def make_orbit_files(self, injected_flux, flareArr, datadir='training_data/'):
        """
        Take a light curve and supplemental information for some target and sector, split the data by orbits, 
        Save the output into an npy file. 

        Parameters
        ----------
        datadir : directory to save orbit data into
        
        Returns
        -------
        """
        if injected_flux is None:
            df = pd.DataFrame(data=[self.lc.time.value, self.lc.flux.value, self.lc.flux_err.value, self.lc.quality.value, self.pos_corr, self.c_dist, self.crArr, np.zeros(self.crArr.shape)]).T
        else:
            df = pd.DataFrame(data=[self.lc.time.value, injected_flux, self.lc.flux_err.value, self.lc.quality.value, self.pos_corr, self.c_dist, self.crArr, flareArr]).T
        dt = df[0].diff()
        print(df.columns)
        gap_index = df.index[dt == dt.max()].item() # also must be a less handwavy to do all this
        orbit1 = df.iloc[:gap_index]
        orbit2 = df.iloc[gap_index:]
        if datadir is not None:
            np.save(datadir+str(self.targetid)+'_'+str(self.sector)+'_1_data.npy', np.asarray(orbit1))
            np.save(datadir+str(self.targetid)+'_'+str(self.sector)+'_2_data.npy', np.asarray(orbit2))
        return orbit1, orbit2

    def flare_generator(self, all_ampls, all_fwhms, fraction_flare=0.1):
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
            Fraction of the light curve to cover with flares. 
            This is an approximation, as time covered by overlapping flares will be counted by both
        
        Returns
        -------
        flare_flux : ndarray
            Array of flare fluxes with the same length as the input object light curve
        paramsArr : ndarray
            Array of parameters for each generated flare
            [Time of peak, relative amplitude, and FWHM]
        """
        paramsArr = np.array([])
        fwhm_time = 0
        flare_flux = np.zeros((1, len(self.lc.time)))
        
        # while summed fwhm times of flares are less than the desired fraction of time of the orbit to be occupied by a flare, 
        # generate flares & save parameters
        while fwhm_time < (fraction_flare * (len(self.lc.time)*0.00023148148)):
        # while fwhm_time < (fraction_flare * (t.max() - t.min())):
            # generate artificial flare parameters
            tpeak = choice(self.lc.time.value)
            # tpeak = uniform(self.lc.time.value.min(), self.lc.time.value.max()) # random time of peak within the timespan of the light curve
            rand_ind = randint(0, len(all_ampls)) # get random index for flare parameters
            ampl = all_ampls[rand_ind]
            fwhm = all_fwhms[rand_ind]

            fwhm_time += fwhm
            
            # add flare parameters to array
            paramsArr = np.append(paramsArr, np.asarray([tpeak, ampl, fwhm])) 

            # generate the flare, add to flux array
            flare = flare_model(self.lc.time.value, tpeak, fwhm, ampl)
            flare = np.nan_to_num(flare)
            flare_flux = np.add(flare_flux, flare)
        return flare_flux.flatten(), paramsArr
    
    @staticmethod
    def get_flareArr(flare_flux, threshold=0.01):
        """
        Get array of 0s and 1s where flare flux is above a given threshold.
        """
        return np.where(flare_flux > threshold, 1, 0)
    

    def inject_flare(self, flare_flux, normalize=True):
        """
        Parameters
        ----------
        flare_flux : ndarray
            Array of flare fluxes to inject
        normalize : bool
            If True, median-normalize the input light curve
        
        Returns
        -------
            : ndarray
            Array of flare fluxes added to light curve
        """
        if normalize == True:
            flare_flux_noisy = normal(flare_flux, scale=(self.lc.flux_err/np.nanmedian(self.lc.flux)), size=flare_flux.shape)
            # fig, ax = plt.subplots(2, figsize=(13,9), sharex=True)
            # ax[0].plot(self.lc.time.value, flare_flux_noisy, color='tab:green')
            # ax[1].plot(self.lc.time.value, flare_flux, color='black')            
            # plt.show()
            return(np.add(InputTPF.median_normalize(self.lc.flux.value).flatten(), flare_flux_noisy))
        else: 
            flare_flux_noisy = normal(flare_flux, scale=(self.lc.flux_err.value), size=flare_flux.shape)
            return np.add(self.lc.flux.flatten(), flare_flux_noisy.flatten())
        # return np.add(lc_flux.flatten(), flare_flux.flatten())


    def median_normalize(lc_flux):
        return (lc_flux - np.nanmedian(lc_flux)) / np.nanmedian(lc_flux)