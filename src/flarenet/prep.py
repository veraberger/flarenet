"""
general structure: 
import packages
__all__ = ['ClassName'] 
define the class
__init__
other funcs
"""

import numpy as np 
import sklearn
import lightkurve as lk
import matplotlib.pyplot as plt
import astropy.units as u
import pdb

from flare_model import *

__all__ = ['TimeSeries'] 


class TimeSeries:

    def __init__(self, lc):
        """
        will add a docstring. see complete ones below
        """
        self.time = lc.time
        self.flux = lc.sap_flux
        self.flux_err = lc.sap_flux_err
        

    def normalize(self, zcenter=True):
        """ Normalize flux values in a light curve by dividing by the median, 
            with the option to zero-center the light curve.

        Parameters
        ----------
        zcenter : book
            If True, center the light curve at zero by subtracting 1 from the normalized flux.
        Returns
        -------
        lc_copy (or do i leave this blank?) : array-like ? 
            Normalized fluxes and flux_errors
        """
        self.flux = self.flux/np.nanmedian(self.flux)
        self.flux_err = self.flux_err/np.nanmedian(self.flux)
        if zcenter == True:
            self.flux -= 1


    def pad(self, cadences = 1000, mode = 'endpt', side = 'both'):
        """ Extend a light curve by a given number of cadences.

        Parameters
        ----------
        cadences : int
            Number of cadences to extend the light curve by.

        mode : str
            Type of filler flux to pad the light curve.
            For any choice, random gaussian noise will be added. 

            "enpt" : take the flux at the end of the light curve being extended.
            "tess_noise" : take data from TIC 1234567 baseline flux from sector 14 etc. but still would need to scale it to the fluxes of the current lc

        side : str
            Which side of the light curve to pad.
            "both" : both sides.
            "start" : prior to the start of the light curve
            "end" : at the end of the light curve
    
        Returns
        -------
        filled_lc : 2D array ?
            Light curve with extended time, flux, and flux error values.

        Raises
        ------
        ValueError
            If input parameters for `mode` or `side` don't match the given options.

        """
        
        # if mode == 'median':
        #     prelim_pad_flux = np.full((1, cadences), np.nanmedian(self.flux))# extend the first or last data point
        #     pad_err = 0.00002 # need some way to generate uncertainties
        #     # pad_err = np.percentile(self.flux, 5)
        # elif mode == 'tess_noise': 
        #     prelim_pad_flux = 0 # insert data from some slightly noisy tess light curve? 
        #     pad_err = [0] # this will throw an error later bc it is length one, not cadences
        # else:
        #     raise ValueError(f"'{side}' is not an available mode. Choose `median` or `tess_noise`.")
        # pad_flux = np.random.normal(prelim_pad_flux, pad_err) # instead take std of lc near the endpoint and use that as the std for np.random.normal

        if side in ['left', 'both']: 
            t_left = np.linspace(self.time[0]-(cadences*20*u.second), self.time[0]-20*u.second, num=cadences) 
            self.time = np.append(t_left, self.time)
            f_left = np.full((1, cadences), self.flux.value[-1])
            self.flux = np.append(f_left, self.flux)
            self.flux_err = np.append(pad_err, self.flux_err)

        if side in ['right', 'both']: 
            t_right = np.linspace(self.time.max() + 20*u.second, self.time.max()+(cadences*20*u.second), num=cadences) # time must be in seconds for this to work
            self.time = np.append(self.time, t_right)
            f_right = np.full((1, cadences), self.flux.value[0]) # HOW TO ISOLATE A SINGLE FLUX VALUE... 
            self.flux = np.append(self.flux, f_right)
            self.flux_err = np.append(self.flux_err, pad_err)

        elif side not in ['left', 'right', 'both']: 
            raise ValueError(f"'{side}' is not an option. Choose `left`, `right`, or `both`.")
        

    def invert(self):
        """
        Takes a series of floats and computes the inverse, leaving 0 values as 0.

        Returns
        -------

        inv : array-like
            Float values of the inverses of all the input elements.

        """
        maskedarr = np.ma.masked_where(self.flux == 0, self.flux) # there must be a better way to deal w this.
        inv = (1/maskedarr).filled(fill_value=0)
        return inv


    def remove_upper_outliers(self, s=3.0): 
        return lk.LightCurve.remove_outliers(self, sigma_upper = s, sigma_lower = float('inf'))




fig, ax = plt.subplots(2, sharex=True, figsize=(10,8))

tlc = lk.search_lightcurve('TIC 219852882', exptime=20).download_all().stitch()

mylc = TimeSeries(tlc)

ax[0].plot(mylc.time.value, mylc.flux.value, color='tab:purple', marker='s', markersize=1, ls='none')


mylc.normalize()
# plt.plot(mylc.time, mylc.flux)
# plt.show()
mylc.flux= mylc.invert() # there must be a better way to do this? or do you just replace the flux every time



t_ext, f_ext, ferr_ext = mylc.pad(side='both')
ax[1].plot(t_ext, f_ext,  color='tab:green', marker='o', markersize=1, ls='none')
ax[1].set_xlabel('Time [s]')
ax[0].set_ylabel('SAP Flux')
ax[1].set_ylabel('Relative Flux & Inverted')
plt.suptitle('TIC 219852882')
# plt.savefig('figures/manipulating_lcs.png', dpi=200)
plt.show()
