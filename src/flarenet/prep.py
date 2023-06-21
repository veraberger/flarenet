"""
general structure: 
import packages
__all__ = ['ClassName'] 
define the class
__init__
other funcs
"""

import numpy as np 
import pandas as pd
import sklearn
import lightkurve as lk
import matplotlib.pyplot as plt
import astropy.units as u
import glob
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
        if side in ['left', 'both']:
            # t_left = np.linspace(self.time[0]-(cadences*20*u.second), self.time[0]-20*u.second, num=cadences) 
            t_left = np.linspace(self.time[0]-(cadences*20), self.time[0]-20, num=cadences) 
            print(t_left)
            t_ext = np.append(t_left, self.time)
            print(t_ext)
            print(self.time)

            if mode == 'endpt':
                prelim_f_left = np.full((1, cadences), self.flux.value[0])
                std_left = np.std(self.flux[:300]) # arbitrary choice
                f_left = np.random.normal(prelim_f_left, std_left)
            elif mode == 'tess_noise':
                prelim_f_left = np.full((1, cadences), self.flux.value[0]) # actually change this and put in a tess LC
                std_left = np.std(self.flux[:300])
                f_left = np.random.normal(prelim_f_left, std_left)
            else:
                raise ValueError(f"'{mode}' is not an available mode. Choose `endpt` or `tess_noise`.")
            f_ext = np.append(f_left, self.flux.value)

        if side in ['right', 'both']: 
            # t_right = np.linspace(self.time.max() + 20*u.second, self.time.max()+(cadences*20*u.second), num=cadences) 
            t_right = np.linspace(self.time.max() + 20, self.time.max()+(cadences*20), num=cadences) # time must be in seconds for this to work


            if mode == 'endpt':
                prelim_f_right= np.full((1, cadences), self.flux.value[-1])
                std_right = np.std(self.flux[-300:]) # arbitrary choice
                f_right = np.random.normal(prelim_f_right, std_right)
            elif mode == 'tess_noise':
                prelim_f_right= np.full((1, cadences), self.flux.value[-1])
                std_right = np.std(self.flux[-300:]) # arbitrary choice
                f_right = np.random.normal(prelim_f_right, std_right)
            else:
                raise ValueError(f"'{mode}' is not an available mode. Choose `endpt` or `tess_noise`.")

            if side == 'both': 
                t_ext = np.append(t_ext, t_right)
                f_ext = np.append(f_ext, f_right)
            else: 
                t_ext = np.append(self.time, t_right)
                f_ext = np.append(self.flux, f_right)
            # self.flux_err = np.append(self.flux_err, pad_err)
        
        elif side not in ['left', 'right', 'both']: 
            raise ValueError(f"'{side}' is not an option. Choose `left`, `right`, or `both`.")
        print(t_ext)
        return t_ext, f_ext
        

    
    

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

ax[0].plot(mylc.time.value, mylc.flux.value, color='tab:pink', marker='s', markersize=1, ls='none')

print(len(glob.glob('/Users/veraberger/.lightkurve/cache/mastDownload/TESS/*')))

mylc.normalize()
# plt.plot(mylc.time, mylc.flux)
# plt.show()
mylc.flux= mylc.invert() # there must be a better way to do this? or do you just replace the flux every time


# invert or normalize or pad gets rid of the astropy quantity object :(
t_ext, f_ext = mylc.pad(side='left')
ax[1].plot(t_ext.value, f_ext.value, color='tab:brown', marker='s', markersize=1, ls='none')
ax[1].set_xlabel('Time [s]')
ax[0].set_ylabel('SAP Flux')
ax[1].set_ylabel('Relative Flux & Inverted')
plt.suptitle('TIC 219852882')
# plt.savefig('figures/manipulating_lcs.png', dpi=200)
plt.show()
