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
        Initializes the time series class. 
        Creates an object with attributes time, flux, and flux error using attributes from lightkurve objects.
        """
        self.time = lc.time
        self.flux = lc.sap_flux
        self.flux_err = lc.sap_flux_err
        self.id = lc.meta['OBJECT'].split(' ')[1]
        self.sector = lc.meta['SECTOR']
        

    def normalize(self, mode='max'): # also need to normalize flux errors? or just feed them into this function? 
        """ Normalize flux values in a light curve

        Parameters
        ----------
        mode : str
            The norm to use for normalization
            "l1" : Sum of the magnitudes of the values
            "l2" : Euclidean norm
            "max" : Maximum value
            "median" : Median value
        Returns
        -------
        lc_copy (or do i leave this blank?) : array-like ? 
            Normalized fluxes
        """
        if mode in ['l1', 'l2', 'max']:
            self.flux = self.flux.reshape(-1,1)
            return sklearn.preprocessing.normalize(self.flux, norm=mode, axis=0, copy=False, return_norm=False)
        elif mode == 'median':
            self.flux = self.flux/np.nanmedian(self.flux)
            self.flux_err = self.flux_err/np.nanmedian(self.flux)
            self.flux -= 1 # zero-center
        else:
            raise ValueError("{mode} is not an available mode. Pick one of: `l1`, `l2`, `max`, or `median`.")
        


    def pad(self, cadences = 500, mode = 'endpt', side = 'both'):
        """ Extend a light curve by a given number of cadences.
        Parameters
        ----------
        cadences : int
            Number of cadences to extend the light curve by.

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
            If input value for `cadences` is not an integer

        """
        if not type(cadences) == int: 
            raise ValueError('`cadences` must be an integer. Please try again.')
        
        if side in ['left', 'both']:
            t_left = np.linspace(self.time[0]-(cadences*20*u.second), self.time[0]-20*u.second, num=cadences)
            t_ext = np.append(t_left, self.time)

            prelim_f_left = np.full((1, cadences), self.flux.value[0]) # create an array of fluxes w value of the first point in the lc
            std_left = np.std(self.flux[:cadences]) # take std of first 'cadences' number of observations in the lc
            f_left = np.random.normal(prelim_f_left, std_left) # add noise to extended fluxes
            f_ext = np.append(f_left, self.flux.value)

            ferr_left = np.full((1, cadences), std_left) # CHECK THIS...
            ferr_ext = np.append(ferr_left, self.flux_err)


        if side in ['right', 'both']: 
            t_right = np.linspace(self.time.max() + 20*u.second, self.time.max()+(cadences*20*u.second), num=cadences) 
            prelim_f_right= np.full((1, cadences), self.flux.value[-1])
            std_right = np.std(self.flux[-1*cadences:]) 
            f_right = np.random.normal(prelim_f_right, std_right)

            ferr_right = np.full((1, cadences), std_right) # CHECK THIS...

            if side == 'both': 
                t_ext = np.append(t_ext, t_right)
                f_ext = np.append(f_ext, f_right)
                ferr_ext = np.append(ferr_ext, ferr_right)
            else: 
                t_ext = np.append(self.time, t_right)
                f_ext = np.append(self.flux, f_right)
                ferr_ext = np.append(self.flux_err, ferr_right)

        
        elif side not in ['left', 'right', 'both']: 
            raise ValueError(f"'{side}' is not an option. Choose `left`, `right`, or `both`.")
        print(t_ext)
        return t_ext, f_ext, ferr_ext

    
    

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


    def rm_upper_outliers(self, s=3.0):
        return lk.LightCurve.remove_outliers(self, sigma_upper = s, sigma_lower = float('inf'))
    
    def segment(self, outdir='segmented_df', gap=600):
        """ Split a light curve into segments by gaps in observations, 
            create csv files for each segment with time, flux, and flux_error values

        Parameters
        ----------
        outdir : str
            Directory to save files for light curve segments into

        gap : int
            Number of seconds between observations to split a light curve into visits .. how to better explain

        Returns
        -------
        csvs for segmented light curve saved into the outdir directory

        """
        # look at self.time
        newdf = pd.DataFrame(data=np.asarray([self.time.value*86400, self.flux.value, self.flux_err.value]).T, columns=['time_s', 'flux', 'flux_err'])
        # delta_t = pd.Series(self.time.value*86400).diff() # FIGURE THIS OUT W THE UNITS....
        newdf['delta_t'] = newdf['time_s'].diff()
        newdf['delta_t'] = round(newdf['delta_t'], 2)
        gap_indices = newdf.index[newdf['delta_t'] > gap]
        gap_indices.insert(0, 0)
        segment_lengths = []
        for i in range(len(gap_indices)-1): 
            start_index = gap_indices[i]
            end_index = gap_indices[i+1]
            cutdf = newdf.iloc[start_index : end_index]
            segment_lengths.append(len(cutdf))
            cutdf.to_csv(outdir+'/'+self.id+'_'+str(self.sector)+'_'+str(i)+'.csv', sep=',')
        
        # plt.hist(segment_lengths, bins=[50, 500, 1000, 2000, 5000, 10000, 15000, 20000, 25000, 30000])
        # plt.xlabel("segment length (cadences)")
        # # plt.savefig('/Users/veraberger/nasa/figures/histogram_segments_sep_by_10_min.png', dpi=200, bbox_inches='tight')
        # plt.show()
        # print(segment_lengths.count(1))
        # print(len(segment_lengths))
        # return segment_lengths

        # if there are gaps between one point and another that a greater than 600*u.second, split the lc.
        # make a new column which is t_i - t_{i-1} the difference between times at adjacent points. then for pts in that new col. but that's a single sector way of doing it
        # alternatively, ask what the greatest time separation is between two gaps, see if it's greater than my limit, do recursion.
        # save all the segmented time, flux, ferr into new files in outdir/TICID-sector-visitnum.csv
    
