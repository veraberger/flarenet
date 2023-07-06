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



__all__ = ['FlareLightCurve'] 


class FlareLightCurve:
    """
    NEED A DOCSTRING
    note: here self.flux is not sap flux. easy change but remember it
    """

    @staticmethod
    def from_lightkurve(path):
        return lk.read(path) # i'm not sure about this


    def normalize(self, norm='median'):
        """ Normalize flux and flux uncertainty values in a light curve
            might want to remove all the options and just normalize -1 to 1

        Parameters
        ----------
        norm : str
            The norm to use for normalization
            "median" : Median flux
            "max" : Maximum flux
            "l1" : Sum of the magnitudes of the fluxes
            "l2" : Euclidean norm: square root of the sum of squares of the fluxes
            
        Returns
        -------
        what is the notation for this? 
            A copy of the input light curve with normalized flux and flux_err values
        """
        if norm == 'median':
            return lk.normalize(self)  # is this good notation
        elif norm == 'max': # not really this easy. see lk normalize documentation for edge cases
            lc = self.copy()
            max_flux = np.nanmax(lc.flux)
            lc.flux = lc.flux/max_flux
            lc.flux_err = lc.flux_err/max_flux
            return lc
        elif  norm == 'l1' or norm == 'l2':
            raise NotImplementedError
        else:
            raise ValueError(f"{norm} is not an available norm. Pick one of: `l1`, `l2`, `max`, or `median`.")
    

    def invert(self):
        """
       Inverts fluxes and flux errors in a light curve

        Returns
        -------

        inv : array-like
            Float values of the inverses of all the input elements.

        """
        # maskedarr = np.ma.masked_where(self.flux==0, self.flux) # trying to work w fluxes of 0
        # inv = (1/maskedarr).filled(fill_value=0)
        self.flux = 1/self.flux # can we assume that the flux is never 0? although maybe we don't need this func at all anymore
        self.flux_err = self.flux_err/self.flux**2
        return self

    def rm_upper_outliers(self, sigma=3.0):
        """
        Wrapper function for lightkurve remove_outliers function
        to remove upper outliers using sigma clipping.

        Parameters
        ----------
        sigma : float
            Number of standard deviations above the mean for which to remove outliers

        Returns
        -------
            Light curve with upper outliers removed
        """
        return lk.LightCurve.remove_outliers(self, sigma_upper=sigma, sigma_lower=float('inf'))
    
    def segment(self, outdir='segmented_df', gap=600):
        """ This is sort of a placeholder function; 
            I wrote it because I think we will segment light curves by orbit or some gap that is large enough that we do not want to fill it in.
            I doubt that csv file output is the move. 
            Feedback especially appreciated.

            Split a light curve into segments by gaps in observations, 
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
        # i'm not even sure if we need to save the times
        newdf = pd.DataFrame(data=np.asarray([self.time.value*86400, self.flux.value, self.flux_err.value]).T, columns=['time_s', 'flux', 'flux_err']) # assumes time in seconds..
        # look at gaps between each observation
        newdf['delta_t'] = newdf['time_s'].diff()
        newdf['delta_t'] = round(newdf['delta_t'], 2)
        # find indices where gap is large enough to warrant splitting the lc
        gap_indices = newdf.index[newdf['delta_t']>gap]
        gap_indices.insert(0, 0)
        segment_lengths = []
        for i in range(len(gap_indices)-1): 
            start_index = gap_indices[i]
            end_index = gap_indices[i+1]
            cutdf = newdf.iloc[start_index:end_index]
            segment_lengths.append(len(cutdf))
            cutdf.to_csv(outdir+'/'+self.id+'_'+str(self.sector)+'_'+str(i)+'.csv', sep=',')

    def get_windows(self, n=600):
        """
        Takes in a light curve and returns windows of size n
        Currently redundant bc it saves n times as many points as we begin with.

        Parameters
        ----------
        n : int
            Window size
        
        Returns
        -------
        flux_data: an np.ndarray of flux and flux error values shape n_flux_points x window_length
                    Where necessary this will be padded with the endpoint of the light curve, with added random gaussian noise.
        """
        
        f_left = np.random.normal(np.full((1, n//2), self.flux.value[0]), np.std(self.flux[:n//2])) 
        f_right = np.random.normal(np.full((1, n//2), self.flux.value[-1]), np.std(self.flux[-1*n//2:])) 
        padded_flux = np.hstack([f_left, self.flux.value, f_right])
        flux_data = np.asarray([padded_flux[idx:idx+n] for idx in
                                 np.arange(0, len(self.flux.value))-n//2])
        return flux_data