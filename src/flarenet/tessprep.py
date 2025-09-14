import numpy as np
import os
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

def get_TESS_data(ticid : Union[str, int, SkyCoord],
        sector : int = None,
        cloud : bool = False,
        exptime : int = 20, 
        tpf : bool = True,
    ):
    """Search for target pixel file (TPF) data for a specified target

    Parameters
    ----------
    ticid : str, int, or SkyCoord
        identifying name or coordinates for your target star
    sector : int, optional
        sector of TESS observations to serach for, default None will search for all available sectors
    cloud : bool, optional
        Whether to access data from AWS cloud storage, default False will search the MAST server
    exptime : int, optional
        desired exposure time for the TESS observations, by default 20
    tpf : bool
        flag to indicate whether to return TESS TPF data. If false, will download lightcurve data instead. 

    Returns
    -------
    tpf
        a lightkurve TPF
    """
    if isinstance(ticid, int):
        ticid = f"TIC {ticid}"
    if isinstance(ticid, str):
        if ticid[:3] != 'TIC':
            ticid = f"TIC {ticid}"

    if cloud == True:
        sr = lksearch.TESSSearch(ticid, sector=sector, exptime=exptime, pipeline='SPOC')
        if tpf:
            cloud_uri = sr.cubedata.cloud_uris
        else:
            cloud_uri = sr.timeseries.cloud_uris
        tess_data = lk.io.read(cloud_uri[0])
    else:
        sr = lksearch.TESSSearch(ticid, sector=sector, exptime=exptime, pipeline='SPOC')
        if tpf:
            local_path = sr.cubedata[0].download()['Local Path'].values[0]
        else:
            local_path = sr.timeseries[0].download()['Local Path'].values[0]
        tess_data = lk.io.read(local_path)
        
    return tess_data

class TessStar(object):
    """
    Contains functions to search for, download, and prepare TESS 20-s data for the flarenet model.
    After initializing the object, you can inject flares into the data, plot the lightcurve, 
    and save data as a csv for use by the flarenet model. 
    """

    def __init__(self, 
                 ticid : Union[str, int, SkyCoord],
                 sector : int = None, 
                 exptime : Union[int, float] = 20, 
                 add_cosmic_rays : bool = True, 
                 cloud : bool = False,
                 ):
        """
        Instantializes the object by downloading the TPF through lightkurve 

        Parameters
        ----------
        ticid : str, int, or astropy.coordinates.SkyCoord object
            Input target ID. See lightkurve search_lightcurve API for details.
        sector : int
            TESS sector within which to search for data.
            If None, then what? Download all and stitch, or raise an error?
        exptime : int, float
            Cadence of data product in seconds. 
        add_cosmic_rays : bool
            If True, add back in the cosmic rays otherwise removed by NASA's TESS processing pipeline. 
            Defaults to True.
        cloud : bool
            If true, download TPF files from the cloud storage location
    
        """
        self.ticid = ticid
        self.sector = sector
        self.exptime = exptime 
        self.add_cosmic_rays = add_cosmic_rays
        self.add_flares = False
        self.cloud = cloud
        self._prepare_TESS_data() # get TESS data and (optionally) inject CRs back into data
        '''if train:
            self.inject_training_flares(save_plots=save_plots)


        self.save_data(
            output_dir = 'training_data',
            plot = save_plots)'''
        
    
    def plot_lc(self, 
                save_plot=True,
                output_dir='training_data/plots'
                ):
        fig, ax = plt.subplots(1, figsize=(12,4))
        ax.set_title(f"{self.ticid} Sector {self.sector}")
        ax.plot(self.lc.time.value, self.lc.flux.value, zorder=0, label='pdcsap data')
        if self.add_flares: #If we injected flares, plot that information
            flare_mask = self.flare_labels == 1
            ax.scatter(self.lc.time[flare_mask].value, self.lc.flux[flare_mask].value, c='C1', s=3, label='injected_flares', zorder=2)
        
            
        if self.add_cosmic_rays:
            cr_mask = [m == 1 for m in self.cr_flags]
            ax.scatter(self.lc.time[cr_mask].value, self.lc.flux[cr_mask].value, marker='x',c='C2', s=3, label=f"{sum(cr_mask)} cosmic rays",zorder=1)
        ax.set_xlim(self.lc.time.value[0], self.lc.time.value[-1])
        plt.legend()
        if save_plot:
            plt.savefig(f"{PACKAGEDIR}/{output_dir}/{self.ticid}_{self.sector}.png")
        else:
            plt.show()



    def _prepare_TESS_data(self,
                      ):
        """
        Downloads a TPF using lightkurve, optionally injects cosmic rays, 
        computes a light curve, and returns each.
        
        Returns
        -------
        Nothing; however, this function add the lc to the class properties.
        The final lc is normalized, and 'flux' corresponds to the normalized PDCSAP_FLUX value

        """
        if self.add_cosmic_rays:
            tpf = get_TESS_data(self.ticid, sector=self.sector, exptime=self.exptime, cloud=self.cloud, tpf=True)
            if not self.exptime == 20:
                raise TypeError("Cosmic ray injection is only possible for 20-second datasets.")
            # inject cosmic rays
            cosmic_ray_cube = get_cosmicrays(tpf)
            tpf = tpf+cosmic_ray_cube
            lc = tpf.to_lightcurve(aperture_mask=tpf.pipeline_mask).normalize()
            #to_lightcurve does not save all meta data, so add the relevant stuff back in
            lc.meta['TESSMAG'] = tpf.get_header(ext=0)['TESSMAG']
            lc.meta['TEFF'] = tpf.get_header(ext=0)['TEFF']
            lc.meta['RADIUS'] = tpf.get_header(ext=0)['RADIUS']
            lc.meta['CDPP1_0'] = tpf.get_header(ext=1)['CDPP1_0']
            lc.meta['CROWDSAP'] = tpf.get_header(ext=1)['CROWDSAP']
            lc.meta['FLFRCSAP'] = tpf.get_header(ext=1)['FLFRCSAP']
            lc.meta['LOGG'] = tpf.get_header(ext=0)['LOGG']
            lc_cr_arr = list(map(int, (np.nansum(cosmic_ray_cube[:, tpf.pipeline_mask], axis=1)>0)))

            self.cr_flags = lc_cr_arr
        else:
            lc = get_TESS_data(self.ticid, sector=self.sector, exptime=self.exptime, cloud=self.cloud, tpf=False)
            lc = lc.normalize()
            self.cr_flags = np.zeros_like(lc.flux.value)

        #Get lc standard deviation before adding flares. Detrend the lc before getting the std
        flat = lc.flatten(return_trend=False)
        self.lc_std = np.nanstd(flat['flux'])
 
        self.lc = lc
        #return lc

        
    def inject_training_flares(self, 
                        save_plot : bool = False,
                        num_flares : Union[int, str] = 100,
                        output_dir : str = f"training_data",
                        verbose : int = 1,
                        ):
        """
        This function can be used to generate traning data by injecting flares into quiet TESS lightcurves
        
        Parameters:
        -----------
        save_plots : bool = False
            Save plots of the injected lightcurves
        num_flares : Union[int, str] = 100,
            Number of flares to inject for each lightcurve. 
            Default is 100, with covers ~10 of a TESS sector
        output_dir : str
            path to the directory to store information (flare properties and optionally plots) regarding the injected flares
        verbose : bool
            If 1, print to screen information about the status of the injection
        
        Returns:
        -------
        None; however, lightcurves injected flares will be saved as csv files
        """
        if not os.path.exists(f"{PACKAGEDIR}/{output_dir}"):
            os.mkdir(f"{PACKAGEDIR}/{output_dir}")
        '''quietdf = pd.read_csv(f"{PACKAGEDIR}/supplemental_files/ids_sectors_quietlcs.txt", sep=' ', header=0, usecols=['TIC', 'sector'])
        if isinstance(num_lcs, str):
            if num_lcs.lower() == 'all':
                num_lcs = len(quietdf)
                print(f"Injecting flares into all {len(quietdf)} lightcurves")
            else:
                print("Must specify the number of lcs to produce. Set num_lcs to 'all' if you want to use all available targets.")'''
        #idxs = np.random.choice(len(quietdf), size=num_lcs, replace=False)
        #for ii, idx in enumerate(idxs):

        if verbose:
            print(f"Creating flares for {self.ticid} Sector {self.sector}")


        self.get_metadata()
        # inject flares
        flares, params, valid_flare_indices = generate_flares(self.lc.time.value, num_flares = num_flares) 

        if not os.path.exists(f"{PACKAGEDIR}/{output_dir}/artificial_flare_params"):
            os.mkdir(f"{PACKAGEDIR}/{output_dir}/artificial_flare_params")
        np.save(f"{PACKAGEDIR}/{output_dir}/artificial_flare_params/{self.ticid}_{self.sector}_flareparams.npy", params)
        
        # get array of 0 (no flare) or 1 (flare)
        # Threshold is 1.5 standard deviation BEFORE cosmic rays were added back in
        threshold = 1.5 * self.lc_std #(self.lc_std / np.nanmedian(self.lc.flux)).value
        flare_flags = np.where(flares > threshold, 1, 0)
        flare_flags[~valid_flare_indices] = 0

        self.lc['modified_flux'] = self.lc.flux.value  + (flares * self.lc.flux.value)
        self.flare_labels = flare_flags
        self.add_flares = True
        #inject_flares(flares, self.lc.flux.value) # inject them onto the lc
        


        if save_plot:
            if not os.path.exists(f"{PACKAGEDIR}/{output_dir}/plots"):
                os.mkdir(f"{PACKAGEDIR}/{output_dir}/plots")
            fig, ax = plt.subplots(2, figsize=(14,5), sharex=True, sharey=True)
            ax[0].plot(self.lc.time.value, self.lc.flux.value, color='black')
            ax[1].plot(self.lc.time.value, self.lc.modified_flux, color='tomato')   
            ax[0].set_title("original lc")  
            ax[1].set_title("added flares")  
            #ax[1].set_ylim(-5,15)    
            ax[0].set_xlim(self.lc.time.value[0], self.lc.time.value[-1])
            plt.savefig(f"{PACKAGEDIR}/{output_dir}/plots/{self.ticid}_{self.sector}_addedflares.png")
            #plt.show()
            plt.close()
        
        if verbose:
            print(f"Flares successfully injected in TIC {self.ticid} Sector {self.sector}")



    

    def get_metadata(self): 
        """ 
        Get relevant metadata from the headers of a TESS TargetPixelFile object
            If outdir is specified, save the array into an .npy file


        Returns
        -------
        metaArr : array
            Right ascension, declination, TESS magnitude, effective temperature, stellar radius, sector, CDPP, crowdsap, camera, CCD, and log10 surface gravity
        """
        

        return np.array([self.lc.ra, self.lc.dec,  
                         self.lc.meta['TESSMAG'], 
                         self.lc.meta['TEFF'],  
                         self.lc.meta['RADIUS'], 
                         self.lc.sector, 
                         self.lc.meta['CDPP1_0'],  
                         self.lc.meta['CROWDSAP'], 
                         self.lc.camera, self.lc.ccd,  
                         self.lc.meta['LOGG']])
    

    def save_data(self, 
            train = True,
            #plot = False
            ):
        """
        Take a light curve and supplemental information for some target and sector
        Save the output into an npy file. 

        Parameters
        ----------
        train : bool
            Whether data being saves will be used for model training or prediction
        
        Returns
        -------
        fname :
            file path to the saved csv file
        """

        if train:
            output_dir = f'training_data/labeled_data'
        else:
            output_dir = f'prediction_data'
        if hasattr(self.lc, 'modified_flux'): # If flares have been added (for training)
            df = pd.DataFrame(data={'time':self.lc.time.value, 
                                    'og_flux':self.lc.flux.value, 
                                    'flux_err':self.lc.flux_err.value, 
                                    'quality':self.lc.quality.value, 
                                    'cr_flags':self.cr_flags, 
                                    'flux': self.lc.modified_flux, 
                                    'flare_flags':self.flare_labels
                                    })

        else:
            df = pd.DataFrame(data={'time':self.lc.time.value, 
                                    'flux':self.lc.flux.value, 
                                    'flux_err':self.lc.flux_err.value, 
                                    'quality':self.lc.quality.value, 
                                    'cr_flags':self.cr_flags, 
                                    'flare_flags': np.zeros_like(self.lc.flux.value)}
            )


        fname = f"{PACKAGEDIR}/{output_dir}/{self.ticid}_{self.sector}_data.csv"
        df.to_csv(fname, index=False)

        return fname
        

    

            

    
