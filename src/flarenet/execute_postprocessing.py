import numpy as np
import pandas as pd
#from astropy.io import ascii
import os
from .flare_model import generate_flares
from .postprocessing import *
from .utils import *
from . import PACKAGEDIR



def create_trainingset(num_lcs : int = 1, 
                    save_plots : bool = False,
                    num_flares : Union[int, str] = 100,
                    output_dir : str = "training_data/injected_flares",
                    add_pulsation : bool = False,
                    add_rrlyrae : bool = False,
                    cloud : bool = False, 
                    ):
    """
    This function can be used to generate traning data by injecting flares into quiet TESS lightcurves
    
    Parameters:
    -----------
    num_lcs: int, str
        Number of stars to inject flares into. Can input 'all' for all quiet lcs
    save_plots : bool = False
        Save plots of the injected lightcurves
    num_flares : Union[int, str] = 100,
        Number of flares to inject for each lightcurve. 
        Default is 100, with covers ~10 of a TESS sector
    output_dir : str
        path to the directory to store the lightcurves with injected flares
    add_pulsation : bool 
        Whether to inject additional randomly generated pulstion. Default is false
    add_rrlyrae : bool
        Whether to inject additional rr lyrae-type variability. Default is false
    cloud : bool
        Whether to use data stored by MAST on AWS. 
        If false, the files will be downloaded locally. 
    
    
    Returns:
    -------
    None; however, lightcurves injected flares will be saved as csv files
    """
    if not os.path.exists(f"{PACKAGEDIR}/{output_dir}"):
        os.mkdir(f"{PACKAGEDIR}/{output_dir}")
    quietdf = pd.read_csv(f"{PACKAGEDIR}/supplemental_files/ids_sectors_quietlcs.txt", sep=' ', header=0, usecols=['TIC', 'sector'])
    if isinstance(num_lcs, str):
        if num_lcs.lower() == 'all':
            num_lcs = len(quietdf)
            print(f"Injecting flares into all {len(quietdf)} lightcurves")
        else:
            print("Must specify the number of lcs to produce. Set num_lcs to 'all' if you want to use all available targets.")
    idxs = np.random.choice(len(quietdf), size=num_lcs, replace=False)
    for ii, idx in enumerate(idxs):
        id = quietdf.iloc[idx]['TIC']
        sector = quietdf.iloc[idx]['sector']

        print(f"Creating flares for {id} Sector {sector}")
        extra_fname = ''
        if add_pulsation:
            extra_fname = extra_fname + '_pulsations'
        if add_rrlyrae:
            extra_fname = extra_fname + '_rrlyrae'


        mytpf = TessStar(f"TIC {id}", sector=sector, exptime=20, download_dir='tpfs/', cloud = cloud)
        mytpf.get_metadata()
        # inject flares such that approximately <flare_frac> of the LC is covered in flares
        flares, params = generate_flares(mytpf.lc.time.value, num_flares = num_flares) 

        if not os.path.exists(f"{PACKAGEDIR}/{output_dir}/artificial_flare_params"):
            os.mkdir(f"{PACKAGEDIR}/{output_dir}/artificial_flare_params")
        np.save(f"{PACKAGEDIR}/{output_dir}/artificial_flare_params/{mytpf.ticid}_{mytpf.sector}{extra_fname}_flareparams.npy", params)
        
        # get array of 0 (no flare) or 1 (flare)
        # Threshold is 1 standard deviation BEFORE cosmic rays were added back in
        threshold = (mytpf.lc_std / np.nanmedian(mytpf.lc.flux)).value
        #print(f"Threshold: {threshold}")
        flare_flags = np.where(flares > threshold, 1, 0)

        modified_flux = inject_flares(flares, mytpf.lc.flux.value) # inject them onto the lc
        # Inject one signal mimicing an asteroid moving over the star in each lightcurve. 
        modified_flux = inject_asteroid_crossing(mytpf.lc.time.value, modified_flux)

        # Optionally add other types of variability to help make the model more robust
        if add_pulsation:
            modified_flux = inject_stellar_pulsations(mytpf.lc.time.value, modified_flux)

        if add_rrlyrae:
            modified_flux = inject_rr_lyrae(mytpf.lc.time.value, modified_flux)

        if save_plots:
            fig, ax = plt.subplots(2, figsize=(14,5), sharex=True)
            ax[0].plot(mytpf.lc.time.value, mytpf.lc.flux.value, color='black')
            ax[1].plot(mytpf.lc.time.value, modified_flux, color='tomato')   
            ax[0].set_title("original lc")  
            ax[1].set_title("added flares")  
            ax[1].set_ylim(-5,15)    
            plt.savefig(f"{PACKAGEDIR}/training_data/injected_flares/{mytpf.ticid}_addedflares{extra_fname}.png")
            plt.close()

        # split into orbits and save .npy files. code currently splits by largest gap in the data
        mytpf.make_orbit_files(output_dir=output_dir, plot=save_plots,extra_fname_descriptor=extra_fname, 
                               modified_flux=modified_flux, labels = flare_flags)



def create_predictionset(ticid : int, 
                            sector : int = None,
                            save_plots : bool = False,
                            output_dir : str = "prediction_data/" ):

    # If sector not provided, this will just return the first available sector
    mytpf = TessStar(f"TIC {ticid}", sector=sector, exptime=20, download_dir='tpfs/')
    mytpf.make_orbit_files(output_dir=output_dir, plot=save_plots)

        



