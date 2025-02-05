import numpy as np
import pandas as pd
#from astropy.io import ascii
import os
from .postprocessing import *
from .utils import normalize_flux
from . import PACKAGEDIR


# # **** read in gunther flares file for flare parameters ****
# flaredf = ascii.read('supplemental_files/gunther_flares_all.txt')
# all_ampls = flaredf['Amp']
# all_fwhms = flaredf['FWHM']

def make_injected_flare_trainingset(num_lcs : int = 1, 
                                    save_plots : bool = False,
                                    #flare_frac : float = 0.1,
                                    num_flares : int = 100,
                                    output_dir : str = "training_data/injected_flares",
                                    add_pulsation : bool = False,
                                    add_rrlyrae : bool = False,
                                    cloud : bool = False, 
                                    ):
    """
    This function can be used to generate traning data by injecting flares into quiet TESS lightcurves
    
    Parameters:
    -----------
    num_lcs: int
        Number of stars to inject flares into. 
    verbose: bool
        True prints out information about the processing and provides diagnostic plots 
    flare_frac: float or list of floats of length num_lcs
        The fraction of the lightcurve to be covered with flares
    
    
    Returns:
    -------
    None; however, the datasets with injected flares will be saved npy files
    """
    if not os.path.exists(f"{PACKAGEDIR}/{output_dir}"):
        os.mkdir(f"{PACKAGEDIR}/{output_dir}")
    quietdf = pd.read_csv(f"{PACKAGEDIR}/supplemental_files/ids_sectors_quietlcs.txt", sep=' ', header=0, usecols=['TIC', 'sector'])
    idxs = np.random.choice(len(quietdf), size=num_lcs, replace=False)
    for ii, idx in enumerate(idxs):
    #for ii, (id, sector) in enumerate(np.random.choice(zip(quietdf['TIC'], quietdf['sector'])), size=num_lcs, replace=False):
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
        flares, params = mytpf.generate_flares(num_flares = num_flares) 
        if not os.path.exists(f"{PACKAGEDIR}/{output_dir}/artificial_flare_params"):
            os.mkdir(f"{PACKAGEDIR}/{output_dir}/artificial_flare_params")
        np.save(f"{PACKAGEDIR}/{output_dir}/artificial_flare_params/{mytpf.ticid}_{mytpf.sector}{extra_fname}_flareparams.npy", params)
        
        # get array of 0 (no flare) or 1 (flare)
        # Threshold is 1 standard deviation BEFORE cosmic rays were added back in
        threshold = (mytpf.lc_std / np.nanmedian(mytpf.lc.flux)).value
        #print(f"Threshold: {threshold}")
        
        mytpf.flare_flags = np.where(flares > threshold, 1, 0)
        #mytpf.get_flare_flags(flares, threshold=threshold) 

        mytpf.inject_flares(flares) # inject them onto the lc
        # Inject a signal mimicing an asteroid moving over the star. 
        mytpf.inject_asteroid_crossing()


        if add_pulsation:
            mytpf.inject_stellar_pulsations()

        if add_rrlyrae:
            mytpf.inject_rr_lyrae()

        if save_plots:
            fig, ax = plt.subplots(2, figsize=(14,5), sharex=True)
            ax[0].plot(mytpf.lc.time.value, mytpf.lc.flux.value, color='black')
            ax[1].plot(mytpf.lc.time.value, mytpf.flux_with_flares, color='tomato')   
            ax[0].set_title("original lc")  
            ax[1].set_title("added flares")  
            ax[1].set_ylim(-10,10)    
            plt.savefig(f"{PACKAGEDIR}/training_data/injected_flares/{mytpf.ticid}_addedflares{extra_fname}.png")
            plt.close()

        # split into orbits and save .npy files. code currently splits by largest gap in the data
        mytpf.make_orbit_files(output_dir=output_dir, plot=save_plots,extra_fname_descriptor=extra_fname)



def make_prediction_dataset(ticid : int, 
                            sector : int = None,
                            save_plots : bool = False,
                            output_dir : str = "prediction_data/" ):

    # If sector not provided, this will just return the first available sector
    mytpf = TessStar(f"TIC {ticid}", sector=sector, exptime=20, download_dir='tpfs/')
    mytpf.make_orbit_files(output_dir=output_dir, plot=save_plots)

        


def add_flares(target_id, sector, num_flares, output_dir = None, plot=False, extra_fname_descriptor=''):
    mytpf = TessStar(f"TIC {target_id}", sector=sector, exptime=20, download_dir='tpfs/')
   #try:
    mytpf.get_metadata()
    # inject flares such that 10% of the LC is covered in flares (determined by FWHM, so much more of the lc actually has a flare)
    flares, params = mytpf.generate_flares(num_flares=num_flares) 
    if not os.path.exists(f"{PACKAGEDIR}/{output_dir}/artificial_flare_params/"):
        os.mkdir(f"{PACKAGEDIR}/{output_dir}/artificial_flare_params/")
    np.save(f"{PACKAGEDIR}/{output_dir}/artificial_flare_params/{mytpf.ticid}_{mytpf.sector}_{extra_fname_descriptor}_flareparams.npy", params)
    
    # get array of 0 (no flare) or 1 (flare)
    # Threshold is 1 standard deviation BEFORE cosmic rays were added back in
    threshold = (mytpf.lc_std / np.nanmedian(mytpf.lc.flux)).value
    print(f"Threshold: {threshold}")
    mytpf.get_flare_flags(flares, threshold=threshold) 

    mytpf.inject_flares(flares, plot=plot) # inject them onto the lc
    # split into orbits and save .npy files. code currently splits by largest gap in the data
    mytpf.make_orbit_files(output_dir=output_dir, plot=plot)





    
    #except: # get rid of this and have it fail gracefully for errors we can anticipate
    #    print(f"Cannot process file for {target_id}, Sector {sector}") 


    
'''cols = ['TIC', 'sector']
# read in list of TIC IDs and corresponding sectors for quiet light curves in our trianing set
quietdf = pd.read_csv('supplemental_files/ids_sectors_quietlcs.txt', sep=' ', header=0, usecols=cols)
for (id, sector) in zip(quietdf['TIC'], quietdf['sector']):

    # # download tpf, convert to LC with cosmic rays, plot LC w/ CRs marked in red
    mytpf = TessStar('TIC '+str(id), sector=sector, exptime=20, download_dir='tpfs/')

    try:
        mytpf.get_metadata()
        # inject flares such that 10% of the LC is covered in flares (determined by FWHM, so much more of the lc actually has a flare)
        flares, params = mytpf.generate_flares(all_ampls, all_fwhms, fraction_flare=0.1) 
        np.save('artificial_flare_params/'+str(mytpf.targetid)+'_'+str(mytpf.sector)+'_flareparams.npy', params)
        
        # get array of 0 (no flare) or 1 (flare)
        # Threshold is 1 standard deviation BEFORE cosmic rays were added back in
        flare_flags = TessStar.get_flare_flags(flares, threshold=(mytpf.lc_std / np.median(mytpf.lc.flux)).value) 

        inj_flux = mytpf.inject_flares(flares) # inject them onto the lc
        inj_flux_normal = normalize_flux(inj_flux, type='standard')

        # split into orbits and save .npy files. code currently splits by greatest gap in the data, not great
        orbit1, orbit2 = mytpf.make_orbit_files(injected_flux=inj_flux, flare_flags=flare_flags, datadir='training_data/') #


    
    except: # get rid of this and have it fail gracefully for errors we can anticipate
        print(id, sector) 
    

    """
    # plot the LC, CRs, and centroid shift
    fix, ax = plt.subplots(3, figsize=(10, 7), sharex=True)
    ax[0].plot(mytpf.lc.time.value, mytpf.lc.flux.value, color='black', label='lc with CRs')
    ax[0].plot(mytpf.lc[mytpf.cr_flags==1].time.value, mytpf.lc[mytpf.cr_flags==1].flux.value, color='black', label='lc with CRs')
    ax[1].plot(mytpf.tpf.time.value, mytpf.cr_flags, ls='none', marker='o', markersize=1,  color='tab:purple', label='CRs')
    ax[2].plot(mytpf.tpf.time.value, mytpf.c_dist, label='centroid shift')
    ax[0].legend()
    ax[1].legend()
    ax[2].legend()
    ax[1].plot(mylc.time.value[cr_flags==1], mylc.flux.value[cr_flags==1], color='r', marker='o', ls='none')
    plt.show()
    """
'''


