import numpy as np
import pandas as pd
from astropy.io import ascii

from postprocessing import *


# # **** read in gunther flares file for flare parameters ****
flaredf = ascii.read('supplemental_files/gunther_flares_all.txt')
all_ampls = flaredf['Amp']
all_fwhms = flaredf['FWHM']


cols = ['TIC', 'sector']
# read in list of TIC IDs and corresponding sectors for quiet light curves in our trianing set
quietdf = pd.read_csv('supplemental_files/ids_sectors_quietlcs.txt', sep=' ', header=0, usecols=cols)
for (id, sector) in zip(quietdf['TIC'], quietdf['sector']):

    # # download tpf, convert to LC with cosmic rays, plot LC w/ CRs marked in red
    mytpf = TessStar('TIC '+str(id), sector=sector, exptime=20, download_dir='tpfs/')

    try:
        mytpf.get_metadata()
        # could probably be more creative and have flares and flareArr of 0s and 1s be made concurrently, and stop generating flares when some % of the points in flareArr have a 1 in them
        flares, params = mytpf.generate_flares(all_ampls, all_fwhms, fraction_flare=0.1) # inject flares such that 10% of the LC is covered in flares (determined by FWHM, so much more of the lc actually has a flare)
        np.save('artificial_flare_params/'+str(mytpf.targetid)+'_'+str(mytpf.sector)+'_flareparams.npy', params)

        flareArr = TessStar.get_flareArr(flares, threshold=0.008) # get array of flares of lc length to inject onto the lc
        inj_flux = mytpf.inject_flares(flares) # inject them onto the lc

        # split into orbits and save .npy files. code currently splits by greatest gap in the data, not great
        orbit1, orbit2 = mytpf.make_orbit_files(injected_flux=inj_flux, flareArr=flareArr, datadir='training_data/') #

        """
        # plot injected flares and where flares are marked to exist  
        print(len(flareArr[flareArr==1])/len(flareArr))
        fig, ax = plt.subplots(2, figsize=(13,9), sharex=True, sharey=True)
        ax[0].plot(mytpf.lc.time.value, inj_flux, color='black')
        ax[0].plot(mytpf.lc[flareArr==1].time.value, inj_flux[flareArr==1], color='red', marker='o', ls='none', markersize=2)
        ax[1].plot(mytpf.lc.time.value, flares, color='black')
        plt.show()
        """
    
    except: # get rid of this and have it fail gracefully for errors we can anticipate
        print(id, sector) 
    

    """
    # plot the LC, CRs, and centroid shift
    fix, ax = plt.subplots(3, figsize=(10, 7), sharex=True)
    ax[0].plot(mytpf.lc.time.value, mytpf.lc.flux.value, color='black', label='lc with CRs')
    ax[0].plot(mytpf.lc[mytpf.crArr==1].time.value, mytpf.lc[mytpf.crArr==1].flux.value, color='black', label='lc with CRs')
    ax[1].plot(mytpf.tpf.time.value, mytpf.crArr, ls='none', marker='o', markersize=1,  color='tab:purple', label='CRs')
    ax[2].plot(mytpf.tpf.time.value, mytpf.c_dist, label='centroid shift')
    ax[0].legend()
    ax[1].legend()
    ax[2].legend()
    ax[1].plot(mylc.time.value[crArr==1], mylc.flux.value[crArr==1], color='r', marker='o', ls='none')
    plt.show()
    """

    
