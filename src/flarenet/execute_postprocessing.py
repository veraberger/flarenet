import numpy as np
import pandas as pd
from astropy.io import ascii

from postprocessing import *


# # ******* gunther flares ******
flaredf = ascii.read('supplemental_files/gunther_flares_all.txt')
all_ampls = flaredf['Amp']
all_fwhms = flaredf['FWHM']


cols = ['TIC', 'sector']
quietdf = pd.read_csv('supplemental_files/ids_sectors_quietlcs.txt', sep=' ', header=0, usecols=cols)
for (id, sector) in zip(quietdf['TIC'], quietdf['sector']):

    # # download tpf, convert to LC with cosmic rays, plot LC w/ CRs marked in red
    mytpf = InputTPF('TIC '+str(id), sector=sector, exptime=20, download_dir='tpfs/')

    try:
        mytpf.get_metadata()

        flares, params = mytpf.flare_generator(all_ampls, all_fwhms)
        np.save('artificial_flare_params/'+str(mytpf.targetid)+'_'+str(mytpf.sector)+'_flareparams.npy', params)

        flareArr = InputTPF.get_flareArr(flares, threshold=0.008)
        inj_flux = mytpf.inject_flare(flares)

        # split into orbits and save .npy files. code currently splits by greatest gap in the data, not great
        orbit1, orbit2 = mytpf.make_orbit_files(injected_flux=inj_flux, flareArr=flareArr, datadir='training_data/')

        """
        # plot injected flares and where flares are marked to exist  
        print(len(flareArr[flareArr==1])/len(flareArr))
        fig, ax = plt.subplots(2, figsize=(13,9), sharex=True, sharey=True)
        ax[0].plot(mytpf.lc.time.value, inj_flux, color='black')
        ax[0].plot(mytpf.lc[flareArr==1].time.value, inj_flux[flareArr==1], color='red', marker='o', ls='none', markersize=2)
        ax[1].plot(mytpf.lc.time.value, flares, color='black')
        plt.show()
        """
    
    except:
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

    
