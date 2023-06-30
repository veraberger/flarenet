import numpy as np 
import pandas as pd
import sklearn
import lightkurve as lk
import matplotlib.pyplot as plt
import astropy.units as u
import glob
import pdb

from flare_model import *
from prep import *

"""
Show the compatibility of the postprocessing functions in prep.py
ideally: 
can take a lk object, 
initialize a TimeSeries object,
turn it into segments,
detrend, 
normalize,
pad
turn into windows?
"""

# lk object
tlc = lk.search_lightcurve('TIC 219852882', exptime=20).download_all().stitch()

# initialize a TimeSeries object
mylc = TimeSeries(tlc)

# turn it into segments
# SHOULD THESE BE CSVS, LISTS OF ARRAYS, ...
mylc.segment()

# detrend
# mylc.detrend() # PLACEHOLDER

# normalize
mylc.normalize()

# pad
# t_ext, f_ext, ferr_ext = mylc.pad(side='both') # THIS CURRENTLY THROWS AN ERROR BC THE PADDING FLUX & ERRORS ARE UNITLESS QUANTITIES, NOT ELECTRON / S


fig, ax = plt.subplots(2, figsize=(10, 4))

ax[0].plot(mylc.time.value, mylc.flux.value, color='tab:pink', marker='s', markersize=1, ls='none')
# plt.plot(mylc.time, mylc.flux)
# plt.show()
# mylc.flux= mylc.invert() # there must be a better way to do this? or do you just replace the flux every time


# # invert or normalize or pad gets rid of the astropy quantity object :(
# t_ext, f_ext, ferr_ext = mylc.pad(side='left')
# print(t_ext, f_ext, ferr_ext)



# ax[1].plot(t_ext.value, f_ext.value, color='tab:brown', marker='s', markersize=1, ls='none')
ax[1].set_xlabel('Time [s]')
ax[0].set_ylabel('SAP Flux')
ax[1].set_ylabel('Relative Flux ')
plt.suptitle('TIC 219852882')
# plt.savefig('figures/manipulating_lcs.png', dpi=200)
plt.show()

