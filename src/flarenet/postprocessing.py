import numpy as np
import lightkurve as lk
import pandas as pd
from sklearn.preprocessing import RobustScaler
from flare_model import * 
from cosmic_ray_extension import *
import matplotlib.pyplot as plt

all = ['InputTPF']

class InputTPF(object):
    """
    Contains all the functions to search for & download a lightkurve TargetPixelFile, 
    extract time series and meta data for input into our model, ...
    """

    def __init__(self, ticid, sector, exptime=20, download_dir='tpfs/'):
        """
        Download a TPF 
        """
        self.tpf = InputTPF.download_tpf(ticid, sector, exptime=exptime, download_dir=download_dir)
        self.targetid = self.tpf.targetid
        self.time =  self.tpf.time
        self.flux = self.tpf.flux
        self.flux_err = self.tpf.flux_err
        self.exptime = exptime
        self.sector = sector
        self.ra = self.tpf.ra
        self.dec = self.tpf.dec
        self.centroid_col, self.centroid_row = self.tpf.estimate_centroids(aperture_mask='default', method='moments')
        self.header = self.tpf.get_header()
        self.tessmag = self.tpf.get_header()['TESSMAG']
        self.teff = self.tpf.get_header()['TEFF']
        self.radius = self.tpf.get_header()['RADIUS']
        self.cdpp1_0 = self.tpf.hdu[1].header['CDPP1_0']
        self.crowdsap = self.tpf.hdu[1].header['CROWDSAP']
        self.camera = self.tpf.camera
        self.ccd = self.tpf.ccd
        self.logg = self.tpf.get_header()['LOGG']
        self.pos_corr1 = self.tpf.pos_corr1
        self.pos_corr2 = self.tpf.pos_corr2

    @staticmethod
    def download_tpf(ticid, sector, exptime=20, download_dir='tpfs'):
        return lk.search_targetpixelfile(ticid, mission='TESS', author='SPOC', exptime=exptime, sector=sector).download(download_dir=download_dir)

    def _get_metadata(self, outdir='/Users/veraberger/nasa/meta_training/'): 
        """ Get relevant metadata from the headers of a TESS TargetPixelFile object
            If outdir is specified, save the array into an .npy file

        Parameters
        ----------
        outdir : str
            Directory to save metadata into
        
        Returns
        -------
        nothing - do i say that or just leave this blank?
        """
        metaArr = np.array([self.ra, self.dec,  self.tessmag, self.teff,  self.radius, self.sector, self.cdpp1_0,  self.crowdsap, self.camera, self.ccd,  self.logg])
        # metaArr = np.array([tpf.ra, tpf.dec,  tpf.get_header()['TESSMAG'], tpf.get_header()['TEFF'],  tpf.get_header()['RADIUS'], tpf.sector, tpf.hdu[1].header['CDPP1_0'],  tpf.hdu[1].header['CROWDSAP'], tpf.camera, tpf.ccd,  tpf.get_header()['LOGG']])
        if outdir is not None:
            np.save(outdir+str(self.targetid)+'_'+str(self.sector)+'_meta.npy', metaArr)
        else:
            return metaArr

    def _get_centroid_shift(self):
        """
        Retrieves the centroid time series from a TargetPixelFile, 
        returns the magnitude of the 2D shift of the centroid from the median

        Parameters
        ----------
        outdir : str
            Directory to save metadata into
        
        Returns
        -------
        Square root of sum of squares of the centroid row and column time series
        """
        c_cols = self.centroid_col-np.nanmedian(self.centroid_col)
        c_rows = self.centroid_row-np.nanmedian(self.centroid_row)
        return np.sqrt(c_cols**2+c_rows**2).value

    def _get_pos_corr(self):
        # magnitude of vector given by poscorr 1 and 2
        return np.sqrt(self.pos_corr1**2+self.pos_corr2**2) 
    
    def tpf_to_cr_lc(self, cr=True):
        """
        Create a lightcurve from the target pixel file using lightkurve to_lightcurve,
        and by default inject cosmic rays originally removed by TESS processing back into the light curve.
        
        Parameters
        ----------
        cr : bool
            If True, inject cosmic rays into light curve.
        
        Returns
        -------
        lc : TESS lightcurve object
        crArr : array of 0s and 1s marking where in the light curve cosmic rays have been identified
        """
        lc = self.tpf.to_lightcurve(aperture_mask=self.tpf.pipeline_mask)
        if cr == True:
            if not self.exptime == 20:
                raise TypeError("Cosmic ray injection is only possible for 20-second datasets.")
            cosmic_ray_cube = load_cosmicray_extension(self.tpf) 
            tpf_cr = (self.tpf+cosmic_ray_cube)
            lc_cr = tpf_cr.to_lightcurve(aperture_mask=self.tpf.pipeline_mask)
            crArr = np.where(lc_cr.flux - lc.flux != 0, 1, 0)
            return lc_cr, crArr
        else: 
            return lc

    @staticmethod
    def make_orbit_files(lc, crArr, pos_corr, c_dist,  datadir='data_training_unnorm/'):
        """
        Take a light curve and supplemental information for some target and sector, split the data by orbits, 
        Save the output into an npy file. 

        Parameters
        ----------
        lc : Lightkurve.LightCurve object with length n
        crArr : 1 x n array with 1s when a cosmic ray is identified, and 0s otherwise
        pos_corr : 1 x n array representing the local motion of the star
        c_dist : 1 x n array representing the shift of the centroid from its median value over time
        datadir : directory to save orbit data into
        
        Returns
        -------
        """
        df = pd.DataFrame(data=[lc.time.value, lc.flux.value, lc.flux_err.value, lc.quality.value, pos_corr, c_dist, crArr]).T
        dt = df[0].diff()
        gap_index = df.index[dt == dt.max()].item()
        orbit1 = df.iloc[:gap_index]
        orbit2 = df.iloc[gap_index:]
        np.save(datadir+str(lc.targetid)+'_'+str(lc.sector)+'_1_data.npy', np.asarray(orbit1))
        np.save(datadir+str(lc.targetid)+'_'+str(lc.sector)+'_2_data.npy', np.asarray(orbit2))

    @staticmethod
    def training_data_generator(lc_list, batch_size=32, window_size=100):
        # here lc_list is a list of filenames in the training set
        for fname in lc_list:
            target_data = np.load(fname)

            target_lc = target_data[1] # flux array
            labels = target_data[7] # flare array 

            lc_length = len(target_lc)
            valid_indices = np.arange(int(window_size/2), int(lc_length-window_size/2), dtype=int)

            # Note I've shuffled the indices for training so the batch isn't all sampling the same part of the lc
            # We don't need/want to do this for prediction. This can become an input flag (unless we design a separate generator for the prediction)
            np.random.shuffle(valid_indices)
            j = 0 #J loops through the time series for a single target
            while j+batch_size <= len(valid_indices):
                data = np.empty((batch_size, window_size, 1))
                #print(data.shape)
                label = np.empty((batch_size), dtype=int)
                for k in range(batch_size):
                    #print(i, j, k)
                    X = target_lc[valid_indices[j+k]-int(window_size/2) : valid_indices[j+k]+int(window_size/2)].reshape(window_size,1)
                    data[k,] = RobustScaler().fit_transform(X)
                    label[k] = np.asarray(labels[j+k])
                    #print(data[k,])
                    #print(label)
                #print(data.shape, label)
                yield data, label
                j = j + batch_size




# # download tpf, convert to LC with cosmic rays, plot LC w/ CRs marked in red
# mytpf = InputTPF(ticid='TIC 10863087', sector=30, exptime=20, download_dir='/Users/veraberger/nasa/tpfs/')
# mytpf._get_metadata()
# mylc, crArr = mytpf.tpf_to_cr_lc()
# InputTPF.make_orbit_files(mylc, crArr, mytpf._get_pos_corr(), mytpf._get_centroid_shift())
# print(InputTPF.training_data_generator([]))
# plt.plot(mylc.time.value, mylc.flux.value, color='black')
# plt.plot(mylc.time.value[crArr==1], mylc.flux.value[crArr==1], color='r', marker='o', ls='none')
# plt.show()


