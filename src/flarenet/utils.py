import numpy as np
import lightkurve as lk
import warnings



def download_tpf(
        ticid : str, 
        sector : int = None,
        exptime : int = 20, 
        download_dir: str = 'TPFs'):
    """
    Downloads a TPF using lightkurve, injects cosmic rays by default, computes a light curve,
    and returns each.

    Parameters
    ----------
    ticid : TIC ID of the target star, eg "TIC 261136679"
    sector : Observing sector for the target
    exptime : observation exposure time

    
    Returns
    -------
    tpf : lightkurve TESSTargetPixelFile object

    """  
    tpf_sr = lk.search_targetpixelfile(ticid, mission='TESS', author='SPOC', exptime=exptime, sector=sector)
    # raise error if there's no file to download
    if len(tpf_sr.table) == 0:
        raise ValueError(f"Unable to find data for target ID {ticid} and sector {sector} with {exptime} exposure time.")
    tpf_sr.table["dataURL"] = tpf_sr.table["dataURI"]
    tpf = tpf_sr.download(download_dir=download_dir)
    return tpf



def normalize_flux(lc_flux : np.ndarray,
                type : str ='median'):
    """
    Normalization for flux
    
    Parameters
    ----------
    lc_flux : 1D array of flux values
    type : Supported types are 'median' or 'standard'

    Returns
    -------
    nparray containing normalized flux values
    """
    if type == 'median':
        normalized = lc_flux / np.nanmedian(lc_flux)
    elif type == 'standard':
        normalized = (lc_flux - np.nanmean(lc_flux)) / np.nanstd(lc_flux)
    return normalized
    



def get_cosmicrays(tpf):
    """Function to access and return the cosmic ray extension

    Unlike other TESS data products, the TESS 20-second target pixel files
    do not have on-board cosmic ray correction. Instead, cosmic rays are identified
    by the pipeline and removed from the data. The removed flux is saved
    in an extension in the FITS file. 
    See https://heasarc.gsfc.nasa.gov/docs/tess/TESS-CosmicRayPrimer.html
    for details
    
    Parameters:
    -----------
    tpf: lk.TargetPixelFile
        Input Target Pixel File
    
    Returns:
    -------
    cr: np.ndarray
        Array containing cosmic ray fluxes of shape tpf.shape
    """
    cadenceno = np.asarray(tpf.cadenceno)

    # Get cosmic ray information
    l = np.where([hdu.name == 'TARGET COSMIC RAY' for hdu in tpf.hdu])[0]
    if len(l) != 1:
        #raise ValueError("TPF product has no cosmic ray extension.")
        warnings.warn(f"No Cosmic Ray Extension found for {tpf.ticid}. Returning cube of 0s. ")
        return np.zeros(tpf.shape)
    hdu = tpf.hdu[l[0]]
    c, x, y, f = [hdu.data[attr].copy() for attr in ['CADENCENO', 'RAWX', 'RAWY', 'COSMIC_RAY']]
    if len(c) == 0:
        warnings.warn(f"No Cosmic Rays identified for {tpf.ticid}. Returning cube of 0s. ")
        np.zeros(tpf.shape)
        #raise ValueError("No cosmic rays identified. Is this a 20-second dataset?")
    x -= tpf.column
    y -= tpf.row

    # Mask down to only cosmic rays in cadences within the TPF
    k = np.in1d(c, cadenceno)
    c, x, y, f = c[k], x[k], y[k], f[k]

    # Map cosmic ray cadences to array indices
    r = {cadenceno[idx]:idx for idx in range(tpf.shape[0])}
    inv = np.asarray([r[c1] for c1 in c])

    # Build a cube
    cr = np.zeros(tpf.shape)
    cr[inv, y, x] = f
    return cr