import numpy as np
#import lightkurve as lk
import warnings


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


def inject_asteroid_crossing(time_arr, flux_arr):
    """
    Simulate an asteroid passing in front of a star in TESS.
    Parameters:
    t (array): Time array
    """
    # Ranges loosely based on https://iopscience.iop.org/article/10.3847/1538-4357/ace9df
    amp = np.random.uniform(0.01, 0.2)
    sig = np.random.uniform(0.01, .2)
    t_mid = np.random.choice(time_arr)
    
    #signal = amp * np.exp(-((t - t_mid) ** 2) / (2 * sig ** 2)) # Regular gaussian
    # Make a 'flat-top' gaussian instead
    signal = amp * np.exp(-((time_arr - t_mid) / (2 * sig))**4)
    return flux_arr + signal

def inject_stellar_pulsations(time_arr, flux_arr):
    periods = np.random.uniform(0.1, 5, size=3)
    amplitudes = np.random.uniform(0.01, 0.2, size=3)
    phases = np.random.uniform(0, 2*np.pi, size=3)

    signal = np.zeros_like(time_arr)
    for period, amplitude, phase in zip(periods, amplitudes, phases):
        signal += amplitude * np.sin(2 * np.pi * (time_arr / period + phase))
    signal /= (1 + np.median(signal))
    return flux_arr + signal

def inject_rr_lyrae(time_arr, flux_arr):
    """
    Simulate an RR Lyrae variable star light curve.
    Parameters:
    t (array): Time array
    period (float): Pulsation period in days
    amplitude (float): Amplitude of the variation
    rise_fraction (float): Fraction of the period spent in the rising phase
    phase_offset (float): Phase offset for the pulsation
    """
    period =  np.random.uniform(0.5, 35),
    amplitude = np.random.uniform(0.1, 0.5)
    rise_fraction = np.random.uniform(0.1, 0.3)
    phase_offset = np.random.uniform(0, 1)

    
    phase = ((time_arr / period) + phase_offset) % 1
    signal = np.zeros_like(time_arr)
    # Rising phase (usually steeper)
    rising = phase < rise_fraction
    signal[rising] = amplitude * (phase[rising] / rise_fraction)
    # Falling phase
    falling = ~rising
    signal[falling] = amplitude * (1 - ((phase[falling] - rise_fraction) / (1 - rise_fraction)))
    # Add some asymmetry and non-linearity to make it more realistic
    signal = signal - 0.1 * amplitude * np.sin(4 * np.pi * phase)
    # Normalize around 1
    signal /= np.median(signal)
    return flux_arr + signal - 1


def inject_flares(
                    flare_flux : np.ndarray, 
                    flux_arr : np.ndarray, 
                    ): 
    """
    Parameters
    ----------
    flare_flux : ndarray
        Array of flare fluxes to inject
    
    Returns
    -------
        : ndarray
        Array of flare fluxes added to light curve
    """

    flux_with_flares = flux_arr + (flare_flux * flux_arr)
    return normalize_flux(flux_with_flares, type='standard')