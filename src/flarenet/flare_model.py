import numpy as np
from scipy import special
from scipy.stats import binned_statistic
from . import PACKAGEDIR
from astropy.io import ascii
from numpy.random import choice, randint
import warnings

def flare_eqn(t,ampl):
    '''
    The equation that defines the shape for the Continuous Flare Model
    References
    ----------
    Taken from https://github.com/lupitatovar/Llamaradas-Estelares/
    See: https://iopscience.iop.org/article/10.3847/1538-3881/ac6fe6#ajac6fe6s4
    Parameters
    ----------
    t : 1-d array
        The time array to evaluate the flare over
    ampl : float
        The amplitude of the flare
    Returns
    -------
    flare : the flare model scaled by the input amplitude
    '''
    #Values were fit & calculated using MCMC 256 walkers and 30000 steps
    #print(tpeak,fwhm, ampl )
    A,B,C,D1,D2,f1 = [0.9687734504375167,-0.251299705922117,0.22675974948468916,
                      0.15551880775110513,1.2150539528490194,0.12695865022878844]

    # We include the corresponding errors for each parameter from the MCMC analysis

    A_err,B_err,C_err,D1_err,D2_err,f1_err = [0.007941622683556804,0.0004073709715788909,0.0006863488251125649,
                                              0.0013498012884345656,0.00453458098656645,0.001053149344530907 ]

    f2 = 1-f1
    with warnings.catch_warnings(): # ignore RuntimeWarning: overflow encountered in exp
        warnings.simplefilter("ignore")
        eqn = ((1 / 2) * np.sqrt(np.pi) * A * C * f1 * np.exp(-D1 * t + ((B / C) + (D1 * C / 2)) ** 2)
                            * special.erfc(((B - t) / C) + (C * D1 / 2))) + ((1 / 2) * np.sqrt(np.pi) * A * C * f2
                            * np.exp(-D2 * t+ ((B / C) + (D2 * C / 2)) ** 2) * special.erfc(((B - t) / C) + (C * D2 / 2)))
    # Can get numerically unstable, so mask out extreme values. 
    eqn[ eqn < 1e-8] = 0
    eqn[np.isnan(eqn)] = 0
    return eqn * ampl

def flare_model(t,tpeak, fwhm, ampl, upsample=False, uptime=10):
    '''
    The Continuous Flare Model evaluated for single-peak (classical) flare events.
    Use this function for fitting classical flares with most curve_fit
    tools. Reference: Tovar Mendoza et al. (2022) DOI 10.3847/1538-3881/ac6fe6
    References
    --------------
    Tovar Mendoza et al. (2022) DOI 10.3847/1538-3881/ac6fe6
    Davenport et al. (2014) http://arxiv.org/abs/1411.3723
    Jackman et al. (2018) https://arxiv.org/abs/1804.03377
    Parameters
    ----------
    t : 1-d array
        The time array to evaluate the flare over
    tpeak : float
        The center time of the flare peak
    fwhm : float
        The Full Width at Half Maximum, timescale of the flare
    ampl : float
        The amplitude of the flare
    upsample : bool
        Flag to increase the time sampling of the flare
    uptime : int
        If upsample is TRUE, uptime indicates the factor to increase the time samples by
    Returns
    -------
    flare : 1-d array
        The flux of the flare model evaluated at each input time
        A continuous flare template whose shape is defined by the convolution of a Gaussian and double exponential
        and can be parameterized by three parameters: center time (tpeak), FWHM, and ampitude
    '''

    t_new = (t-tpeak)/fwhm
    

    if upsample:
        dt = np.nanmedian(np.diff(np.abs(t_new)))
        timeup = np.linspace(min(t_new) - dt, max(t_new) + dt, t_new.size * uptime)

        flareup = flare_eqn(timeup,ampl)

        # and now downsample back to the original time...

        downbins = np.concatenate((t_new - dt / 2.,[max(t_new) + dt / 2.]))
        flare,_,_ = binned_statistic(timeup, flareup, statistic='mean',bins=np.sort(downbins))
    else:

        flare = flare_eqn(t_new,ampl)

    return flare

def generate_flares(
                    t,
                    num_flares : int = 100,
                    window_size : int = None, 
                    ): 
    """
    Take time and flux time series 
    Generate and inject artificial flares into randomized points in the light curve

    Parameters
    ----------
    num_flares : float
        Number of flares to inject
        based on the average duration, 100 flares equates to ~10% of a TESS sector
    window_size : int
        For ML training, we may want to require flare peaks to be in the window,
        which may not be the case for long duration flares. This sets the flare flag
        to FALSE for points beyond window_size/2 from the start of the flare.
    Returns
    -------
    flare_flux : ndarray
        Array of flare fluxes with the same length as the input object light curve
    paramsArr : ndarray
        Array of parameters for each generated flare
        [Time of peak, relative amplitude, and FWHM]
    """
    # # **** read in gunther flares file for flare parameters ****
    flaredf = ascii.read(f"{PACKAGEDIR}/supplemental_files/gunther_flares_all.txt")
    all_ampls = flaredf['Amp']
    all_fwhms = flaredf['FWHM']


    
    paramsArr = np.array([])
    flare_flux = np.zeros(len(t))
    valid_flare_flags = np.zeros(len(t), dtype=bool)

    nf = 0
    while nf < num_flares:
        # generate artificial flare parameters
        tpeak = choice(t)
        
        rand_ind = randint(0, len(all_ampls)) # get random index for flare parameters
        ampl = all_ampls[rand_ind]
        fwhm = all_fwhms[rand_ind]

        
        # add flare parameters to array
        paramsArr = np.append(paramsArr, np.asarray([tpeak, ampl, fwhm])) 
        # generate the flare, add to flux array
        flare = flare_model(t, tpeak, fwhm, ampl)
        flare = np.nan_to_num(flare)
        flare_flags = flare > 0
        in_flare = [i for i in range(len(flare)) if flare[i] > 0]
        # Optionally, if the flare is longer than the ML window,
        # we may want to make sure the flarepeak is in the window
        if window_size:
            if len(in_flare) > window_size / 2:
                flare_flags[in_flare[int(window_size/2):]] = False

        flare_flux += flare
        nf += 1

        valid_flare_flags += flare_flags
    return flare_flux, paramsArr, valid_flare_flags