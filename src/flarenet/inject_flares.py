import numpy as np
from scipy import special
from sklearn.preprocessing import RobustScaler
from scipy.stats import binned_statistic
from astroquery.mast import Catalogs
from numpy.random import uniform


def flare_generator(lcfile, iterations=50, fraction_flare=0.3, flaredir='flareArrs/', paramdir='artificial_flare_params/'):
    """
    Take a ticid_orbitnum_data.npy file with time and flux in the first and second column, respectively
    For a given number of iterations, 
    Generate and inject artificial flares into randomized points in the light curve, 
    such that approximately 50% of the timespan is occupied by a flare
    use fwhm as duration
    keep injecting until t1/2 sum reaches 30% of the time
    pre-generate a giant list of possible flare parameters
    use gunther fwhm/ampl
    """
    # read in the time series
    datafile = np.load(lcfile)
    t = datafile[0]
    f = datafile[1]
    ticid = lcfile.split('_')[0]

    # normalize & zero-center the flux
    f = RobustScaler().fit_transform(f)

    # initialize arrays of multi-flare light curves
    # flareArr = np.empty((num_flares, len(f)))
    flareArr = np.array()
    paramsArr = np.array()

    for i in range(iterations):
        fwhm_time = 0
        new_f = np.zeros((1, len(f)))
        # while summed fwhm times of flares are less than the desired fraction of time of the orbit to be occupied by a flare, 
        # generate flares & save parameters
        while fwhm_time < fraction_flare * (t.max() - t.min()):
            # generate artificial flare parameters
            tpeak = uniform(t.min(), t.max())
            # draw a random energy and convert to relative amplitude
            e_flare = _get_random_energy()
            ed_flare = energy_to_ed(e_flare, get_dist_cm(ticid))
            ampl = ed_to_rel_ampl(ed_flare) # since the flux is normalized and zero-centered, I think we can treat this as The amplitude
            fwhm = ampl_to_fwhm(ampl) # help

            fwhm_time += fwhm 
            
            # add flare parameters to array
            paramsArr.append(np.asarray([i, tpeak, ampl, fwhm, e_flare, ed_flare])) 

            # compute the flare flux, add to flux array
            flare = flare_model(t, tpeak, fwhm, ampl) 
            new_f += flare
        flareArr[i] = new_f
    np.save(flaredir+lcfile[:-8]+'flares.npy', flareArr)
    np.save(paramdir+lcfile[:-8]+'flareparams.npy', paramsArr)


def get_dist_cm(targetid, radius='0.1 arcsec'):
    """
    Query the TESS Input Catalogue to extract a distance of the input target,
    and convert the distance from parsec to cm

    Parameters
    ----------
    targetid : 
        Target ID (does it need to be a TIC ID?) for the object of interest
    radius : str or astropy Quantity object
        Radius around which to search for target information
        Default is 0.1 arcsecond.
    
    Returns
    -------
    dist : float
        Distance of the target in cm, 
        approximated by the median value of the distances returned by astroquery search...

    """
    catalog_data = Catalogs.query_object(targetid, catalog="Tic", radius=radius)
    return np.nanmedian(catalog_data['d'])*3.08568e18 # this is a choice

def _get_random_energy(emin=1e23, emax=1e38, distribution='log-uniform'):
    """
    Draw a flare energy at random from a log-uniform or uniform distribution.

    Parameters
    ----------
    emin : float
        Lower bound for flare energy
    emax : float
        Upper bound for flare energy
    distribution : str
        Distribution from which to make a random selection, defaulting to log-uniform.
        'log-uniform', 'uniform'
    
    Returns
    -------
    E : float
        Flare energy drawn from the input distribution
    """
    if distribution == 'log-uniform':
        log_rand_e = uniform(low=np.log10(emin), high=np.log10(emax))
        return 10**log_rand_e
    elif distribution == 'uniform':
        return uniform(emin, emax)
    
def energy_to_ed(energy, dist):
    """
    Convert energy to equivalent duration by dviding by the luminosity of the star.

    Parameters
    ----------
    energy : float
        Energy of the flare IN WHAT BAND
    dist : float
        Distance of the target, measured in centimeters
    
    Returns
    -------
    ED : float
        Equivalent duration of the flare, measured in seconds
    """
    lum_params = 4*np.pi*dist**2
    return energy / lum_params

def ed_to_rel_ampl(ed):
    """
    Use the empirically derived correlation between flare equivalent duration (ED) and relative amplitude
    for ultracool dwarfs by Ilin et al. (2023) DOI 10.1093/mnras/stad1690

    Parameters
    ----------
    ed : float
        Equivalent duration of flare in seconds
    
    Returns
    -------
    Amplitude of the flare relative to the baseline flux f_0
    (f_flare - f_0) / f_0
    """
    return 10**((np.log10(ed)-2.67)/1.01) # should i add some noise?


def ampl_to_fwhm(ampl):
    raise NotImplementedError


def flare_eqn(t, tpeak, fwhm, ampl):
    '''
    The equation that defines the shape for the Continuous Flare Model
    '''
    #Values were fit & calculated using MCMC 256 walkers and 30000 steps

    A, B, C, D1, D2, f1 = [0.9687734504375167, -0.251299705922117, 0.22675974948468916,
                      0.15551880775110513, 1.2150539528490194, 0.12695865022878844]

    # We include the corresponding errors for each parameter from the MCMC analysis
    A_err, B_err, C_err, D1_err, D2_err, f1_err = [0.007941622683556804,0.0004073709715788909,0.0006863488251125649,
                                              0.0013498012884345656,0.00453458098656645,0.001053149344530907 ]

    f2 = 1-f1

    eqn = ((1/2)*np.sqrt(np.pi)*A*C*f1*np.exp(-D1*t+((B/C)+(D1*C/2))**2)
        *special.erfc(((B-t)/C)+(C*D1/2)))+((1/2)*np.sqrt(np.pi)*A*C*f2
        *np.exp(-D2*t+((B/C)+(D2*C/2))**2)*special.erfc(((B-t)/C)+(C*D2/2)))
    return eqn*ampl


def flare_model(t, tpeak, fwhm, ampl, upsample=False, uptime=10):
    '''
    The Continuous Flare Model evaluated for single-peak (classical) flare events.
    From Tovar Mendoza et al. (2022) DOI 10.3847/1538-3881/ac6fe6

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
   
    Returns
    -------
    flare : 1-d array
        The flux of the flare model evaluated at each time
        A continuous flare template whose shape is defined by the convolution of a Gaussian and double exponential
        and can be parameterized by three parameters: center time (tpeak), FWHM, and ampitude
    '''
    t_new = (t-tpeak)/fwhm
    if upsample:
        dt = np.nanmedian(np.diff(np.abs(t_new)))
        timeup = np.linspace(min(t_new -dt, max(t_new)+dt, t_new.size*uptime))

        flareup = flare_eqn(timeup,tpeak,fwhm,ampl)

        # and now downsample back to the original time...

        downbins = np.concatenate((t_new-dt/2., [max(t_new)+dt/2.]))
        flare,_,_ = binned_statistic(timeup, flareup, statistic='mean',bins=np.sort(downbins))
    else:
        flare = flare_eqn(t_new,tpeak,fwhm,ampl)
    return flare




