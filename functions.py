import numpy as np
from scipy.stats import norm
from scipy.special import i0


def T2star_decay(TE, S0, T2):
    return S0 * np.exp(- TE / T2)

def T2star_decay_noise(TE, S0, T2, sigma):
    decay = T2star_decay(TE, S0, T2)
    real_noise = sigma * norm.rvs(size = len(TE))
    imag_noise = sigma * norm.rvs(size = len(TE))

    return decay + real_noise + 1j*imag_noise

def SNRGain_LLS(TEs, T2):
    # Analytical Solution for SNR Gain for LLS
    N = len(TEs)
    return N / np.sqrt( np.sum((np.exp(TEs/T2))**2) )

def wLLS(X, TEs, T2):
    weight = np.exp(- TEs / T2)
    return X.sum() / weight.sum()


def LLS(X, TEs, T2):
    N = len(TEs)
    cTE = X * np.exp(TEs / T2)
    return cTE.sum() / N


def LL_exp_gauss_fixT2(params, args):
    S0 = params[0]

    TEs = args[0]
    ydata_noise = args[1]
    T2 = args[3]

    yPred = T2star_decay(TEs, S0, T2)

    LL = np.sum((ydata_noise - yPred)**2)

    return LL


def LL_exp_rice_fixT2(params, args):
    S0 = params[0]

    TEs = args[0]
    ydata_noise = args[1]
    sigma    = args[2]
    T2 = args[3]


    yPred = T2star_decay(TEs, S0, T2)

    term1 = (yPred*ydata_noise)/sigma**2
    term2 = yPred**2 / (2*sigma**2)

    # If the SNR becomes too high, the zero order modified Bessel function becomes inf
    # To prevent this, in highh SNR cases, we will use Gaussian estimators
    if term1.max() <= 709:
        LL = -(np.sum( np.log( i0(term1))) - np.sum(term2))
    else:
        # print('SNR Too High: Using Gaussian MLE Approximatipon')
        LL = np.sum((ydata_noise - yPred)**2)


    return LL


def G_SNR(T2s, TEs):
    """
    Calculate the SNR Gain from Multi Echo dMRI Acqusition using 
    LLS Estimation 
    Parameters
    ----------
    T2s : float
        Underlying Tissue T2 Star
    TEs : numpy array
        Array containing the Echo Times of the 
        Multi Echo Acquisition.

    Returns
    -------
    float
        SNR Gain Factor.

    """
    N = len(TEs)
    TEs = TEs-TEs[0]
    
    return N / np.sqrt(np.sum(np.exp(TEs / T2s)**2))


def MacroR2s(f_in, R2s_ex, R2s_in, D_fast, D_slow, b):
    """
    Calculation of Macrostructural T2*
    Based on Kleban et al, 2020

    Parameters
    ----------
    f_in : float
        Relative intra axonal volume fraction between 0 and 1.
    T2s_ex : TYPE
        Extra Axonal T2* [s].
    T2s_in : float
        Intra Axonal T2* [s].
    D_fast : TYPE
        Diffusivity of fast diffusion compartment [mm**2/s].
    D_slow : float
        Diffusivity of slow diffusion compartment [mm**2/s].
    b : TYPE
        b-value of diffusion weighting [s/mm2].

    Returns
    -------
    None.

    """
    
    # Estimate Total Diffusion Attenation
    diff_atten = f_in*np.exp(-b*D_slow) + (1-f_in)*np.exp(-b*D_fast)
    
    R2s_macro =     f_in * np.exp(-b * D_slow) * R2s_in / diff_atten\
                + (1-f_in) * np.exp(-b * D_fast) * R2s_ex / diff_atten
    
    return R2s_macro
