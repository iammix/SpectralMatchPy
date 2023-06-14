import numpy as np
from scipy import integrate
from sm_wavelet import *

def single_component(s, fs, dso, To, T1=0, T2=0, zi=0.05, NS=100, blcorrection=True, plots=False):
    """
    Continuoys wavelet transformation based modification of a single component
    from a historic records to obtain spectrally equivalent acceleration series

    Parameters
    ----------
    s
    fs
    dso
    To
    T1
    T2
    zi
    NS
    blcorrection
    plots
    """


    n = np.size(s)
    dt = 1 / fs
    t = np.linspace(0, (n-1) * dt, n)
    FF1 = min(4/(n*dt), 0.1)
    FF2 = 1 / (2 * dt)

    Tsortindex = np.argsort(To)
    To = To[Tsortindex]
    dso = dso[Tsortindex]

    T1, T2, FF1 = periodrange(T1, T2, To, FF1, FF2)

    omega = np.pi
    zeta = 0.05
    freqs = np.geomspace(FF2, FF1, NS)
    T = 1 / freqs
    scales = omega / (2 * np.pi * freqs)
    C = cwtzm(s, fs, scales, omega, zeta)

    D, sr = details(t, s, C ,scales, omega, zeta)

    PSAs, _, _, _, _ = responsespectrum(T, s, zi, dt)
    PSAsr, _, _, _, _ = responsespectrum(T, sr, zi, dt)

    ds = np.interp(T, To, dso, left=np.nan, right=np.nan)
    Tlocs = np.nonzero((T>=T1) & T<=T2))
    nTlocs = np.size(Tlocs)

