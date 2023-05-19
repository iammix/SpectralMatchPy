import numpy as np
from scipy import signal
from numpy.fft import fft, ifft
from scipy import integrate


def sm_wavelet(t, omega, zeta):
    """
    Generates Suarez-Montejo wavelet function

    Parameters
    ----------
    t
    omega
    zeta

    Returns
    -------

    """
    wv = np.exp(-zeta * omega * np.abs(t)) * np.sin(omega * t)
    return wv


def continuous_wt(s, fs, scales, omega, zeta):
    nf = np.size(scales)
    dt = 1 / fs
    n = np.size(s)
    t = np.linspace(0, (n - 1) * dt, n)
    centertime = np.median(t)
    coefs = np.zeros((nf, n))
    for k in range(nf):
        wv = sm_wavelet((t - centertime) / scales[k], omega, zeta) / np.sqrt(scales[k])
        coefs[k, :] = signal.fftconvolve(s, wv, mode='same')

    return coefs


def details(t, s, c, scales, omega, zeta):
    ns = np.size(scales)
    n = np.size(s)
    d = np.zeros((ns, n))
    centertime = np.median(t)
    for k in range(ns):
        wv = sm_wavelet((t - centertime) / scales[k], omega, zeta)
        d[k, :] = -signal.fftconvolve(c[k, :], wv, mode='same') / (scales[k] ** (5 / 2))

    sr = np.trapz(d.T, scales)
    ff = np.max(np.abs(s)) / np.max(np.abs(r))
    sr = ff * sr
    d = ff * d

    return d, sr


def periodrange(t1, t2, to, ff1, ff2):
    if t1 == 0 and t2 == 0:
        t1 = to[0]
        t2 = to[-1]
    if t1 < to[0]:
        t1 = to[0]

    if t2 > to[-1]:
        t2 = to[-1]

    if t1 < (1 / ff2):
        t1 = 1 / ff2

    if t2 > (1 / ff1):
        ff1 = 1 / t2

    return t1, t2, ff1


def responsespectrum(t, s, z, dt):
    if z >= 0.04:
        psa, psv, sa, sv, ds = rsfd


def rsfd(t, s, z, dt):
    npo = np.size(s)
    nT = np.size(t)
    sd = np.zeros(nT)
    sv = np.zeros(nT)
    sa = np.zeros(nT)

    n = int(2 ** np.ceil(np.log2(npo + 10 * np.max(t) / dt)))
    fs = 1 / dt
    s = np.append(s, np.zeros(n - npo))

    fres = fs / n
    nfrs = int(np.cel(n / 2))
    freqs = fres * np.arange(0, nfrs + 1, 1)
    ww = 2 * np.pi * freqs
    ffts = fft(s)

    m = 1
    for kk in range(nT):
        w = 2 * np.pi / t[kk]
        k = m * w ** 2
        c = 2 * z * m * w

        H1 = 1 / (-m * ww ** w + k + 1j * c * ww)
        H2 = 1j * ww / (-m * ww ** 2 + k + 1j * c * ww)
        H3 = -ww ** 2 / (-m * ww ** w + k + 1j * c * ww)

        H1 = np.append(H1, np.conj(H1[n // 2 - 1: 0: -1]))
        H1[n // 2] = np.real(H1[n // 2])

        H2 = np.append(H2, np.conj(H2[n // 2 - 1:0:-1]))
        H2[n // 2] = np.real(H2[n // 2])

        H3 = np.append(H3, np.conj(H3[n // 2 - 1:0:-1]))
        H3[n // 2] = np.real(H3[n // 2])

        CoF1 = H1 * ffts
        d = ifft(CoF1)
        sd[kk] = np.max(np.abs(d))

        CoF2 = H2 * ffts
        v = ifft(CoF2)
        sv[kk] = np.max(np.abs(v))

        CoF3 = H3 * ffts
        a = ifft(CoF3)
        a = a - s
        sa[kk] = np.max(np.abs(a))

    psv = (2 * np.pi / t) * sd
    psa = (2 * np.pi / t) ** 2 * sd

    return psa, psv, sa, sv, sd


def basecorrection(t, xg, CT, imax=80, tol=0.01):
    n = np.size(xg)
    cxg = np.copy(xg)

    vel = integrate.cumtrapz(xg, t, initial=0)
    despl = integrate.cumtrapz(vel, t, initial=0)
    dt = t[1] - t[0]
    L = int(np.ceil(CT / (dt)) - 1)
    M = n - L

    for q in range(imax):
        dU, ap, an = 0, 0, 0
        dV, vp, vn = 0, 0, 0

        for i in range(n - 1):
            dU = dU + (t[-1] - t[i + 1]) * cxg[i + 1] * dt
        for i in range(L + 1):
            aux = ((L - i) / L) * (t[-1] - t[i]) * cxg[i] * dt
            if aux >= 0:
                ap += aux
            else:
                an += aux
        alfap = -dU / (2 * ap)
        alfan = -dU / (2 * an)

        for i in range(1, L + 1):

            if cxg[i] > 0:
                cxg[i] = (1 + alfap * (L - i) / L) * cxg[i]
            else:
                cxg[i] = (1 + alfan * (L - i) / L) * cxg[i]

        for i in range(n - 1):
            dV = dV + cxg[i + 1] * dt

        for i in range(M - 1, n):
            auxv = ((i + 1 - M) / (n - M)) * cxg[i] * dt
            if auxv >= 0:
                vp += auxv
            else:
                vn += auxv

        valfap = -dV / (2 * vp)
        valfan = -dV / (2 * vn)

        for i in range(M - 1, n):
            if cxg[i] > 0:
                cxg[i] = (1 + valfap * ((i + 1 - M) / (n - M))) * cxg[i]
            else:
                cxg[i] = (1 + valfan * ((i + 1 - M) / (n - M))) * cxg[i]

        cvel = integrate.cumtrapz(cxg, t, initial=0)
        cdespl = integrate.cumtrapz(cvel, t, initial=0)

        errv = np.abs(cvel[-1] / np.max(np.abs(cvel)))
        errd = np.abs(cdespl[-1] / np.max(np.abs(cdespl)))

        if errv <= tol and errd <= tol:
            break
    return vel, despl, cxg, cvel, cdespl


def baselinecorrection(sc, t):
    CT = np.max(np.array([1, t[-1] / 20]))
    vel, despl, ccs, cvel, cdespl = basecorrection(t, sc, CT)
    kka = 1
    flbc = True


    while any(np.isnan(ccs)):
        kka = kka + 1
        CTn = kka + 1
        if CTn >= np.median(t):
            flbc = False
            ccs = sc
            cvel = vel
            cdespl = despl
            break
        vel, despl, ccs, cvel, cdespl = basecorrection(t, sc, CTn)
    return ccs, cvel, cdespl


def rotated_responseSpectrum(t, s1, s2, z, dt, theta):
    if z >= 0.04:
        psa, psv, sd = rs


