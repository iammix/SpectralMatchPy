import numpy as np
from typing import List, Tuple, Any
from numpy import ndarray

__all__ = ['ec8_rs']


def ec8_rs(agr: int, ground_type: str, resp_type: int, orientation: str = 'horizontal', importance_class: int = 2,
           damping: float = 5, periods: List = None) -> tuple[Any, ndarray]:
    """
    Calculates the Design Spectrum of Eurocode 8 given the corresponding inputs [CT].

    Parameters
    ----------
    agr : int
        Peak Ground Acceleration
    ground_type : str
        Ground type. Selection between A, B, C, D
    resp_type : int
        Type of spectrum. Selection between 1 or 2
    orientation : str
        horizontal orientation is only implemented
    importance_class : int
        Importance class
    damping : float
        Damping ration for the design spectrum generation
    periods

    Returns
    -------
    periods : List
        List of periods
    PGA : List
        List of Peak Ground Acceleration corresponding to the period list


    Examples
    --------
    >>> np.fft.fft(np.exp(2j * np.pi * np.arange(8) / 8))
    array([-2.33486982e-16+1.14423775e-17j,  8.00000000e+00-1.25557246e-15j,
            2.33486982e-16+2.33486982e-16j,  0.00000000e+00+1.22464680e-16j,
           -1.14423775e-17+2.33486982e-16j,  0.00000000e+00+5.20784380e-16j,
            1.14423775e-17+1.14423775e-17j,  0.00000000e+00+1.22464680e-16j])

    In this example, real input has an FFT which is Hermitian, i.e., symmetric
    in the real part and anti-symmetric in the imaginary part, as described in
    the `numpy.fft` documentation:

    >>> import matplotlib.pyplot as plt
    >>> t = np.arange(256)
    >>> sp = np.fft.fft(np.sin(t))
    >>> freq = np.fft.fftfreq(t.shape[-1])
    >>> plt.plot(freq, sp.real, freq, sp.imag)
    [<matplotlib.lines.Line2D object at 0x...>, <matplotlib.lines.Line2D object at 0x...>]
    >>> plt.show()


    References
    ----------
    .. [CT] Cooley, James W., and John W. Tukey, 1965, "An algorithm for the
            machine calculation of complex Fourier series," *Math. Comput.*
            19: 297-301.
    """

    importance_factor = {1: 0.8,
                         2: 1,
                         3: 1.2,
                         4: 1.4}[importance_class]
    ag = agr * importance_factor
    ground_type = ground_type.upper()
    if resp_type == 1:
        S = {'A': 1.0,
             'B': 1.2,
             'C': 1.15,
             'D': 1.35,
             'E': 1.4}[ground_type]
        if orientation == 'horizontal':
            TB = {'A': 0.15, 'B': 0.15, 'C': 0.2, 'D': 0.2, 'E': 0.15}[ground_type]
            TC = {'A': 0.4, 'B': 0.5, 'C': 0.6, 'D': 0.8, 'E': 0.5}[ground_type]
            TD = {'A': 2.0, 'B': 2.0, 'C': 2.0, 'D': 2.0, 'E': 2.0}[ground_type]
        elif orientation == 'vertical':
            raise NotImplemented('Vertical Orientation is not Implemented yet!')
    elif resp_type == 2:
        S = {'A': 1.0, 'B': 1.35, 'C': 1.5, 'D': 1.8, 'E': 1.6}[ground_type]
        if orientation == "horizontal":
            TB = {'A': 0.05, 'B': 0.05, 'C': 0.1, 'D': 0.1, 'E': 0.05}[ground_type]
            TC = {'A': 0.25, 'B': 0.25, 'C': 0.25, 'D': 0.30, 'E': 0.25}[ground_type]
            TD = {'A': 1.2, 'B': 1.2, 'C': 1.2, 'D': 1.2, 'E': 1.2}[ground_type]
        elif orientation == "vertical":
            raise NotImplementedError('Vertical Orientation is not Implemented yet!')

    eta = np.sqrt(10.0 / (5 + damping))

    if periods is None:
        periods = np.concatenate([[0.0], np.arange(0.04, 4.02, 0.02)])
    else:
        periods = np.sort(np.unique(list(periods) + [TB, TC, TD]))
    periods = periods[periods <= 4.0]
    values = np.zeros(len(periods), 'd')

    if orientation == "horizontal":
        for i, T in enumerate(periods):
            if 0 <= T < TB:
                values[i] = ag * S * (1 + (T / TB) * (eta * 2.5 - 1))
            elif TB <= T < TC:
                values[i] = ag * S * eta * 2.5
            elif TC <= T < TD:
                values[i] = ag * S * eta * 2.5 * (TC / T)
            elif T >= TD:
                values[i] = ag * S * eta * 2.5 * ((TC * TD) / T ** 2)
    elif orientation == "vertical":
        raise NotImplementedError('Vertical Orientation is not Implemented yet!')
    return periods, values
