import numpy as np
from typing import List, Tuple, Any
from numpy import ndarray
import matplotlib.pyplot as plt

__all__ = ['ec8_rs', 'processNGAfile', 'processTwoCfile', 'processOneCfile']


def processTwoCfile(filepath: str, scalefactor: float = 1) -> Tuple[List, List]:
    """
    This function process acceleration history data saved in Two column format.


    Parameters
    ----------
    filepath : str
        File path in order to read the file data
    scalefactor : float
        Scale factor in which the acceleration values will be multiplied.

    Returns
    -------
    time : List
        A list of time intervals
    accel : List
        A list of acceleration data points
    """
    data = np.loadtxt(filepath)
    return data[:, 0], data[:, 1] * scalefactor


def processOneCfile(filepath: str, dt: float, scalefactor: float = 1) -> Tuple[List, List, float]:
    """
    This function process acceleration history data saved in One column format.

    Parameters
    ----------
    filepath : str
        File path in order to read the file data
    dt : float
        Time interval of the acceleration history data
    scalefactor : float
        Scale factor in which the acceleration values will be multiplied.

    Returns
    -------
    time : List
        A list of time intervals
    accel : List
        A list of acceleration data points

    """
    with open(filepath, 'r') as file:
        lines = file.readlines()
    file.close()
    accel = [float(line.split('\n')[0]) * scalefactor for line in lines]
    time = [i * dt for i in range(len(accel))]
    return time, accel, dt


def processNGAfile(filepath, scalefactor: float = 1):
    """
    This function process acceleration history for NGA data file (.AT2 format)
    to a single column value and return the total number of data points and time interval of the recording.

    Parameters
    ----------
    filepath : str
        File path in order to read the file data
    scalefactor : float
        Scale factor in which the acceleration values will be multiplied.

    Returns
    -------
    time : List
        A list of time intervals
    accel : List
        A list of acceleration data points
    """
    with open(filepath, 'r') as file:
        lines = file.readlines()
    file.close()
    accel = []
    time = []
    dt_line = lines[3].split(' ')
    dt_line = [item for item in dt_line if item != '']
    dt = float(dt_line[1])
    time_counter = 0
    for value in lines[4:]:
        data = value.split(' ')
        data = [item for item in data if item != '']
        for dat in data:
            accel.append(float(dat) * scalefactor)
            time.append(dt * time_counter)
            time_counter += 1
    return time, accel, dt


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
