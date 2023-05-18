import numpy as np

def ec8_rs(agr, ground_type, resp_type, orientation='horizontal', importance_class=2, damping=5, periods=None):
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