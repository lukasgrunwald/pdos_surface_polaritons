"""
Relations between kz values above and below the substrate surface. These are give
"""
import numpy as np
import scipy.constants as consts

def epsilon(omega, wLO, wTO, epsInf = 1.0, tau = 0.0):
    return epsInf * (wLO ** 2 - omega ** 2) / (wTO ** 2 - omega ** 2 - tau**2)

def normFac(omega, wLO, wTO, epsInf = 1.0):
    frac = (wLO**2 - wTO**2) / (wTO**2 - omega**2)**2
    return epsInf * (1 + wTO**2 * frac)

def kDFromK(kVal, omega, wLO, wTO, epsInf):
    return np.sqrt((epsilon(omega, wLO, wTO, epsInf) - 1) * omega ** 2 / consts.c ** 2 + kVal ** 2)

def kDFromKEva(kVal, omega, wLO, wTO, epsInf):
    return np.sqrt((epsilon(omega, wLO, wTO, epsInf) - 1) * omega ** 2 / consts.c ** 2 - kVal ** 2)

def kDFromKRes(kVal, omega, wLO, wTO, epsInf):
    return np.sqrt((1 - epsilon(omega, wLO, wTO, epsInf)) * omega ** 2 / consts.c ** 2 - kVal ** 2)

def kDFromKSurf(kVal, omega, wLO, wTO, epsInf):
    return np.sqrt((1 - epsilon(omega, wLO, wTO, epsInf)) * omega ** 2 / consts.c ** 2 + kVal ** 2)
