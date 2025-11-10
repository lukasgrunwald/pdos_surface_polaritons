"""
Normalization and Co. for the dielectric function
"""
import numpy as np
import scipy.constants as consts

def normFac(omega, wLO, wTO, epsInf = 1.0):
    frac = (wLO**2 - wTO**2) / (wTO**2 - omega**2)**2
    return epsInf * (1 + wTO**2 * frac)

def epsilon(omega, wLO, wTO, epsInf = 1.0, tau = 0.0):
    return epsInf * (wLO ** 2 - omega ** 2) / (wTO ** 2 - omega ** 2 - tau**2)

def kDFromK(kVal, omega, wLO, wTO, epsInf):
    return np.sqrt((epsilon(omega, wLO, wTO, epsInf) - 1) * omega ** 2 / consts.c ** 2 + kVal ** 2)

def kDFromKEva(kVal, omega, wLO, wTO, epsInf):
    return np.sqrt((epsilon(omega, wLO, wTO, epsInf) - 1) * omega ** 2 / consts.c ** 2 - kVal ** 2)

def kDFromKRes(kVal, omega, wLO, wTO, epsInf):
    return np.sqrt((1 - epsilon(omega, wLO, wTO, epsInf)) * omega ** 2 / consts.c ** 2 - kVal ** 2)

def kDFromKSurf(kVal, omega, wLO, wTO, epsInf):
    return np.sqrt((1 - epsilon(omega, wLO, wTO, epsInf)) * omega ** 2 / consts.c ** 2 + kVal ** 2)
