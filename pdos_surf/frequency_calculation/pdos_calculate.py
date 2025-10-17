import numpy as np
import h5py

import pdos_surf.momentum_relations as epsFunc

from pdos_surf.pdos_functions import _dosTEModes as dosTE
from pdos_surf.pdos_functions import _dosTEEvaModes as dosTEEva
from pdos_surf.pdos_functions import _dosTEResModes as dosTERes
from pdos_surf.pdos_functions import _dosTMModes as dosTM
from pdos_surf.pdos_functions import _dosTMEvaModes as dosTMEva
from pdos_surf.pdos_functions import dosTMResModes as dosTMRes
from pdos_surf.pdos_functions import _dosTMSurfModes as dosTMSurf

# import plotAsOfFreq as plotFreq

# —————————————————————————————————————— Te-Modes —————————————————————————————————————— #
# All TE modes are perpendicular, such that we don't need to distinguish

def getDosTE(omega, zArr, L, wLO, wTO, epsInf):
    eps = epsFunc.epsilon(omega, wLO, wTO, epsInf)
    if(eps > 0):
        return dosTE.calcDosTE(zArr, L, omega, wLO, wTO, epsInf)
    else:
        return np.zeros(len(zArr))

def getDosTEEva(omega, zArr, L, wLO, wTO, epsInf):
    eps = epsFunc.epsilon(omega, wLO, wTO, epsInf)
    if( eps > 1 ):
        return dosTEEva.calcDosTE(zArr, L, omega, wLO, wTO, epsInf)
    else:
        return np.zeros(len(zArr))

def getDosTERes(omega, zArr, L, wLO, wTO, epsInf):
    eps = epsFunc.epsilon(omega, wLO, wTO, epsInf)
    if(eps < 1.):
        return dosTERes.calcDosTE(zArr, L, omega, wLO, wTO, epsInf)
    else:
        return np.zeros(len(zArr))

# —————————————————————————————————————— Tm-Modes —————————————————————————————————————— #

def getDosTM(omega, zArr, L, wLO, wTO, epsInf):
    """Full PDOS"""
    eps = epsFunc.epsilon(omega, wLO, wTO, epsInf)
    if(eps > 0):
        return dosTM.calcDosTM(zArr, L, omega, wLO, wTO, epsInf)
    else:
        return np.zeros(len(zArr))

def getDosTMRes(omega, zArr, L, wLO, wTO, epsInf):

    eps = epsFunc.epsilon(omega, wLO, wTO, epsInf)
    if(eps < 1.):
        return dosTMRes.calcDosTM(zArr, L, omega, wLO, wTO, epsInf)
    else:
        return np.zeros(len(zArr))

def getDosTMEva(omega, zArr, L, wLO, wTO, epsInf):
    eps = epsFunc.epsilon(omega, wLO, wTO, epsInf)
    if( eps > 1 ):
        return dosTMEva.calcDosTM(zArr, L, omega, wLO, wTO, epsInf)
    else:
        return np.zeros(len(zArr))

# —————————————————————————————————— TM-Modes resolved ————————————————————————————————— #

def getDosTMParaPerp(omega, zArr, L, wLO, wTO, epsInf):
    """Parallel and perpendicularly resolved contribution"""
    eps = epsFunc.epsilon(omega, wLO, wTO, epsInf)
    if(eps > 0):
        return dosTM.calcDosTMParaPerp(zArr, L, omega, wLO, wTO, epsInf)
    else:
        return (np.zeros(len(zArr)), np.zeros(len(zArr)))

def getDosTMEvaParaPerp(omega, zArr, L, wLO, wTO, epsInf):

    eps = epsFunc.epsilon(omega, wLO, wTO, epsInf)
    if( eps > 1 ):
        return dosTMEva.calcDosTMParaPerp(zArr, L, omega, wLO, wTO, epsInf)
    else:
        return (np.zeros(len(zArr)), np.zeros(len(zArr)))

def getDosTMResParaPerp(omega, zArr, L, wLO, wTO, epsInf):
    eps = epsFunc.epsilon(omega, wLO, wTO, epsInf)
    if(eps < 1.):
        return dosTMRes.calcDosTMParaPerp(zArr, L, omega, wLO, wTO, epsInf)
    else:
        return (np.zeros(len(zArr)), np.zeros(len(zArr)))

# ——————————————————————————————————————— Surface —————————————————————————————————————— #
#! I think the surface implementation currently takes both the parallel and the orthogonal one!
def getDosTMSurf(omega, zArr, L, wLO, wTO, epsInf):
    wInf = np.sqrt(epsInf * wLO**2 + wTO**2) / np.sqrt(epsInf + 1)
    if(omega < wInf and omega > wTO):
        return dosTMSurf.calcDosTM(zArr, L, omega, wLO, wTO, epsInf)
    else:
        return np.zeros(len(zArr))

def produceFreqDataSurf(omegaArr, zArr, L, wLO, wTO, epsInf, filename):

    dosTMSurfVals = getDosTMSurf(zArr, L, omegaArr, wLO, wTO, epsInf)

    dosSurf = dosTMSurfVals
    h5f = h5py.File(filename, 'w')
    h5f.create_dataset('dosTM', data=dosSurf)
    h5f.close()
