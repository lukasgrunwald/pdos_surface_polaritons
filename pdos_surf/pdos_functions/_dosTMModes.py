import numpy as np
import matplotlib.pyplot as plt
from matplotlib import gridspec
from scipy.integrate import quad

from matplotlib import ticker
from matplotlib.colors import LogNorm
import matplotlib.patches as patches
import matplotlib as mpl
import matplotlib.cm as cm
import matplotlib.colors
import h5py
from matplotlib import gridspec
from matplotlib.patches import ConnectionPatch
import scipy.integrate as integrate
import scipy.optimize as opt
import scipy.constants as consts
from time import perf_counter


import pdos_surf.allowed_kz as allowed_kz
import pdos_surf.momentum_relations as epsFunc

fontsize = 8

mpl.rcParams['font.family'] = 'Helvetica'
mpl.rcParams['lines.linewidth'] = 2
mpl.rcParams['lines.markersize'] = 8
mpl.rcParams['font.size'] = 8  # <-- change fonsize globally
mpl.rcParams['legend.fontsize'] = 8
mpl.rcParams['axes.titlesize'] = 8
mpl.rcParams['axes.labelsize'] = 8
mpl.rcParams['xtick.major.size'] = 3
mpl.rcParams['ytick.major.size'] = 3
mpl.rcParams['xtick.major.width'] = .5
mpl.rcParams['ytick.major.width'] = .5
mpl.rcParams['xtick.direction'] = 'out'
mpl.rcParams['ytick.direction'] = 'out'
mpl.rcParams['figure.titlesize'] = 8
mpl.rc('text', usetex=True)

mpl.rcParams['text.latex.preamble'] =  r'\usepackage[helvet]{sfmath}'


def norm_sqrt(kVal, L, omega, wLO, wTO, epsInf):
    kDVal = epsFunc.kDFromK(kVal, omega, wLO, wTO, epsInf)
    normPrefac = epsFunc.normFac(omega, wLO, wTO, epsInf)
    eps = epsFunc.epsilon(omega, wLO, wTO, epsInf)
    brack11 = omega**2 / (consts.c**2 * kVal**2)
    brack12 = (omega**2 / (consts.c**2 * kVal**2) - 2) * np.sin(kVal * L) / (kVal * L)
    term1 = (brack11 + brack12) * np.sin(kDVal * L / 2.)**2
    brack21 = eps * omega**2 / (consts.c**2 * kDVal**2)
    brack22 = (eps * omega**2 / (consts.c**2 * kDVal**2) - 2) * np.sin(kDVal * L) / (kDVal * L)
    term2 = normPrefac * (brack21 + brack22) * np.sin(kVal * L / 2.)**2

    return L / 4. * (term1 + term2)

def waveFunctionPosPara(zArr, kArr, kzArrDel, L, omega, wLO, wTO, epsInf):
    NSqr = norm_sqrt(kArr, L, omega, wLO, wTO, epsInf)
    #kzArrDel = findAllowedKsSPhP.findKsDerivativeW(kArr, L, omega, wLO, wTO, epsInf, "TM")
    kDArr = epsFunc.kDFromK(kArr, omega, wLO, wTO, epsInf)
    func = np.sin(kArr[None, :] * (L / 2 - zArr[:, None])) * np.sin(kDArr[None, :] * L / 2.)
    diffFac = (1. - consts.c ** 2 * kArr[None, :] / omega * kzArrDel[None, :])
    return np.sum(1. / NSqr[None, :] * func**2 * diffFac, axis = 1)

def waveFunctionNegPara(zArr, kArr, kzArrDel, L, omega, wLO, wTO, epsInf):
    NSqr = norm_sqrt(kArr, L, omega, wLO, wTO, epsInf)
    #kzArrDel = findAllowedKsSPhP.findKsDerivativeW(kArr, L, omega, wLO, wTO, epsInf, "TM")
    kDArr = epsFunc.kDFromK(kArr, omega, wLO, wTO, epsInf)
    func = np.sin(kDArr[None, :] * (L / 2. + zArr[:, None])) * np.sin(kArr[None, :] * L / 2.)
    diffFac = (1. - consts.c ** 2 * kArr[None, :] / omega * kzArrDel[None, :])
    return np.sum(1. / NSqr[None, :] * func**2 * diffFac, axis = 1)

def waveFunctionPosPerp(zArr, kArr, kzArrDel, L, omega, wLO, wTO, epsInf):
    NSqr = norm_sqrt(kArr, L, omega, wLO, wTO, epsInf)
    #kzArrDel = findAllowedKsSPhP.findKsDerivativeW(kArr, L, omega, wLO, wTO, epsInf, "TM")
    kDArr = epsFunc.kDFromK(kArr, omega, wLO, wTO, epsInf)
    func = np.sqrt(omega ** 2 / (consts.c ** 2 * kArr[None, :] ** 2) - 1) * np.cos(kArr[None, :] * (L / 2 - zArr[:, None])) * np.sin(kDArr[None, :] * L / 2.)
    diffFac = (1. - consts.c ** 2 * kArr[None, :] / omega * kzArrDel[None, :])
    return np.sum(1. / NSqr[None, :] * func**2 * diffFac, axis = 1)

def waveFunctionNegPerp(zArr, kArr, kzArrDel, L, omega, wLO, wTO, epsInf):
    NSqr = norm_sqrt(kArr, L, omega, wLO, wTO, epsInf)
    #kzArrDel = findAllowedKsSPhP.findKsDerivativeW(kArr, L, omega, wLO, wTO, epsInf, "TM")
    kDArr = epsFunc.kDFromK(kArr, omega, wLO, wTO, epsInf)
    eps = epsFunc.epsilon(omega, wLO, wTO, epsInf)
    func = - np.sqrt(eps * omega**2 / (consts.c**2 * kDArr[None, :]**2) - 1) * np.cos(kDArr[None, :] * (L / 2. + zArr[:, None])) * np.sin(kArr[None, :] * L / 2.)
    diffFac = (1. - consts.c ** 2 * kArr[None, :] / omega * kzArrDel[None, :])
    return np.sum(1. / NSqr[None, :] * func**2 * diffFac, axis = 1)

def calcDosTM(zArr, L, omega, wLO, wTO, epsInf):

    t_start = perf_counter()
    kArr = allowed_kz.computeAllowedKs(L, omega, wLO, wTO, epsInf, "TM")
    kzArrDel = allowed_kz.findKsDerivativeW(kArr, L, omega, wLO, wTO, epsInf, "TM")
    t_stop = perf_counter()
    #print("Time to find kVals TM: {}".format(t_stop - t_start))

    #print("len kArr TM = {}".format(len(kArr)))

    if(len(kArr) == 0):
        return 0

    indNeg = np.where(zArr < 0)
    indPos = np.where(zArr >= 0)
    zPosArr = zArr[indPos]
    zNegArr = zArr[indNeg]

    t_start = perf_counter()
    dosPos = waveFunctionPosPara(zPosArr, kArr, kzArrDel, L, omega, wLO, wTO, epsInf)
    dosNeg = waveFunctionNegPara(zNegArr, kArr, kzArrDel, L, omega, wLO, wTO, epsInf)
    dosPos += waveFunctionPosPerp(zPosArr, kArr, kzArrDel, L, omega, wLO, wTO, epsInf)
    dosNeg += waveFunctionNegPerp(zNegArr, kArr, kzArrDel, L, omega, wLO, wTO, epsInf)
    t_stop = perf_counter()
    #print("Time to find perform dos sums TM: {}".format(t_stop - t_start))
    dos = np.pi * consts.c / (2. * omega) * np.append(dosNeg, dosPos)
    return dos

def calcDosTMParaPerp(zArr, L, omega, wLO, wTO, epsInf):

    t_start = perf_counter()
    kArr = allowed_kz.computeAllowedKs(L, omega, wLO, wTO, epsInf, "TM")
    kzArrDel = allowed_kz.findKsDerivativeW(kArr, L, omega, wLO, wTO, epsInf, "TM")
    t_stop = perf_counter()
    #print("Time to find kVals TM: {}".format(t_stop - t_start))

    #print("len kArr TM = {}".format(len(kArr)))

    if(len(kArr) == 0):
        return 0

    indNeg = np.where(zArr < 0)
    indPos = np.where(zArr >= 0)
    zPosArr = zArr[indPos]
    zNegArr = zArr[indNeg]

    t_start = perf_counter()
    dosPosPara = waveFunctionPosPara(zPosArr, kArr, kzArrDel, L, omega, wLO, wTO, epsInf)
    dosNegPara = waveFunctionNegPara(zNegArr, kArr, kzArrDel, L, omega, wLO, wTO, epsInf)
    dosPosPerp = waveFunctionPosPerp(zPosArr, kArr, kzArrDel, L, omega, wLO, wTO, epsInf)
    dosNegPerp = waveFunctionNegPerp(zNegArr, kArr, kzArrDel, L, omega, wLO, wTO, epsInf)
    t_stop = perf_counter()
    #print("Time to find perform dos sums TM: {}".format(t_stop - t_start))
    dosPara = np.pi * consts.c / (2. * omega) * np.append(dosNegPara, dosPosPara)
    dosPerp = np.pi * consts.c / (2. * omega) * np.append(dosNegPerp, dosPosPerp)
    return (dosPara, dosPerp)

def plotDosTMSPhP(zArr, dos, L, omega, wLO, wTO, epsInf):

    fig = plt.figure(figsize=(3., 2.), dpi=800)
    gs = gridspec.GridSpec(1, 1,
                           wspace=0.35, hspace=0., top=0.9, bottom=0.25, left=0.22, right=0.96)
    ax = plt.subplot(gs[0, 0])

    eps = epsFunc.epsilon(omega, wLO, wTO, epsInf)

    ax.plot(zArr, dos, color='peru', lw=1., label = "DOS from Box")
    ax.axhline(0.5, lw = 0.5, color = 'gray', zorder = -666)
    ax.axhline(0.5 * np.sqrt(eps)**1, lw = 0.5, color = 'gray')

    ax.axvline(0., lw = 0.5, color = 'gray')

    ax.set_xlim(np.amin(zArr), np.amax(zArr))


    ax.set_xticks([np.amin(zArr), 0, np.amax(zArr)])
    ax.set_xticklabels([r"$-\frac{L}{2}$", r"$0$", r"$\frac{L}{2}$"])

    ax.set_xlabel(r"$z[\frac{c}{\omega}]$")
    ax.set_ylabel(r"$\rho / \rho_0$")

    #legend = ax.legend(fontsize=fontsize, loc='lower right', bbox_to_anchor=(1.0, 0.1), edgecolor='black', ncol=1)
    #legend.get_frame().set_alpha(0.)
    #legend.get_frame().set_boxstyle('Square', pad=0.1)
    #legend.get_frame().set_linewidth(0.0)

    plt.savefig("./SPhPPlotsSaved/dosTM.png")

def createPlotDosTM():

    omega = 1 * 1e11
    wLO = 3. * 1e12
    wTO = 1. * 1e12
    epsInf = 2.
    L = .5

    zArr = np.linspace(-L / 2., L / 2., 1000)

    dos = calcDosTM(zArr, L, omega, wLO, wTO, epsInf)
    plotDosTMSPhP(zArr, dos, L, omega, wLO, wTO, epsInf)
